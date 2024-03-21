package app.aaps.plugins.aps.openAPSAIMI

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.LongSparseArray
import androidx.preference.PreferenceCategory
import androidx.preference.PreferenceFragmentCompat
import androidx.preference.PreferenceManager
import androidx.preference.PreferenceScreen
import androidx.preference.SwitchPreference
import app.aaps.core.data.aps.SMBDefaults
import app.aaps.core.data.model.GlucoseUnit
import app.aaps.core.data.plugin.PluginType
import app.aaps.core.data.time.T
import app.aaps.core.interfaces.aps.APS
import app.aaps.core.interfaces.aps.APSResult
import app.aaps.core.interfaces.aps.AutosensResult
import app.aaps.core.interfaces.aps.CurrentTemp
import app.aaps.core.interfaces.aps.OapsProfile
import app.aaps.core.interfaces.bgQualityCheck.BgQualityCheck
import app.aaps.core.interfaces.configuration.Config
import app.aaps.core.interfaces.constraints.Constraint
import app.aaps.core.interfaces.constraints.ConstraintsChecker
import app.aaps.core.interfaces.constraints.PluginConstraints
import app.aaps.core.interfaces.db.PersistenceLayer
import app.aaps.core.interfaces.db.ProcessedTbrEbData
import app.aaps.core.interfaces.iob.GlucoseStatusProvider
import app.aaps.core.interfaces.iob.IobCobCalculator
import app.aaps.core.interfaces.logging.AAPSLogger
import app.aaps.core.interfaces.logging.LTag
import app.aaps.core.interfaces.notifications.Notification
import app.aaps.core.interfaces.plugin.ActivePlugin
import app.aaps.core.interfaces.plugin.PluginBase
import app.aaps.core.interfaces.plugin.PluginDescription
import app.aaps.core.interfaces.profile.Profile
import app.aaps.core.interfaces.profile.ProfileFunction
import app.aaps.core.interfaces.profile.ProfileUtil
import app.aaps.core.interfaces.profiling.Profiler
import app.aaps.core.interfaces.resources.ResourceHelper
import app.aaps.core.interfaces.rx.bus.RxBus
import app.aaps.core.interfaces.rx.events.EventAPSCalculationFinished
import app.aaps.core.interfaces.stats.TddCalculator
import app.aaps.core.interfaces.ui.UiInteraction
import app.aaps.core.interfaces.utils.DateUtil
import app.aaps.core.interfaces.utils.HardLimits
import app.aaps.core.interfaces.utils.Round
import app.aaps.core.keys.AdaptiveIntentPreference
import app.aaps.core.keys.BooleanKey
import app.aaps.core.keys.DoubleKey
import app.aaps.core.keys.IntKey
import app.aaps.core.keys.IntentKey
import app.aaps.core.keys.Preferences
import app.aaps.core.keys.UnitDoubleKey
import app.aaps.core.objects.aps.DetermineBasalResult
import app.aaps.core.objects.constraints.ConstraintObject
import app.aaps.core.objects.extensions.convertedToAbsolute
import app.aaps.core.objects.extensions.getPassedDurationToTimeInMinutes
import app.aaps.core.objects.extensions.plannedRemainingMinutes
import app.aaps.core.objects.extensions.put
import app.aaps.core.objects.extensions.store
import app.aaps.core.objects.extensions.target
import app.aaps.core.utils.MidnightUtils
import app.aaps.core.validators.AdaptiveDoublePreference
import app.aaps.core.validators.AdaptiveIntPreference
import app.aaps.core.validators.AdaptiveUnitPreference
import app.aaps.core.validators.AdaptiveSwitchPreference
import app.aaps.plugins.aps.OpenAPSFragment
import app.aaps.plugins.aps.R
import app.aaps.plugins.aps.events.EventOpenAPSUpdateGui
import app.aaps.plugins.aps.events.EventResetOpenAPSGui
import app.aaps.plugins.aps.openAPS.TddStatus
import dagger.android.HasAndroidInjector
import org.json.JSONObject
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.floor
import kotlin.math.ln

@Singleton
open class OpenAPSAIMIPlugin  @Inject constructor(
    private val injector: HasAndroidInjector,
    aapsLogger: AAPSLogger,
    private val rxBus: RxBus,
    private val constraintsChecker: ConstraintsChecker,
    rh: ResourceHelper,
    private val profileFunction: ProfileFunction,
    private val profileUtil: ProfileUtil,
    config: Config,
    private val activePlugin: ActivePlugin,
    private val iobCobCalculator: IobCobCalculator,
    private val hardLimits: HardLimits,
    private val preferences: Preferences,
    protected val dateUtil: DateUtil,
    private val processedTbrEbData: ProcessedTbrEbData,
    private val persistenceLayer: PersistenceLayer,
    private val glucoseStatusProvider: GlucoseStatusProvider,
    private val tddCalculator: TddCalculator,
    private val bgQualityCheck: BgQualityCheck,
    private val uiInteraction: UiInteraction,
    private val determineBasalAIMI: DetermineBasalaimiSMB,
    private val profiler: Profiler,
) : PluginBase(
    PluginDescription()
        .mainType(PluginType.APS)
        .fragmentClass(OpenAPSFragment::class.java.name)
        .pluginIcon(app.aaps.core.ui.R.drawable.ic_generic_icon)
        .pluginName(R.string.openapsaimi)
        .shortName(R.string.oaps_aimi_shortname)
        .preferencesId(PluginDescription.PREFERENCE_SCREEN)
        .preferencesVisibleInSimpleMode(false)
        .showInList(config.APS)
        .description(R.string.description_openapsaimi)
        .setDefault(),
    aapsLogger, rh
), APS, PluginConstraints {

    // last values
    override var lastAPSRun: Long = 0
    override val algorithm = APSResult.Algorithm.SMB
    override var lastAPSResult: DetermineBasalResult? = null
    override fun supportsDynamicIsf(): Boolean = preferences.get(BooleanKey.ApsUseDynamicSensitivity)

    /*override fun getIsfMgdl(multiplier: Double, timeShift: Int, caller: String): Double? {
        val start = dateUtil.now()
        val sensitivity = calculateVariableIsf(start, bg = null)
        if (sensitivity.second == null)
            uiInteraction.addNotificationValidTo(
                Notification.DYN_ISF_FALLBACK, start,
                rh.gs(R.string.fallback_to_isf_no_tdd), Notification.INFO, dateUtil.now() + T.mins(1).msecs()
            )
        else
            uiInteraction.dismissNotification(Notification.DYN_ISF_FALLBACK)
        profiler.log(LTag.APS, String.format("getIsfMgdl() %s %f %s %s", sensitivity.first, sensitivity.second, dateUtil.dateAndTimeAndSecondsString(start), caller), start)
        return sensitivity.second?.let { it * multiplier }
    }*/
    override fun getIsfMgdl(multiplier: Double, timeShift: Int, caller: String): Double? {
        val start = dateUtil.now()
        val sensitivity = calculateVariableIsf(start, bg = null)
        /*if (sensitivity.second == null) {
            aapsLogger.debug(LTag.APS, "getIsfMgdl $caller falling back to static ISF due to no TDD data")
        } else {
            uiInteraction.dismissNotification(Notification.DYN_ISF_FALLBACK)
        }*/
        uiInteraction.dismissNotification(Notification.DYN_ISF_FALLBACK)
        profiler.log(LTag.APS, String.format("getIsfMgdl() %s %f %s %s", sensitivity.first, sensitivity.second, dateUtil.dateAndTimeAndSecondsString(start), caller), start)
        return sensitivity.second?.let { it * multiplier }
    }

    override fun getIsfMgdl(timestamp: Long, bg: Double, multiplier: Double, timeShift: Int, caller: String): Double? {
        val start = dateUtil.now()
        val sensitivity = calculateVariableIsf(timestamp, bg)
        profiler.log(LTag.APS, String.format("getIsfMgdl() %s %f %s %s", sensitivity.first, sensitivity.second, dateUtil.dateAndTimeAndSecondsString(timestamp), caller), start)
        return sensitivity.second?.let { it * multiplier }
    }

    override fun specialEnableCondition(): Boolean {
        return try {
            activePlugin.activePump.pumpDescription.isTempBasalCapable
        } catch (ignored: Exception) {
            // may fail during initialization
            true
        }
    }

    override fun specialShowInListCondition(): Boolean {
        val pump = activePlugin.activePump
        return pump.pumpDescription.isTempBasalCapable
    }

    override fun preprocessPreferences(preferenceFragment: PreferenceFragmentCompat) {
        super.preprocessPreferences(preferenceFragment)
        val uamEnabled = if (preferences.get(BooleanKey.ApsUseSmb)) {
            preferences.get(BooleanKey.ApsUseUam)
        } else {
            preferences.get(BooleanKey.ApsUseSmb)
        }
        val smbAlwaysEnabled = if (preferences.get(BooleanKey.ApsUseSmb)) {
            preferences.get(BooleanKey.ApsUseSmbAlways)
        } else {
            !preferences.get(BooleanKey.ApsUseSmb)
        }
        val advancedFiltering = activePlugin.activeBgSource.advancedFilteringSupported()
        val autoSensOrDynIsfSensEnabled = if (preferences.get(BooleanKey.ApsUseDynamicSensitivity)) {
            preferences.get(BooleanKey.ApsDynIsfAdjustSensitivity)
        } else {
            preferences.get(BooleanKey.ApsUseAutosens)
        }
        preferenceFragment.findPreference<SwitchPreference>(rh.gs(app.aaps.core.keys.R.string.key_openaps_allow_smb_with_COB))?.isVisible = !smbAlwaysEnabled || !advancedFiltering
        preferenceFragment.findPreference<SwitchPreference>(rh.gs(app.aaps.core.keys.R.string.key_openaps_allow_smb_with_low_temp_target))?.isVisible = !smbAlwaysEnabled || !advancedFiltering
        preferenceFragment.findPreference<SwitchPreference>(rh.gs(app.aaps.core.keys.R.string.key_openaps_enable_smb_after_carbs))?.isVisible = !smbAlwaysEnabled || !advancedFiltering
        preferenceFragment.findPreference<SwitchPreference>(rh.gs(app.aaps.core.keys.R.string.key_openaps_resistance_lowers_target))?.isVisible = autoSensOrDynIsfSensEnabled
        preferenceFragment.findPreference<SwitchPreference>(rh.gs(app.aaps.core.keys.R.string.key_openaps_sensitivity_raises_target))?.isVisible = autoSensOrDynIsfSensEnabled
        preferenceFragment.findPreference<AdaptiveIntPreference>(rh.gs(app.aaps.core.keys.R.string.key_openaps_uam_smb_max_minutes))?.isVisible = uamEnabled
    }
    private fun adjustFactorsdynisfBasedOnBgAndHypo(
        dynISFadjust: Double
    ): Float {
        var bg = glucoseStatusProvider.glucoseStatusData
        val hypoAdjustment = if (bg!!.glucose < 110 || iobCobCalculator.calculateIobFromBolus().iob > 3 * preferences.get(DoubleKey.OApsAIMIMaxSMB)) 0.8f else 1.0f // Réduire les facteurs si hypo récente
        val factorAdjustment = if ( bg.glucose < 120) 0.1f else 0.2f
        val bgAdjustment = 1.0f + (Math.log(Math.abs(bg.delta) + 1) - 1)  * factorAdjustment
        val isfadjust = if (bg.delta < 0) {bgAdjustment / dynISFadjust} else {dynISFadjust * bgAdjustment * hypoAdjustment}
        return isfadjust.toFloat()
    }
    private val dynIsfCache = LongSparseArray<Double>()
    private fun calculateVariableIsf(timestamp: Long, bg: Double?): Pair<String, Double?> {
        if (!preferences.get(BooleanKey.ApsUseDynamicSensitivity)) return Pair("OFF", null)

        val result = persistenceLayer.getApsResultCloseTo(timestamp)
        if (result?.variableSens != null) {
            //aapsLogger.debug("calculateVariableIsf $caller DB  ${dateUtil.dateAndTimeAndSecondsString(timestamp)} ${result.variableSens}")
            return Pair("DB", result.variableSens)
        }

        val glucose = bg ?: glucoseStatusProvider.glucoseStatusData?.glucose ?: return Pair("GLUC", null)
        // Round down to 30 min and use it as a key for caching
        // Add BG to key as it affects calculation
        val key = timestamp - timestamp % T.mins(30).msecs() + glucose.toLong()
        val cached = dynIsfCache[key]
        if (cached != null && timestamp < dateUtil.now()) {
            //aapsLogger.debug("calculateVariableIsf $caller HIT ${dateUtil.dateAndTimeAndSecondsString(timestamp)} $cached")
            return Pair("HIT", cached)
        }
        val tdd7P: Double = preferences.get(DoubleKey.OApsAIMITDD7)
        // no cached result found, let's calculate the value
        val tdd1D = tddCalculator.averageTDD(tddCalculator.calculate(timestamp, 1, allowMissingDays = false))?.data?.totalAmount ?: tdd7P
        val tdd7D = tddCalculator.averageTDD(tddCalculator.calculate(timestamp, 7, allowMissingDays = false))?.data?.totalAmount ?: tdd7P
        val tddLast24H = tddCalculator.calculateDaily(-24, 0)?.totalAmount ?: tdd7P
        val tddLast4H = tddCalculator.calculateDaily(-4, 0)?.totalAmount ?: ((tdd7P / 24) * 4)
        val tddLast8to4H = tddCalculator.calculateDaily(-8, -4)?.totalAmount ?: ((tdd7P / 24) * 4)
        val insulinDivisor = when {
            activePlugin.activeInsulin.peak > 65 -> 55 // lyumjev peak: 45
            activePlugin.activeInsulin.peak > 50 -> 65 // ultra rapid peak: 55
            else                                 -> 75 // rapid peak: 75
        }
        val dynISFadjust: Double = (preferences.get(IntKey.OApsAIMIDynISFAdjustment).toDouble() / 100.0)
        val dynISFadjusthyper: Double = (preferences.get(IntKey.OApsAIMIDynISFAdjustmentHyper).toDouble() / 100.0)
        val mealTimeDynISFAdjFactor: Double = (preferences.get(IntKey.OApsAIMImealAdjISFFact).toDouble() / 100.0)
        val snackTimeDynISFAdjFactor: Double = (preferences.get(IntKey.OApsAIMISnackAdjISFFact).toDouble() / 100.0)
        val sleepTimeDynISFAdjFactor: Double = (preferences.get(IntKey.OApsAIMIsleepAdjISFFact).toDouble() / 100.0)
        val hcTimeDynISFAdjFactor: Double = (preferences.get(IntKey.OApsAIMIHighCarbAdjISFFact).toDouble() / 100.0)
        val therapy = Therapy(persistenceLayer).also {
            it.updateStatesBasedOnTherapyEvents()
        }

        val sleepTime = therapy.sleepTime
        val snackTime = therapy.snackTime
        val sportTime = therapy.sportTime
        val lowCarbTime = therapy.lowCarbTime
        val highCarbTime = therapy.highCarbTime
        val mealTime = therapy.mealTime

        val tddStatus = TddStatus(tdd1D, tdd7D, tddLast24H, tddLast4H, tddLast8to4H)
        val tddWeightedFromLast8H = ((1.4 * tddStatus.tddLast4H) + (0.6 * tddStatus.tddLast8to4H)) * 3
        //val tdd = (tddWeightedFromLast8H * 0.33) + (tddStatus.tdd7D * 0.34) + (tddStatus.tdd1D * 0.33) * preferences.get(IntKey.ApsDynIsfAdjustmentFactor) / 100.0
        var tdd = (tddWeightedFromLast8H * 0.33) + (tddStatus.tdd7D * 0.34) + (tddStatus.tdd1D * 0.33)
        if (bg != null) {
            tdd = when {
                sportTime -> tdd * 1.1
                sleepTime -> tdd * adjustFactorsdynisfBasedOnBgAndHypo(sleepTimeDynISFAdjFactor)
                lowCarbTime -> tdd * 1.1
                snackTime -> tdd * adjustFactorsdynisfBasedOnBgAndHypo(snackTimeDynISFAdjFactor)
                highCarbTime -> tdd * adjustFactorsdynisfBasedOnBgAndHypo(hcTimeDynISFAdjFactor)
                mealTime -> tdd * adjustFactorsdynisfBasedOnBgAndHypo(mealTimeDynISFAdjFactor)
                bg > 180 -> tdd * adjustFactorsdynisfBasedOnBgAndHypo(dynISFadjusthyper)
                else -> tdd * adjustFactorsdynisfBasedOnBgAndHypo(dynISFadjust)
            }
        }
        val isfMgdl = profileFunction.getProfile()?.getProfileIsfMgdl()
        var sensitivity = Round.roundTo(1800 / (tdd * (ln((glucose / insulinDivisor) + 1))), 0.1)
        if (sensitivity < 0) if (isfMgdl != null) {
            sensitivity = profileUtil.fromMgdlToUnits(isfMgdl.toDouble(), profileFunction.getUnits())
        }
        //aapsLogger.debug("calculateVariableIsf $caller CAL ${dateUtil.dateAndTimeAndSecondsString(timestamp)} $sensitivity")
        dynIsfCache.put(key, sensitivity)
        if (dynIsfCache.size() > 1000) dynIsfCache.clear()
        return Pair("CALC", sensitivity)
    }
    override fun invoke(initiator: String, tempBasalFallback: Boolean) {
        aapsLogger.debug(LTag.APS, "invoke from $initiator tempBasalFallback: $tempBasalFallback")
        lastAPSResult = null
        val glucoseStatus = glucoseStatusProvider.glucoseStatusData
        val profile = profileFunction.getProfile()
        val pump = activePlugin.activePump

        if (profile == null) {
            rxBus.send(EventResetOpenAPSGui(rh.gs(app.aaps.core.ui.R.string.no_profile_set)))
            aapsLogger.debug(LTag.APS, rh.gs(app.aaps.core.ui.R.string.no_profile_set))
            return
        }
        if (!isEnabled()) {
            rxBus.send(EventResetOpenAPSGui(rh.gs(R.string.openapsma_disabled)))
            aapsLogger.debug(LTag.APS, rh.gs(R.string.openapsma_disabled))
            return
        }
        if (glucoseStatus == null) {
            rxBus.send(EventResetOpenAPSGui(rh.gs(R.string.openapsma_no_glucose_data)))
            aapsLogger.debug(LTag.APS, rh.gs(R.string.openapsma_no_glucose_data))
            return
        }

        val inputConstraints = ConstraintObject(0.0, aapsLogger) // fake. only for collecting all results

        if (!hardLimits.checkHardLimits(profile.dia, app.aaps.core.ui.R.string.profile_dia, hardLimits.minDia(), hardLimits.maxDia())) return
        if (!hardLimits.checkHardLimits(
                profile.getIcTimeFromMidnight(MidnightUtils.secondsFromMidnight()),
                app.aaps.core.ui.R.string.profile_carbs_ratio_value,
                hardLimits.minIC(),
                hardLimits.maxIC()
            )
        ) return
        if (!hardLimits.checkHardLimits(profile.getIsfMgdl("OpenAPSAIMIPlugin"), app.aaps.core.ui.R.string.profile_sensitivity_value, HardLimits.MIN_ISF, HardLimits.MAX_ISF)) return
        if (!hardLimits.checkHardLimits(profile.getMaxDailyBasal(), app.aaps.core.ui.R.string.profile_max_daily_basal_value, 0.02, hardLimits.maxBasal())) return
        if (!hardLimits.checkHardLimits(pump.baseBasalRate, app.aaps.core.ui.R.string.current_basal_value, 0.01, hardLimits.maxBasal())) return

        // End of check, start gathering data

        val dynIsfMode = preferences.get(BooleanKey.ApsUseDynamicSensitivity)
        val smbEnabled = preferences.get(BooleanKey.ApsUseSmb)
        val advancedFiltering = constraintsChecker.isAdvancedFilteringEnabled().also { inputConstraints.copyReasons(it) }.value()

        val now = dateUtil.now()
        val tb = processedTbrEbData.getTempBasalIncludingConvertedExtended(now)
        val currentTemp = CurrentTemp(
            duration = tb?.plannedRemainingMinutes ?: 0,
            rate = tb?.convertedToAbsolute(now, profile) ?: 0.0,
            minutesrunning = tb?.getPassedDurationToTimeInMinutes(now)
        )
        var minBg = hardLimits.verifyHardLimits(Round.roundTo(profile.getTargetLowMgdl(), 0.1), app.aaps.core.ui.R.string.profile_low_target, HardLimits.LIMIT_MIN_BG[0], HardLimits.LIMIT_MIN_BG[1])
        var maxBg = hardLimits.verifyHardLimits(Round.roundTo(profile.getTargetHighMgdl(), 0.1), app.aaps.core.ui.R.string.profile_high_target, HardLimits.LIMIT_MAX_BG[0], HardLimits.LIMIT_MAX_BG[1])
        var targetBg = hardLimits.verifyHardLimits(profile.getTargetMgdl(), app.aaps.core.ui.R.string.temp_target_value, HardLimits.LIMIT_TARGET_BG[0], HardLimits.LIMIT_TARGET_BG[1])
        var isTempTarget = false
        persistenceLayer.getTemporaryTargetActiveAt(dateUtil.now())?.let { tempTarget ->
            isTempTarget = true
            minBg = hardLimits.verifyHardLimits(tempTarget.lowTarget, app.aaps.core.ui.R.string.temp_target_low_target, HardLimits.LIMIT_TEMP_MIN_BG[0], HardLimits.LIMIT_TEMP_MIN_BG[1])
            maxBg = hardLimits.verifyHardLimits(tempTarget.highTarget, app.aaps.core.ui.R.string.temp_target_high_target, HardLimits.LIMIT_TEMP_MAX_BG[0], HardLimits.LIMIT_TEMP_MAX_BG[1])
            targetBg = hardLimits.verifyHardLimits(tempTarget.target(), app.aaps.core.ui.R.string.temp_target_value, HardLimits.LIMIT_TEMP_TARGET_BG[0], HardLimits.LIMIT_TEMP_TARGET_BG[1])
        }
        val insulin = activePlugin.activeInsulin
        val insulinDivisor = when {
            insulin.peak > 65 -> 55 // rapid peak: 75
            insulin.peak > 50 -> 65 // ultra rapid peak: 55
            else              -> 75 // lyumjev peak: 45
        }

        var autosensResult = AutosensResult()
        var variableSensitivity = 0.0
        var tdd = 0.0
        if (dynIsfMode) {
            // DynamicISF specific
            // without these values DynISF doesn't work properly
            // Current implementation is fallback to SMB if TDD history is not available. Thus calculated here

            val tdd7P: Double = preferences.get(DoubleKey.OApsAIMITDD7)

            var tdd7D = tddCalculator.averageTDD(tddCalculator.calculate(7, allowMissingDays = false))
            //if (tdd7D == 0.0 || tdd7D < tdd7P) tdd7D = tdd7P

            var tdd2Days = tddCalculator.averageTDD(tddCalculator.calculate(2, allowMissingDays = false))?.data?.totalAmount ?: 0.0
            if (tdd2Days == 0.0 || tdd2Days < tdd7P) tdd2Days = tdd7P
            val tdd2DaysPerHour = tdd2Days / 24

            val tddLast4H = tdd2DaysPerHour * 4

            var tddDaily = tddCalculator.averageTDD(tddCalculator.calculate(1, allowMissingDays = false))?.data?.totalAmount ?: 0.0
            if (tddDaily == 0.0 || tddDaily < tdd7P / 2) tddDaily = tdd7P

            var tdd24Hrs = tddCalculator.calculateDaily(-24, 0)?.totalAmount ?: 0.0
            if (tdd24Hrs == 0.0) tdd24Hrs = tdd7P
            val tdd24HrsPerHour = tdd24Hrs / 24
            val tddLast24H = tddCalculator.calculateDaily(-24, 0)
            val tddLast8to4H = tdd24HrsPerHour * 4
            val bg = glucoseStatusProvider.glucoseStatusData?.glucose

            val dynISFadjust: Double = (preferences.get(IntKey.OApsAIMIDynISFAdjustment).toDouble() / 100.0)
            val dynISFadjusthyper: Double = (preferences.get(IntKey.OApsAIMIDynISFAdjustmentHyper).toDouble() / 100.0)
            val mealTimeDynISFAdjFactor: Double = (preferences.get(IntKey.OApsAIMImealAdjISFFact).toDouble() / 100.0)
            val snackTimeDynISFAdjFactor: Double = (preferences.get(IntKey.OApsAIMISnackAdjISFFact).toDouble() / 100.0)
            val sleepTimeDynISFAdjFactor: Double = (preferences.get(IntKey.OApsAIMIsleepAdjISFFact).toDouble() / 100.0)
            val hcTimeDynISFAdjFactor: Double = (preferences.get(IntKey.OApsAIMIHighCarbAdjISFFact).toDouble() / 100.0)

            val therapy = Therapy(persistenceLayer).also {
                it.updateStatesBasedOnTherapyEvents()
            }
            val sleepTime = therapy.sleepTime
            val snackTime = therapy.snackTime
            val sportTime = therapy.sportTime
            val lowCarbTime = therapy.lowCarbTime
            val highCarbTime = therapy.highCarbTime
            val mealTime = therapy.mealTime
            val tddWeightedFromLast8H = ((1.4 * tddLast4H) + (0.6 * tddLast8to4H)) * 3
            tdd = (tddWeightedFromLast8H * 0.33) + (tdd2Days * 0.34) + (tddDaily * 0.33)
            if (bg != null) {
                tdd = when {
                    sportTime -> tdd * 1.1
                    sleepTime -> tdd * adjustFactorsdynisfBasedOnBgAndHypo(sleepTimeDynISFAdjFactor)
                    lowCarbTime -> tdd * 1.1
                    snackTime -> tdd * adjustFactorsdynisfBasedOnBgAndHypo(snackTimeDynISFAdjFactor)
                    highCarbTime -> tdd * adjustFactorsdynisfBasedOnBgAndHypo(hcTimeDynISFAdjFactor)
                    mealTime -> tdd * adjustFactorsdynisfBasedOnBgAndHypo(mealTimeDynISFAdjFactor)
                    bg > 180 -> tdd * adjustFactorsdynisfBasedOnBgAndHypo(dynISFadjusthyper)
                    else -> tdd * adjustFactorsdynisfBasedOnBgAndHypo(dynISFadjust)
                }
            }
            variableSensitivity = Round.roundTo(1800 / (tdd * (ln((glucoseStatus.glucose / insulinDivisor) + 1))), 0.1)
            val isfMgdl = profileFunction.getProfile()?.getProfileIsfMgdl()
            if (variableSensitivity < 0) if (isfMgdl != null) {
                variableSensitivity = profileUtil.fromMgdlToUnits(isfMgdl.toDouble(), profileFunction.getUnits())
            }

// Compare insulin consumption of last 24h with last 7 days average
            val tddRatio = if (preferences.get(BooleanKey.ApsDynIsfAdjustSensitivity)) tdd24Hrs / tdd2Days else 1.0
// Because consumed carbs affects total amount of insulin compensate final ratio by consumed carbs ratio
// take only 60% (expecting 40% basal). We cannot use bolus/total because of SMBs
            val carbsRatio = if (
                preferences.get(BooleanKey.ApsDynIsfAdjustSensitivity) &&
                tddLast24H != null && tddLast24H.carbs != 0.0 &&
                tdd7D != null && tdd7D.data != null && tdd7D.data.carbs != 0.0 &&
                tdd7D.allDaysHaveCarbs
            ) ((tddLast24H.carbs / tdd7D.data.carbs - 1.0) * 0.6) + 1.0 else 1.0
            autosensResult = AutosensResult(
                ratio = tddRatio / carbsRatio,
                ratioFromTdd = tddRatio,
                ratioFromCarbs = carbsRatio
            )
            val iobArray = iobCobCalculator.calculateIobArrayForSMB(autosensResult, SMBDefaults.exercise_mode, SMBDefaults.half_basal_exercise_target, isTempTarget)
            val mealData = iobCobCalculator.getMealDataWithWaitingForCalculationFinish()

            val oapsProfile = OapsProfile(
                dia = 0.0, // not used
                min_5m_carbimpact = 0.0, // not used
                max_iob = constraintsChecker.getMaxIOBAllowed().also { inputConstraints.copyReasons(it) }.value(),
                max_daily_basal = profile.getMaxDailyBasal(),
                max_basal = constraintsChecker.getMaxBasalAllowed(profile).also { inputConstraints.copyReasons(it) }.value(),
                min_bg = minBg,
                max_bg = maxBg,
                target_bg = targetBg,
                carb_ratio = profile.getIc(),
                sens = profile.getIsfMgdl("OpenAPSAIMIPlugin"),
                autosens_adjust_targets = false, // not used
                max_daily_safety_multiplier = preferences.get(DoubleKey.ApsMaxDailyMultiplier),
                current_basal_safety_multiplier = preferences.get(DoubleKey.ApsMaxCurrentBasalMultiplier),
                lgsThreshold = profileUtil.convertToMgdlDetect(preferences.get(UnitDoubleKey.ApsLgsThreshold)).toInt(),
                high_temptarget_raises_sensitivity = false,
                low_temptarget_lowers_sensitivity = false,
                sensitivity_raises_target = preferences.get(BooleanKey.ApsSensitivityRaisesTarget),
                resistance_lowers_target = preferences.get(BooleanKey.ApsResistanceLowersTarget),
                adv_target_adjustments = SMBDefaults.adv_target_adjustments,
                exercise_mode = SMBDefaults.exercise_mode,
                half_basal_exercise_target = SMBDefaults.half_basal_exercise_target,
                maxCOB = SMBDefaults.maxCOB,
                skip_neutral_temps = pump.setNeutralTempAtFullHour(),
                remainingCarbsCap = SMBDefaults.remainingCarbsCap,
                enableUAM = constraintsChecker.isUAMEnabled().also { inputConstraints.copyReasons(it) }.value(),
                A52_risk_enable = SMBDefaults.A52_risk_enable,
                SMBInterval = preferences.get(IntKey.ApsMaxSmbFrequency),
                enableSMB_with_COB = smbEnabled && preferences.get(BooleanKey.ApsUseSmbWithCob),
                enableSMB_with_temptarget = smbEnabled && preferences.get(BooleanKey.ApsUseSmbWithLowTt),
                allowSMB_with_high_temptarget = smbEnabled && preferences.get(BooleanKey.ApsUseSmbWithHighTt),
                enableSMB_always = smbEnabled && preferences.get(BooleanKey.ApsUseSmbAlways) && advancedFiltering,
                enableSMB_after_carbs = smbEnabled && preferences.get(BooleanKey.ApsUseSmbAfterCarbs) && advancedFiltering,
                maxSMBBasalMinutes = preferences.get(IntKey.ApsMaxMinutesOfBasalToLimitSmb),
                maxUAMSMBBasalMinutes = preferences.get(IntKey.ApsUamMaxMinutesOfBasalToLimitSmb),
                bolus_increment = pump.pumpDescription.bolusStep,
                carbsReqThreshold = preferences.get(IntKey.ApsCarbsRequestThreshold),
                current_basal = activePlugin.activePump.baseBasalRate,
                temptargetSet = isTempTarget,
                autosens_max = preferences.get(DoubleKey.AutosensMax),
                out_units = if (profileFunction.getUnits() == GlucoseUnit.MMOL) "mmol/L" else "mg/dl",
                variable_sens = variableSensitivity,
                insulinDivisor = insulinDivisor,
                TDD = if (tdd == 0.0)  preferences.get(DoubleKey.OApsAIMITDD7) else tdd
            )

            val microBolusAllowed = constraintsChecker.isSMBModeEnabled(ConstraintObject(tempBasalFallback.not(), aapsLogger)).also { inputConstraints.copyReasons(it) }.value()
            val flatBGsDetected = bgQualityCheck.state == BgQualityCheck.State.FLAT

            aapsLogger.debug(LTag.APS, ">>> Invoking determine_basal AIMI <<<")
            aapsLogger.debug(LTag.APS, "Glucose status:     $glucoseStatus")
            aapsLogger.debug(LTag.APS, "Current temp:       $currentTemp")
            aapsLogger.debug(LTag.APS, "IOB data:           ${iobArray.joinToString()}")
            aapsLogger.debug(LTag.APS, "Profile:            $oapsProfile")
            aapsLogger.debug(LTag.APS, "Autosens data:      $autosensResult")
            aapsLogger.debug(LTag.APS, "Meal data:          $mealData")
            aapsLogger.debug(LTag.APS, "MicroBolusAllowed:  $microBolusAllowed")
            aapsLogger.debug(LTag.APS, "flatBGsDetected:    $flatBGsDetected")
            aapsLogger.debug(LTag.APS, "DynIsfMode:         $dynIsfMode")

            determineBasalAIMI.determine_basal(
                glucose_status = glucoseStatus,
                currenttemp = currentTemp,
                iob_data_array = iobArray,
                profile = oapsProfile,
                autosens_data = autosensResult,
                meal_data = mealData,
                microBolusAllowed = microBolusAllowed,
                currentTime = now,
                flatBGsDetected = flatBGsDetected,
                dynIsfMode = dynIsfMode
            ).also {
                val determineBasalResult = DetermineBasalResult(injector, it)
                // Preserve input data
                determineBasalResult.inputConstraints = inputConstraints
                determineBasalResult.autosensResult = autosensResult
                determineBasalResult.iobData = iobArray
                determineBasalResult.glucoseStatus = glucoseStatus
                determineBasalResult.currentTemp = currentTemp
                determineBasalResult.oapsProfile = oapsProfile
                determineBasalResult.mealData = mealData
                lastAPSResult = determineBasalResult
                lastAPSRun = now
                aapsLogger.debug(LTag.APS, "Result: $it")
                rxBus.send(EventAPSCalculationFinished())
            }

            rxBus.send(EventOpenAPSUpdateGui())
        }
    }

    override fun isSuperBolusEnabled(value: Constraint<Boolean>): Constraint<Boolean> {
        value.set(false)
        return value
    }

    override fun applyMaxIOBConstraints(maxIob: Constraint<Double>): Constraint<Double> {
        if (isEnabled()) {
            val maxIobPref = preferences.get(DoubleKey.ApsSmbMaxIob)
            maxIob.setIfSmaller(maxIobPref, rh.gs(R.string.limiting_iob, maxIobPref, rh.gs(R.string.maxvalueinpreferences)), this)
            maxIob.setIfSmaller(hardLimits.maxIobSMB(), rh.gs(R.string.limiting_iob, hardLimits.maxIobSMB(), rh.gs(R.string.hardlimit)), this)
        }
        return maxIob
    }

    override fun applyBasalConstraints(absoluteRate: Constraint<Double>, profile: Profile): Constraint<Double> {
        if (isEnabled()) {
            var maxBasal = preferences.get(DoubleKey.ApsMaxBasal)
            if (maxBasal < profile.getMaxDailyBasal()) {
                maxBasal = profile.getMaxDailyBasal()
                absoluteRate.addReason(rh.gs(R.string.increasing_max_basal), this)
            }
            absoluteRate.setIfSmaller(maxBasal, rh.gs(app.aaps.core.ui.R.string.limitingbasalratio, maxBasal, rh.gs(R.string.maxvalueinpreferences)), this)

            // Check percentRate but absolute rate too, because we know real current basal in pump
            val maxBasalMultiplier = preferences.get(DoubleKey.ApsMaxCurrentBasalMultiplier)
            val maxFromBasalMultiplier = floor(maxBasalMultiplier * profile.getBasal() * 100) / 100
            absoluteRate.setIfSmaller(
                maxFromBasalMultiplier,
                rh.gs(app.aaps.core.ui.R.string.limitingbasalratio, maxFromBasalMultiplier, rh.gs(R.string.max_basal_multiplier)),
                this
            )
            val maxBasalFromDaily = preferences.get(DoubleKey.ApsMaxDailyMultiplier)
            val maxFromDaily = floor(profile.getMaxDailyBasal() * maxBasalFromDaily * 100) / 100
            absoluteRate.setIfSmaller(maxFromDaily, rh.gs(app.aaps.core.ui.R.string.limitingbasalratio, maxFromDaily, rh.gs(R.string.max_daily_basal_multiplier)), this)
        }
        return absoluteRate
    }

    override fun isSMBModeEnabled(value: Constraint<Boolean>): Constraint<Boolean> {
        val enabled = preferences.get(BooleanKey.ApsUseSmb)
        if (!enabled) value.set(false, rh.gs(R.string.smb_disabled_in_preferences), this)
        return value
    }

    override fun isUAMEnabled(value: Constraint<Boolean>): Constraint<Boolean> {
        val enabled = preferences.get(BooleanKey.ApsUseUam)
        if (!enabled) value.set(false, rh.gs(R.string.uam_disabled_in_preferences), this)
        return value
    }

    override fun isAutosensModeEnabled(value: Constraint<Boolean>): Constraint<Boolean> {
        if (preferences.get(BooleanKey.ApsUseDynamicSensitivity)) {
            // DynISF mode
            if (!preferences.get(BooleanKey.ApsDynIsfAdjustSensitivity))
                value.set(false, rh.gs(R.string.autosens_disabled_in_preferences), this)
        } else {
            // SMB mode
            val enabled = preferences.get(BooleanKey.ApsUseAutosens)
            if (!enabled) value.set(false, rh.gs(R.string.autosens_disabled_in_preferences), this)
        }
        return value
    }

    override fun configuration(): JSONObject =
        JSONObject()
            .put(BooleanKey.ApsUseDynamicSensitivity, preferences, rh)
            .put(IntKey.ApsDynIsfAdjustmentFactor, preferences, rh)

    override fun applyConfiguration(configuration: JSONObject) {
        configuration
            .store(BooleanKey.ApsUseDynamicSensitivity, preferences, rh)
            .store(IntKey.ApsDynIsfAdjustmentFactor, preferences, rh)
    }
    override fun addPreferenceScreen(preferenceManager: PreferenceManager, parent: PreferenceScreen, context: Context, requiredKey: String?) {
        //if (requiredKey != null && requiredKey != "absorption_smb_advanced") return
        val category = PreferenceCategory(context)
        parent.addPreference(category)
        category.apply {
            key = "AIMI_Settings"
            title = rh.gs(R.string.aimi_preferences)
            initialExpandedChildrenCount = 0
            addPreference(AdaptiveSwitchPreference(ctx = context, booleanKey = BooleanKey.OApsAIMIEnableBasal, title = R.string.Enable_basal_title))
            addPreference(AdaptiveSwitchPreference(ctx = context, booleanKey = BooleanKey.OApsAIMIpregnancy, title = R.string.OApsAIMI_Enable_pregnancy))
            addPreference(AdaptiveSwitchPreference(ctx = context, booleanKey = BooleanKey.OApsAIMIhoneymoon, title = R.string.OApsAIMI_Enable_honeymoon))
            addPreference(AdaptiveSwitchPreference(ctx = context, booleanKey = BooleanKey.OApsAIMIEnableStepsFromWatch, title = R.string.countsteps_watch_title))
            addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.OApsAIMIMaxSMB, dialogMessage = R.string.openapsaimi_maxsmb_summary, title = R.string.openapsaimi_maxsmb_title))
            addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.OApsAIMIweight, dialogMessage = R.string.oaps_aimi_weight_summary, title = R.string.oaps_aimi_weight_title))
            addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.OApsAIMICHO, dialogMessage = R.string.oaps_aimi_cho_summary, title = R.string.oaps_aimi_cho_title))
            addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.OApsAIMITDD7, dialogMessage = R.string.oaps_aimi_tdd7_summary, title = R.string.oaps_aimi_tdd7_title))
            addPreference(AdaptiveIntPreference(ctx = context, intKey = IntKey.OApsAIMIDynISFAdjustment, dialogMessage = R.string.DynISF_Adjust_summary, title = R.string.DynISF_Adjust_title))
            addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.OApsAIMIMorningFactor, dialogMessage = R.string.oaps_aimi_morning_factor_summary, title = R.string.oaps_aimi_morning_factor_title))
            addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.OApsAIMIAfternoonFactor, dialogMessage = R.string.oaps_aimi_afternoon_factor_summary, title = R.string.oaps_aimi_afternoon_factor_title))
            addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.OApsAIMIEveningFactor, dialogMessage = R.string.oaps_aimi_evening_factor_summary, title = R.string.oaps_aimi_evening_factor_title))
            addPreference(preferenceManager.createPreferenceScreen(context).apply {
                key = "high_BG_settings"
                title = "High BG Preferences"
                addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.OApsAIMIHyperFactor, dialogMessage = R.string.oaps_aimi_hyper_factor_summary, title = R.string.oaps_aimi_hyper_factor_title))
                addPreference(AdaptiveIntPreference(ctx = context, intKey = IntKey.OApsAIMIDynISFAdjustmentHyper, dialogMessage = R.string.DynISFAdjusthyper_summary, title = R.string.DynISFAdjusthyper_title))
                addPreference(AdaptiveIntPreference(ctx = context, intKey = IntKey.OApsAIMIHighBGinterval, dialogMessage = R.string.oaps_aimi_HIGHBG_interval_summary, title = R.string.oaps_aimi_HIGHBG_interval_title))
                addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.OApsAIMIHighBGMaxSMB, dialogMessage = R.string.openapsaimi_highBG_maxsmb_summary, title = R.string.openapsaimi_highBG_maxsmb_title))
            })
            addPreference(preferenceManager.createPreferenceScreen(context).apply {
                key = "Training_ML_Modes"
                title = "Training ML and Modes"
                addPreference(AdaptiveSwitchPreference(ctx = context, booleanKey = BooleanKey.OApsAIMIMLtraining, title = R.string.oaps_aimi_enableMlTraining_title))
                addPreference(preferenceManager.createPreferenceScreen(context).apply {
                    key = "mode_meal"
                    title = "Meal Mode settings"
                    addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.OApsAIMIMealPrebolus, dialogMessage = R.string.prebolus_meal_mode_summary, title = R.string.prebolus_meal_mode_title))
                    addPreference(AdaptiveIntPreference(ctx = context, intKey = IntKey.OApsAIMImealAdjISFFact, dialogMessage = R.string.oaps_aimi_mealAdjFact_summary, title = R.string.oaps_aimi_mealAdjFact_title))
                    addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.OApsAIMIMealFactor, dialogMessage = R.string.OApsAIMI_MealFactor_summary, title = R.string.OApsAIMI_MealFactor_title))
                    addPreference(AdaptiveIntPreference(ctx = context, intKey = IntKey.OApsAIMImealinterval, dialogMessage = R.string.oaps_aimi_meal_interval_summary, title = R.string.oaps_aimi_meal_interval_title))
                })
                addPreference(preferenceManager.createPreferenceScreen(context).apply {
                    key = "mode_highcarb"
                    title = "High Carb Mode settings"
                    addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.OApsAIMIHighCarbPrebolus, dialogMessage = R.string.prebolus_highcarb_mode_summary, title = R.string.prebolus_highcarb_mode_title))
                    addPreference(AdaptiveIntPreference(ctx = context, intKey = IntKey.OApsAIMIHighCarbAdjISFFact, dialogMessage = R.string.oaps_aimi_highcarbAdjFact_summary, title = R.string.oaps_aimi_highcarbAdjFact_title))
                    addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.OApsAIMIHCFactor, dialogMessage = R.string.OApsAIMI_HC_Factor_summary, title = R.string.OApsAIMI_HC_Factor_title))
                    addPreference(AdaptiveIntPreference(ctx = context, intKey = IntKey.OApsAIMIHCinterval, dialogMessage = R.string.oaps_aimi_HC_interval_summary, title = R.string.oaps_aimi_HC_interval_title))
                })
                addPreference(preferenceManager.createPreferenceScreen(context).apply {
                    key = "mode_snack"
                    title = "Snack Mode settings"
                    addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.OApsAIMISnackPrebolus, dialogMessage = R.string.prebolus_snack_mode_summary, title = R.string.prebolus_snack_mode_title))
                    addPreference(AdaptiveIntPreference(ctx = context, intKey = IntKey.OApsAIMISnackAdjISFFact, dialogMessage = R.string.oaps_aimi_snackAdjFact_summary, title = R.string.oaps_aimi_snackAdjFact_title))
                    addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.OApsAIMISnackFactor, dialogMessage = R.string.OApsAIMI_snack_Factor_summary, title = R.string.OApsAIMI_snack_Factor_title))
                    addPreference(AdaptiveIntPreference(ctx = context, intKey = IntKey.OApsAIMISnackinterval, dialogMessage = R.string.oaps_aimi_snack_interval_summary, title = R.string.oaps_aimi_snack_interval_title))
                })
                addPreference(preferenceManager.createPreferenceScreen(context).apply {
                    key = "mode_sleep"
                    title = "Sleep Mode settings"
                    addPreference(AdaptiveIntPreference(ctx = context, intKey = IntKey.OApsAIMIsleepAdjISFFact, dialogMessage = R.string.oaps_aimi_sleepAdjFact_summary, title = R.string.oaps_aimi_sleepAdjFact_title))
                    addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.OApsAIMIsleepFactor, dialogMessage = R.string.OApsAIMI_sleep_Factor_summary, title = R.string.OApsAIMI_sleep_Factor_title))
                    addPreference(AdaptiveIntPreference(ctx = context, intKey = IntKey.OApsAIMISleepinterval, dialogMessage = R.string.oaps_aimi_sleep_interval_summary, title = R.string.oaps_aimi_sleep_interval_title))
                })
            })
            addPreference(preferenceManager.createPreferenceScreen(context).apply {
                key = "OAPS_SMB_Settings"
                title = rh.gs(R.string.AAPS_SMB_Settings)
                addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.ApsMaxBasal, dialogMessage = R.string.openapsma_max_basal_summary, title = R.string.openapsma_max_basal_title))
                addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.ApsSmbMaxIob, dialogMessage = R.string.openapssmb_max_iob_summary, title = R.string.openapssmb_max_iob_title))
                addPreference(AdaptiveSwitchPreference(ctx = context, booleanKey = BooleanKey.ApsUseDynamicSensitivity, summary = R.string.use_dynamic_sensitivity_summary, title = R.string.use_dynamic_sensitivity_title))
                addPreference(AdaptiveSwitchPreference(ctx = context, booleanKey = BooleanKey.ApsUseAutosens, title = R.string.openapsama_use_autosens))
                addPreference(AdaptiveIntPreference(ctx = context, intKey = IntKey.ApsDynIsfAdjustmentFactor, dialogMessage = R.string.dyn_isf_adjust_summary, title = R.string.dyn_isf_adjust_title))
                addPreference(AdaptiveUnitPreference(ctx = context, unitKey = UnitDoubleKey.ApsLgsThreshold, dialogMessage = R.string.lgs_threshold_summary, title = R.string.lgs_threshold_title))
                addPreference(AdaptiveSwitchPreference(ctx = context, booleanKey = BooleanKey.ApsDynIsfAdjustSensitivity, summary = R.string.dynisf_adjust_sensitivity_summary, title = R.string.dynisf_adjust_sensitivity))
                addPreference(AdaptiveSwitchPreference(ctx = context, booleanKey = BooleanKey.ApsSensitivityRaisesTarget, summary = R.string.sensitivity_raises_target_summary, title = R.string.sensitivity_raises_target_title))
                addPreference(AdaptiveSwitchPreference(ctx = context, booleanKey = BooleanKey.ApsResistanceLowersTarget, summary = R.string.resistance_lowers_target_summary, title = R.string.resistance_lowers_target_title))
                addPreference(AdaptiveSwitchPreference(ctx = context, booleanKey = BooleanKey.ApsUseSmb, summary = R.string.enable_smb_summary, title = R.string.enable_smb))
                addPreference(AdaptiveSwitchPreference(ctx = context, booleanKey = BooleanKey.ApsUseSmbWithHighTt, summary = R.string.enable_smb_with_high_temp_target_summary, title = R.string.enable_smb_with_high_temp_target))
                addPreference(AdaptiveSwitchPreference(ctx = context, booleanKey = BooleanKey.ApsUseSmbAlways, summary = R.string.enable_smb_always_summary, title = R.string.enable_smb_always))
                addPreference(AdaptiveSwitchPreference(ctx = context, booleanKey = BooleanKey.ApsUseSmbWithCob, summary = R.string.enable_smb_with_cob_summary, title = R.string.enable_smb_with_cob))
                addPreference(AdaptiveSwitchPreference(ctx = context, booleanKey = BooleanKey.ApsUseSmbWithLowTt, summary = R.string.enable_smb_with_temp_target_summary, title = R.string.enable_smb_with_temp_target))
                addPreference(AdaptiveSwitchPreference(ctx = context, booleanKey = BooleanKey.ApsUseSmbAfterCarbs, summary = R.string.enable_smb_after_carbs_summary, title = R.string.enable_smb_after_carbs))
                addPreference(AdaptiveIntPreference(ctx = context, intKey = IntKey.ApsMaxSmbFrequency, title = R.string.smb_interval_summary))
                //addPreference(AdaptiveIntPreference(ctx = context, intKey = IntKey.ApsMaxMinutesOfBasalToLimitSmb, title = R.string.smb_max_minutes_summary))
                //addPreference(AdaptiveIntPreference(ctx = context, intKey = IntKey.ApsUamMaxMinutesOfBasalToLimitSmb, dialogMessage = R.string.uam_smb_max_minutes, title = R.string.uam_smb_max_minutes_summary))
                addPreference(AdaptiveSwitchPreference(ctx = context, booleanKey = BooleanKey.ApsUseUam, summary = R.string.enable_uam_summary, title = R.string.enable_uam))
                addPreference(AdaptiveIntPreference(ctx = context, intKey = IntKey.ApsCarbsRequestThreshold, dialogMessage = R.string.carbs_req_threshold_summary, title = R.string.carbs_req_threshold))
            })
            addPreference(preferenceManager.createPreferenceScreen(context).apply {
                key = "absorption_smb_advanced"
                title = rh.gs(app.aaps.core.ui.R.string.advanced_settings_title)
                addPreference(
                    AdaptiveIntentPreference(
                        ctx = context,
                        intentKey = IntentKey.ApsLinkToDocs,
                        intent = Intent().apply { action = Intent.ACTION_VIEW; data = Uri.parse(rh.gs(R.string.openapsama_link_to_preference_json_doc)) },
                        summary = R.string.openapsama_link_to_preference_json_doc_txt
                    )
                )
                addPreference(AdaptiveSwitchPreference(ctx = context, booleanKey = BooleanKey.ApsAlwaysUseShortDeltas, summary = R.string.always_use_short_avg_summary, title = R.string.always_use_short_avg))
                addPreference(AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.ApsMaxDailyMultiplier, dialogMessage = R.string.openapsama_max_daily_safety_multiplier_summary, title = R.string.openapsama_max_daily_safety_multiplier))
                addPreference(
                    AdaptiveDoublePreference(ctx = context, doubleKey = DoubleKey.ApsMaxCurrentBasalMultiplier, dialogMessage = R.string.openapsama_current_basal_safety_multiplier_summary, title = R.string.openapsama_current_basal_safety_multiplier)
                )
            })
        }
    }
}