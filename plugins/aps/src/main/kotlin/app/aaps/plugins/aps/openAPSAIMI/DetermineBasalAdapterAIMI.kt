package app.aaps.plugins.aps.openAPSAIMI

import android.os.Environment
import app.aaps.core.interfaces.aps.DetermineBasalAdapter
import app.aaps.core.interfaces.constraints.ConstraintsChecker
import app.aaps.core.interfaces.db.GlucoseUnit
import app.aaps.core.interfaces.iob.GlucoseStatus
import app.aaps.core.interfaces.iob.IobCobCalculator
import app.aaps.core.interfaces.iob.IobTotal
import app.aaps.core.interfaces.iob.MealData
import app.aaps.core.interfaces.logging.AAPSLogger
import app.aaps.core.interfaces.logging.LTag
import app.aaps.core.interfaces.plugin.ActivePlugin
import app.aaps.core.interfaces.profile.Profile
import app.aaps.core.interfaces.profile.ProfileFunction
import app.aaps.core.interfaces.sharedPreferences.SP
import app.aaps.core.interfaces.stats.TddCalculator
import app.aaps.core.interfaces.stats.TirCalculator
import app.aaps.core.interfaces.utils.DateUtil
import app.aaps.core.interfaces.utils.SafeParse
import app.aaps.core.main.extensions.convertedToAbsolute
import app.aaps.core.main.extensions.getPassedDurationToTimeInMinutes
import app.aaps.core.main.extensions.plannedRemainingMinutes
import app.aaps.database.entities.UserEntry
import app.aaps.database.impl.AppRepository
import app.aaps.plugins.aps.APSResultObject
import app.aaps.plugins.aps.R
import dagger.android.HasAndroidInjector
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.io.File
import javax.inject.Inject
import kotlin.math.roundToInt
import java.util.Calendar
import kotlin.math.min
import org.tensorflow.lite.Interpreter
import java.time.LocalTime
import kotlin.math.ln



class DetermineBasalAdapterAIMI internal constructor(private val injector: HasAndroidInjector) : DetermineBasalAdapter {

    @Inject lateinit var aapsLogger: AAPSLogger
    @Inject lateinit var constraintChecker: ConstraintsChecker
    @Inject lateinit var sp: SP
    @Inject lateinit var profileFunction: ProfileFunction
    @Inject lateinit var iobCobCalculator: IobCobCalculator
    @Inject lateinit var activePlugin: ActivePlugin
    @Inject lateinit var repository: AppRepository
    @Inject lateinit var dateUtil: DateUtil
    @Inject lateinit var tddCalculator: TddCalculator
    @Inject lateinit var tirCalculator: TirCalculator

    private var iob = 0.0f
    private var cob = 0.0f
    private var lastCarbAgeMin: Int = 0
    private var futureCarbs = 0.0f
    private var recentNotes: List<UserEntry>? = null
    private var tags0to60minAgo = ""
    private var tags60to120minAgo = ""
    private var tags120to180minAgo = ""
    private var tags180to240minAgo = ""
    private var bg = 0.0f
    private var targetBg = 100.0f
    private var delta = 0.0f
    private var shortAvgDelta = 0.0f
    private var longAvgDelta = 0.0f
    private var accelerating_up: Int = 0
    private var deccelerating_up: Int = 0
    private var accelerating_down: Int = 0
    private var deccelerating_down: Int = 0
    private var stable: Int = 0
    private var maxIob = 0.0f
    private var maxSMB = 1.0f
    private var tdd7DaysPerHour = 0.0f
    private var tdd2DaysPerHour = 0.0f
    private var tddPerHour = 0.0f
    private var tdd24HrsPerHour = 0.0f
    private var hourOfDay: Int = 0
    private var weekend: Int = 0
    private var recentSteps5Minutes: Int = 0
    private var recentSteps10Minutes: Int = 0
    private var recentSteps15Minutes: Int = 0
    private var recentSteps30Minutes: Int = 0
    private var recentSteps60Minutes: Int = 0
    private var recentSteps180Minutes: Int = 0
    private var basalaimi = 0.0f
    private var aimilimit = 0.0f
    private var CI = 0.0f
    private var variableSensitivity = 0.0f
    private var averageBeatsPerMinute = 0.0
    private var averageBeatsPerMinute180 = 0.0
    private var tirlow = 0.0f
    private var profile = JSONObject()
    private var glucoseStatus = JSONObject()
    private var iobData: JSONArray? = null
    private var mealData = JSONObject()
    private var currentTemp = JSONObject()
    private var autosensData = JSONObject()
    private val path = File(Environment.getExternalStorageDirectory().toString())
    private val modelFile = File(path, "AAPS/ml/model.tflite")

    override var currentTempParam: String? = null
    override var iobDataParam: String? = null
    override var glucoseStatusParam: String? = null
    override var profileParam: String? = null
    override var mealDataParam: String? = null
    override var scriptDebug = ""


    private var now: Long = 0

    @Suppress("SpellCheckingInspection")
    override operator fun invoke(): APSResultObject {
        aapsLogger.debug(LTag.APS, ">>> Invoking determine_basal <<<")

        val predictedSMB = calculateSMBFromModel()
        var smbToGive = predictedSMB

        smbToGive = applySafetyPrecautions(smbToGive)
        smbToGive = roundToPoint05(smbToGive)

        logDataToCsv(predictedSMB, smbToGive)

        val constraintStr = " Max IOB: $maxIob <br/> Max SMB: $maxSMB"
        val glucoseStr = " bg: $bg <br/> targetBg: $targetBg <br/> " +
            "delta: $delta <br/> short avg delta: $shortAvgDelta <br/> long avg delta: $longAvgDelta <br/>" +
            " accelerating_up: $accelerating_up <br/> deccelerating_up: $deccelerating_up <br/> accelerating_down: $accelerating_down <br/> deccelerating_down: $deccelerating_down <br/> stable: $stable"
        val iobStr = " IOB: $iob <br/> tdd 7d/h: ${roundToPoint05(tdd7DaysPerHour)} <br/> " +
            "tdd 2d/h : ${roundToPoint05(tdd2DaysPerHour)} <br/> " +
            "tdd daily/h : ${roundToPoint05(tddPerHour)} <br/> " +
            "tdd 24h/h : ${roundToPoint05(tdd24HrsPerHour)}"
        val profileStr = " Hour of day: $hourOfDay <br/> Weekend: $weekend <br/>" +
            " 5 Min Steps: $recentSteps5Minutes <br/> 10 Min Steps: $recentSteps10Minutes <br/> 15 Min Steps: $recentSteps15Minutes <br/>" +
            " 30 Min Steps: $recentSteps30Minutes <br/> 60 Min Steps: $recentSteps60Minutes <br/> 180 Min Steps: $recentSteps180Minutes <br/>"
        var mealStr = " COB: ${cob}g   Future: ${futureCarbs}g <br/> COB Age Min: $lastCarbAgeMin <br/><br/> "
        mealStr += "tags0to60minAgo: ${tags0to60minAgo}<br/> tags60to120minAgo: $tags60to120minAgo<br/> " +
            "tags120to180minAgo: $tags120to180minAgo<br/> tags180to240minAgo: $tags180to240minAgo"
        val reason = "The ai model predicted SMB of ${roundToPoint001(predictedSMB)}u and after safety requirements and rounding to .05, requested ${smbToGive}u to the pump"

        val determineBasalResultAIMISMB = DetermineBasalResultAIMISMB(injector, smbToGive, constraintStr, glucoseStr, iobStr, profileStr, mealStr, reason)

        glucoseStatusParam = glucoseStatus.toString()
        iobDataParam = iobData.toString()
        currentTempParam = currentTemp.toString()
        profileParam = profile.toString()
        mealDataParam = mealData.toString()
        return determineBasalResultAIMISMB
    }

    private fun logDataToCsv(predictedSMB: Float, smbToGive: Float) {
        val dateStr = dateUtil.dateAndTimeString(dateUtil.now())

        val headerRow = "dateStr,dateLong,hourOfDay,weekend," +
            "bg,targetBg,iob,cob,lastCarbAgeMin,futureCarbs,delta,shortAvgDelta,longAvgDelta," +
            "tdd7DaysPerHour,tdd2DaysPerHour,tddPerHour,tdd24HrsPerHour," +
            "recentSteps5Minutes,recentSteps10Minutes,recentSteps15Minutes,recentSteps30Minutes,recentSteps60Minutes,recentSteps180Minutes" +
            "tags0to60minAgo,tags60to120minAgo,tags120to180minAgo,tags180to240minAgo," +
            "predictedSMB,maxIob,maxSMB,smbGiven\n"
        val valuesToRecord = "$dateStr,${dateUtil.now()},$hourOfDay,$weekend," +
            "$bg,$targetBg,$iob,$cob,$lastCarbAgeMin,$futureCarbs,$delta,$shortAvgDelta,$longAvgDelta," +
            "$tdd7DaysPerHour,$tdd2DaysPerHour,$tddPerHour,$tdd24HrsPerHour," +
            "$recentSteps5Minutes,$recentSteps10Minutes,$recentSteps15Minutes,$recentSteps30Minutes,$recentSteps60Minutes,$recentSteps180Minutes" +
            "$tags0to60minAgo,$tags60to120minAgo,$tags120to180minAgo,$tags180to240minAgo," +
            "$predictedSMB,$maxIob,$maxSMB,$smbToGive"

        val file = File(path, "AAPS/aimiSMB_records.csv")
        if (!file.exists()) {
            file.createNewFile()
            file.appendText(headerRow)
        }
        file.appendText(valuesToRecord + "\n")
    }

    private fun applySafetyPrecautions(smbToGiveParam: Float): Float {
        var smbToGive = smbToGiveParam
        // don't exceed max IOB
        if (iob + smbToGive > maxIob) {
            smbToGive = maxIob - iob
        }
        // don't exceed max SMB
        if (smbToGive > maxSMB) {
            smbToGive = maxSMB
        }
        // don't give insulin if below target too aggressive
        val belowTargetAndDropping = bg < targetBg && delta < -2
        val belowTargetAndStableButNoCob = bg < targetBg - 15 && shortAvgDelta <= 2 && cob <= 5
        val belowMinThreshold = bg < 70
        if (belowTargetAndDropping || belowMinThreshold || belowTargetAndStableButNoCob) {
            smbToGive = 0.0f
        }

        // don't give insulin if dropping fast
        val droppingFast = bg < 150 && delta < -5
        val droppingFastAtHigh = bg < 200 && delta < -7
        val droppingVeryFast = delta < -10
        if (droppingFast || droppingFastAtHigh || droppingVeryFast) {
            smbToGive = 0.0f
        }
        if (smbToGive < 0.0f) {
            smbToGive = 0.0f
        }
        return smbToGive
    }

    private fun roundToPoint05(number: Float): Float {
        return (number * 20.0).roundToInt() / 20.0f
    }

    private fun roundToPoint001(number: Float): Float {
        return (number * 1000.0).roundToInt() / 1000.0f
    }

    private fun calculateSMBFromModel(): Float {
        if (!modelFile.exists()) {
            aapsLogger.error(LTag.APS, "NO Model found at AAPS/ml/model.tflite")
            return 0.0f
        }

        val interpreter = Interpreter(modelFile)
        val modelInputs = floatArrayOf(
            hourOfDay.toFloat(), weekend.toFloat(),
            bg, targetBg, iob, cob, lastCarbAgeMin.toFloat(), futureCarbs, delta, shortAvgDelta, longAvgDelta,
            tdd7DaysPerHour, tdd2DaysPerHour, tddPerHour, tdd24HrsPerHour,
            recentSteps5Minutes.toFloat(), recentSteps10Minutes.toFloat(), recentSteps15Minutes.toFloat(), recentSteps30Minutes.toFloat(), recentSteps60Minutes.toFloat()
        )
        val output = arrayOf(floatArrayOf(0.0f))
        interpreter.run(modelInputs, output)
        interpreter.close()
        var smbToGive = output[0][0]
        smbToGive = "%.4f".format(smbToGive.toDouble()).toFloat()
        return smbToGive
    }

    @Suppress("SpellCheckingInspection")
    @Throws(JSONException::class)
    override fun setData(
        profile: Profile,
        maxIob: Double,
        maxBasal: Double,
        minBg: Double,
        maxBg: Double,
        targetBg: Double,
        basalRate: Double,
        iobArray: Array<IobTotal>,
        glucoseStatus: GlucoseStatus,
        mealData: MealData,
        autosensDataRatio: Double,
        tempTargetSet: Boolean,
        microBolusAllowed: Boolean,
        uamAllowed: Boolean,
        advancedFiltering: Boolean,
        flatBGsDetected: Boolean,
        tdd1D: Double?,
        tdd7D: Double?,
        tddLast24H: Double?,
        tddLast4H: Double?,
        tddLast8to4H: Double?
    ) {
        this.now = System.currentTimeMillis()
        this.hourOfDay = Calendar.getInstance().get(Calendar.HOUR_OF_DAY)
        val dayOfWeek = Calendar.getInstance().get(Calendar.DAY_OF_WEEK)
        this.weekend = if (dayOfWeek == Calendar.SUNDAY || dayOfWeek == Calendar.SATURDAY) 1 else 0

        val iobCalcs = iobCobCalculator.calculateIobFromBolus()
        this.iob = iobCalcs.iob.toFloat() + iobCalcs.basaliob.toFloat()
        this.bg = glucoseStatus.glucose.toFloat()
        this.targetBg = targetBg.toFloat()
        this.cob = mealData.mealCOB.toFloat()
        var lastCarbTimestamp = mealData.lastCarbTime

        if(lastCarbTimestamp.toInt() == 0) {
            val oneDayAgoIfNotFound = now - 24 * 60 * 60 * 1000
            lastCarbTimestamp = iobCobCalculator.getMostRecentCarbByDate() ?: oneDayAgoIfNotFound
        }
        this.lastCarbAgeMin = ((now - lastCarbTimestamp) / (60 * 1000)).toDouble().roundToInt()

        if(lastCarbAgeMin < 15 && cob == 0.0f) {
            this.cob = iobCobCalculator.getMostRecentCarbAmount()?.toFloat() ?: 0.0f
        }

        this.futureCarbs = iobCobCalculator.getFutureCob().toFloat()
        val fourHoursAgo = now - 4 * 60 * 60 * 1000
        this.recentNotes = iobCobCalculator.getUserEntryDataWithNotesFromTime(fourHoursAgo)
        this.tags0to60minAgo = parseNotes(0, 60)
        this.tags60to120minAgo = parseNotes(60, 120)
        this.tags120to180minAgo = parseNotes(120, 180)
        this.tags180to240minAgo = parseNotes(180, 240)
        val lastHourTIRAbove = tirCalculator.averageTIR(tirCalculator.calculateHour(72.0, 140.0))?.above()
        val lastHourTIRLow = tirCalculator.averageTIR(tirCalculator.calculateHour(72.0, 140.0))?.below()
        val tirbasal3IR = tirCalculator.averageTIR(tirCalculator.calculate(1, 65.0, 130.0))?.inRange()
        val tirbasal3B = tirCalculator.averageTIR(tirCalculator.calculate(1, 65.0, 130.0))?.below()
        val tirbasal3A = tirCalculator.averageTIR(tirCalculator.calculate(1, 65.0, 130.0))?.above()
        val tirbasalhAP = tirCalculator.averageTIR(tirCalculator.calculateHour(65.0, 115.0))?.above()
        this.delta = glucoseStatus.delta.toFloat()
        this.shortAvgDelta = glucoseStatus.shortAvgDelta.toFloat()
        this.longAvgDelta = glucoseStatus.longAvgDelta.toFloat()

        this.accelerating_up = if (delta > 2 && delta - longAvgDelta > 2) 1 else 0
        this.deccelerating_up = if (delta > 0 && (delta < shortAvgDelta || delta < longAvgDelta)) 1 else 0
        this.accelerating_down = if (delta < -2 && delta - longAvgDelta < -2) 1 else 0
        this.deccelerating_down = if (delta < 0 && (delta > shortAvgDelta || delta > longAvgDelta)) 1 else 0
        this.stable = if (delta>-3 && delta<3 && shortAvgDelta>-3 && shortAvgDelta<3 && longAvgDelta>-3 && longAvgDelta<3) 1 else 0

        var tdd7Days = tddCalculator.averageTDD(tddCalculator.calculate(7, allowMissingDays = false))?.totalAmount?.toFloat() ?: 0.0f
        this.tdd7DaysPerHour = tdd7Days / 24

        val tdd2Days = tddCalculator.averageTDD(tddCalculator.calculate(2, allowMissingDays = false))?.totalAmount?.toFloat() ?: 0.0f
        this.tdd2DaysPerHour = tdd2Days / 24

        val tddDaily = tddCalculator.averageTDD(tddCalculator.calculate(1, allowMissingDays = false))?.totalAmount?.toFloat() ?: 0.0f
        this.tddPerHour = tddDaily / 24

        val tdd24Hrs = tddCalculator.calculateDaily(-24, 0)?.totalAmount?.toFloat() ?: 0.0f
        this.tdd24HrsPerHour = tdd24Hrs / 24
        val insulin = activePlugin.activeInsulin

        val insulinDivisor = when {
            insulin.peak >= 35 -> 55 // lyumjev peak: 45
            insulin.peak > 45  -> 65 // ultra rapid peak: 55
            else               -> 75 // rapid peak: 75
        }
        this.recentSteps5Minutes = StepService.getRecentStepCount5Min()
        this.recentSteps10Minutes = StepService.getRecentStepCount10Min()
        this.recentSteps15Minutes = StepService.getRecentStepCount15Min()
        this.recentSteps30Minutes = StepService.getRecentStepCount30Min()
        this.recentSteps60Minutes = StepService.getRecentStepCount60Min()
        this.recentSteps180Minutes = StepService.getRecentStepCount180Min()

        this.maxIob = sp.getDouble(R.string.key_openapssmb_max_iob, 5.0).toFloat()
        this.maxSMB = sp.getDouble(R.string.key_openapsaimi_max_smb, 1.0).toFloat()
        if (tdd2Days != null && tdd2Days != 0.0f) {
            this.CI = 450 / tdd2Days
        } else {
            val tdd7Key = SafeParse.stringToDouble(sp.getString(R.string.key_tdd7, "50"))
            this.CI = (450 / tdd7Key).toFloat()
        }

        val choKey = SafeParse.stringToDouble(sp.getString(R.string.key_cho, "50"))
        if (CI != 0.0f && CI != Float.POSITIVE_INFINITY && CI != Float.NEGATIVE_INFINITY) {
            this.aimilimit = (choKey / CI).toFloat()
        } else {
            this.aimilimit = (choKey / profile.getIc()).toFloat()
        }
        // profile.dia
        val abs = iobCobCalculator.calculateAbsoluteIobFromBaseBasals(System.currentTimeMillis())
        val absIob = abs.iob
        val absNet = abs.netInsulin
        val absBasal = abs.basaliob
        val tddWeightedFromLast8H: Float? = if (tddLast4H != null && tddLast8to4H != null) {
            (((1.4 * tddLast4H).toFloat()) + (0.6 * tddLast8to4H).toFloat()) * 3
        } else {
            null
        }

// Use null-safe operator to avoid unnecessary unwrapping
        tdd7Days = tdd7Days?.let { it } ?: return

// Use when expression to make the code more readable

       /* val tdd = when {
            tddWeightedFromLast8H != null && !tddWeightedFromLast8H.isNaN() &&
                tdd1D != null && !tdd1D.isNaN() &&
                tdd7Days != null && !tdd7Days.isNaN() && tdd7Days != 0.0f && lastHourTIRLow!! > 0 -> ((tddWeightedFromLast8H * 0.20) + (tdd7Days * 0.45) + (tdd1D * 0.35)) * 0.85
            tddWeightedFromLast8H != null && !tddWeightedFromLast8H.isNaN() &&
                tdd1D != null && !tdd1D.isNaN() &&
                tdd7Days != null && !tdd7Days.isNaN() && tdd7Days != 0.0f && lastHourTIRAbove!! > 0 -> ((tddWeightedFromLast8H * 0.20) + (tdd7Days * 0.45) + (tdd1D * 0.35)) * 1.15
            tddWeightedFromLast8H != null && !tddWeightedFromLast8H.isNaN() &&
                tdd1D != null && !tdd1D.isNaN() &&
                tdd7Days != null && !tdd7Days.isNaN() && tdd7Days != 0.0f -> (tddWeightedFromLast8H * 0.20) + (tdd7Days * 0.45) + (tdd1D * 0.35)

            else -> {
                tddWeightedFromLast8H ?: 0.0 // or any default value you want
            }
        }*/
        val commonConditions = tddWeightedFromLast8H != null && !tddWeightedFromLast8H.isNaN() && tdd1D != null && !tdd1D.isNaN() && tdd7Days != null && !tdd7Days.isNaN() && tdd7Days != 0.0f

        val calculatedTdd = (tddWeightedFromLast8H ?: 0.0f) * 0.20 + (tdd7Days ?: 0.0f) * 0.45 + (tdd1D ?: 0.0) * 0.35
        this.tirlow = lastHourTIRLow.INT
        val tdd = when {
            commonConditions && lastHourTIRLow!! > 0.0f -> calculatedTdd * 0.85
            commonConditions && lastHourTIRAbove!! > 0.0f-> calculatedTdd * 1.15
            commonConditions -> calculatedTdd
            else -> tddWeightedFromLast8H ?: 0.0f
        }


        aapsLogger.debug(LTag.APS, "IOB options : bolus iob: ${iobCalcs.iob} basal iob : ${iobCalcs.basaliob}")
        aapsLogger.debug(LTag.APS, "IOB options : calculateAbsoluteIobFromBaseBasals iob: $absIob net : $absNet basal : $absBasal")
        val tddDouble = tdd.toDoubleSafely()
        val glucoseDouble = glucoseStatus.glucose?.toDoubleSafely()
        val insulinDivisorDouble = insulinDivisor?.toDoubleSafely()

        if (tddDouble != null && glucoseDouble != null && insulinDivisorDouble != null) {
            variableSensitivity = (1800 / (tdd.toDouble() * (ln((glucoseStatus.glucose / insulinDivisor) + 1)))).toFloat()

            // Ajout d'un log pour vérifier la valeur de variableSensitivity après le calcul
            val variableSensitivityDouble = variableSensitivity.toDoubleSafely()
            if (variableSensitivityDouble != null) {
                if (recentSteps5Minutes > 100 && recentSteps10Minutes > 200 && bg < 130 && delta < 10|| recentSteps180Minutes > 1500 && bg < 130 && delta < 10) variableSensitivity *= 1.5f
                if (recentSteps30Minutes > 500 && recentSteps5Minutes >= 0 && recentSteps5Minutes < 100 && bg < 130 && delta < 10) variableSensitivity *= 1.3f
            }
        } else {
            variableSensitivity = profile.getIsfMgdl().toFloat()
        }

        this.profile = JSONObject()
        this.profile.put("max_iob", maxIob)
        this.profile.put("dia", min(profile.dia, 3.0))
        this.profile.put("type", "current")
        this.profile.put("max_daily_basal", profile.getMaxDailyBasal())
        this.profile.put("max_basal", maxBasal)
        this.profile.put("min_bg", minBg)
        this.profile.put("max_bg", maxBg)
        this.profile.put("target_bg", targetBg)
        this.profile.put("carb_ratio", profile.getIc())
        this.profile.put("sens", profile.getIsfMgdl())
        this.profile.put("max_daily_safety_multiplier", sp.getInt(R.string.key_openapsama_max_daily_safety_multiplier, 3))
        this.profile.put("current_basal_safety_multiplier", sp.getDouble(R.string.key_openapsama_current_basal_safety_multiplier, 4.0))
        this.profile.put("skip_neutral_temps", true)
        this.profile.put("current_basal", basalRate)
        this.profile.put("temptargetSet", tempTargetSet)
        this.profile.put("autosens_adjust_targets", sp.getBoolean(R.string.key_openapsama_autosens_adjusttargets, true))

        if (profileFunction.getUnits() == GlucoseUnit.MMOL) {
            this.profile.put("out_units", "mmol/L")
        }

        val tb = iobCobCalculator.getTempBasalIncludingConvertedExtended(now)
        currentTemp = JSONObject()
        currentTemp.put("temp", "absolute")
        currentTemp.put("duration", tb?.plannedRemainingMinutes ?: 0)
        currentTemp.put("rate", tb?.convertedToAbsolute(now, profile) ?: 0.0)
        // as we have non default temps longer than 30 minutes
        if (tb != null) currentTemp.put("minutesrunning", tb.getPassedDurationToTimeInMinutes(now))

        iobData = iobCobCalculator.convertToJSONArray(iobArray)
        this.glucoseStatus = JSONObject()
        this.glucoseStatus.put("glucose", glucoseStatus.glucose)
        if (sp.getBoolean(R.string.key_always_use_shortavg, false)) {
            this.glucoseStatus.put("delta", glucoseStatus.shortAvgDelta)
        } else {
            this.glucoseStatus.put("delta", glucoseStatus.delta)
        }
        this.glucoseStatus.put("short_avgdelta", glucoseStatus.shortAvgDelta)
        this.glucoseStatus.put("long_avgdelta", glucoseStatus.longAvgDelta)
        this.mealData = JSONObject()
        this.mealData.put("carbs", mealData.carbs)
        this.mealData.put("mealCOB", mealData.mealCOB)
        if (constraintChecker.isAutosensModeEnabled().value()) {
            autosensData.put("ratio", autosensDataRatio)
        } else {
            autosensData.put("ratio", 1.0)
        }
    }
    fun Number.toDoubleSafely(): Double? {
        val doubleValue = this.toDouble()
        return doubleValue.takeIf { !it.isNaN() && !it.isInfinite() }
    }
    fun parseNotes(startMinAgo: Int, endMinAgo: Int): String {
        val olderTimeStamp = now - endMinAgo * 60 * 1000
        val moreRecentTimeStamp = now - startMinAgo * 60 * 1000
        var notes = ""
        recentNotes?.forEach { note ->
            if(note.timestamp > olderTimeStamp
                && note.timestamp <= moreRecentTimeStamp
                && !note.note.lowercase().contains("low treatment")
                && !note.note.lowercase().contains("less aggressive")
                && !note.note.lowercase().contains("more aggressive")
                && !note.note.lowercase().contains("too aggressive")
            ) {
                notes += if(notes.isEmpty()) "" else " "
                notes += note.note
            }
        }
        notes = notes.lowercase()
        notes.replace(","," ")
        notes.replace("."," ")
        notes.replace("!"," ")
        notes.replace("a"," ")
        notes.replace("an"," ")
        notes.replace("and"," ")
        notes.replace("\\s+"," ")
        return notes
    }

    init {
        injector.androidInjector().inject(this)
    }
}


