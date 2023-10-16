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
import app.aaps.core.interfaces.utils.Round
import app.aaps.core.interfaces.utils.SafeParse
import app.aaps.core.main.extensions.convertedToAbsolute
import app.aaps.core.main.extensions.getPassedDurationToTimeInMinutes
import app.aaps.core.main.extensions.plannedRemainingMinutes
import app.aaps.database.ValueWrapper
import app.aaps.database.entities.Bolus
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
import java.text.DecimalFormat
import java.text.DecimalFormatSymbols
import java.time.LocalTime
import java.util.Locale
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
    private var lastsmbtime = 0
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
    private var basalSMB = 0.0f
    private var aimilimit = 0.0f
    private var basaloapsaimirate = 0.0f
    private var CI = 0.0f
    private var variableSensitivity = 0.0f
    private var averageBeatsPerMinute = 0.0
    private var averageBeatsPerMinute180 = 0.0
    private var b30upperbg = 0.0
    private var b30upperdelta = 0.0
    private var profile = JSONObject()
    private var glucoseStatus = JSONObject()
    private var iobData: JSONArray? = null
    private var mealData = JSONObject()
    private var currentTemp = JSONObject()
    private var autosensData = JSONObject()
    private val path = File(Environment.getExternalStorageDirectory().toString())
    private val modelFile = File(path, "AAPS/ml/model.tflite")
    private val modelHBFile = File(path, "AAPS/ml/modelHB.tflite")

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
        var morningfactor = SafeParse.stringToDouble(sp.getString(R.string.key_oaps_aimi_morning_factor, "100")) / 100.0
        var afternoonfactor = SafeParse.stringToDouble(sp.getString(R.string.key_oaps_aimi_afternoon_factor, "100")) / 100.0
        var eveningfactor = SafeParse.stringToDouble(sp.getString(R.string.key_oaps_aimi_evening_factor, "100")) / 100.0
        if (hourOfDay in 1..11){
            smbToGive *= morningfactor.toFloat()
        }else if (hourOfDay in 12..18){
            smbToGive *= afternoonfactor.toFloat()
        }else if (hourOfDay in 18..23){
            smbToGive *= eveningfactor.toFloat()
        }
        smbToGive = applySafetyPrecautions(smbToGive)
        smbToGive = roundToPoint05(smbToGive)

        logDataToCsv(predictedSMB, smbToGive)
        logDataToCsvHB(predictedSMB,smbToGive)

        val constraintStr = " Max IOB: $maxIob <br/> Max SMB: $maxSMB"
        val glucoseStr = " bg: $bg <br/> targetBg: $targetBg <br/> " +
            "delta: $delta <br/> short avg delta: $shortAvgDelta <br/> long avg delta: $longAvgDelta <br/>" +
            " accelerating_up: $accelerating_up <br/> deccelerating_up: $deccelerating_up <br/> accelerating_down: $accelerating_down <br/> deccelerating_down: $deccelerating_down <br/> stable: $stable"
        val iobStr = " IOB: $iob <br/> tdd 7d/h: ${roundToPoint05(tdd7DaysPerHour)} <br/> " +
            "tdd 2d/h : ${roundToPoint05(tdd2DaysPerHour)} <br/> " +
            "tdd daily/h : ${roundToPoint05(tddPerHour)} <br/> " +
            "tdd 24h/h : ${roundToPoint05(tdd24HrsPerHour)}" +
            "basalaimi : $basalaimi <br/> basalsmb : $basalSMB <br/>"
        val profileStr = " Hour of day: $hourOfDay <br/> Weekend: $weekend <br/>" +
            " 5 Min Steps: $recentSteps5Minutes <br/> 10 Min Steps: $recentSteps10Minutes <br/> 15 Min Steps: $recentSteps15Minutes <br/>" +
            " 30 Min Steps: $recentSteps30Minutes <br/> 60 Min Steps: $recentSteps60Minutes <br/> 180 Min Steps: $recentSteps180Minutes <br/>" +
            "ISF : $variableSensitivity <br/> Heart Beat per minute(average past 5 minutes) : $averageBeatsPerMinute <br/> Heart Beat per minute(average past 180 minutes) : $averageBeatsPerMinute180"
        var mealStr = " COB: ${cob}g   Future: ${futureCarbs}g <br/> COB Age Min: $lastCarbAgeMin <br/><br/> "
        mealStr += "tags0to60minAgo: ${tags0to60minAgo}<br/> tags60to120minAgo: $tags60to120minAgo<br/> " +
            "tags120to180minAgo: $tags120to180minAgo<br/> tags180to240minAgo: $tags180to240minAgo"
        val reason = "The ai model predicted SMB of ${roundToPoint001(predictedSMB)}u and after safety requirements and rounding to .05, requested ${smbToGive}u to the pump" +
         ",<br/> Version du plugin OpenApsAIMI.1 ML.2, 12 octobre 2023"
        val determineBasalResultAIMISMB = DetermineBasalResultAIMISMB(injector, smbToGive,basaloapsaimirate, constraintStr, glucoseStr, iobStr, profileStr, mealStr, reason)

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
            "recentSteps5Minutes,recentSteps10Minutes,recentSteps15Minutes,recentSteps30Minutes,recentSteps60Minutes,recentSteps180Minutes," +
            "tags0to60minAgo,tags60to120minAgo,tags120to180minAgo,tags180to240minAgo," +
            "predictedSMB,maxIob,maxSMB,smbGiven\n"
        val valuesToRecord = "$dateStr,${dateUtil.now()},$hourOfDay,$weekend," +
            "$bg,$targetBg,$iob,$cob,$lastCarbAgeMin,$futureCarbs,$delta,$shortAvgDelta,$longAvgDelta," +
            "$tdd7DaysPerHour,$tdd2DaysPerHour,$tddPerHour,$tdd24HrsPerHour," +
            "$recentSteps5Minutes,$recentSteps10Minutes,$recentSteps15Minutes,$recentSteps30Minutes,$recentSteps60Minutes,$recentSteps180Minutes," +
            "$tags0to60minAgo,$tags60to120minAgo,$tags120to180minAgo,$tags180to240minAgo," +
            "$predictedSMB,$maxIob,$maxSMB,$smbToGive"

        val file = File(path, "AAPS/oapsaimi_records.csv")
        if (!file.exists()) {
            file.createNewFile()
            file.appendText(headerRow)
        }
        file.appendText(valuesToRecord + "\n")
    }
    private fun logDataToCsvHB(predictedSMB: Float, smbToGive: Float) {
        val dateStr = dateUtil.dateAndTimeString(dateUtil.now())

        val headerRow = "dateStr,dateLong,hourOfDay,weekend," +
            "bg,targetBg,iob,cob,lastCarbAgeMin,futureCarbs,delta,shortAvgDelta,longAvgDelta," +
            "accelerating_up,deccelerating_up,accelerating_down,deccelerating_down,stable," +
            "tdd7DaysPerHour,tdd2DaysPerHour,tddDailyPerHour,tdd24HrsPerHour," +
            "recentSteps5Minutes,recentSteps10Minutes,recentSteps15Minutes,recentSteps30Minutes,recentSteps60Minutes,averageBeatsPerMinute, averageBeatsPerMinute180," +
            "tags0to60minAgo,tags60to120minAgo,tags120to180minAgo,tags180to240minAgo," +
            "variableSensitivity,lastbolusage,predictedSMB,maxIob,maxSMB,smbGiven\n"
        val valuesToRecord = "$dateStr,${dateUtil.now()},$hourOfDay,$weekend," +
            "$bg,$targetBg,$iob,$cob,$lastCarbAgeMin,$futureCarbs,$delta,$shortAvgDelta,$longAvgDelta," +
            "$accelerating_up,$deccelerating_up,$accelerating_down,$deccelerating_down,$stable," +
            "$tdd7DaysPerHour,$tdd2DaysPerHour,$tddPerHour,$tdd24HrsPerHour," +
            "$recentSteps5Minutes,$recentSteps10Minutes,$recentSteps15Minutes,$recentSteps30Minutes,$recentSteps60Minutes,$recentSteps180Minutes," +
            "$averageBeatsPerMinute, $averageBeatsPerMinute180," +
            "$tags0to60minAgo,$tags60to120minAgo,$tags120to180minAgo,$tags180to240minAgo," +
            "$variableSensitivity,$predictedSMB,$maxIob,$maxSMB,$smbToGive"

        val file = File(path, "AAPS/oapsaimiHB_records.csv")
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
        if (delta < b30upperdelta && delta > 2 && bg < b30upperbg && lastsmbtime < 15){
            smbToGive = 0.0f
        }else if (delta < b30upperdelta && delta > 2 && bg < b30upperbg && lastsmbtime > 15){
            smbToGive = basalSMB
        }
        val safetysmb = recentSteps180Minutes > 1500 && bg < 130
        if (safetysmb){
            smbToGive /= 2
        }
        if (recentSteps5Minutes > 100 && recentSteps30Minutes > 500 && lastsmbtime < 20){
            smbToGive = 0.0f
        }
        // don't give insulin if dropping fast
        val droppingFast = bg < 150 && delta < -5
        val droppingFastAtHigh = bg < 200 && delta < -7
        val droppingVeryFast = delta < -10
        if (droppingFast || droppingFastAtHigh || droppingVeryFast) {
            smbToGive = 0.0f
        }
        if ((smbToGive < 0.0f || smbToGive === 0.0f ) && delta > 6 && bg > 130 && lastsmbtime > 15) {
            smbToGive = basalSMB
        } else if (smbToGive < 0.0f)
            smbToGive = 0.0f
        return smbToGive
    }

    private fun roundToPoint05(number: Float): Float {
        return (number * 20.0).roundToInt() / 20.0f
    }

    private fun roundToPoint001(number: Float): Float {
        return (number * 1000.0).roundToInt() / 1000.0f
    }


    private fun calculateSMBFromModel(): Float {
    var selectedModelFile: File?
    var modelInputs: FloatArray

    when {
        modelHBFile.exists() -> {
            selectedModelFile = modelHBFile
            modelInputs = floatArrayOf(
                hourOfDay.toFloat(), weekend.toFloat(),
                bg, targetBg, iob, delta, shortAvgDelta, longAvgDelta,
                tdd7DaysPerHour, tdd2DaysPerHour, tddPerHour, tdd24HrsPerHour, averageBeatsPerMinute.toFloat()
            )
        }
        modelFile.exists() -> {
            selectedModelFile = modelFile
            modelInputs = floatArrayOf(
                hourOfDay.toFloat(), weekend.toFloat(),
                bg, targetBg, iob, delta, shortAvgDelta, longAvgDelta,
                tdd7DaysPerHour, tdd2DaysPerHour, tddPerHour, tdd24HrsPerHour
            )
        }
        else -> {
            aapsLogger.error(LTag.APS, "NO Model found at specified location")
            return 0.0F
        }
    }

    val interpreter = Interpreter(selectedModelFile!!)
    val output = arrayOf(floatArrayOf(0.0F))
    interpreter.run(modelInputs, output)
    interpreter.close()
    var smbToGive = output[0][0].toString().replace(',', '.').toDouble()

    val formatter = DecimalFormat("#.####", DecimalFormatSymbols(Locale.US))
    smbToGive = formatter.format(smbToGive).toDouble()

    return smbToGive.toFloat()
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
        this.delta = glucoseStatus.delta.toFloat()
        this.shortAvgDelta = glucoseStatus.shortAvgDelta.toFloat()
        this.longAvgDelta = glucoseStatus.longAvgDelta.toFloat()

        this.accelerating_up = if (delta > 2 && delta - longAvgDelta > 2) 1 else 0
        this.deccelerating_up = if (delta > 0 && (delta < shortAvgDelta || delta < longAvgDelta)) 1 else 0
        this.accelerating_down = if (delta < -2 && delta - longAvgDelta < -2) 1 else 0
        this.deccelerating_down = if (delta < 0 && (delta > shortAvgDelta || delta > longAvgDelta)) 1 else 0
        this.stable = if (delta>-3 && delta<3 && shortAvgDelta>-3 && shortAvgDelta<3 && longAvgDelta>-3 && longAvgDelta<3) 1 else 0
        var tdd7P = SafeParse.stringToDouble(sp.getString(R.string.key_tdd7, "50"))
        var tdd7Days = tddCalculator.averageTDD(tddCalculator.calculate(7, allowMissingDays = false))?.totalAmount?.toFloat() ?: 0.0f
        if (tdd7Days == 0.0f) tdd7Days = tdd7P.toFloat()
        this.tdd7DaysPerHour = tdd7Days / 24

        var tdd2Days = tddCalculator.averageTDD(tddCalculator.calculate(2, allowMissingDays = false))?.totalAmount?.toFloat() ?: 0.0f
        if (tdd2Days == 0.0f) tdd2Days = tdd7P.toFloat()
        this.tdd2DaysPerHour = tdd2Days / 24
        val tddLast4H = tdd2DaysPerHour.toDouble() * 4
        var tddDaily = tddCalculator.averageTDD(tddCalculator.calculate(1, allowMissingDays = false))?.totalAmount?.toFloat() ?: 0.0f
        if (tddDaily == 0.0f) tddDaily = tdd7P.toFloat()
        this.tddPerHour = tddDaily / 24

        val tdd24Hrs = tddCalculator.calculateDaily(-24, 0)?.totalAmount?.toFloat() ?: 0.0f
        this.tdd24HrsPerHour = tdd24Hrs / 24
        val tddLast8to4H  = tdd24HrsPerHour.toDouble() * 4
        val insulin = activePlugin.activeInsulin

        val insulinDivisor = when {
            insulin.peak >= 35 -> 55 // lyumjev peak: 45
            insulin.peak > 45  -> 65 // ultra rapid peak: 55
            else               -> 75 // rapid peak: 75
        }
        val tddWeightedFromLast8H = ((1.4 * tddLast4H) + (0.6 * tddLast8to4H)) * 3
        var tdd = (tddWeightedFromLast8H * 0.33) + (tdd7Days.toDouble() * 0.34) + (tddDaily.toDouble() * 0.33)
        val dynISFadjust = SafeParse.stringToDouble(sp.getString(R.string.key_DynISFAdjust, "100")) / 100.0
        tdd *= dynISFadjust

        this.variableSensitivity = Round.roundTo(1800 / (tdd * (ln((glucoseStatus.glucose / insulinDivisor) + 1))), 0.1).toFloat()
        var beatsPerMinuteValues: List<Int>
        var beatsPerMinuteValues180: List<Int>
        val timeMillisNow = System.currentTimeMillis()
        val timeMillis5 = System.currentTimeMillis() - 5 * 60 * 1000 // 5 minutes en millisecondes
        val timeMillis10 = System.currentTimeMillis() - 10 * 60 * 1000 // 10 minutes en millisecondes
        val timeMillis15 = System.currentTimeMillis() - 15 * 60 * 1000 // 15 minutes en millisecondes
        val timeMillis30 = System.currentTimeMillis() - 30 * 60 * 1000 // 30 minutes en millisecondes
        val timeMillis60 = System.currentTimeMillis() - 60 * 60 * 1000 // 60 minutes en millisecondes
        val timeMillis180 = System.currentTimeMillis() - 180 * 60 * 1000 // 180 minutes en millisecondes
        val stepsCountList5 = repository.getLastStepsCountFromTimeToTime(timeMillis5, timeMillisNow)
        val stepsCount5 = stepsCountList5?.steps5min ?: 0

        val stepsCountList10 = repository.getLastStepsCountFromTimeToTime(timeMillis10, timeMillisNow)
        val stepsCount10 = stepsCountList10?.steps10min ?: 0

        val stepsCountList15 = repository.getLastStepsCountFromTimeToTime(timeMillis15, timeMillisNow)
        val stepsCount15 = stepsCountList15?.steps15min ?: 0

        val stepsCountList30 = repository.getLastStepsCountFromTimeToTime(timeMillis30, timeMillisNow)
        val stepsCount30 = stepsCountList30?.steps30min ?: 0

        val stepsCountList60 = repository.getLastStepsCountFromTimeToTime(timeMillis60, timeMillisNow)
        val stepsCount60 = stepsCountList60?.steps60min ?: 0

        val stepsCountList180 = repository.getLastStepsCountFromTimeToTime(timeMillis180, timeMillisNow)
        val stepsCount180 = stepsCountList180?.steps180min ?: 0
        if (sp.getBoolean(R.string.count_steps_watch, false)===true) {
            this.recentSteps5Minutes = stepsCount5
            this.recentSteps10Minutes = stepsCount10
            this.recentSteps15Minutes = stepsCount15
            this.recentSteps30Minutes = stepsCount30
            this.recentSteps60Minutes = stepsCount60
            this.recentSteps180Minutes = stepsCount180
        }else{
            this.recentSteps5Minutes = StepService.getRecentStepCount5Min()
            this.recentSteps10Minutes = StepService.getRecentStepCount10Min()
            this.recentSteps15Minutes = StepService.getRecentStepCount15Min()
            this.recentSteps30Minutes = StepService.getRecentStepCount30Min()
            this.recentSteps60Minutes = StepService.getRecentStepCount60Min()
            this.recentSteps180Minutes = StepService.getRecentStepCount180Min()
        }
        try {
            val heartRates = repository.getHeartRatesFromTimeToTime(timeMillis5,timeMillisNow)
            beatsPerMinuteValues = heartRates.map { it.beatsPerMinute.toInt() } // Extract beatsPerMinute values from heartRates
            this.averageBeatsPerMinute = if (beatsPerMinuteValues.isNotEmpty()) {
                beatsPerMinuteValues.average()
            } else {
                80.0 // or some other default value
            }

        } catch (e: Exception) {
            // Log that watch is not connected
            //println("Watch is not connected. Using default values for heart rate data.")
            // Réaffecter les variables à leurs valeurs par défaut
            beatsPerMinuteValues = listOf(80)
            this.averageBeatsPerMinute = 80.0
        }
        try {

            val heartRates180 = repository.getHeartRatesFromTimeToTime(timeMillis180,timeMillisNow)
            beatsPerMinuteValues180 = heartRates180.map { it.beatsPerMinute.toInt() } // Extract beatsPerMinute values from heartRates
            this.averageBeatsPerMinute180 = if (beatsPerMinuteValues180.isNotEmpty()) {
                beatsPerMinuteValues180.average()
            } else {
                10.0 // or some other default value
            }

        } catch (e: Exception) {
            // Log that watch is not connected
            //println("Watch is not connected. Using default values for heart rate data.")
            // Réaffecter les variables à leurs valeurs par défaut
            beatsPerMinuteValues180 = listOf(10)
            this.averageBeatsPerMinute180 = 10.0
        }
        if (tdd2Days != null && tdd2Days != 0.0f) {
            this.basalaimi = (tdd2Days / SafeParse.stringToDouble(sp.getString(R.string.key_aimiweight, "50"))).toFloat()
        } else {
            this.basalaimi = (tdd7P / SafeParse.stringToDouble(sp.getString(R.string.key_aimiweight, "50"))).toFloat()
        }
        if (tdd2Days != null && tdd2Days != 0.0f) {
            this.CI = 450 / tdd2Days
        } else {

            this.CI = (450 / tdd7P).toFloat()
        }

        val choKey = SafeParse.stringToDouble(sp.getString(R.string.key_cho, "50"))
        if (CI != 0.0f && CI != Float.POSITIVE_INFINITY && CI != Float.NEGATIVE_INFINITY) {
            this.aimilimit = (choKey / CI).toFloat()
        } else {
            this.aimilimit = (choKey / profile.getIc()).toFloat()
        }
        val timenow = LocalTime.now()
        val sixAM = LocalTime.of(6, 0)
        if (averageBeatsPerMinute != 0.0) {
            if (averageBeatsPerMinute >= averageBeatsPerMinute180 && recentSteps5Minutes > 100 && recentSteps10Minutes > 200) {
                this.basalaimi = (basalaimi * 0.65).toFloat()
            } else if (averageBeatsPerMinute180 != 10.0 && averageBeatsPerMinute > averageBeatsPerMinute180 && bg >= 130 && recentSteps10Minutes === 0 && timenow > sixAM) {
                this.basalaimi = (basalaimi * 1.3).toFloat()
            } else if (averageBeatsPerMinute180 != 10.0 && averageBeatsPerMinute < averageBeatsPerMinute180 && recentSteps10Minutes === 0 && bg >= 130) {
                this.basalaimi = (basalaimi * 1.2).toFloat()
            }
        }
        this.b30upperbg = SafeParse.stringToDouble(sp.getString(R.string.key_B30_upperBG, "130"))
        this.b30upperdelta = SafeParse.stringToDouble(sp.getString(R.string.key_B30_upperdelta, "10"))
        val b30duration = SafeParse.stringToDouble(sp.getString(R.string.key_B30_duration, "20"))

        this.basalSMB = (((basalaimi * delta) / 60) * b30duration).toFloat()

        if (delta < b30upperdelta && delta > 1 && bg < b30upperbg && lastsmbtime > 10){
           this.basaloapsaimirate = basalSMB
        }else{
            this.basaloapsaimirate = 0.0f
        }

        val variableSensitivityDouble = variableSensitivity.toDoubleSafely()
        if (variableSensitivityDouble != null) {
            if (recentSteps5Minutes > 100 && recentSteps10Minutes > 200 && bg < 130 && delta < 10|| recentSteps180Minutes > 1500 && bg < 130 && delta < 10) variableSensitivity *= 1.5f
            if (recentSteps30Minutes > 500 && recentSteps5Minutes >= 0 && recentSteps5Minutes < 100 && bg < 130 && delta < 10) variableSensitivity *= 1.3f

    } else {
        variableSensitivity = profile.getIsfMgdl().toFloat()
    }
        val getlastBolusSMB = repository.getLastBolusRecordOfTypeWrapped(Bolus.Type.SMB).blockingGet()
        val lastBolusSMBTime = if (getlastBolusSMB is ValueWrapper.Existing) getlastBolusSMB.value.timestamp else 0L
        this.lastsmbtime = ((now - lastBolusSMBTime) / (60 * 1000)).toDouble().roundToInt().toLong().toInt()

    this.maxIob = sp.getDouble(R.string.key_openapssmb_max_iob, 5.0).toFloat()
        this.maxSMB = sp.getDouble(R.string.key_openapsaimi_max_smb, 1.0).toFloat()

        // profile.dia
        val abs = iobCobCalculator.calculateAbsoluteIobFromBaseBasals(System.currentTimeMillis())
        val absIob = abs.iob
        val absNet = abs.netInsulin
        val absBasal = abs.basaliob

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
        this.profile.put("carb_ratio", CI)
        this.profile.put("sens", variableSensitivity)
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


    private fun determineNoteBasedOnBg(bg: Double): String {
        return when {
            bg > 170 -> "more aggressive"
            bg in 90.0..100.0 -> "less aggressive"
            bg in 80.0..89.9 -> "too aggressive" // Vous pouvez ajuster ces valeurs selon votre logique
            bg < 80 -> "low treatment"
            else -> "normal" // Vous pouvez définir un autre message par défaut pour les cas non couverts
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
        var recentNotes2: MutableList<String> = mutableListOf()

        val autoNote = determineNoteBasedOnBg(bg.toDouble())
        recentNotes2.add(autoNote)

        recentNotes?.forEach { note ->
            if(note.timestamp > olderTimeStamp
                && note.timestamp <= moreRecentTimeStamp
                && !note.note.lowercase().contains("low treatment")
                && !note.note.lowercase().contains("less aggressive")
                && !note.note.lowercase().contains("more aggressive")
                && !note.note.lowercase().contains("too aggressive")
            ) {
                notes += if(notes.isEmpty()) recentNotes2 else " "
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


