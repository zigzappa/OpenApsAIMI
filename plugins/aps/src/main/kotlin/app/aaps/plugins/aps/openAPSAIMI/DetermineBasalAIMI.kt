package app.aaps.plugins.aps.openAPSAIMI

import android.os.Environment
import app.aaps.core.data.model.BS
import app.aaps.core.data.model.UE
import app.aaps.core.interfaces.aps.APSResult
import app.aaps.core.interfaces.aps.AutosensResult
import app.aaps.core.interfaces.aps.CurrentTemp
import app.aaps.core.interfaces.aps.GlucoseStatus
import app.aaps.core.interfaces.aps.IobTotal
import app.aaps.core.interfaces.aps.MealData
import app.aaps.core.interfaces.aps.OapsProfile
import app.aaps.core.interfaces.aps.Predictions
import app.aaps.core.interfaces.aps.RT
import app.aaps.core.interfaces.db.PersistenceLayer
import app.aaps.core.interfaces.profile.ProfileUtil
import app.aaps.core.interfaces.stats.TddCalculator
import app.aaps.core.interfaces.stats.TirCalculator
import app.aaps.core.interfaces.utils.Round
import app.aaps.core.interfaces.utils.DateUtil
import app.aaps.core.keys.BooleanKey
import app.aaps.core.keys.DoubleKey
import app.aaps.core.keys.IntKey
import app.aaps.core.keys.Preferences
import org.tensorflow.lite.Interpreter
import java.io.File
import java.text.DecimalFormat
import java.text.DecimalFormatSymbols
import java.time.Instant
import java.time.LocalTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Calendar
import java.util.Locale
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt

@Singleton
class DetermineBasalaimiSMB @Inject constructor(
    private val profileUtil: ProfileUtil
) {
    @Inject lateinit var preferences: Preferences
    @Inject lateinit var persistenceLayer: PersistenceLayer
    @Inject lateinit var tddCalculator: TddCalculator
    @Inject lateinit var tirCalculator: TirCalculator
    @Inject lateinit var dateUtil: DateUtil
    private val consoleError = mutableListOf<String>()
    private val consoleLog = mutableListOf<String>()
    private val path = File(Environment.getExternalStorageDirectory().toString())
    private val modelFile = File(path, "AAPS/ml/model.tflite")
    private val modelFileUAM = File(path, "AAPS/ml/modelUAM.tflite")
    private val csvfile = File(path, "AAPS/oapsaimiML2_records.csv")
    private var predictedSMB = 0.0f
    private var variableSensitivity = 0.0f
    private var averageBeatsPerMinute = 0.0
    private var averageBeatsPerMinute10 = 0.0
    private var averageBeatsPerMinute60 = 0.0
    private var averageBeatsPerMinute180 = 0.0
    private var now = System.currentTimeMillis()
    private var iob = 0.0f
    private var cob = 0.0f
    private var predictedBg = 0.0f
    private var lastCarbAgeMin: Int = 0
    private var futureCarbs = 0.0f
    private var enablebasal: Boolean = false
    private var recentNotes: List<UE>? = null
    private var tags0to60minAgo = ""
    private var tags60to120minAgo = ""
    private var tags120to180minAgo = ""
    private var tags180to240minAgo = ""
    private var tir1DAYabove: Double = 0.0
    private var currentTIRLow: Double = 0.0
    private var currentTIRRange: Double = 0.0
    private var currentTIRAbove: Double = 0.0
    private var lastHourTIRLow: Double = 0.0
    private var lastHourTIRLow100: Double = 0.0
    private var lastHourTIRabove170: Double = 0.0
    private var bg = 0.0
    private var targetBg = 100.0f
    private var normalBgThreshold = 140.0f
    private var delta = 0.0f
    private var shortAvgDelta = 0.0f
    private var longAvgDelta = 0.0f
    private var lastsmbtime = 0
    private var accelerating_up: Int = 0
    private var deccelerating_up: Int = 0
    private var accelerating_down: Int = 0
    private var deccelerating_down: Int = 0
    private var stable: Int = 0
    private var maxIob = 0.0
    private var maxSMB = 1.0
    private var maxSMBHB = 1.0
    private var lastBolusSMBUnit = 0.0f
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
    private var sleepTime = false
    private var sportTime = false
    private var snackTime = false
    private var lowCarbTime = false
    private var highCarbTime = false
    private var mealTime = false
    private var fastingTime = false
    private var stopTime = false
    private var iscalibration = false
    private var mealruntime: Long = 0
    private var highCarbrunTime: Long = 0
    private var snackrunTime: Long = 0
    private var intervalsmb = 5

    private fun Double.toFixed2(): String = DecimalFormat("0.00#").format(round(this, 2))

    fun round_basal(value: Double): Double = value

    // Rounds value to 'digits' decimal places
    // different for negative numbers fun round(value: Double, digits: Int): Double = BigDecimal(value).setScale(digits, RoundingMode.HALF_EVEN).toDouble()
    fun round(value: Double, digits: Int): Double {
        if (value.isNaN()) return Double.NaN
        val scale = 10.0.pow(digits.toDouble())
        return Math.round(value * scale) / scale
    }

    fun Double.withoutZeros(): String = DecimalFormat("0.##").format(this)
//    fun round(value: Double): Int = value.roundToInt()
fun round(value: Double): Int {
    if (value.isNaN()) return 0
    val scale = 10.0.pow(2.0)
    return (Math.round(value * scale) / scale).toInt()
}

    // we expect BG to rise or fall at the rate of BGI,
    // adjusted by the rate at which BG would need to rise /
    // fall to get eventualBG to target over 2 hours
    fun calculate_expected_delta(targetBg: Double, eventualBg: Double, bgi: Double): Double {
        // (hours * mins_per_hour) / 5 = how many 5 minute periods in 2h = 24
        val fiveMinBlocks = (2 * 60) / 5
        val targetDelta = targetBg - eventualBg
        return /* expectedDelta */ round(bgi + (targetDelta / fiveMinBlocks), 1)
    }

    fun convert_bg(value: Double): String =
        profileUtil.fromMgdlToStringInUnits(value).replace("-0.0", "0.0")
    //DecimalFormat("0.#").format(profileUtil.fromMgdlToUnits(value))
    //if (profile.out_units === "mmol/L") round(value / 18, 1).toFixed(1);
    //else Math.round(value);

    fun enable_smb(profile: OapsProfile, microBolusAllowed: Boolean, meal_data: MealData, target_bg: Double): Boolean {
        // disable SMB when a high temptarget is set
        if (!microBolusAllowed) {
            consoleError.add("SMB disabled (!microBolusAllowed)")
            return false
        } else if (!profile.allowSMB_with_high_temptarget && profile.temptargetSet && target_bg > 100) {
            consoleError.add("SMB disabled due to high temptarget of $target_bg")
            return false
        }

        // enable SMB/UAM if always-on (unless previously disabled for high temptarget)
        if (profile.enableSMB_always) {
            consoleError.add("SMB enabled due to enableSMB_always")
            return true
        }

        // enable SMB/UAM (if enabled in preferences) while we have COB
        if (profile.enableSMB_with_COB && meal_data.mealCOB != 0.0) {
            consoleError.add("SMB enabled for COB of ${meal_data.mealCOB}")
            return true
        }

        // enable SMB/UAM (if enabled in preferences) for a full 6 hours after any carb entry
        // (6 hours is defined in carbWindow in lib/meal/total.js)
        if (profile.enableSMB_after_carbs && meal_data.carbs != 0.0) {
            consoleError.add("SMB enabled for 6h after carb entry")
            return true
        }

        // enable SMB/UAM (if enabled in preferences) if a low temptarget is set
        if (profile.enableSMB_with_temptarget && (profile.temptargetSet && target_bg < 100)) {
            consoleError.add("SMB enabled for temptarget of ${convert_bg(target_bg)}")
            return true
        }

        consoleError.add("SMB disabled (no enableSMB preferences active or no condition satisfied)")
        return false
    }

    fun reason(rT: RT, msg: String) {
        if (rT.reason.toString().isNotEmpty()) rT.reason.append(". ")
        rT.reason.append(msg)
        consoleError.add(msg)
    }

    private fun getMaxSafeBasal(profile: OapsProfile): Double =
        min(profile.max_basal, min(profile.max_daily_safety_multiplier * profile.max_daily_basal, profile.current_basal_safety_multiplier * profile.current_basal))

    fun setTempBasal(_rate: Double, duration: Int, profile: OapsProfile, rT: RT, currenttemp: CurrentTemp): RT {
        //var maxSafeBasal = Math.min(profile.max_basal, 3 * profile.max_daily_basal, 4 * profile.current_basal);

        val maxSafeBasal = getMaxSafeBasal(profile)
        var rate = _rate

        if (rate < 0) rate = 0.0
        else if (rate > maxSafeBasal) rate = maxSafeBasal

        val suggestedRate = round_basal(rate)
        if (currenttemp.duration > (duration - 10) && currenttemp.duration <= 120 && suggestedRate <= currenttemp.rate * 1.2 && suggestedRate >= currenttemp.rate * 0.8 && duration > 0) {
            rT.reason.append(" ${currenttemp.duration}m left and ${currenttemp.rate.withoutZeros()} ~ req ${suggestedRate.withoutZeros()}U/hr: no temp required")
            return rT
        }

        if (suggestedRate == profile.current_basal) {
            if (profile.skip_neutral_temps) {
                if (currenttemp.duration > 0) {
                    reason(rT, "Suggested rate is same as profile rate, a temp basal is active, canceling current temp")
                    rT.duration = 0
                    rT.rate = 0.0
                    return rT
                } else {
                    reason(rT, "Suggested rate is same as profile rate, no temp basal is active, doing nothing")
                    return rT
                }
            } else {
                reason(rT, "Setting neutral temp basal of ${profile.current_basal}U/hr")
                rT.duration = duration
                rT.rate = suggestedRate
                return rT
            }
        } else {
            rT.duration = duration
            rT.rate = suggestedRate
            return rT
        }
    }
    private fun logDataMLToCsv(predictedSMB: Float, smbToGive: Float) {

        val usFormatter = DateTimeFormatter.ofPattern("MM/dd/yyyy HH:mm")
        val dateStr = dateUtil.dateAndTimeString(dateUtil.now()).format(usFormatter)


        val headerRow = "dateStr, bg, iob, cob, delta, shortAvgDelta, longAvgDelta, tdd7DaysPerHour, tdd2DaysPerHour, tddPerHour, tdd24HrsPerHour, predictedSMB, smbGiven\n"
        val valuesToRecord = "$dateStr," +
            "$bg,$iob,$cob,$delta,$shortAvgDelta,$longAvgDelta," +
            "$tdd7DaysPerHour,$tdd2DaysPerHour,$tddPerHour,$tdd24HrsPerHour," +
            "$predictedSMB,$smbToGive"

        val file = File(path, "AAPS/oapsaimiML2_records.csv")
        if (!file.exists()) {
            file.createNewFile()
            file.appendText(headerRow)
        }
        file.appendText(valuesToRecord + "\n")
    }
    private fun logDataToCsv(predictedSMB: Float, smbToGive: Float) {

        val usFormatter = DateTimeFormatter.ofPattern("MM/dd/yyyy HH:mm")
        val dateStr = dateUtil.dateAndTimeString(dateUtil.now()).format(usFormatter)


        val headerRow = "dateStr,dateLong,hourOfDay,weekend," +
            "bg,targetBg,iob,cob,lastCarbAgeMin,futureCarbs,delta,shortAvgDelta,longAvgDelta," +
            "tdd7DaysPerHour,tdd2DaysPerHour,tddPerHour,tdd24HrsPerHour," +
            "recentSteps5Minutes,recentSteps10Minutes,recentSteps15Minutes,recentSteps30Minutes,recentSteps60Minutes,recentSteps180Minutes," +
            "tags0to60minAgo,tags60to120minAgo,tags120to180minAgo,tags180to240minAgo," +
            "predictedSMB,maxIob,maxSMB,smbGiven\n"
        val valuesToRecord = "$dateStr,$hourOfDay,$weekend," +
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
        val (conditionResult, _) = isCriticalSafetyCondition()
        if (conditionResult) return 0.0f

        if (isSportSafetyCondition()) return 0.0f
        // Ajustements basés sur des conditions spécifiques
        smbToGive = applySpecificAdjustments(smbToGive)

        smbToGive = finalizeSmbToGive(smbToGive)
        // Appliquer les limites maximum
        smbToGive = applyMaxLimits(smbToGive)

        return smbToGive
    }
    private fun applyMaxLimits(smbToGive: Float): Float {
        var result = smbToGive

        // Vérifiez d'abord si smbToGive dépasse maxSMB
        if (result > maxSMB) {
            result = maxSMB.toFloat()
        }
        // Ensuite, vérifiez si la somme de iob et smbToGive dépasse maxIob
        if (iob + result > maxIob) {
            result = maxIob.toFloat() - iob
        }

        return result
    }

    private fun isMealModeCondition(): Boolean{
        val pbolusM: Double = preferences.get(DoubleKey.OApsAIMIMealPrebolus)
        val modeMealPB = mealruntime in 0..7 && lastBolusSMBUnit != pbolusM.toFloat() && mealTime
        return modeMealPB
    }
    private fun isHighCarbModeCondition(): Boolean{
        val pbolusHC: Double = preferences.get(DoubleKey.OApsAIMIHighCarbPrebolus)
        val modeHcPB = highCarbrunTime in 0..7 && lastBolusSMBUnit != pbolusHC.toFloat() && highCarbTime
        return modeHcPB
    }

    private fun issnackModeCondition(): Boolean{
        val pbolussnack: Double = preferences.get(DoubleKey.OApsAIMISnackPrebolus)
        val modesnackPB = snackrunTime in 0..7 && lastBolusSMBUnit != pbolussnack.toFloat() && snackTime
        return modesnackPB
    }
    private fun roundToPoint05(number: Float): Float {
        return (number * 20.0).roundToInt() / 20.0f
    }

    private fun roundToPoint001(number: Float): Float {
        return (number * 1000.0).roundToInt() / 1000.0f
    }
    private fun isCriticalSafetyCondition(): Pair<Boolean, String> {
        val conditionsTrue = mutableListOf<String>()
        val nosmb = iob >= 2*maxSMB && bg < 110 && delta < 10 && !isMealModeCondition() && !isHighCarbModeCondition()
        if (nosmb) conditionsTrue.add("nosmb")
        val fasting = fastingTime
        if (fasting) conditionsTrue.add("fasting")
        val nightTrigger = LocalTime.now().run { (hour in 23..23 || hour in 0..6) } && delta > 10 && cob === 0.0f
        if (nightTrigger) conditionsTrue.add("nightTrigger")
        val belowMinThreshold = bg < 120 && delta < 10 && !isMealModeCondition() && !isHighCarbModeCondition()
        if (belowMinThreshold) conditionsTrue.add("belowMinThreshold")
        val isNewCalibration = iscalibration && delta > 10
        if (isNewCalibration) conditionsTrue.add("isNewCalibration")
        val belowTargetAndDropping = bg < targetBg && delta < -2 && !isMealModeCondition() && !isHighCarbModeCondition()
        if (belowTargetAndDropping) conditionsTrue.add("belowTargetAndDropping")
        val belowTargetAndStableButNoCob = bg < targetBg - 15 && shortAvgDelta <= 2 && cob <= 10 && !isMealModeCondition() && !isHighCarbModeCondition()
        if (belowTargetAndStableButNoCob) conditionsTrue.add("belowTargetAndStableButNoCob")
        val droppingFast = bg < 150 && delta < -5
        if (droppingFast) conditionsTrue.add("droppingFast")
        val droppingFastAtHigh = bg < 240 && delta < -7
        if (droppingFastAtHigh) conditionsTrue.add("droppingFastAtHigh")
        val droppingVeryFast = delta < -11
        if (droppingVeryFast) conditionsTrue.add("droppingVeryFast")
        val prediction = predictedBg < targetBg && bg < 135
        if (prediction) conditionsTrue.add("prediction")
        val interval = predictedBg < targetBg && delta > 10 && iob >= maxSMB/2 && lastsmbtime < 10
        if (interval) conditionsTrue.add("interval")
        val targetinterval = targetBg >= 120 && delta > 0 && iob >= maxSMB/2 && lastsmbtime < 15
        if (targetinterval) conditionsTrue.add("targetinterval")
        val stablebg = delta>-3 && delta<3 && shortAvgDelta>-3 && shortAvgDelta<3 && longAvgDelta>-3 && longAvgDelta<3 && bg < 180 && !isMealModeCondition() && !isHighCarbModeCondition()
        if (stablebg) conditionsTrue.add("stablebg")
        val acceleratingDown = delta < -2 && delta - longAvgDelta < -2 && lastsmbtime < 15
        if (acceleratingDown) conditionsTrue.add("acceleratingDown")
        val decceleratingdown = delta < 0 && (delta > shortAvgDelta || delta > longAvgDelta) && lastsmbtime < 15
        if (decceleratingdown) conditionsTrue.add("decceleratingdown")
        val result = belowTargetAndDropping || belowTargetAndStableButNoCob ||
            droppingFast || droppingFastAtHigh || droppingVeryFast || prediction || interval || targetinterval ||
            fasting || nosmb || nightTrigger || isNewCalibration || stablebg || belowMinThreshold || acceleratingDown || decceleratingdown

        val conditionsTrueString = if (conditionsTrue.isNotEmpty()) {
            conditionsTrue.joinToString(", ")
        } else {
            "No conditions met"
        }

        return Pair(result, conditionsTrueString)
    }
    private fun isSportSafetyCondition(): Boolean {
        val sport = targetBg >= 140 && recentSteps5Minutes >= 200 && recentSteps10Minutes >= 500
        val sport1 = targetBg >= 140 && recentSteps5Minutes >= 200 && averageBeatsPerMinute > averageBeatsPerMinute60
        val sport2 = recentSteps5Minutes >= 200 && averageBeatsPerMinute > averageBeatsPerMinute60
        val sport3 = recentSteps5Minutes >= 200 && recentSteps10Minutes >= 500
        val sport4 = targetBg >= 140
        val sport5= sportTime

        return sport || sport1 || sport2 || sport3 || sport4 || sport5

    }
    private fun applySpecificAdjustments(smbToGive: Float): Float {
        var result = smbToGive
        val intervalSMBsnack = preferences.get(IntKey.OApsAIMISnackinterval)
        val intervalSMBmeal = preferences.get(IntKey.OApsAIMImealinterval)
        val intervalSMBsleep = preferences.get(IntKey.OApsAIMISleepinterval)
        val intervalSMBhc = preferences.get(IntKey.OApsAIMIHCinterval)
        val intervalSMBhighBG = preferences.get(IntKey.OApsAIMIHighBGinterval)
        val belowTargetAndDropping = bg < targetBg

        if (shouldApplyIntervalAdjustment(intervalSMBsnack, intervalSMBmeal, intervalSMBsleep, intervalSMBhc,intervalSMBhighBG)) {
            result = 0.0f
        } else if (shouldApplySafetyAdjustment()) {
            result /= 2
            this.intervalsmb = 10
        } else if (shouldApplyTimeAdjustment()) {
            result = 0.0f
            this.intervalsmb = 10
        }

        if (shouldApplyStepAdjustment()) {
            result = 0.0f
        }
        if (belowTargetAndDropping) result /= 2

        return result
    }

    private fun shouldApplyIntervalAdjustment(intervalSMBsnack: Int, intervalSMBmeal: Int, intervalSMBsleep: Int, intervalSMBhc: Int, intervalSMBhighBG: Int): Boolean {
        return (lastsmbtime < intervalSMBsnack && snackTime) || (lastsmbtime < intervalSMBmeal && mealTime) ||
            (lastsmbtime < intervalSMBsleep && sleepTime) || (lastsmbtime < intervalSMBhc && highCarbTime) || (lastsmbtime < intervalSMBhighBG && bg > 180)
    }

    private fun shouldApplySafetyAdjustment(): Boolean {
        val safetysmb = recentSteps180Minutes > 1500 && bg < 140
        return (safetysmb || lowCarbTime) && lastsmbtime >= 15
    }

    private fun shouldApplyTimeAdjustment(): Boolean {
        val safetysmb = recentSteps180Minutes > 1500 && bg < 140
        return (safetysmb || lowCarbTime) && lastsmbtime < 15
    }

    private fun shouldApplyStepAdjustment(): Boolean {
        return recentSteps5Minutes > 100 && recentSteps30Minutes > 500 && lastsmbtime < 20
    }
    private fun finalizeSmbToGive(smbToGive: Float): Float {
        var result = smbToGive
        // Assurez-vous que smbToGive n'est pas négatif
        if (result < 0.0f) {
            result = 0.0f
        }
        return result
    }
    private fun calculateSMBFromModel(): Float {
        val selectedModelFile: File?
        val modelInputs: FloatArray

        when {
            cob > 0 && lastCarbAgeMin < 240 && modelFile.exists() -> {
                selectedModelFile = modelFile
                modelInputs = floatArrayOf(
                    hourOfDay.toFloat(), weekend.toFloat(),
                    bg.toFloat(), targetBg, iob, cob, lastCarbAgeMin.toFloat(), futureCarbs, delta, shortAvgDelta, longAvgDelta
                )
            }

            modelFileUAM.exists()   -> {
                selectedModelFile = modelFileUAM
                modelInputs = floatArrayOf(
                    hourOfDay.toFloat(), weekend.toFloat(),
                    bg.toFloat(), targetBg, iob, delta, shortAvgDelta, longAvgDelta,
                    tdd7DaysPerHour, tdd2DaysPerHour, tddPerHour, tdd24HrsPerHour,
                    recentSteps5Minutes.toFloat(),recentSteps10Minutes.toFloat(),recentSteps15Minutes.toFloat(),recentSteps30Minutes.toFloat(),recentSteps60Minutes.toFloat(),recentSteps180Minutes.toFloat()
                )
            }

            else                 -> {
                return 0.0F
            }
        }

        val interpreter = Interpreter(selectedModelFile)
        val output = arrayOf(floatArrayOf(0.0F))
        interpreter.run(modelInputs, output)
        interpreter.close()
        var smbToGive = output[0][0].toString().replace(',', '.').toDouble()

        val formatter = DecimalFormat("#.####", DecimalFormatSymbols(Locale.US))
        smbToGive = formatter.format(smbToGive).toDouble()

        return smbToGive.toFloat()
    }
    private fun neuralnetwork5(delta: Float, shortAvgDelta: Float, longAvgDelta: Float, predictedSMB: Float, basalaimi: Float): Pair<Float, Float> {
        val minutesToConsider = if (tir1DAYabove > 15) 15000.0 else 5000.0
        val linesToConsider = (minutesToConsider / 5).toInt()
        var totalDifference: Float
        val maxIterations = 10000.0
        var differenceWithinRange = false
        var finalRefinedSMB: Float = calculateSMBFromModel()
        val maxGlobalIterations = 5 // Nombre maximum d'itérations globales
        var globalConvergenceReached = false
        var refineBasalAimi = basalaimi

        for (globalIteration in 1..maxGlobalIterations) {
            var globalIterationCount = 0
            var iterationCount = 0

            while (globalIterationCount < maxGlobalIterations && !globalConvergenceReached) {

                val allLines = csvfile.readLines()
                val headerLine = allLines.first()
                val headers = headerLine.split(",").map { it.trim() }
                val colIndices = listOf("bg", "iob", "cob", "delta", "shortAvgDelta", "longAvgDelta", "predictedSMB").map { headers.indexOf(it) }
                val targetColIndex = headers.indexOf("smbGiven")

                val lines = if (allLines.size > linesToConsider) allLines.takeLast(linesToConsider + 1) else allLines // +1 pour inclure l'en-tête

                val inputs = mutableListOf<FloatArray>()
                val targets = mutableListOf<DoubleArray>()
                var isAggressiveResponseNeeded = false
                for (line in lines.drop(1)) { // Ignorer l'en-tête
                    val cols = line.split(",").map { it.trim() }

                    val input = colIndices.mapNotNull { index -> cols.getOrNull(index)?.toFloatOrNull() }.toFloatArray()
                    // Calculez et ajoutez l'indicateur de tendance directement dans 'input'
                    val trendIndicator = when {
                        delta > 0 && shortAvgDelta > 0 && longAvgDelta > 0 -> 1
                        delta < 0 && shortAvgDelta < 0 && longAvgDelta < 0 -> -1
                        else                                               -> 0
                    }
                    val enhancedInput = input.copyOf(input.size + 1)
                    enhancedInput[input.size] = trendIndicator.toFloat()

                    val targetValue = cols.getOrNull(targetColIndex)?.toDoubleOrNull()
                    if (enhancedInput.size == colIndices.size + 1 && targetValue != null) {
                        inputs.add(enhancedInput)
                        targets.add(doubleArrayOf(targetValue))
                    }
                }

                if (inputs.isEmpty() || targets.isEmpty()) {
                    return Pair(predictedSMB, basalaimi)
                }
                val epochs = 250.0
                val learningRate = if (tir1DAYabove > 15) 0.0001 else 0.001
                // Déterminer la taille de l'ensemble de validation
                val validationSize = (inputs.size * 0.1).toInt() // Par exemple, 10% pour la validation

                // Diviser les données en ensembles d'entraînement et de validation
                val validationInputs = inputs.takeLast(validationSize)
                val validationTargets = targets.takeLast(validationSize)
                val trainingInputs = inputs.take(inputs.size - validationSize)
                val trainingTargets = targets.take(targets.size - validationSize)
                val maxChangePercent = 1.0f

                // Création et entraînement du réseau de neurones
                val neuralNetwork = aimiNeuralNetwork(inputs.first().size, 5, 1)
                neuralNetwork.train(trainingInputs, trainingTargets, validationInputs, validationTargets, epochs, learningRate)

                val inputForPrediction = inputs.last()
                val prediction = neuralNetwork.predict(inputForPrediction)
                do {
                    totalDifference = 0.0f

                    for (enhancedInput in inputs) {
                        val predictedrefineSMB = finalRefinedSMB// Prédiction du modèle TFLite
                        val refinedSMB = refineSMB(predictedrefineSMB, neuralNetwork, enhancedInput)
                        val refinedBasalAimi = refineBasalaimi(refineBasalAimi, neuralNetwork, enhancedInput)
                        if (delta > 10 && bg > 120 && iob < 1.5) {
                            isAggressiveResponseNeeded = true
                        }

                        refineBasalAimi = refinedBasalAimi
                        val change = refineBasalAimi - basalaimi
                        val maxChange = basalaimi * maxChangePercent
                        refineBasalAimi = if (kotlin.math.abs(change) > maxChange) {
                            basalaimi + kotlin.math.sign(change) * maxChange
                        } else {
                            basalaimi
                        }
                        val difference = kotlin.math.abs(predictedrefineSMB - refinedSMB)
                        totalDifference += difference
                        if (difference in 0.0..2.5) {
                            finalRefinedSMB = if (refinedSMB > 0.0f) refinedSMB else 0.0f
                            differenceWithinRange = true
                            break
                        }
                    }
                    if (isAggressiveResponseNeeded && (finalRefinedSMB <= 0.5 || refineBasalAimi <= 0.5)) {
                        finalRefinedSMB = maxSMB.toFloat() / 2
                        refineBasalAimi = maxSMB.toFloat()
                    } else if (!isAggressiveResponseNeeded && delta > 3 && bg > 130) {
                        refineBasalAimi = basalaimi * delta
                    }
                    iterationCount++
                    if (differenceWithinRange || iterationCount >= maxIterations) {
                        break
                    }
                } while (true)
                if (differenceWithinRange || iterationCount >= maxIterations) {
                    globalConvergenceReached = true
                }


                globalIterationCount++
            }
        }
        return Pair (if (globalConvergenceReached) finalRefinedSMB else predictedSMB,refineBasalAimi)
    }
    private fun calculateGFactor(delta: Float, lastHourTIRabove170: Double, bg: Float): Double {
        val deltaFactor = delta / 10 // Ajuster selon les besoins
        val bgFactor = if (bg > 170) 1.2 else if (bg < 120) 0.8 else 1.0

        // Introduire un facteur basé sur lastHourTIRabove170
        val tirFactor = 1.0 + lastHourTIRabove170 * 0.05 // Exemple: 5% d'augmentation pour chaque unité de lastHourTIRabove170

        // Combinez les facteurs pour obtenir un ajustement global
        return deltaFactor * bgFactor * tirFactor
    }
    private fun adjustFactorsBasedOnBgAndHypo(
        morningFactor: Float,
        afternoonFactor: Float,
        eveningFactor: Float
    ): Triple<Double, Double, Double> {
        val hypoAdjustment = if (bg < 120 || (iob > 3 * maxSMB)) 0.8f else 1.0f
        val factorAdjustment = if (bg < 120) 0.2f else 0.3f
        val bgAdjustment = 1.0f + (Math.log(Math.abs(delta.toDouble()) + 1) - 1) * factorAdjustment
        val scalingFactor = 1.0f - (bg - targetBg).toFloat() / (200 - targetBg) * 0.5f

        if (delta < 0)
            return Triple(
                morningFactor / bgAdjustment,
                afternoonFactor / bgAdjustment,
                eveningFactor / bgAdjustment
            )
        else
            return Triple(
                morningFactor * bgAdjustment * hypoAdjustment * scalingFactor,
                afternoonFactor * bgAdjustment * hypoAdjustment * scalingFactor,
                eveningFactor * bgAdjustment * hypoAdjustment * scalingFactor)

    }
    private fun calculateAdjustedDelayFactor(
        bg: Float, recentSteps180Minutes: Int, averageBeatsPerMinute60: Float, averageBeatsPerMinute180: Float
    ): Float {
        // Seuil pour une activité physique significative basée sur les étapes
        val stepActivityThreshold = 1500

        // Seuil d'augmentation de la fréquence cardiaque indiquant une activité accrue
        val heartRateIncreaseThreshold = 1.2  // par exemple, une augmentation de 20%

        // Seuil à partir duquel l'efficacité de l'insuline commence à diminuer
        val insulinSensitivityDecreaseThreshold = 1.5 * normalBgThreshold

        // Déterminer si une activité physique significative a eu lieu
        val increasedPhysicalActivity = recentSteps180Minutes > stepActivityThreshold

        // Calculer le changement relatif de la fréquence cardiaque
        val heartRateChange = averageBeatsPerMinute / averageBeatsPerMinute10

        // Indicateur d'une augmentation possible de la fréquence cardiaque due à l'exercice
        val increasedHeartRateActivity = heartRateChange >= heartRateIncreaseThreshold

        // Calculer le facteur de base avant de prendre en compte l'activité physique
        val baseFactor = when {
            bg <= normalBgThreshold -> 1f
            bg <= insulinSensitivityDecreaseThreshold -> 1f - ((bg - normalBgThreshold) / (insulinSensitivityDecreaseThreshold - normalBgThreshold))
            else -> 0.5f // Arbitraire, à ajuster en fonction de la physiologie individuelle
        }

        // Si une activité physique est détectée (soit par les étapes, soit par la fréquence cardiaque),
        // nous ajustons le facteur de retard pour augmenter la sensibilité à l'insuline.
        return if (increasedPhysicalActivity || increasedHeartRateActivity) {
            (baseFactor.toFloat() * 0.8f).coerceAtLeast(0.5f)
        } else {
            baseFactor.toFloat()
        }
    }

    private fun calculateInsulinEffect(
        bg: Float,
        iob: Float,
        variableSensitivity: Float,
        cob: Float,
        normalBgThreshold: Float,
        recentSteps180Min: Int,
        averageBeatsPerMinute: Float,
        averageBeatsPerMinute10: Float,
        insulinDivisor: Float
    ): Float {
        // Calculer l'effet initial de l'insuline
        var insulinEffect = iob * variableSensitivity / insulinDivisor

        // Si des glucides sont présents, nous pourrions vouloir ajuster l'effet de l'insuline pour tenir compte de l'absorption des glucides.
        if (cob > 0) {
            // Ajustement hypothétique basé sur la présence de glucides. Ce facteur doit être déterminé par des tests/logique métier.
            insulinEffect *= 0.9f
        }
        val physicalActivityFactor = 1.0f - recentSteps180Min / 10000f
        insulinEffect *= physicalActivityFactor
        // Calculer le facteur de retard ajusté en fonction de l'activité physique
        val adjustedDelayFactor = calculateAdjustedDelayFactor(
            normalBgThreshold,
            recentSteps180Min,
            averageBeatsPerMinute,
            averageBeatsPerMinute10
        )

        // Appliquer le facteur de retard ajusté à l'effet de l'insuline
        insulinEffect *= adjustedDelayFactor
        if (bg > normalBgThreshold) {
            insulinEffect *= 1.1f
        }

        return insulinEffect
    }
    private fun predictFutureBg(
        bg: Float,
        iob: Float,
        variableSensitivity: Float,
        cob: Float,
        CI: Float,
        mealTime: Boolean,
        highcarbTime: Boolean,
        snackTime: Boolean,
        profile: OapsProfile
    ): Float {
        val (averageCarbAbsorptionTime, carbTypeFactor, estimatedCob) = when {
            highcarbTime -> Triple(3.5f, 0.75f, 100f) // Repas riche en glucides
            snackTime -> Triple(1.5f, 1.25f, 15f) // Snack
            mealTime -> Triple(2.5f, 1.0f, 55f) // Repas normal
            else -> Triple(2.5f, 1.0f, cob) // Valeur par défaut si aucun type de repas spécifié
        }
        val absorptionTimeInMinutes = averageCarbAbsorptionTime * 60

        val insulinEffect = calculateInsulinEffect(
            bg, iob, variableSensitivity, cob, normalBgThreshold, recentSteps180Minutes,
            averageBeatsPerMinute.toFloat(), averageBeatsPerMinute10.toFloat(),profile.insulinDivisor.toFloat()
        )

        val carbEffect = if (absorptionTimeInMinutes != 0f && CI > 0f) {
            (estimatedCob / absorptionTimeInMinutes) * CI * carbTypeFactor
        } else {
            0f
        }

        var futureBg = bg - insulinEffect + carbEffect
        if (futureBg < 39f) {
            futureBg = 39f
        }

        return futureBg
    }

    private fun calculateSmoothBasalRate(
        tdd2Days: Float, // Total Daily Dose (TDD) pour le jour le plus récent
        tdd7Days: Float, // TDD pour le jour précédent
        currentBasalRate: Float // Le taux de basal actuel
    ): Float {
        // Poids pour le lissage. Plus la valeur est proche de 1, plus l'influence du jour le plus récent est grande.
        val weightRecent = 0.6f
        val weightPrevious = 1.0f - weightRecent

        // Calculer la TDD moyenne pondérée
        val weightedTdd = (tdd2Days * weightRecent) + (tdd7Days * weightPrevious)

        // Ajuster la basale en fonction de la TDD moyenne pondérée
        // Cette formule peut être ajustée en fonction de la logique souhaitée
        val adjustedBasalRate = currentBasalRate * (weightedTdd / tdd2Days)

        // Retourner la nouvelle basale lissée
        return adjustedBasalRate
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
    private fun processNotesAndCleanUp(notes: String): String {
        return notes.lowercase()
            .replace(",", " ")
            .replace(".", " ")
            .replace("!", " ")
            //.replace("a", " ")
            .replace("an", " ")
            .replace("and", " ")
            .replace("\\s+", " ")
    }
    private fun parseNotes(startMinAgo: Int, endMinAgo: Int): String {
        val olderTimeStamp = now - endMinAgo * 60 * 1000
        val moreRecentTimeStamp = now - startMinAgo * 60 * 1000
        var notes = ""
        val recentNotes2: MutableList<String> = mutableListOf()
        val autoNote = determineNoteBasedOnBg(bg.toDouble())
        recentNotes2.add(autoNote)
        notes += autoNote  // Ajout de la note auto générée

        recentNotes?.forEach { note ->
            if(note.timestamp > olderTimeStamp && note.timestamp <= moreRecentTimeStamp) {
                val noteText = note.note.lowercase()
                if (noteText.contains("sleep") || noteText.contains("sport") || noteText.contains("snack") ||
                    noteText.contains("lowcarb") || noteText.contains("highcarb") || noteText.contains("meal") || noteText.contains("fasting") ||
                    noteText.contains("low treatment") || noteText.contains("less aggressive") ||
                    noteText.contains("more aggressive") || noteText.contains("too aggressive") ||
                    noteText.contains("normal")) {

                    notes += if (notes.isEmpty()) recentNotes2 else " "
                    notes += note.note
                    recentNotes2.add(note.note)
                }
            }
        }

        notes = processNotesAndCleanUp(notes)
        return notes
    }
    fun determine_basal(
        glucose_status: GlucoseStatus, currenttemp: CurrentTemp, iob_data_array: Array<IobTotal>, profile: OapsProfile, autosens_data: AutosensResult, meal_data: MealData,
        microBolusAllowed: Boolean, currentTime: Long, flatBGsDetected: Boolean, dynIsfMode: Boolean
    ): RT {
        consoleError.clear()
        consoleLog.clear()
        var rT = RT(
            algorithm = APSResult.Algorithm.AIMI,
            runningDynamicIsf = dynIsfMode,
            timestamp = currentTime,
            consoleLog = consoleLog,
            consoleError = consoleError
        )
        val honeymoon = preferences.get(BooleanKey.OApsAIMIhoneymoon)
        this.bg = glucose_status.glucose
        val getlastBolusSMB = persistenceLayer.getNewestBolusOfType(BS.Type.SMB)
        val lastBolusSMBTime = getlastBolusSMB?.timestamp ?: 0L
        val lastBolusSMBMinutes = lastBolusSMBTime / 60000
        this.lastBolusSMBUnit = getlastBolusSMB?.amount?.toFloat() ?: 0.0F
        val diff = Math.abs(now - lastBolusSMBTime)
        this.lastsmbtime = (diff / (60 * 1000)).toInt()
        this.maxIob = preferences.get(DoubleKey.ApsSmbMaxIob)
        this.maxSMB = preferences.get(DoubleKey.OApsAIMIMaxSMB)
        this.maxSMBHB = preferences.get(DoubleKey.OApsAIMIHighBGMaxSMB)
        this.maxSMB = if (bg > 180) maxSMBHB else maxSMB
        this.tir1DAYabove = tirCalculator.averageTIR(tirCalculator.calculate(1, 65.0, 180.0))?.abovePct()!!
        this.currentTIRLow = tirCalculator.averageTIR(tirCalculator.calculateDaily(65.0, 180.0))?.belowPct()!!
        this.currentTIRRange = tirCalculator.averageTIR(tirCalculator.calculateDaily(65.0, 180.0))?.inRangePct()!!
        this.currentTIRAbove = tirCalculator.averageTIR(tirCalculator.calculateDaily(65.0, 180.0))?.abovePct()!!
        this.lastHourTIRLow = tirCalculator.averageTIR(tirCalculator.calculateHour(80.0,140.0))?.belowPct()!!
        val lastHourTIRAbove = tirCalculator.averageTIR(tirCalculator.calculateHour(72.0, 140.0))?.abovePct()
        this.lastHourTIRLow100 = tirCalculator.averageTIR(tirCalculator.calculateHour(100.0,140.0))?.belowPct()!!
        this.lastHourTIRabove170 = tirCalculator.averageTIR(tirCalculator.calculateHour(100.0,170.0))?.abovePct()!!
        val tirbasal3IR = tirCalculator.averageTIR(tirCalculator.calculate(3, 65.0, 130.0))?.inRangePct()
        val tirbasal3B = tirCalculator.averageTIR(tirCalculator.calculate(3, 65.0, 130.0))?.belowPct()
        val tirbasal3A = tirCalculator.averageTIR(tirCalculator.calculate(3, 65.0, 130.0))?.abovePct()
        val tirbasalhAP = tirCalculator.averageTIR(tirCalculator.calculateHour(65.0, 115.0))?.abovePct()
        this.enablebasal = preferences.get(BooleanKey.OApsAIMIEnableBasal)
        //this.now = System.currentTimeMillis()
        val calendarInstance = Calendar.getInstance()
        this.hourOfDay = calendarInstance[Calendar.HOUR_OF_DAY]
        val dayOfWeek = calendarInstance[Calendar.DAY_OF_WEEK]
        this.weekend = if (dayOfWeek == Calendar.SUNDAY || dayOfWeek == Calendar.SATURDAY) 1 else 0
        var lastCarbTimestamp = meal_data.lastCarbTime
        if (lastCarbTimestamp.toInt() == 0) {
            val oneDayAgoIfNotFound = now - 24 * 60 * 60 * 1000
            lastCarbTimestamp = persistenceLayer.getMostRecentCarbByDate() ?: oneDayAgoIfNotFound
        }
        this.lastCarbAgeMin = ((now - lastCarbTimestamp) / (60 * 1000)).toInt()

        this.futureCarbs = persistenceLayer.getFutureCob().toFloat()
        if (lastCarbAgeMin < 15 && cob == 0.0f) {
            this.cob = persistenceLayer.getMostRecentCarbAmount()?.toFloat() ?: 0.0f
        }

        val fourHoursAgo = now - 4 * 60 * 60 * 1000
        this.recentNotes = persistenceLayer.getUserEntryDataFromTime(fourHoursAgo).blockingGet()

        this.tags0to60minAgo = parseNotes(0, 60)
        this.tags60to120minAgo = parseNotes(60, 120)
        this.tags120to180minAgo = parseNotes(120, 180)
        this.tags180to240minAgo = parseNotes(180, 240)
        this.delta = glucose_status.delta.toFloat()
        this.shortAvgDelta = glucose_status.shortAvgDelta.toFloat()
        this.longAvgDelta = glucose_status.longAvgDelta.toFloat()
        val therapy = Therapy(persistenceLayer).also {
            it.updateStatesBasedOnTherapyEvents()
        }
        this.sleepTime = therapy.sleepTime
        this.snackTime = therapy.snackTime
        this.sportTime = therapy.sportTime
        this.lowCarbTime = therapy.lowCarbTime
        this.highCarbTime = therapy.highCarbTime
        this.mealTime = therapy.mealTime
        this.fastingTime = therapy.fastingTime
        this.stopTime = therapy.stopTime
        this.mealruntime = therapy.getTimeElapsedSinceLastEvent("meal")
        this.highCarbrunTime = therapy.getTimeElapsedSinceLastEvent("highcarb")
        this.snackrunTime = therapy.getTimeElapsedSinceLastEvent("snack")
        this.iscalibration = therapy.calibartionTime
        this.accelerating_up = if (delta > 2 && delta - longAvgDelta > 2) 1 else 0
        this.deccelerating_up = if (delta > 0 && (delta < shortAvgDelta || delta < longAvgDelta)) 1 else 0
        this.accelerating_down = if (delta < -2 && delta - longAvgDelta < -2) 1 else 0
        this.deccelerating_down = if (delta < 0 && (delta > shortAvgDelta || delta > longAvgDelta)) 1 else 0
        this.stable = if (delta>-3 && delta<3 && shortAvgDelta>-3 && shortAvgDelta<3 && longAvgDelta>-3 && longAvgDelta<3 && bg < 180) 1 else 0
         if (isMealModeCondition()){
             val pbolusM: Double = preferences.get(DoubleKey.OApsAIMIMealPrebolus)
                 rT.units = pbolusM
                 rT.reason.append("Microbolusing Meal Mode ${pbolusM}U. ")
             return rT
         }
        if (isHighCarbModeCondition()){
            val pbolusHC: Double = preferences.get(DoubleKey.OApsAIMIHighCarbPrebolus)
            rT.units = pbolusHC
            rT.reason.append("Microbolusing High Carb Mode ${pbolusHC}U. ")
            return rT
        }
        if (issnackModeCondition()){
            val pbolussnack: Double = preferences.get(DoubleKey.OApsAIMISnackPrebolus)
            rT.units = pbolussnack
            rT.reason.append("Microbolusing High Carb Mode ${pbolussnack}U. ")
            return rT
        }

        var nowMinutes = calendarInstance[Calendar.HOUR_OF_DAY] + calendarInstance[Calendar.MINUTE] / 60.0 + calendarInstance[Calendar.SECOND] / 3600.0
        nowMinutes = (kotlin.math.round(nowMinutes * 100) / 100).toDouble()  // Arrondi à 2 décimales
        val circadianSensitivity = (0.00000379 * nowMinutes.pow(5)) -
            (0.00016422 * nowMinutes.pow(4)) +
            (0.00128081 * nowMinutes.pow(3)) +
            (0.02533782 * nowMinutes.pow(2)) -
            (0.33275556 * nowMinutes) +
            1.38581503

        val circadianSmb = kotlin.math.round(
            ((0.00000379 * delta * nowMinutes.pow(5)) -
                (0.00016422 * delta * nowMinutes.pow(4)) +
                (0.00128081 * delta * nowMinutes.pow(3)) +
                (0.02533782 * delta * nowMinutes.pow(2)) -
                (0.33275556 * delta * nowMinutes) +
                1.38581503) * 100
        ) / 100  // Arrondi à 2 décimales
        // TODO eliminate
        val deliverAt = currentTime

        // TODO eliminate
        val profile_current_basal = round_basal(profile.current_basal)
        var basal = profile_current_basal

        // TODO eliminate
        val systemTime = currentTime

        // TODO eliminate
        val bgTime = glucose_status.date
        val minAgo = round((systemTime - bgTime) / 60.0 / 1000.0, 1)
        // TODO eliminate
        //bg = glucose_status.glucose.toFloat()
        //this.bg = bg.toFloat()
        // TODO eliminate
        val noise = glucose_status.noise
        // 38 is an xDrip error state that usually indicates sensor failure
        // all other BG values between 11 and 37 mg/dL reflect non-error-code BG values, so we should zero temp for those
        if (bg <= 10 || bg == 38.0 || noise >= 3) {  //Dexcom is in ??? mode or calibrating, or xDrip reports high noise
            rT.reason.append("CGM is calibrating, in ??? state, or noise is high")
        }
        if (minAgo > 12 || minAgo < -5) { // Dexcom data is too old, or way in the future
            rT.reason.append("If current system time $systemTime is correct, then BG data is too old. The last BG data was read ${minAgo}m ago at $bgTime")
            // if BG is too old/noisy, or is changing less than 1 mg/dL/5m for 45m, cancel any high temps and shorten any long zero temps
        } else if (bg > 60 && flatBGsDetected) {
            rT.reason.append("Error: CGM data is unchanged for the past ~45m")
        }
        if (bg <= 10 || bg == 38.0 || noise >= 3 || minAgo > 12 || minAgo < -5 || (bg > 60 && flatBGsDetected)) {
            if (currenttemp.rate > basal) { // high temp is running
                rT.reason.append(". Replacing high temp basal of ${currenttemp.rate} with neutral temp of $basal")
                rT.deliverAt = deliverAt
                rT.duration = 30
                rT.rate = basal
                return rT
            } else if (currenttemp.rate == 0.0 && currenttemp.duration > 30) { //shorten long zero temps to 30m
                rT.reason.append(". Shortening " + currenttemp.duration + "m long zero temp to 30m. ")
                rT.deliverAt = deliverAt
                rT.duration = 30
                rT.rate = 0.0
                return rT
            } else { //do nothing.
                rT.reason.append(". Temp ${currenttemp.rate} <= current basal ${round(basal, 2)}U/hr; doing nothing. ")
                return rT
            }
        }

        // TODO eliminate
        val max_iob = profile.max_iob // maximum amount of non-bolus IOB OpenAPS will ever deliver
        this.maxIob = max_iob
        // if min and max are set, then set target to their average
        var target_bg = (profile.min_bg + profile.max_bg) / 2
        var min_bg = profile.min_bg
        var max_bg = profile.max_bg

        var sensitivityRatio: Double
        val high_temptarget_raises_sensitivity = profile.exercise_mode || profile.high_temptarget_raises_sensitivity
        val normalTarget = 100 // evaluate high/low temptarget against 100, not scheduled target (which might change)
        // when temptarget is 160 mg/dL, run 50% basal (120 = 75%; 140 = 60%),  80 mg/dL with low_temptarget_lowers_sensitivity would give 1.5x basal, but is limited to autosens_max (1.2x by default)
        val halfBasalTarget = profile.half_basal_exercise_target

        when {
            !profile.temptargetSet && recentSteps5Minutes >= 0 && (recentSteps30Minutes >= 500 || recentSteps180Minutes > 1500) && recentSteps10Minutes > 0 -> {
                this.targetBg = 130.0f
            }
            !profile.temptargetSet && predictedBg >= 120 && delta > 5 -> {
                var hyperTarget = max(65.0, profile.target_bg - (bg - profile.target_bg) / 3).toInt()
                hyperTarget = (hyperTarget * kotlin.math.min(circadianSensitivity, 1.0)).toInt()
                hyperTarget = max(hyperTarget, 65)
                this.targetBg = hyperTarget.toFloat()
                target_bg = hyperTarget.toDouble()
                val c = (halfBasalTarget - normalTarget).toDouble()
                sensitivityRatio = c / (c + target_bg - normalTarget)
                // limit sensitivityRatio to profile.autosens_max (1.2x by default)
                sensitivityRatio = min(sensitivityRatio, profile.autosens_max)
                sensitivityRatio = round(sensitivityRatio, 2)
                consoleLog.add("Sensitivity ratio set to $sensitivityRatio based on temp target of $target_bg; ")
            }
            !profile.temptargetSet && circadianSmb > 0.1 && predictedBg < 130 -> {
                val hypoTarget = 100 * kotlin.math.max(1.0, circadianSensitivity)
                this.targetBg = (hypoTarget + circadianSmb).toFloat()
                target_bg = targetBg.toDouble()
                val c = (halfBasalTarget - normalTarget).toDouble()
                sensitivityRatio = c / (c + target_bg - normalTarget)
                // limit sensitivityRatio to profile.autosens_max (1.2x by default)
                sensitivityRatio = min(sensitivityRatio, profile.autosens_max)
                sensitivityRatio = round(sensitivityRatio, 2)
                consoleLog.add("Sensitivity ratio set to $sensitivityRatio based on temp target of $target_bg; ")
            }
            else -> {
                val defaultTarget = profile.target_bg
                this.targetBg = defaultTarget.toFloat()
                target_bg = targetBg.toDouble()
            }
        }
        if (high_temptarget_raises_sensitivity && profile.temptargetSet && target_bg > normalTarget
            || profile.low_temptarget_lowers_sensitivity && profile.temptargetSet && target_bg < normalTarget
        ) {
            // w/ target 100, temp target 110 = .89, 120 = 0.8, 140 = 0.67, 160 = .57, and 200 = .44
            // e.g.: Sensitivity ratio set to 0.8 based on temp target of 120; Adjusting basal from 1.65 to 1.35; ISF from 58.9 to 73.6
            //sensitivityRatio = 2/(2+(target_bg-normalTarget)/40);
            val c = (halfBasalTarget - normalTarget).toDouble()
            sensitivityRatio = c / (c + target_bg - normalTarget)
            // limit sensitivityRatio to profile.autosens_max (1.2x by default)
            sensitivityRatio = min(sensitivityRatio, profile.autosens_max)
            sensitivityRatio = round(sensitivityRatio, 2)
            consoleLog.add("Sensitivity ratio set to $sensitivityRatio based on temp target of $target_bg; ")
        } else {
            sensitivityRatio = autosens_data.ratio
            consoleLog.add("Autosens ratio: $sensitivityRatio; ")
        }
        basal = profile.current_basal * sensitivityRatio
        basal = round_basal(basal)
        if (basal != profile_current_basal)
            consoleLog.add("Adjusting basal from $profile_current_basal to $basal; ")
        else
            consoleLog.add("Basal unchanged: $basal; ")

        // adjust min, max, and target BG for sensitivity, such that 50% increase in ISF raises target from 100 to 120
        if (profile.temptargetSet) {
            //console.log("Temp Target set, not adjusting with autosens; ");
        } else {
            if (profile.sensitivity_raises_target && autosens_data.ratio < 1 || profile.resistance_lowers_target && autosens_data.ratio > 1) {
                // with a target of 100, default 0.7-1.2 autosens min/max range would allow a 93-117 target range
                min_bg = round((min_bg - 60) / autosens_data.ratio, 0) + 60
                max_bg = round((max_bg - 60) / autosens_data.ratio, 0) + 60
                var new_target_bg = round((target_bg - 60) / autosens_data.ratio, 0) + 60
                // don't allow target_bg below 80
                new_target_bg = max(80.0, new_target_bg)
                if (target_bg == new_target_bg)
                    consoleLog.add("target_bg unchanged: $new_target_bg; ")
                else
                    consoleLog.add("target_bg from $target_bg to $new_target_bg; ")

                target_bg = new_target_bg
            }
        }

        val iobArray = iob_data_array
        val iob_data = iobArray[0]
        this.iob = iob_data.iob.toFloat()

        val tick: String

        tick = if (glucose_status.delta > -0.5) {
            "+" + round(glucose_status.delta)
        } else {
            round(glucose_status.delta).toString()
        }
        val minDelta = min(glucose_status.delta, glucose_status.shortAvgDelta)
        val minAvgDelta = min(glucose_status.shortAvgDelta, glucose_status.longAvgDelta)
        val maxDelta = max(glucose_status.delta, max(glucose_status.shortAvgDelta, glucose_status.longAvgDelta))
        val tdd7P: Double = preferences.get(DoubleKey.OApsAIMITDD7)
        var tdd7Days = profile.TDD
        if (tdd7Days == 0.0 || tdd7Days == null || tdd7Days < tdd7P) tdd7Days = tdd7P
        this.tdd7DaysPerHour = (tdd7Days / 24).toFloat()
        var tdd2Days = tddCalculator.averageTDD(tddCalculator.calculate(2, allowMissingDays = false))?.data?.totalAmount?.toFloat() ?: 0.0f
        if (tdd2Days == 0.0f || tdd2Days == null || tdd2Days < tdd7P) tdd2Days = tdd7P.toFloat()
        this.tdd2DaysPerHour = tdd2Days / 24
        var tddDaily = tddCalculator.averageTDD(tddCalculator.calculate(1, allowMissingDays = false))?.data?.totalAmount?.toFloat() ?: 0.0f
        if (tddDaily == 0.0f || tddDaily == null || tddDaily < tdd7P/2) tddDaily = tdd7P.toFloat()
        this.tddPerHour = tddDaily / 24

        var tdd24Hrs = tddCalculator.calculateDaily(-24, 0)?.totalAmount?.toFloat() ?: 0.0f
        if (tdd24Hrs == 0.0f || tdd24Hrs == null) tdd24Hrs = tdd7P.toFloat()
        this.tdd24HrsPerHour = tdd24Hrs / 24
        var sens = profile.variable_sens
        this.variableSensitivity = sens.toFloat()
        consoleError.add("CR:${profile.carb_ratio}")
        this.predictedBg = predictFutureBg(bg.toFloat(), iob, variableSensitivity, cob, CI,mealTime,highCarbTime,snackTime,profile)
        val insulinEffect = calculateInsulinEffect(bg.toFloat(),iob,variableSensitivity,cob,normalBgThreshold,recentSteps180Minutes,averageBeatsPerMinute.toFloat(),averageBeatsPerMinute10.toFloat(),profile.insulinDivisor.toFloat())

        val now = System.currentTimeMillis()
        val timeMillis5 = now - 5 * 60 * 1000 // 5 minutes en millisecondes
        val timeMillis10 = now - 10 * 60 * 1000 // 10 minutes en millisecondes
        val timeMillis15 = now - 15 * 60 * 1000 // 15 minutes en millisecondes
        val timeMillis30 = now - 30 * 60 * 1000 // 30 minutes en millisecondes
        val timeMillis60 = now - 60 * 60 * 1000 // 60 minutes en millisecondes
        val timeMillis180 = now - 180 * 60 * 1000 // 180 minutes en millisecondes

        val allStepsCounts = persistenceLayer.getStepsCountFromTimeToTime(timeMillis180, now)

        if (preferences.get(BooleanKey.OApsAIMIEnableStepsFromWatch)) {
        allStepsCounts.forEach { stepCount ->
            val timestamp = stepCount.timestamp
            if (timestamp >= timeMillis5) {
                this.recentSteps5Minutes = stepCount.steps5min
            }
            if (timestamp >= timeMillis10) {
                this.recentSteps10Minutes = stepCount.steps10min
            }
            if (timestamp >= timeMillis15) {
                this.recentSteps15Minutes = stepCount.steps15min
            }
            if (timestamp >= timeMillis30) {
                this.recentSteps30Minutes = stepCount.steps30min
            }
            if (timestamp >= timeMillis60) {
                this.recentSteps60Minutes = stepCount.steps60min
            }
            if (timestamp >= timeMillis180) {
                this.recentSteps180Minutes = stepCount.steps180min
            }
        }
        }else{
            this.recentSteps5Minutes = StepService.getRecentStepCount5Min()
            this.recentSteps10Minutes = StepService.getRecentStepCount10Min()
            this.recentSteps15Minutes = StepService.getRecentStepCount15Min()
            this.recentSteps30Minutes = StepService.getRecentStepCount30Min()
            this.recentSteps60Minutes = StepService.getRecentStepCount60Min()
            this.recentSteps180Minutes = StepService.getRecentStepCount180Min()
        }

        try {
            val heartRates5 = persistenceLayer.getHeartRatesFromTimeToTime(timeMillis5,now)
            this.averageBeatsPerMinute = heartRates5.map { it.beatsPerMinute.toInt() }.average()

        } catch (e: Exception) {

            averageBeatsPerMinute = 80.0
        }
        try {
            val heartRates10 = persistenceLayer.getHeartRatesFromTimeToTime(timeMillis60,now)
            this.averageBeatsPerMinute10 = heartRates10.map { it.beatsPerMinute.toInt() }.average()

        } catch (e: Exception) {

            averageBeatsPerMinute10 = 80.0
        }
        try {
            val heartRates60 = persistenceLayer.getHeartRatesFromTimeToTime(timeMillis60,now)
            this.averageBeatsPerMinute60 = heartRates60.map { it.beatsPerMinute.toInt() }.average()

        } catch (e: Exception) {

            averageBeatsPerMinute60 = 80.0
        }
        try {

            val heartRates180 = persistenceLayer.getHeartRatesFromTimeToTime(timeMillis180,now)
            this.averageBeatsPerMinute180 = heartRates180.map { it.beatsPerMinute.toInt() }.average()

        } catch (e: Exception) {

            averageBeatsPerMinute180 = 80.0
        }
        if (tdd7Days.toFloat() != 0.0f) {
            basalaimi = (tdd7Days / preferences.get(DoubleKey.OApsAIMIweight)).toFloat()
        }
        this.basalaimi = calculateSmoothBasalRate(tdd7P.toFloat(),tdd7Days.toFloat(),basalaimi)
        if (tdd7Days.toFloat() != 0.0f) {
            this.CI = (450 / tdd7Days).toFloat()
        }

        val choKey: Double = preferences.get(DoubleKey.OApsAIMICHO)
        if (CI != 0.0f && CI != Float.POSITIVE_INFINITY && CI != Float.NEGATIVE_INFINITY) {
            this.aimilimit = (choKey / CI).toFloat()
        } else {
            this.aimilimit = (choKey / profile.carb_ratio).toFloat()
        }
        val timenow = LocalTime.now().hour
        val sixAMHour = LocalTime.of(6, 0).hour
        if (averageBeatsPerMinute != 0.0) {
            this.basalaimi = when {
                averageBeatsPerMinute >= averageBeatsPerMinute180 && recentSteps5Minutes > 100 && recentSteps10Minutes > 200 -> (basalaimi * 0.65).toFloat()
                averageBeatsPerMinute180 != 80.0 && averageBeatsPerMinute > averageBeatsPerMinute180 && bg >= 130 && recentSteps10Minutes === 0 && timenow > sixAMHour -> (basalaimi * 1.2).toFloat()
                averageBeatsPerMinute180 != 80.0 && averageBeatsPerMinute < averageBeatsPerMinute180 && recentSteps10Minutes === 0 && bg >= 110 -> (basalaimi * 1.1).toFloat()
                else -> basalaimi
            }
        }

        val pregnancyEnable = preferences.get(BooleanKey.OApsAIMIpregnancy)

        if (tirbasal3B != null && pregnancyEnable) {
            if (tirbasal3IR != null) {
                if (tirbasalhAP != null && tirbasalhAP >= 5) {
                    this.basalaimi = (basalaimi * 2.0).toFloat()
                } else if (lastHourTIRAbove != null && lastHourTIRAbove >= 2) {
                    this.basalaimi = (basalaimi * 1.8).toFloat()
                }else if (timenow < sixAMHour){
                    this.basalaimi = (basalaimi * 1.4).toFloat()
                }else if (timenow > sixAMHour) {
                    this.basalaimi = (basalaimi * 1.6).toFloat()
                } else if ((tirbasal3B <= 5) && (tirbasal3IR >= 70 && tirbasal3IR <= 80)) {
                    this.basalaimi = (basalaimi * 1.1).toFloat()
                } else if (tirbasal3B <= 5 && tirbasal3IR <= 70) {
                    this.basalaimi = (basalaimi * 1.3).toFloat()
                } else if (tirbasal3B > 5 && tirbasal3A!! < 5) {
                    this.basalaimi = (basalaimi * 0.85).toFloat()
                }
            }
        }

        this.variableSensitivity = max(
            profile.sens.toFloat() / 4.0f,
            sens.toFloat() * calculateGFactor(delta, lastHourTIRabove170, bg.toFloat()).toFloat()
        )

        if (recentSteps5Minutes > 100 && recentSteps10Minutes > 200 && bg < 130 && delta < 10 || recentSteps180Minutes > 1500 && bg < 130 && delta < 10) {
            this.variableSensitivity *= 1.5f * calculateGFactor(delta, lastHourTIRabove170, bg.toFloat()).toFloat()
        }
        if (recentSteps30Minutes > 500 && recentSteps5Minutes >= 0 && recentSteps5Minutes < 100 && bg < 130 && delta < 10) {
            this.variableSensitivity *= 1.3f * calculateGFactor(delta, lastHourTIRabove170, bg.toFloat()).toFloat()
        }
        if (variableSensitivity < 2) variableSensitivity = profile.sens.toFloat()
        if (variableSensitivity > (3 * profile.sens.toFloat())) variableSensitivity = profile.sens.toFloat() * 3

        sens = variableSensitivity.toDouble()
        //calculate BG impact: the amount BG "should" be rising or falling based on insulin activity alone
        val bgi = round((-iob_data.activity * sens * 5), 2)
        // project deviations for 30 minutes
        var deviation = round(30 / 5 * (minDelta - bgi))
        // don't overreact to a big negative delta: use minAvgDelta if deviation is negative
        if (deviation < 0) {
            deviation = round((30 / 5) * (minAvgDelta - bgi))
            // and if deviation is still negative, use long_avgdelta
            if (deviation < 0) {
                deviation = round((30 / 5) * (glucose_status.longAvgDelta - bgi))
            }
        }
        // calculate the naive (bolus calculator math) eventual BG based on net IOB and sensitivity
        val naive_eventualBG = round(bg - (iob_data.iob * sens), 0)
        // and adjust it for the deviation above
        var eventualBG = naive_eventualBG + deviation

        // raise target for noisy / raw CGM data
        if (bg > max_bg && profile.adv_target_adjustments && !profile.temptargetSet) {
            // with target=100, as BG rises from 100 to 160, adjustedTarget drops from 100 to 80
            val adjustedMinBG = round(max(80.0, min_bg - (bg - min_bg) / 3.0), 0)
            val adjustedTargetBG = round(max(80.0, target_bg - (bg - target_bg) / 3.0), 0)
            val adjustedMaxBG = round(max(80.0, max_bg - (bg - max_bg) / 3.0), 0)
            // if eventualBG, naive_eventualBG, and target_bg aren't all above adjustedMinBG, don’t use it
            //console.error("naive_eventualBG:",naive_eventualBG+", eventualBG:",eventualBG);
            if (eventualBG > adjustedMinBG && naive_eventualBG > adjustedMinBG && min_bg > adjustedMinBG) {
                consoleLog.add("Adjusting targets for high BG: min_bg from $min_bg to $adjustedMinBG; ")
                min_bg = adjustedMinBG
            } else {
                consoleLog.add("min_bg unchanged: $min_bg; ")
            }
            // if eventualBG, naive_eventualBG, and target_bg aren't all above adjustedTargetBG, don’t use it
            if (eventualBG > adjustedTargetBG && naive_eventualBG > adjustedTargetBG && target_bg > adjustedTargetBG) {
                consoleLog.add("target_bg from $target_bg to $adjustedTargetBG; ")
                target_bg = adjustedTargetBG
            } else {
                consoleLog.add("target_bg unchanged: $target_bg; ")
            }
            // if eventualBG, naive_eventualBG, and max_bg aren't all above adjustedMaxBG, don’t use it
            if (eventualBG > adjustedMaxBG && naive_eventualBG > adjustedMaxBG && max_bg > adjustedMaxBG) {
                consoleError.add("max_bg from $max_bg to $adjustedMaxBG")
                max_bg = adjustedMaxBG
            } else {
                consoleError.add("max_bg unchanged: $max_bg")
            }
        }

        val expectedDelta = calculate_expected_delta(target_bg, eventualBG, bgi)

        // min_bg of 90 -> threshold of 65, 100 -> 70 110 -> 75, and 130 -> 85
        var threshold = min_bg - 0.5 * (min_bg - 40)
        if (profile.lgsThreshold != null) {
            val lgsThreshold = profile.lgsThreshold ?: error("lgsThreshold missing")
            if (lgsThreshold > threshold) {
                consoleError.add("Threshold set from ${convert_bg(threshold)} to ${convert_bg(lgsThreshold.toDouble())}; ")
                threshold = lgsThreshold.toDouble()
            }
        }
        this.predictedSMB = calculateSMBFromModel()
        if ((preferences.get(BooleanKey.OApsAIMIMLtraining) === true) && csvfile.exists()){
            val allLines = csvfile.readLines()
            val minutesToConsider: Double = preferences.get(DoubleKey.OApsAIMIMlminutesTraining)
            val linesToConsider = (minutesToConsider / 5).toInt()
            if (allLines.size > linesToConsider) {
                //this.predictedSMB = neuralnetwork5(delta, shortAvgDelta, longAvgDelta)
                val (refinedSMB, refinedBasalaimi) = neuralnetwork5(delta, shortAvgDelta, longAvgDelta, predictedSMB, basalaimi)
                rT.reason.append("neuralnetwork SMB: $refinedSMB Basal: $refinedBasalaimi")
                this.predictedSMB = refinedSMB
                this.basalaimi = refinedBasalaimi
                basal = if (honeymoon && bg < 170) basalaimi * 0.8 else basalaimi.toDouble()
                basal = round_basal(basal)
            }
            rT.reason.append("csvfile ${csvfile.exists()}")
        }else {
            rT.reason.append("ML Decision data training","ML decision has no enough data to refine the decision")
        }
        var smbToGive = if (honeymoon && bg < 170) predictedSMB * 0.8f else predictedSMB

        val morningfactor: Double = preferences.get(DoubleKey.OApsAIMIMorningFactor) / 100.0
        val afternoonfactor: Double = preferences.get(DoubleKey.OApsAIMIAfternoonFactor) / 100.0
        val eveningfactor: Double = preferences.get(DoubleKey.OApsAIMIEveningFactor) / 100.0
        val hyperfactor: Double = preferences.get(DoubleKey.OApsAIMIHyperFactor) / 100.0
        val highcarbfactor: Double = preferences.get(DoubleKey.OApsAIMIHCFactor) / 100.0
        val mealfactor: Double = preferences.get(DoubleKey.OApsAIMIMealFactor) / 100.0
        val snackfactor: Double = preferences.get(DoubleKey.OApsAIMISnackFactor) / 100.0
        val sleepfactor: Double = preferences.get(DoubleKey.OApsAIMIsleepFactor) / 100.0

        val adjustedFactors = adjustFactorsBasedOnBgAndHypo(
                morningfactor.toFloat(), afternoonfactor.toFloat(), eveningfactor.toFloat()
            )

        val (adjustedMorningFactor, adjustedAfternoonFactor, adjustedEveningFactor) = adjustedFactors

        // Appliquer les ajustements en fonction de l'heure de la journée
        smbToGive = when {
            highCarbTime -> smbToGive * highcarbfactor.toFloat()
            mealTime -> smbToGive * mealfactor.toFloat()
            snackTime -> smbToGive * snackfactor.toFloat()
            sleepTime -> smbToGive * sleepfactor.toFloat()
            hourOfDay in 1..11 -> smbToGive * adjustedMorningFactor.toFloat()
            hourOfDay in 12..18 -> smbToGive * adjustedAfternoonFactor.toFloat()
            hourOfDay in 19..23 -> smbToGive * adjustedEveningFactor.toFloat()
            bg > 180 -> (smbToGive * hyperfactor).toFloat()
            else -> smbToGive
        }
        rT.reason.append("adjustedMorningFactor $adjustedMorningFactor")
        rT.reason.append("adjustedAfternoonFactor $adjustedAfternoonFactor")
        rT.reason.append("adjustedEveningFactor $adjustedEveningFactor")


        smbToGive = applySafetyPrecautions(smbToGive)
        smbToGive = roundToPoint05(smbToGive)
        logDataMLToCsv(predictedSMB, smbToGive)
        logDataToCsv(predictedSMB, smbToGive)
        logDataToCsvHB(predictedSMB, smbToGive)
        if (mealTime && mealruntime < 30){
            rT.rate = if (basal == 0.0) (profile_current_basal * 10) else round_basal(basal * 10)
            rT.deliverAt = deliverAt
            rT.duration = 30
            rT.reason.append("${currenttemp.duration}m@${(currenttemp.rate).toFixed2()} AI Force basal because mealTime.")
            return rT
        }else if (highCarbTime && highCarbrunTime < 60){
            rT.rate = if (basal == 0.0) (profile_current_basal * 10) else round_basal(basal * 10)
            rT.deliverAt = deliverAt
            rT.duration = 30
            rT.reason.append("${currenttemp.duration}m@${(currenttemp.rate).toFixed2()} AI Force basal because highcarb.")
            return rT
        }else if(bg > 80 && bg < 100 && delta > 2 && delta < 8 && !honeymoon){
            rT.rate = if (basal == 0.0) (profile_current_basal * delta) else round_basal(basal * delta)
            rT.deliverAt = deliverAt
            rT.duration = 30
            rT.reason.append("${currenttemp.duration}m@${(currenttemp.rate).toFixed2()} AI Force basal because bg is between 80 and 100 with a small delta.")
        }else if(bg > 80 && bg < 100 && delta > 2 && delta < 8 && honeymoon){
            rT.rate = if (basal == 0.0) (profile_current_basal * 2) else round_basal(basal * 2)
            rT.deliverAt = deliverAt
            rT.duration = 30
            rT.reason.append("${currenttemp.duration}m@${(currenttemp.rate).toFixed2()} AI Force basal because bg is between 80 and 100 with a small delta.")
        }else if (bg > 180 && delta > 2 && smbToGive == 0.0f && !honeymoon){
            rT.rate = if (basal == 0.0) (profile_current_basal * 10) else round_basal(basal * 10)
            rT.deliverAt = deliverAt
            rT.duration = 30
            rT.reason.append("${currenttemp.duration}m@${(currenttemp.rate).toFixed2()} AI Force basal because bg is greater than 180 and SMB = 0U.")
            return rT
        }else if (bg > 180 && delta > 2 && smbToGive == 0.0f && honeymoon) {
            rT.rate = if (basal == 0.0) (profile_current_basal * 3) else round_basal(basal * 3)
            rT.deliverAt = deliverAt
            rT.duration = 30
            rT.reason.append("${currenttemp.duration}m@${(currenttemp.rate).toFixed2()} AI Force basal because bg is greater than 180 and SMB = 0U.")
            return rT
        }

        rT = RT(
            algorithm = APSResult.Algorithm.AIMI,
            runningDynamicIsf = dynIsfMode,
            timestamp = currentTime,
            bg = bg,
            tick = tick,
            eventualBG = eventualBG,
            targetBG = target_bg,
            insulinReq = 0.0,
            deliverAt = deliverAt, // The time at which the microbolus should be delivered
            sensitivityRatio = sensitivityRatio, // autosens ratio (fraction of normal basal)
            consoleLog = consoleLog,
            consoleError = consoleError,
            variable_sens = variableSensitivity.toDouble()
        )

        // generate predicted future BGs based on IOB, COB, and current absorption rate

        var COBpredBGs = mutableListOf<Double>()
        var aCOBpredBGs = mutableListOf<Double>()
        var IOBpredBGs = mutableListOf<Double>()
        var UAMpredBGs = mutableListOf<Double>()
        var ZTpredBGs = mutableListOf<Double>()
        COBpredBGs.add(bg)
        aCOBpredBGs.add(bg)
        IOBpredBGs.add(bg)
        ZTpredBGs.add(bg)
        UAMpredBGs.add(bg)

        var enableSMB = enable_smb(profile, microBolusAllowed, meal_data, target_bg)

        // enable UAM (if enabled in preferences)
        val enableUAM = profile.enableUAM

        //console.error(meal_data);
        // carb impact and duration are 0 unless changed below
        var ci: Double
        val cid: Double
        // calculate current carb absorption rate, and how long to absorb all carbs
        // CI = current carb impact on BG in mg/dL/5m
        ci = round((minDelta - bgi), 1)
        val uci = round((minDelta - bgi), 1)
        // ISF (mg/dL/U) / CR (g/U) = CSF (mg/dL/g)

        // TODO: remove commented-out code for old behavior
        //if (profile.temptargetSet) {
        // if temptargetSet, use unadjusted profile.sens to allow activity mode sensitivityRatio to adjust CR
        //var csf = profile.sens / profile.carb_ratio;
        //} else {
        // otherwise, use autosens-adjusted sens to counteract autosens meal insulin dosing adjustments
        // so that autotuned CR is still in effect even when basals and ISF are being adjusted by autosens
        //var csf = sens / profile.carb_ratio;
        //}
        // use autosens-adjusted sens to counteract autosens meal insulin dosing adjustments so that
        // autotuned CR is still in effect even when basals and ISF are being adjusted by TT or autosens
        // this avoids overdosing insulin for large meals when low temp targets are active
        val csf = sens / profile.carb_ratio
        consoleError.add("profile.sens: ${profile.sens}, sens: $sens, CSF: $csf")

        val maxCarbAbsorptionRate = 30 // g/h; maximum rate to assume carbs will absorb if no CI observed
        // limit Carb Impact to maxCarbAbsorptionRate * csf in mg/dL per 5m
        val maxCI = round(maxCarbAbsorptionRate * csf * 5 / 60, 1)
        if (ci > maxCI) {
            consoleError.add("Limiting carb impact from $ci to $maxCI mg/dL/5m ( $maxCarbAbsorptionRate g/h )")
            ci = maxCI
        }
        var remainingCATimeMin = 3.0 // h; duration of expected not-yet-observed carb absorption
        // adjust remainingCATime (instead of CR) for autosens if sensitivityRatio defined
        remainingCATimeMin = remainingCATimeMin / sensitivityRatio
        // 20 g/h means that anything <= 60g will get a remainingCATimeMin, 80g will get 4h, and 120g 6h
        // when actual absorption ramps up it will take over from remainingCATime
        val assumedCarbAbsorptionRate = 20 // g/h; maximum rate to assume carbs will absorb if no CI observed
        var remainingCATime = remainingCATimeMin
        if (meal_data.carbs != 0.0) {
            // if carbs * assumedCarbAbsorptionRate > remainingCATimeMin, raise it
            // so <= 90g is assumed to take 3h, and 120g=4h
            remainingCATimeMin = Math.max(remainingCATimeMin, meal_data.mealCOB / assumedCarbAbsorptionRate)
            val lastCarbAge = round((systemTime - meal_data.lastCarbTime) / 60000.0)
            //console.error(meal_data.lastCarbTime, lastCarbAge);

            val fractionCOBAbsorbed = (meal_data.carbs - meal_data.mealCOB) / meal_data.carbs
            remainingCATime = remainingCATimeMin + 1.5 * lastCarbAge / 60
            remainingCATime = round(remainingCATime, 1)
            //console.error(fractionCOBAbsorbed, remainingCATimeAdjustment, remainingCATime)
            consoleError.add("Last carbs " + lastCarbAge + "minutes ago; remainingCATime:" + remainingCATime + "hours;" + round(fractionCOBAbsorbed * 100) + "% carbs absorbed")
        }

        // calculate the number of carbs absorbed over remainingCATime hours at current CI
        // CI (mg/dL/5m) * (5m)/5 (m) * 60 (min/hr) * 4 (h) / 2 (linear decay factor) = total carb impact (mg/dL)
        val totalCI = Math.max(0.0, ci / 5 * 60 * remainingCATime / 2)
        // totalCI (mg/dL) / CSF (mg/dL/g) = total carbs absorbed (g)
        val totalCA = totalCI / csf
        val remainingCarbsCap: Int // default to 90
        remainingCarbsCap = min(90, profile.remainingCarbsCap)
        var remainingCarbs = max(0.0, meal_data.mealCOB - totalCA)
        remainingCarbs = Math.min(remainingCarbsCap.toDouble(), remainingCarbs)
        // assume remainingCarbs will absorb in a /\ shaped bilinear curve
        // peaking at remainingCATime / 2 and ending at remainingCATime hours
        // area of the /\ triangle is the same as a remainingCIpeak-height rectangle out to remainingCATime/2
        // remainingCIpeak (mg/dL/5m) = remainingCarbs (g) * CSF (mg/dL/g) * 5 (m/5m) * 1h/60m / (remainingCATime/2) (h)
        val remainingCIpeak = remainingCarbs * csf * 5 / 60 / (remainingCATime / 2)
        //console.error(profile.min_5m_carbimpact,ci,totalCI,totalCA,remainingCarbs,remainingCI,remainingCATime);

        // calculate peak deviation in last hour, and slope from that to current deviation
        val slopeFromMaxDeviation = round(meal_data.slopeFromMaxDeviation, 2)
        // calculate lowest deviation in last hour, and slope from that to current deviation
        val slopeFromMinDeviation = round(meal_data.slopeFromMinDeviation, 2)
        // assume deviations will drop back down at least at 1/3 the rate they ramped up
        val slopeFromDeviations = Math.min(slopeFromMaxDeviation, -slopeFromMinDeviation / 3)
        //console.error(slopeFromMaxDeviation);

        val aci = 10
        //5m data points = g * (1U/10g) * (40mg/dL/1U) / (mg/dL/5m)
        // duration (in 5m data points) = COB (g) * CSF (mg/dL/g) / ci (mg/dL/5m)
        // limit cid to remainingCATime hours: the reset goes to remainingCI
        if (ci == 0.0) {
            // avoid divide by zero
            cid = 0.0
        } else {
            cid = min(remainingCATime * 60 / 5 / 2, Math.max(0.0, meal_data.mealCOB * csf / ci))
        }
        val acid = max(0.0, meal_data.mealCOB * csf / aci)
        // duration (hours) = duration (5m) * 5 / 60 * 2 (to account for linear decay)
        consoleError.add("Carb Impact: ${ci} mg/dL per 5m; CI Duration: ${round(cid * 5 / 60 * 2, 1)} hours; remaining CI (~2h peak): ${round(remainingCIpeak, 1)} mg/dL per 5m")
        //console.error("Accel. Carb Impact:",aci,"mg/dL per 5m; ACI Duration:",round(acid*5/60*2,1),"hours");
        var minIOBPredBG = 999.0
        var minCOBPredBG = 999.0
        var minUAMPredBG = 999.0
        var minGuardBG: Double
        var minCOBGuardBG = 999.0
        var minUAMGuardBG = 999.0
        var minIOBGuardBG = 999.0
        var minZTGuardBG = 999.0
        var minPredBG: Double
        var avgPredBG: Double
        var IOBpredBG: Double = eventualBG
        var maxIOBPredBG = bg
        var maxCOBPredBG = bg
        //var maxUAMPredBG = bg
        //var maxPredBG = bg;
        //var eventualPredBG = bg
        val lastIOBpredBG: Double
        var lastCOBpredBG: Double? = null
        var lastUAMpredBG: Double? = null
        //var lastZTpredBG: Int
        var UAMduration = 0.0
        var remainingCItotal = 0.0
        val remainingCIs = mutableListOf<Int>()
        val predCIs = mutableListOf<Int>()
        var UAMpredBG: Double? = null
        var COBpredBG: Double? = null
        var aCOBpredBG: Double?
        iobArray.forEach { iobTick ->
            //console.error(iobTick);
            val predBGI: Double = round((-iobTick.activity * sens * 5), 2)
            val IOBpredBGI: Double = round((-iobTick.activity * (1800 / (profile.TDD * (ln((max(IOBpredBGs[IOBpredBGs.size - 1], 39.0) / profile.insulinDivisor) + 1)))) * 5), 2)
            iobTick.iobWithZeroTemp ?: error("iobTick.iobWithZeroTemp missing")
            val predZTBGI = round((-iobTick.iobWithZeroTemp!!.activity * (1800 / (profile.TDD * (ln((max(ZTpredBGs[ZTpredBGs.size - 1], 39.0) / profile.insulinDivisor) + 1)))) * 5), 2)
            val predUAMBGI = round((-iobTick.activity * (1800 / (profile.TDD * (ln((max(UAMpredBGs[UAMpredBGs.size - 1], 39.0) / profile.insulinDivisor) + 1)))) * 5), 2)
            // for IOBpredBGs, predicted deviation impact drops linearly from current deviation down to zero
            // over 60 minutes (data points every 5m)
            val predDev: Double = ci * (1 - min(1.0, IOBpredBGs.size / (60.0 / 5.0)))
            IOBpredBG = IOBpredBGs[IOBpredBGs.size - 1] + IOBpredBGI + predDev
            // calculate predBGs with long zero temp without deviations
            val ZTpredBG = ZTpredBGs[ZTpredBGs.size - 1] + predZTBGI
            // for COBpredBGs, predicted carb impact drops linearly from current carb impact down to zero
            // eventually accounting for all carbs (if they can be absorbed over DIA)
            val predCI: Double = max(0.0, max(0.0, ci) * (1 - COBpredBGs.size / max(cid * 2, 1.0)))
            val predACI = max(0.0, max(0, aci) * (1 - COBpredBGs.size / max(acid * 2, 1.0)))
            // if any carbs aren't absorbed after remainingCATime hours, assume they'll absorb in a /\ shaped
            // bilinear curve peaking at remainingCIpeak at remainingCATime/2 hours (remainingCATime/2*12 * 5m)
            // and ending at remainingCATime h (remainingCATime*12 * 5m intervals)
            val intervals = Math.min(COBpredBGs.size.toDouble(), ((remainingCATime * 12) - COBpredBGs.size))
            val remainingCI = Math.max(0.0, intervals / (remainingCATime / 2 * 12) * remainingCIpeak)
            remainingCItotal += predCI + remainingCI
            remainingCIs.add(round(remainingCI))
            predCIs.add(round(predCI))
            //console.log(round(predCI,1)+"+"+round(remainingCI,1)+" ");
            COBpredBG = COBpredBGs[COBpredBGs.size - 1] + predBGI + min(0.0, predDev) + predCI + remainingCI
            aCOBpredBG = aCOBpredBGs[aCOBpredBGs.size - 1] + predBGI + min(0.0, predDev) + predACI
            // for UAMpredBGs, predicted carb impact drops at slopeFromDeviations
            // calculate predicted CI from UAM based on slopeFromDeviations
            val predUCIslope = max(0.0, uci + (UAMpredBGs.size * slopeFromDeviations))
            // if slopeFromDeviations is too flat, predicted deviation impact drops linearly from
            // current deviation down to zero over 3h (data points every 5m)
            val predUCImax = max(0.0, uci * (1 - UAMpredBGs.size / max(3.0 * 60 / 5, 1.0)))
            //console.error(predUCIslope, predUCImax);
            // predicted CI from UAM is the lesser of CI based on deviationSlope or DIA
            val predUCI = min(predUCIslope, predUCImax)
            if (predUCI > 0) {
                //console.error(UAMpredBGs.length,slopeFromDeviations, predUCI);
                UAMduration = round((UAMpredBGs.size + 1) * 5 / 60.0, 1)
            }
            UAMpredBG = UAMpredBGs[UAMpredBGs.size - 1] + predUAMBGI + min(0.0, predDev) + predUCI
            //console.error(predBGI, predCI, predUCI);
            // truncate all BG predictions at 4 hours
            if (IOBpredBGs.size < 48) IOBpredBGs.add(IOBpredBG)
            if (COBpredBGs.size < 48) COBpredBGs.add(COBpredBG!!)
            if (aCOBpredBGs.size < 48) aCOBpredBGs.add(aCOBpredBG!!)
            if (UAMpredBGs.size < 48) UAMpredBGs.add(UAMpredBG!!)
            if (ZTpredBGs.size < 48) ZTpredBGs.add(ZTpredBG)
            // calculate minGuardBGs without a wait from COB, UAM, IOB predBGs
            if (COBpredBG!! < minCOBGuardBG) minCOBGuardBG = round(COBpredBG!!).toDouble()
            if (UAMpredBG!! < minUAMGuardBG) minUAMGuardBG = round(UAMpredBG!!).toDouble()
            if (IOBpredBG < minIOBGuardBG) minIOBGuardBG = IOBpredBG
            if (ZTpredBG < minZTGuardBG) minZTGuardBG = round(ZTpredBG, 0)

            // set minPredBGs starting when currently-dosed insulin activity will peak
            // look ahead 60m (regardless of insulin type) so as to be less aggressive on slower insulins
            // add 30m to allow for insulin delivery (SMBs or temps)
            val insulinPeakTime = 90
            val insulinPeak5m = (insulinPeakTime / 60.0) * 12.0
            //console.error(insulinPeakTime, insulinPeak5m, profile.insulinPeakTime, profile.curve);

            // wait 90m before setting minIOBPredBG
            if (IOBpredBGs.size > insulinPeak5m && (IOBpredBG < minIOBPredBG)) minIOBPredBG = round(IOBpredBG, 0)
            if (IOBpredBG > maxIOBPredBG) maxIOBPredBG = IOBpredBG
            // wait 85-105m before setting COB and 60m for UAM minPredBGs
            if ((cid != 0.0 || remainingCIpeak > 0) && COBpredBGs.size > insulinPeak5m && (COBpredBG!! < minCOBPredBG)) minCOBPredBG = round(COBpredBG!!, 0)
            if ((cid != 0.0 || remainingCIpeak > 0) && COBpredBG!! > maxIOBPredBG) maxCOBPredBG = COBpredBG!!
            if (enableUAM && UAMpredBGs.size > 12 && (UAMpredBG!! < minUAMPredBG)) minUAMPredBG = round(UAMpredBG!!, 0)
            //if (enableUAM && UAMpredBG!! > maxIOBPredBG) maxUAMPredBG = UAMpredBG!!
        }
        // set eventualBG to include effect of carbs
        //console.error("PredBGs:",JSON.stringify(predBGs));
        if (meal_data.mealCOB > 0) {
            consoleError.add("predCIs (mg/dL/5m):" + predCIs.joinToString(separator = " "))
            consoleError.add("remainingCIs:      " + remainingCIs.joinToString(separator = " "))
        }
        rT.predBGs = Predictions()
        IOBpredBGs = IOBpredBGs.map { round(min(401.0, max(39.0, it)), 0) }.toMutableList()
        for (i in IOBpredBGs.size - 1 downTo 13) {
            if (IOBpredBGs[i - 1] != IOBpredBGs[i]) break
            else IOBpredBGs.removeLast()
        }
        rT.predBGs?.IOB = IOBpredBGs.map { it.toInt() }
        lastIOBpredBG = round(IOBpredBGs[IOBpredBGs.size - 1]).toDouble()
        ZTpredBGs = ZTpredBGs.map { round(min(401.0, max(39.0, it)), 0) }.toMutableList()
        for (i in ZTpredBGs.size - 1 downTo 7) {
            // stop displaying ZTpredBGs once they're rising and above target
            if (ZTpredBGs[i - 1] >= ZTpredBGs[i] || ZTpredBGs[i] <= target_bg) break
            else ZTpredBGs.removeLast()
        }
        rT.predBGs?.ZT = ZTpredBGs.map { it.toInt() }
        if (meal_data.mealCOB > 0) {
            aCOBpredBGs = aCOBpredBGs.map { round(min(401.0, max(39.0, it)), 0) }.toMutableList()
            for (i in aCOBpredBGs.size - 1 downTo 13) {
                if (aCOBpredBGs[i - 1] != aCOBpredBGs[i]) break
                else aCOBpredBGs.removeLast()
            }
        }
        if (meal_data.mealCOB > 0 && (ci > 0 || remainingCIpeak > 0)) {
            COBpredBGs = COBpredBGs.map { round(min(401.0, max(39.0, it)), 0) }.toMutableList()
            for (i in COBpredBGs.size - 1 downTo 13) {
                if (COBpredBGs[i - 1] != COBpredBGs[i]) break
                else COBpredBGs.removeLast()
            }
            rT.predBGs?.COB = COBpredBGs.map { it.toInt() }
            lastCOBpredBG = COBpredBGs[COBpredBGs.size - 1]
            eventualBG = max(eventualBG, round(COBpredBGs[COBpredBGs.size - 1], 0))
        }
        if (ci > 0 || remainingCIpeak > 0) {
            if (enableUAM) {
                UAMpredBGs = UAMpredBGs.map { round(min(401.0, max(39.0, it)), 0) }.toMutableList()
                for (i in UAMpredBGs.size - 1 downTo 13) {
                    if (UAMpredBGs[i - 1] != UAMpredBGs[i]) break
                    else UAMpredBGs.removeLast()
                }
                rT.predBGs?.UAM = UAMpredBGs.map { it.toInt() }
                lastUAMpredBG = UAMpredBGs[UAMpredBGs.size - 1]
                eventualBG = max(eventualBG, round(UAMpredBGs[UAMpredBGs.size - 1], 0))
            }

            // set eventualBG based on COB or UAM predBGs
            rT.eventualBG = eventualBG
        }

        consoleError.add("UAM Impact: $uci mg/dL per 5m; UAM Duration: $UAMduration hours")
        consoleLog.add("EventualBG is $eventualBG ;")

        minIOBPredBG = max(39.0, minIOBPredBG)
        minCOBPredBG = max(39.0, minCOBPredBG)
        minUAMPredBG = max(39.0, minUAMPredBG)
        minPredBG = round(minIOBPredBG, 0)

        val fSensBG = min(minPredBG, bg)

        var future_sens = 0.0

        if (bg > target_bg && glucose_status.delta < 3 && glucose_status.delta > -3 && glucose_status.shortAvgDelta > -3 && glucose_status.shortAvgDelta < 3 && eventualBG > target_bg && eventualBG < bg) {
            future_sens = (1800 / (ln((((fSensBG * 0.5) + (bg * 0.5)) / profile.insulinDivisor) + 1) * profile.TDD))
            future_sens = round(future_sens, 1)
            consoleLog.add("Future state sensitivity is $future_sens based on eventual and current bg due to flat glucose level above target")
            rT.reason.append("Dosing sensitivity: $future_sens using eventual BG;")
        } else if (glucose_status.delta > 0 && eventualBG > target_bg || eventualBG > bg) {
            future_sens = (1800 / (ln((bg / profile.insulinDivisor) + 1) * profile.TDD))
            future_sens = round(future_sens, 1)
            consoleLog.add("Future state sensitivity is $future_sens using current bg due to small delta or variation")
            rT.reason.append("Dosing sensitivity: $future_sens using current BG;")
        } else {
            future_sens = (1800 / (ln((fSensBG / profile.insulinDivisor) + 1) * profile.TDD))
            future_sens = round(future_sens, 1)
            consoleLog.add("Future state sensitivity is $future_sens based on eventual bg due to -ve delta")
            rT.reason.append("Dosing sensitivity: $future_sens using eventual BG;")
        }


        val fractionCarbsLeft = meal_data.mealCOB / meal_data.carbs
        // if we have COB and UAM is enabled, average both
        if (minUAMPredBG < 999 && minCOBPredBG < 999) {
            // weight COBpredBG vs. UAMpredBG based on how many carbs remain as COB
            avgPredBG = round((1 - fractionCarbsLeft) * UAMpredBG!! + fractionCarbsLeft * COBpredBG!!, 0)
            // if UAM is disabled, average IOB and COB
        } else if (minCOBPredBG < 999) {
            avgPredBG = round((IOBpredBG + COBpredBG!!) / 2.0, 0)
            // if we have UAM but no COB, average IOB and UAM
        } else if (minUAMPredBG < 999) {
            avgPredBG = round((IOBpredBG + UAMpredBG!!) / 2.0, 0)
        } else {
            avgPredBG = round(IOBpredBG, 0)
        }
        // if avgPredBG is below minZTGuardBG, bring it up to that level
        if (minZTGuardBG > avgPredBG) {
            avgPredBG = minZTGuardBG
        }

        // if we have both minCOBGuardBG and minUAMGuardBG, blend according to fractionCarbsLeft
        if ((cid > 0.0 || remainingCIpeak > 0)) {
            if (enableUAM) {
                minGuardBG = fractionCarbsLeft * minCOBGuardBG + (1 - fractionCarbsLeft) * minUAMGuardBG
            } else {
                minGuardBG = minCOBGuardBG
            }
        } else if (enableUAM) {
            minGuardBG = minUAMGuardBG
        } else {
            minGuardBG = minIOBGuardBG
        }
        minGuardBG = round(minGuardBG, 0)
        //console.error(minCOBGuardBG, minUAMGuardBG, minIOBGuardBG, minGuardBG);

        var minZTUAMPredBG = minUAMPredBG
        // if minZTGuardBG is below threshold, bring down any super-high minUAMPredBG by averaging
        // this helps prevent UAM from giving too much insulin in case absorption falls off suddenly
        if (minZTGuardBG < threshold) {
            minZTUAMPredBG = (minUAMPredBG + minZTGuardBG) / 2.0
            // if minZTGuardBG is between threshold and target, blend in the averaging
        } else if (minZTGuardBG < target_bg) {
            // target 100, threshold 70, minZTGuardBG 85 gives 50%: (85-70) / (100-70)
            val blendPct = (minZTGuardBG - threshold) / (target_bg - threshold)
            val blendedMinZTGuardBG = minUAMPredBG * blendPct + minZTGuardBG * (1 - blendPct)
            minZTUAMPredBG = (minUAMPredBG + blendedMinZTGuardBG) / 2.0
            //minZTUAMPredBG = minUAMPredBG - target_bg + minZTGuardBG;
            // if minUAMPredBG is below minZTGuardBG, bring minUAMPredBG up by averaging
            // this allows more insulin if lastUAMPredBG is below target, but minZTGuardBG is still high
        } else if (minZTGuardBG > minUAMPredBG) {
            minZTUAMPredBG = (minUAMPredBG + minZTGuardBG) / 2.0
        }
        minZTUAMPredBG = round(minZTUAMPredBG, 0)
        //console.error("minUAMPredBG:",minUAMPredBG,"minZTGuardBG:",minZTGuardBG,"minZTUAMPredBG:",minZTUAMPredBG);
        // if any carbs have been entered recently
        if (meal_data.carbs != 0.0) {

            // if UAM is disabled, use max of minIOBPredBG, minCOBPredBG
            if (!enableUAM && minCOBPredBG < 999) {
                minPredBG = round(max(minIOBPredBG, minCOBPredBG), 0)
                // if we have COB, use minCOBPredBG, or blendedMinPredBG if it's higher
            } else if (minCOBPredBG < 999) {
                // calculate blendedMinPredBG based on how many carbs remain as COB
                val blendedMinPredBG = fractionCarbsLeft * minCOBPredBG + (1 - fractionCarbsLeft) * minZTUAMPredBG
                // if blendedMinPredBG > minCOBPredBG, use that instead
                minPredBG = round(max(minIOBPredBG, max(minCOBPredBG, blendedMinPredBG)), 0)
                // if carbs have been entered, but have expired, use minUAMPredBG
            } else if (enableUAM) {
                minPredBG = minZTUAMPredBG
            } else {
                minPredBG = minGuardBG
            }
            // in pure UAM mode, use the higher of minIOBPredBG,minUAMPredBG
        } else if (enableUAM) {
            minPredBG = round(max(minIOBPredBG, minZTUAMPredBG), 0)
        }
        // make sure minPredBG isn't higher than avgPredBG
        minPredBG = min(minPredBG, avgPredBG)

        consoleLog.add("minPredBG: $minPredBG minIOBPredBG: $minIOBPredBG minZTGuardBG: $minZTGuardBG")
        if (minCOBPredBG < 999) {
            consoleLog.add(" minCOBPredBG: $minCOBPredBG")
        }
        if (minUAMPredBG < 999) {
            consoleLog.add(" minUAMPredBG: $minUAMPredBG")
        }
        consoleError.add(" avgPredBG: $avgPredBG COB: ${meal_data.mealCOB} / ${meal_data.carbs}")
        // But if the COB line falls off a cliff, don't trust UAM too much:
        // use maxCOBPredBG if it's been set and lower than minPredBG
        if (maxCOBPredBG > bg) {
            minPredBG = min(minPredBG, maxCOBPredBG)
        }

        rT.COB = meal_data.mealCOB
        rT.IOB = iob_data.iob
        rT.reason.append(
            "COB: ${round(meal_data.mealCOB, 1).withoutZeros()}, Dev: ${convert_bg(deviation.toDouble())}, BGI: ${convert_bg(bgi)}, ISF: ${convert_bg(sens)}, CR: ${
                round(profile.carb_ratio, 2)
                    .withoutZeros()
            }, Target: ${convert_bg(target_bg)}, minPredBG ${convert_bg(minPredBG)}, minGuardBG ${convert_bg(minGuardBG)}, IOBpredBG ${convert_bg(lastIOBpredBG)}"
        )
        if (lastCOBpredBG != null) {
            rT.reason.append(", COBpredBG " + convert_bg(lastCOBpredBG.toDouble()))
        }
        if (lastUAMpredBG != null) {
            rT.reason.append(", UAMpredBG " + convert_bg(lastUAMpredBG.toDouble()))
        }
        rT.reason.append("; ")
        // use naive_eventualBG if above 40, but switch to minGuardBG if both eventualBGs hit floor of 39
        var carbsReqBG = naive_eventualBG
        if (carbsReqBG < 40) {
            carbsReqBG = min(minGuardBG, carbsReqBG)
        }
        var bgUndershoot: Double = threshold - carbsReqBG
        // calculate how long until COB (or IOB) predBGs drop below min_bg
        var minutesAboveMinBG = 240
        var minutesAboveThreshold = 240
        if (meal_data.mealCOB > 0 && (ci > 0 || remainingCIpeak > 0)) {
            for (i in COBpredBGs.indices) {
                //console.error(COBpredBGs[i], min_bg);
                if (COBpredBGs[i] < min_bg) {
                    minutesAboveMinBG = 5 * i
                    break
                }
            }
            for (i in COBpredBGs.indices) {
                //console.error(COBpredBGs[i], threshold);
                if (COBpredBGs[i] < threshold) {
                    minutesAboveThreshold = 5 * i
                    break
                }
            }
        } else {
            for (i in IOBpredBGs.indices) {
                //console.error(IOBpredBGs[i], min_bg);
                if (IOBpredBGs[i] < min_bg) {
                    minutesAboveMinBG = 5 * i
                    break
                }
            }
            for (i in IOBpredBGs.indices) {
                //console.error(IOBpredBGs[i], threshold);
                if (IOBpredBGs[i] < threshold) {
                    minutesAboveThreshold = 5 * i
                    break
                }
            }
        }

        if (enableSMB && minGuardBG < threshold) {
            consoleError.add("minGuardBG ${convert_bg(minGuardBG)} projected below ${convert_bg(threshold)} - disabling SMB")
            //rT.reason += "minGuardBG "+minGuardBG+"<"+threshold+": SMB disabled; ";
            enableSMB = false
        }
        if (maxDelta > 0.20 * bg) {
            consoleError.add("maxDelta ${convert_bg(maxDelta)} > 20% of BG ${convert_bg(bg)} - disabling SMB")
            rT.reason.append("maxDelta " + convert_bg(maxDelta) + " > 20% of BG " + convert_bg(bg) + ": SMB disabled; ")
            enableSMB = false
        }

        consoleError.add("BG projected to remain above ${convert_bg(min_bg)} for $minutesAboveMinBG minutes")
        if (minutesAboveThreshold < 240 || minutesAboveMinBG < 60) {
            consoleError.add("BG projected to remain above ${convert_bg(threshold)} for $minutesAboveThreshold minutes")
        }
        // include at least minutesAboveThreshold worth of zero temps in calculating carbsReq
        // always include at least 30m worth of zero temp (carbs to 80, low temp up to target)
        val zeroTempDuration = minutesAboveThreshold
        // BG undershoot, minus effect of zero temps until hitting min_bg, converted to grams, minus COB
        val zeroTempEffectDouble = profile.current_basal * sens * zeroTempDuration / 60
        // don't count the last 25% of COB against carbsReq
        val COBforCarbsReq = max(0.0, meal_data.mealCOB - 0.25 * meal_data.carbs)
        val carbsReq = round(((bgUndershoot - zeroTempEffectDouble) / csf - COBforCarbsReq))
        val zeroTempEffect = round(zeroTempEffectDouble)
        consoleError.add("naive_eventualBG: $naive_eventualBG bgUndershoot: $bgUndershoot zeroTempDuration $zeroTempDuration zeroTempEffect: $zeroTempEffect carbsReq: $carbsReq")
        if (carbsReq >= profile.carbsReqThreshold && minutesAboveThreshold <= 45) {
            rT.carbsReq = carbsReq
            rT.carbsReqWithin = minutesAboveThreshold
            rT.reason.append("$carbsReq add\'l carbs req w/in ${minutesAboveThreshold}m; ")
        }

        // don't low glucose suspend if IOB is already super negative and BG is rising faster than predicted
        if (bg < threshold && iob_data.iob < -profile.current_basal * 20 / 60 && minDelta > 0 && minDelta > expectedDelta) {
            rT.reason.append("IOB ${iob_data.iob} < ${round(-profile.current_basal * 20 / 60, 2)}")
            rT.reason.append(" and minDelta ${convert_bg(minDelta)} > expectedDelta ${convert_bg(expectedDelta)}; ")
            // predictive low glucose suspend mode: BG is / is projected to be < threshold
        } else if (bg < threshold || minGuardBG < threshold) {
            rT.reason.append("minGuardBG " + convert_bg(minGuardBG) + "<" + convert_bg(threshold))
            bgUndershoot = target_bg - minGuardBG
            val worstCaseInsulinReq = bgUndershoot / sens
            var durationReq = round(60 * worstCaseInsulinReq / profile.current_basal)
            durationReq = round(durationReq / 30.0) * 30
            // always set a 30-120m zero temp (oref0-pump-loop will let any longer SMB zero temp run)
            durationReq = min(60, max(30, durationReq))
            return setTempBasal(0.0, durationReq, profile, rT, currenttemp)
        }

        // if not in LGS mode, cancel temps before the top of the hour to reduce beeping/vibration
        // console.error(profile.skip_neutral_temps, rT.deliverAt.getMinutes());
        val minutes = Instant.ofEpochMilli(rT.deliverAt!!).atZone(ZoneId.systemDefault()).toLocalDateTime().minute
        if (profile.skip_neutral_temps && minutes >= 55) {
            rT.reason.append("; Canceling temp at " + minutes + "m past the hour. ")
            return setTempBasal(0.0, 0, profile, rT, currenttemp)
        }

        if (eventualBG < min_bg) { // if eventual BG is below target:
            rT.reason.append("Eventual BG ${convert_bg(eventualBG)} < ${convert_bg(min_bg)}")
            // if 5m or 30m avg BG is rising faster than expected delta
            if (minDelta > expectedDelta && minDelta > 0 && carbsReq == 0) {
                // if naive_eventualBG < 40, set a 30m zero temp (oref0-pump-loop will let any longer SMB zero temp run)
                if (naive_eventualBG < 40) {
                    rT.reason.append(", naive_eventualBG < 40. ")
                    return setTempBasal(0.0, 30, profile, rT, currenttemp)
                }
                if (glucose_status.delta > minDelta) {
                    rT.reason.append(", but Delta ${convert_bg(tick.toDouble())} > expectedDelta ${convert_bg(expectedDelta)}")
                } else {
                    rT.reason.append(", but Min. Delta ${minDelta.toFixed2()} > Exp. Delta ${convert_bg(expectedDelta)}")
                }
                if (currenttemp.duration > 15 && (round_basal(basal) == round_basal(currenttemp.rate))) {
                    rT.reason.append(", temp " + currenttemp.rate + " ~ req " + round(basal, 2).withoutZeros() + "U/hr. ")
                    return rT
                } else {
                    rT.reason.append("; setting current basal of ${round(basal, 2)} as temp. ")
                    return setTempBasal(basal, 30, profile, rT, currenttemp)
                }
            }

            // calculate 30m low-temp required to get projected BG up to target
            // multiply by 2 to low-temp faster for increased hypo safety
            //var insulinReq = 2 * min(0.0, (eventualBG - target_bg) / future_sens)
            var insulinReq = 2 * min(0.0, smbToGive.toDouble())
            insulinReq = round(insulinReq, 2)
            // calculate naiveInsulinReq based on naive_eventualBG
            var naiveInsulinReq = min(0.0, (naive_eventualBG - target_bg) / sens)
            naiveInsulinReq = round(naiveInsulinReq, 2)
            if (minDelta < 0 && minDelta > expectedDelta) {
                // if we're barely falling, newinsulinReq should be barely negative
                val newinsulinReq = round((insulinReq * (minDelta / expectedDelta)), 2)
                //console.error("Increasing insulinReq from " + insulinReq + " to " + newinsulinReq);
                insulinReq = newinsulinReq
            }
            // rate required to deliver insulinReq less insulin over 30m:
            var rate = basal + (2 * insulinReq)
            rate = round_basal(rate)

            // if required temp < existing temp basal
            val insulinScheduled = currenttemp.duration * (currenttemp.rate - basal) / 60
            // if current temp would deliver a lot (30% of basal) less than the required insulin,
            // by both normal and naive calculations, then raise the rate
            val minInsulinReq = Math.min(insulinReq, naiveInsulinReq)
            if (insulinScheduled < minInsulinReq - basal * 0.3) {
                rT.reason.append(", ${currenttemp.duration}m@${(currenttemp.rate).toFixed2()} is a lot less than needed. ")
                return setTempBasal(rate, 30, profile, rT, currenttemp)
            }
            if (currenttemp.duration > 5 && rate >= currenttemp.rate * 0.8) {
                rT.reason.append(", temp ${currenttemp.rate} ~< req ${round(rate, 2)}U/hr. ")
                return rT
            } else {
                // calculate a long enough zero temp to eventually correct back up to target
                if (rate <= 0) {
                    bgUndershoot = (target_bg - naive_eventualBG)
                    val worstCaseInsulinReq = bgUndershoot / sens
                    var durationReq = round(60 * worstCaseInsulinReq / profile.current_basal)
                    if (durationReq < 0) {
                        durationReq = 0
                        // don't set a temp longer than 120 minutes
                    } else {
                        durationReq = round(durationReq / 30.0) * 30
                        durationReq = min(60, max(0, durationReq))
                    }
                    //console.error(durationReq);
                    if (durationReq > 0) {
                        rT.reason.append(", setting ${durationReq}m zero temp. ")
                        return setTempBasal(rate, durationReq, profile, rT, currenttemp)
                    }
                } else {
                    rT.reason.append(", setting ${round(rate, 2)}U/hr. ")
                }
                return setTempBasal(rate, 30, profile, rT, currenttemp)
            }
        }

        // if eventual BG is above min but BG is falling faster than expected Delta
        if (minDelta < expectedDelta) {
            // if in SMB mode, don't cancel SMB zero temp
            if (!(microBolusAllowed && enableSMB)) {
                if (glucose_status.delta < minDelta) {
                    rT.reason.append(
                        "Eventual BG ${convert_bg(eventualBG)} > ${convert_bg(min_bg)} but Delta ${convert_bg(tick.toDouble())} < Exp. Delta ${
                            convert_bg(expectedDelta)
                        }"
                    )
                } else {
                    rT.reason.append("Eventual BG ${convert_bg(eventualBG)} > ${convert_bg(min_bg)} but Min. Delta ${minDelta.toFixed2()} < Exp. Delta ${convert_bg(expectedDelta)}")
                }
                if (currenttemp.duration > 15 && (round_basal(basal) == round_basal(currenttemp.rate))) {
                    rT.reason.append(", temp " + currenttemp.rate + " ~ req " + round(basal, 2).withoutZeros() + "U/hr. ")
                    return rT
                } else {
                    rT.reason.append("; setting current basal of ${round(basal, 2)} as temp. ")
                    return setTempBasal(basal, 30, profile, rT, currenttemp)
                }
            }
        }
        // eventualBG or minPredBG is below max_bg
        if (min(eventualBG, minPredBG) < max_bg) {
            // if in SMB mode, don't cancel SMB zero temp
            if (!(microBolusAllowed && enableSMB)) {
                rT.reason.append("${convert_bg(eventualBG)}-${convert_bg(minPredBG)} in range: no temp required")
                if (currenttemp.duration > 15 && (round_basal(basal) == round_basal(currenttemp.rate))) {
                    rT.reason.append(", temp ${currenttemp.rate} ~ req ${round(basal, 2).withoutZeros()}U/hr. ")
                    return rT
                } else {
                    rT.reason.append("; setting current basal of ${round(basal, 2)} as temp. ")
                    return setTempBasal(basal, 30, profile, rT, currenttemp)
                }
            }
        }

        val (conditionResult, conditionsTrue) = isCriticalSafetyCondition()
        val lineSeparator = System.lineSeparator()
        val logAIMI = """
    |The ai model predicted SMB of ${predictedSMB}u and after safety requirements and rounding to .05, requested ${smbToGive}u to the pump<br>$lineSeparator
    |Version du plugin OpenApsAIMI-MT.2 ML.2, 21 Mars 2024<br>$lineSeparator
    |adjustedFactors: $adjustedFactors<br>$lineSeparator
    |
    |Max IOB: $maxIob<br>$lineSeparator
    |Max SMB: $maxSMB<br>$lineSeparator
    |sleep: $sleepTime<br>$lineSeparator
    |sport: $sportTime<br>$lineSeparator
    |snack: $snackTime<br>$lineSeparator
    |lowcarb: $lowCarbTime<br>$lineSeparator
    |highcarb: $highCarbTime<br>$lineSeparator
    |meal: $mealTime<br>$lineSeparator
    |fastingtime: $fastingTime<br>$lineSeparator
    |intervalsmb: $intervalsmb<br>$lineSeparator
    |mealruntime: $mealruntime<br>$lineSeparator
    |highCarbrunTime: $highCarbrunTime<br>$lineSeparator
    |
    |insulinEffect: $insulinEffect
    |bg: $bg
    |targetBG: $targetBg
    |futureBg: $predictedBg
    |delta: $delta<br>$lineSeparator
    |short avg delta: $shortAvgDelta
    |long avg delta: $longAvgDelta<br>$lineSeparator
    |accelerating_up: $accelerating_up
    |deccelerating_up: $deccelerating_up
    |accelerating_down: $accelerating_down
    |deccelerating_down: $deccelerating_down
    |stable: $stable<br>$lineSeparator
    |
    |IOB: $iob<br>$lineSeparator
    |tdd 7d/h: ${roundToPoint05(tdd7DaysPerHour)}
    |tdd 2d/h: ${roundToPoint05(tdd2DaysPerHour)}
    |tdd daily/h: ${roundToPoint05(tddPerHour)}
    |tdd 24h/h: ${roundToPoint05(tdd24HrsPerHour)}<br>$lineSeparator
    |enablebasal: $enablebasal<br>|basalaimi: $basalaimi<br>$lineSeparator
    |ISF: $variableSensitivity<br>$lineSeparator
    |
    |Hour of day: $hourOfDay<br>$lineSeparator
    |Weekend: $weekend<br>$lineSeparator
    |5 Min Steps: $recentSteps5Minutes
    |10 Min Steps: $recentSteps10Minutes
    |15 Min Steps: $recentSteps15Minutes
    |30 Min Steps: $recentSteps30Minutes
    |60 Min Steps: $recentSteps60Minutes
    |180 Min Steps: $recentSteps180Minutes<br>$lineSeparator
    |Heart Beat(average past 5 minutes): $averageBeatsPerMinute
    |Heart Beat(average past 10 minutes): $averageBeatsPerMinute10
    |Heart Beat(average past 60 minutes): $averageBeatsPerMinute60
    |Heart Beat(average past 180 minutes): $averageBeatsPerMinute180<br>$lineSeparator
    |COB: ${cob}g Future: ${futureCarbs}g<br>
    |COB Age Min: $lastCarbAgeMin<br>$lineSeparator
    |
    |tags0to60minAgo: ${tags0to60minAgo}
    |tags60to120minAgo: $tags60to120minAgo
    |tags120to180minAgo: $tags120to180minAgo
    |tags180to240minAgo: $tags180to240minAgo<br>$lineSeparator
    |currentTIRLow: $currentTIRLow
    |currentTIRRange: $currentTIRRange
    |currentTIRAbove: $currentTIRAbove
    |lastHourTIRLow: $lastHourTIRLow<br>$lineSeparator
    |lastHourTIRLow100: $lastHourTIRLow100
    |lastHourTIRabove170: $lastHourTIRabove170<br>$lineSeparator
    |isCriticalSafetyCondition: $conditionResult, True Conditions: $conditionsTrue<br>$lineSeparator
    |lastBolusSMBMinutes: $lastBolusSMBMinutes<br>$lineSeparator
    |lastsmbtime: $lastsmbtime<br>$lineSeparator
    |lastCarbAgeMin: $lastCarbAgeMin<br>$lineSeparator
""".trimMargin()

        rT.reason.append(logAIMI)
        // eventual BG is at/above target
        // if iob is over max, just cancel any temps
        if (eventualBG >= max_bg) {
            rT.reason.append("Eventual BG " + convert_bg(eventualBG) + " >= " + convert_bg(max_bg) + ", ")
        }
        if (iob_data.iob > max_iob) {
            rT.reason.append("IOB ${round(iob_data.iob, 2)} > max_iob $max_iob")
            if (currenttemp.duration > 15 && (round_basal(basal) == round_basal(currenttemp.rate))) {
                rT.reason.append(", temp ${currenttemp.rate} ~ req ${round(basal, 2).withoutZeros()}U/hr. ")
                return rT
            } else {
                rT.reason.append("; setting current basal of ${round(basal, 2)} as temp. ")
                return setTempBasal(basal, 30, profile, rT, currenttemp)
            }
        } else { // otherwise, calculate 30m high-temp required to get projected BG down to target
            // insulinReq is the additional insulin required to get minPredBG down to target_bg
            //console.error(minPredBG,eventualBG);
            //var insulinReq = round((min(minPredBG, eventualBG) - target_bg) / future_sens, 2)
            var insulinReq = smbToGive.toDouble()
            // if that would put us over max_iob, then reduce accordingly
            /*if (insulinReq > max_iob - iob_data.iob) {
                rT.reason.append("max_iob $max_iob, ")
                insulinReq = max_iob - iob_data.iob
            }*/

            // rate required to deliver insulinReq more insulin over 30m:
            var rate = basal + (2 * insulinReq)
            rate = round_basal(rate)
            insulinReq = round(insulinReq, 3)
            rT.insulinReq = insulinReq
            //console.error(iob_data.lastBolusTime);
            // minutes since last bolus
            val lastBolusAge = round((systemTime - iob_data.lastBolusTime) / 60000.0, 1)
            //console.error(lastBolusAge);
            //console.error(profile.temptargetSet, target_bg, rT.COB);
            // only allow microboluses with COB or low temp targets, or within DIA hours of a bolus
            val maxBolus: Double
            if (microBolusAllowed && enableSMB) {
                // never bolus more than maxSMBBasalMinutes worth of basal
                val mealInsulinReq = round(meal_data.mealCOB / profile.carb_ratio, 3)
                if (iob_data.iob > mealInsulinReq && iob_data.iob > 0) {
                    consoleError.add("IOB ${iob_data.iob} > COB ${meal_data.mealCOB}; mealInsulinReq = $mealInsulinReq")
                    consoleError.add("profile.maxUAMSMBBasalMinutes: ${profile.maxUAMSMBBasalMinutes} profile.current_basal: ${profile.current_basal}")
                    maxBolus = round(profile.current_basal * profile.maxUAMSMBBasalMinutes / 60, 1)
                } else {
                    consoleError.add("profile.maxSMBBasalMinutes: ${profile.maxSMBBasalMinutes} profile.current_basal: ${profile.current_basal}")
                    maxBolus = round(profile.current_basal * profile.maxSMBBasalMinutes / 60, 1)
                }
                // bolus 1/2 the insulinReq, up to maxBolus, rounding down to nearest bolus increment
                //val roundSMBTo = 1 / profile.bolus_increment
                //insulinReq = smbToGive.toDouble()
                //val microBolus = Math.floor(Math.min(insulinReq / 2, maxBolus) * roundSMBTo) / roundSMBTo
                val microBolus = insulinReq
                // calculate a long enough zero temp to eventually correct back up to target
                val smbTarget = target_bg
                val worstCaseInsulinReq = (smbTarget - (naive_eventualBG + minIOBPredBG) / 2.0) / sens
                var durationReq = round(30 * worstCaseInsulinReq / profile.current_basal)

                // if insulinReq > 0 but not enough for a microBolus, don't set an SMB zero temp
                if (insulinReq > 0 && microBolus < profile.bolus_increment) {
                    durationReq = 0
                }

                var smbLowTempReq = 0.0
                if (durationReq <= 0) {
                    durationReq = 0
                    // don't set an SMB zero temp longer than 60 minutes
                } else if (durationReq >= 30) {
                    durationReq = round(durationReq / 30.0) * 30
                    durationReq = min(60, max(0, durationReq))
                } else {
                    // if SMB durationReq is less than 30m, set a nonzero low temp
                    smbLowTempReq = round(basal * durationReq / 30.0, 2)
                    durationReq = 30
                }
                rT.reason.append(" insulinReq $insulinReq")
                if (microBolus >= maxSMB) {
                    rT.reason.append("; maxBolus $maxSMB")
                }
                if (durationReq > 0) {
                    rT.reason.append("; setting ${durationReq}m low temp of ${smbLowTempReq}U/h")
                }
                rT.reason.append(". ")

                // allow SMBIntervals between 1 and 10 minutes
                //val SMBInterval = min(10, max(1, profile.SMBInterval))
                val SMBInterval = min(20, max(1, intervalsmb))
                val nextBolusMins = round(SMBInterval - lastBolusAge, 0)
                val nextBolusSeconds = round((SMBInterval - lastBolusAge) * 60, 0) % 60
                //console.error(naive_eventualBG, insulinReq, worstCaseInsulinReq, durationReq);
                consoleError.add("naive_eventualBG $naive_eventualBG,${durationReq}m ${smbLowTempReq}U/h temp needed; last bolus ${lastBolusAge}m ago; maxBolus: $maxBolus")
                if (lastBolusAge > SMBInterval) {
                    if (microBolus > 0) {
                        rT.units = microBolus
                        rT.reason.append("Microbolusing ${microBolus}U. ")
                    }
                } else {
                    rT.reason.append("Waiting " + nextBolusMins + "m " + nextBolusSeconds + "s to microbolus again. ")
                }
                //rT.reason += ". ";

                // if no zero temp is required, don't return yet; allow later code to set a high temp
                if (durationReq > 0) {
                    rT.rate = smbLowTempReq
                    rT.duration = durationReq
                    return rT
                }

            }
            val (conditionResult, _) = isCriticalSafetyCondition()

            val maxSafeBasal = getMaxSafeBasal(profile)
            if (mealTime && mealruntime < 30){
                rate = if (basal == 0.0) (profile_current_basal * 10) else round_basal(basal * 10)
                rT.reason.append("${currenttemp.duration}m@${(currenttemp.rate).toFixed2()} AI Force basal because mealTime ${round(rate, 2)}U/hr. ")
                return setTempBasal(rate, 30, profile, rT, currenttemp)
            }else if (highCarbTime && highCarbrunTime < 60){
                rate = if (basal == 0.0) (profile_current_basal * 10) else round_basal(basal * 10)
                rT.reason.append("${currenttemp.duration}m@${(currenttemp.rate).toFixed2()} AI Force basal because highcarbTime ${round(rate, 2)}U/hr. ")
                return setTempBasal(rate, 30, profile, rT, currenttemp)
            }else if (bg > 180){
                rate = if (basal == 0.0) (profile_current_basal * 10) else round_basal(basal * 10)
                rT.reason.append("${currenttemp.duration}m@${(currenttemp.rate).toFixed2()} AI Force basal because bg > 200 ${round(rate, 2)}U/hr. ")
                return setTempBasal(rate, 30, profile, rT, currenttemp)
            }else if (honeymoon && delta > 2 && bg > 90 && bg < 110) {
                rate = profile_current_basal
                rT.reason.append("${currenttemp.duration}m@${(currenttemp.rate).toFixed2()} AI Force basal because honeymoon ${round(rate, 2)}U/hr. ")
                return setTempBasal(rate, 30, profile, rT, currenttemp)
            }else if (honeymoon && delta > 0 && bg > 110 && eventualBG > 100){
                rate = if (basal == 0.0) (profile_current_basal * delta) else round_basal(basal * delta)
                rT.reason.append("${currenttemp.duration}m@${(currenttemp.rate).toFixed2()} AI Force basal because honeymoon ${round(rate, 2)}U/hr. ")
                return setTempBasal(rate, 30, profile, rT, currenttemp)
            }else if (pregnancyEnable && delta > 0 && bg > 110 && !honeymoon) {
                rate = if (basal == 0.0) (profile_current_basal * 10) else round_basal(basal * 10)
                rT.reason.append("${currenttemp.duration}m@${(currenttemp.rate).toFixed2()} AI Force basal because pregnancy ${round(rate, 2)}U/hr. ")
                return setTempBasal(rate, 30, profile, rT, currenttemp)
            }else if (delta > 0 && bg > 80 && eventualBG > 65 && !honeymoon){
                rate = if (basal == 0.0) (profile_current_basal * delta) else round_basal(basal * delta)
                rT.reason.append("${currenttemp.duration}m@${(currenttemp.rate).toFixed2()} AI Force basal ${round(rate, 2)}U/hr. ")
                return setTempBasal(rate, 30, profile, rT, currenttemp)
            } else if (conditionResult && delta > 0 && bg > 80 && !honeymoon){
                rate = if (basal == 0.0) (profile_current_basal * delta) else round_basal(basal * delta)
                rT.reason.append("${currenttemp.duration}m@${(currenttemp.rate).toFixed2()} AI Force basal when isCriticalSafetyCondition is true ${round(rate, 2)}U/hr. ")
                return setTempBasal(rate, 30, profile, rT, currenttemp)
            }

            if (rate > maxSafeBasal) {
                rT.reason.append("adj. req. rate: ${round(rate, 2)} to maxSafeBasal: ${maxSafeBasal.withoutZeros()}, ")
                rate = round_basal(maxSafeBasal)
            }

            val insulinScheduled = currenttemp.duration * (currenttemp.rate - basal) / 60
            if (insulinScheduled >= insulinReq * 2) { // if current temp would deliver >2x more than the required insulin, lower the rate
                rT.reason.append("${currenttemp.duration}m@${(currenttemp.rate).toFixed2()} > 2 * insulinReq. Setting temp basal of ${round(rate, 2)}U/hr. ")
                return setTempBasal(rate, 30, profile, rT, currenttemp)
            }

            if (currenttemp.duration == 0) { // no temp is set
                rT.reason.append("no temp, setting " + round(rate, 2).withoutZeros() + "U/hr. ")
                return setTempBasal(rate, 30, profile, rT, currenttemp)
            }

            if (currenttemp.duration > 5 && (round_basal(rate) <= round_basal(currenttemp.rate))) { // if required temp <~ existing temp basal
                rT.reason.append("temp ${(currenttemp.rate).toFixed2()} >~ req ${round(rate, 2).withoutZeros()}U/hr. ")
                return rT
            }

            // required temp > existing temp basal
            rT.reason.append("temp ${currenttemp.rate.toFixed2()} < ${round(rate, 2).withoutZeros()}U/hr. ")
            return setTempBasal(rate, 30, profile, rT, currenttemp)
        }
    }
}
