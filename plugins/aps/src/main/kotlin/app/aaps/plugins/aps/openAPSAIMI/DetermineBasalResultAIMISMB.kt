package app.aaps.plugins.aps.openAPSAIMI

import android.text.Spanned
import app.aaps.core.interfaces.aps.VariableSensitivityResult
import app.aaps.core.interfaces.logging.LTag
import app.aaps.core.utils.HtmlHelper
import app.aaps.plugins.aps.APSResultObject
import app.aaps.plugins.aps.openAPSSMB.DetermineBasalResultSMB
import dagger.android.HasAndroidInjector
import org.json.JSONException
import org.json.JSONObject
import kotlin.math.round

class DetermineBasalResultAIMISMB private constructor(injector: HasAndroidInjector) : DetermineBasalResultSMB(injector), VariableSensitivityResult {

    var constraintStr: String = ""
    var glucoseStr: String = ""
    var iobStr: String = ""
    var profileStr: String = ""
    var mealStr: String = ""
    var delta:Float = 0.0f
    var bg:Float = 0.0f
    override var targetBG: Double = 0.0
    var basalaimi:Float = 0.0f
    var enablebasal:Boolean = false
    override var variableSens: Double? = null
    //val apsResultObject = APSResultObject(injector)


    internal constructor(
        injector: HasAndroidInjector,
        requestedSMB: Float,
        constraintStr: String,
        glucoseStr: String,
        iobStr: String,
        profileStr: String,
        mealStr: String,
        reason: String
    ) : this(injector) {
        this.constraintStr = constraintStr
        this.glucoseStr = glucoseStr
        this.iobStr = iobStr
        this.profileStr = profileStr
        this.mealStr = mealStr


        fun extractGlucoseValues() {
            val lines = glucoseStr.split("<br/>")
            lines.forEach { line ->
                when {
                    line.startsWith("bg: ") -> this.bg = line.substringAfter("bg: ").toFloatOrNull() ?: 0.0f
                    line.startsWith("delta: ") -> this.delta = line.substringAfter("delta: ").toFloatOrNull() ?: 0.0f
                    line.startsWith("targetBG: ") -> this.targetBG = (line.substringAfter("targetBg: ").toFloatOrNull() ?: 0.0f).toDouble()
                }
            }
        }
        fun extractIobValues() {
            val lines = iobStr.split("<br/>")
            lines.forEach { line ->
                when {
                    line.startsWith("basalaimi: ") -> this.basalaimi= line.substringAfter("basalaimi: ").toFloatOrNull() ?: 0.0f
                    line.startsWith("enablebasal: ") -> this.enablebasal= line.substringAfter("enablebasal: ").toBoolean()
                }
            }
        }


        this.date = dateUtil.now()
        extractGlucoseValues()
        extractIobValues()

        //updateAPSResult(apsResultObject)
            this.isTempBasalRequested = true

            if (this.delta <= 0.0f && this.bg <= 140.0f) {
                this.rate = 0.0
                this.duration = 120
                //this.deliverAt = dateUtil.now()
                //this.isChangeRequested
            }else if(this.delta > 0.0f && this.bg > 80 && enablebasal){
                this.rate = basalaimi.toDouble()
                this.duration = 30
                //this.deliverAt = dateUtil.now()
                //this.isChangeRequested
            }

        this.smb = requestedSMB.toDouble()
        if (requestedSMB > 0) {
            this.deliverAt = dateUtil.now()
        }
        //updateAPSResult(apsResultObject)
        this.reason = reason
    }

    override fun toSpanned(): Spanned {
        val result = "$constraintStr<br/><br/>$glucoseStr<br/><br/>$iobStr" +
            "<br/><br/>$profileStr<br/><br/>$mealStr<br/><br/>$reason"
        return HtmlHelper.fromHtml(result)
    }
    /*private fun updateAPSResult(apsResult: APSResultObject) {

        val newRate = round(this.basalaimi.toDouble() * this.delta)
        val newDuration = 30
        val newtargetBG = this.targetBG
        if (this.delta <= 0.0f && this.bg <= 140.0f) {
            apsResult.rate = 0.0
            apsResult.duration = newDuration
            this.isTempBasalRequested
            this.isChangeRequested
        }else if(this.delta > 0.0f && this.bg > 80){
            apsResult.rate = newRate.toDouble()
            apsResult.duration = newDuration
            this.isTempBasalRequested
            this.isChangeRequested
        }
        apsResult.targetBG = newtargetBG.toDouble()

    }*/

    override fun newAndClone(injector: HasAndroidInjector): DetermineBasalResultSMB {
        val newResult = DetermineBasalResultAIMISMB(injector)
        doClone(newResult)
        newResult.rate = this.rate
        newResult.duration = this.duration
        newResult.targetBG = this.targetBG
        return newResult
    }

    override fun json(): JSONObject? {
        val result = "$constraintStr<br/><br/>$glucoseStr<br/><br/>$iobStr" +
            "<br/><br/>$profileStr<br/><br/>$mealStr<br/><br/>$reason"
        val json = JSONObject()
        try {
            // Ajout des données dans l'objet JSON
            //jsonData.put("reason", result)
            if (isChangeRequested) {
                //json.put("rate", rate)
                //json.put("duration", duration)
                json.put("reason", result)
            }

        } catch (e: JSONException) {
            aapsLogger.error(LTag.APS, "Error creating JSON object", e)
            return null
        }
        return json
    }


    init {
        this.date = dateUtil.now()
        //updateAPSResult(apsResultObject)
        hasPredictions = true
    }


}