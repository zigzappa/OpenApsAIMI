package app.aaps.plugins.aps.openAPSAIMI

import android.text.Spanned
import app.aaps.core.interfaces.aps.VariableSensitivityResult
import app.aaps.core.interfaces.logging.LTag
import app.aaps.core.utils.HtmlHelper
import app.aaps.plugins.aps.openAPSSMB.DetermineBasalResultSMB
import dagger.android.HasAndroidInjector
import org.json.JSONException
import org.json.JSONObject

class DetermineBasalResultAIMISMB private constructor(injector: HasAndroidInjector) : DetermineBasalResultSMB(injector), VariableSensitivityResult {

    var constraintStr: String = ""
    var glucoseStr: String = ""
    var iobStr: String = ""
    var profileStr: String = ""
    var mealStr: String = ""
    var delta:Float = 0.0f
    var bg:Float = 0.0f
    var basalaimi:Float = 0.0f
    override var variableSens: Double? = null

    internal constructor(
        injector: HasAndroidInjector,
        requestedSMB: Float,
        delta: Float,
        bg: Float,
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
                }
            }
        }
        fun extractIobValues() {
            val lines = iobStr.split("<br/>")
            lines.forEach { line ->
                when {
                    line.startsWith("basalaimi: ") -> basalaimi= line.substringAfter("basalaimi: ").toFloatOrNull() ?: 0.0f
                }
            }
        }

        this.date = dateUtil.now()
        extractGlucoseValues()
            this.isTempBasalRequested = true
            if (this.delta <= 0.0f && this.bg <= 140.0f) {
                this.rate = 0.0
                this.duration = 120
            }else if(this.delta > 0.0f && this.bg > 80){
                this.rate = basalaimi.toDouble()
                this.duration = 30
            }

        this.smb = requestedSMB.toDouble()
        if (requestedSMB > 0) {
            this.deliverAt = dateUtil.now()
        }

        this.reason = reason
    }

    override fun toSpanned(): Spanned {
        val result = "$constraintStr<br/><br/>$glucoseStr<br/><br/>$iobStr" +
            "<br/><br/>$profileStr<br/><br/>$mealStr<br/><br/>$reason"
        return HtmlHelper.fromHtml(result)
    }

    override fun newAndClone(injector: HasAndroidInjector): DetermineBasalResultSMB {
        val newResult = DetermineBasalResultAIMISMB(injector)
        doClone(newResult)
        return newResult
    }

    override fun json(): JSONObject? {
        val result = "$constraintStr<br/><br/>$glucoseStr<br/><br/>$iobStr" +
            "<br/><br/>$profileStr<br/><br/>$mealStr<br/><br/>$reason"
        val jsonData = JSONObject()
        try {
            // Ajout des données dans l'objet JSON
            jsonData.put("reason", result)

        } catch (e: JSONException) {
            aapsLogger.error(LTag.APS, "Error creating JSON object", e)
            return null
        }
        return jsonData
    }


    init {
        hasPredictions = true
    }
}