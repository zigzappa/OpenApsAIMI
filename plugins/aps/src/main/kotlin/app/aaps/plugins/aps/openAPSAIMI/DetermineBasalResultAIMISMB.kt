package app.aaps.plugins.aps.openAPSAIMI

import android.text.Spanned
import app.aaps.core.interfaces.aps.VariableSensitivityResult
import app.aaps.core.interfaces.logging.LTag
import app.aaps.core.interfaces.utils.SafeParse
import app.aaps.core.utils.HtmlHelper
import app.aaps.plugins.aps.R
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
    var basaloapsaimirate: Float = 0.0f
    override var variableSens: Double? = null

    internal constructor(
        injector: HasAndroidInjector,
        requestedSMB: Float,
        basaloapsaimirate: Float,
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
        this.basaloapsaimirate = basaloapsaimirate

        this.date = dateUtil.now()

        this.isTempBasalRequested = true
        this.rate = basaloapsaimirate.toDouble()
        this.duration = SafeParse.stringToDouble(sp.getString(R.string.key_B30_duration, "20")).toInt()

        this.smb = requestedSMB.toDouble()
        if (requestedSMB > 0) {
            this.deliverAt = dateUtil.now()
        }

        this.reason = reason
    }

    override fun toSpanned(): Spanned {
        val result = "$constraintStr<br/><br/>$glucoseStr<br/><br/>$iobStr" +
            "<br/><br/>$profileStr<br/><br/>$mealStr<br/><br/><br/>$reason"
        return HtmlHelper.fromHtml(result)
    }

    override fun newAndClone(injector: HasAndroidInjector): DetermineBasalResultSMB {
        val newResult = DetermineBasalResultAIMISMB(injector)
        doClone(newResult)
        return newResult
    }

    override fun json(): JSONObject? {
        try {
            return JSONObject(json.toString())
        } catch (e: JSONException) {
            aapsLogger.error(LTag.APS, "Unhandled exception", e)
        }
        return null
    }

    init {
        hasPredictions = true
    }
}