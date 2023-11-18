package app.aaps.plugins.aps.openAPSAIMI

import android.annotation.SuppressLint
import app.aaps.database.entities.TherapyEvent
import app.aaps.database.impl.AppRepository
import io.reactivex.rxjava3.core.Single
import java.util.concurrent.TimeUnit
class therapy (private val appRepository: AppRepository){

    var sleepTime = false
    var sportTime = false
    var snackTime = false
    var lowCarbTime = false
    var highCarbTime = false

    @SuppressLint("CheckResult")
    fun updateStatesBasedOnTherapyEvents() {
        // Mise à jour de l'état de sleepTime
        sleepTime = findActiveSleepEvents(System.currentTimeMillis()).blockingGet()
        sportTime = findActiveSportEvents(System.currentTimeMillis()).blockingGet()
        snackTime = findActiveSnackEvents(System.currentTimeMillis()).blockingGet()
        lowCarbTime = findActiveLowCarbEvents(System.currentTimeMillis()).blockingGet()
        highCarbTime = findActiveHighCarbEvents(System.currentTimeMillis()).blockingGet()
    }

    private fun findActiveSleepEvents(timestamp: Long): Single<Boolean> {
        val fromTime = timestamp - TimeUnit.DAYS.toMillis(1) // les dernières 24 heures
        return appRepository.getTherapyEventDataFromTime(fromTime, TherapyEvent.Type.NOTE, true)
            .map { events ->
                events.any { event ->
                    event.note?.contains("sleep", ignoreCase = true) == true &&
                        System.currentTimeMillis() <= (event.timestamp + event.duration)
                }
            }
    }

    private fun findActiveSportEvents(timestamp: Long): Single<Boolean> {
        val fromTime = timestamp - TimeUnit.DAYS.toMillis(1) // les dernières 24 heures
        return appRepository.getTherapyEventDataFromTime(fromTime, TherapyEvent.Type.NOTE, true)
            .map { events ->
                events.any { event ->
                    event.note?.contains("sport", ignoreCase = true) == true &&
                        System.currentTimeMillis() <= (event.timestamp + event.duration)
                }
            }
    }
    private fun findActiveSnackEvents(timestamp: Long): Single<Boolean> {
        val fromTime = timestamp - TimeUnit.DAYS.toMillis(1) // les dernières 24 heures
        return appRepository.getTherapyEventDataFromTime(fromTime, TherapyEvent.Type.NOTE, true)
            .map { events ->
                events.any { event ->
                    event.note?.contains("snack", ignoreCase = true) == true &&
                        System.currentTimeMillis() <= (event.timestamp + event.duration)
                }
            }
    }

    private fun findActiveLowCarbEvents(timestamp: Long): Single<Boolean> {
        val fromTime = timestamp - TimeUnit.DAYS.toMillis(1) // les dernières 24 heures
        return appRepository.getTherapyEventDataFromTime(fromTime, TherapyEvent.Type.NOTE, true)
            .map { events ->
                events.any { event ->
                    event.note?.contains("lowcarb", ignoreCase = true) == true &&
                        System.currentTimeMillis() <= (event.timestamp + event.duration)
                }
            }
    }

    private fun findActiveHighCarbEvents(timestamp: Long): Single<Boolean> {
        val fromTime = timestamp - TimeUnit.DAYS.toMillis(1) // les dernières 24 heures
        return appRepository.getTherapyEventDataFromTime(fromTime, TherapyEvent.Type.NOTE, true)
            .map { events ->
                events.any { event ->
                    event.note?.contains("highcarb", ignoreCase = true) == true &&
                        System.currentTimeMillis() <= (event.timestamp + event.duration)
                }
            }
    }

    private fun isEventActive(event: TherapyEvent, currentTime: Long): Boolean {
        val eventEndTime = event.timestamp + event.duration
        return currentTime <= eventEndTime
    }
}