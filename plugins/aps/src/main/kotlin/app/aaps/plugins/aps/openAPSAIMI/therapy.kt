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
    var mealTime = false
    var fastingTime = false

    @SuppressLint("CheckResult")
    fun updateStatesBasedOnTherapyEvents() {
        // Mise à jour de l'état de sleepTime
        sleepTime = findActiveSleepEvents(System.currentTimeMillis()).blockingGet()
        sportTime = findActiveSportEvents(System.currentTimeMillis()).blockingGet()
        snackTime = findActiveSnackEvents(System.currentTimeMillis()).blockingGet()
        lowCarbTime = findActiveLowCarbEvents(System.currentTimeMillis()).blockingGet()
        highCarbTime = findActiveHighCarbEvents(System.currentTimeMillis()).blockingGet()
        mealTime = findActiveMealEvents(System.currentTimeMillis()).blockingGet()
        fastingTime = findActiveMealEvents(System.currentTimeMillis()).blockingGet()
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
    private fun findActiveMealEvents(timestamp: Long): Single<Boolean> {
        val fromTime = timestamp - TimeUnit.DAYS.toMillis(1) // les dernières 24 heures
        return appRepository.getTherapyEventDataFromTime(fromTime, TherapyEvent.Type.NOTE, true)
            .map { events ->
                events.any { event ->
                    event.note?.contains("meal", ignoreCase = true) == true &&
                        System.currentTimeMillis() <= (event.timestamp + event.duration)
                }
            }
    }
    private fun findActiveFastingEvents(timestamp: Long): Single<Boolean> {
        val fromTime = timestamp - TimeUnit.DAYS.toMillis(1) // les dernières 24 heures
        return appRepository.getTherapyEventDataFromTime(fromTime, TherapyEvent.Type.NOTE, true)
            .map { events ->
                events.any { event ->
                    event.note?.contains("fasting", ignoreCase = true) == true &&
                        System.currentTimeMillis() <= (event.timestamp + event.duration)
                }
            }
    }

    fun getTimeElapsedSinceLastEvent(keyword: String): Long {
        val fromTime = System.currentTimeMillis() - TimeUnit.MINUTES.toMillis(60)
        val events = appRepository.getTherapyEventDataFromTime(fromTime, TherapyEvent.Type.NOTE, true).blockingGet()

        val lastEvent = events.filter { it.note?.contains(keyword, ignoreCase = true) == true }
            .maxByOrNull { it.timestamp }
        lastEvent?.let {
            // Calculer et retourner le temps écoulé en minutes depuis l'événement
            return (System.currentTimeMillis() - it.timestamp) / 60000  // Convertir en minutes
        }
        return -1  // Retourner -1 si aucun événement n'a été trouvé
    }

    private fun isEventActive(event: TherapyEvent, currentTime: Long): Boolean {
        val eventEndTime = event.timestamp + event.duration
        return currentTime <= eventEndTime
    }
}