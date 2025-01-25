package app.aaps.database.transactions

import app.aaps.database.entities.TotalDailyDose
import app.aaps.database.entities.embedments.InterfaceIDs

/**
 * Creates or updates the TotalDailyDose from data caching
 */
class InsertOrUpdateCachedTotalDailyDoseTransaction(
    private val tdd: TotalDailyDose
) : Transaction<InsertOrUpdateCachedTotalDailyDoseTransaction.TransactionResult>() {

    override fun run(): TransactionResult {
        val result = TransactionResult()
        var current: TotalDailyDose? = null
        // search by timestamp
        current = database.totalDailyDoseDao.findByPumpTimestamp(tdd.timestamp, InterfaceIDs.PumpType.CACHE)
        if (current == null) {
            database.totalDailyDoseDao.insertNewEntry(tdd)
            result.inserted.add(tdd)
        } else {
            current.basalAmount = tdd.basalAmount
            current.bolusAmount = tdd.bolusAmount
            current.totalAmount = tdd.totalAmount
            current.carbs = tdd.carbs
            database.totalDailyDoseDao.updateExistingEntry(current)
            result.updated.add(current)
        }

        return result
    }

    class TransactionResult {

        val inserted = mutableListOf<TotalDailyDose>()
        val updated = mutableListOf<TotalDailyDose>()
    }
}