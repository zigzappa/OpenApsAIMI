package app.aaps.plugins.aps.openAPSAIMI

import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

class aimiNeuralNetwork(private val inputSize: Int, private val hiddenSize: Int, private val outputSize: Int) {
    private val weightsInputHidden = Array(inputSize) { DoubleArray(hiddenSize) { Math.random() } }
    private val biasHidden = DoubleArray(hiddenSize) { Math.random() }
    private val weightsHiddenOutput = DoubleArray(hiddenSize) { Math.random() }
    private val biasOutput = DoubleArray(outputSize) { Math.random() }
    var lastTrainingException: Exception? = null
    var trainingLossHistory: MutableList<Double> = mutableListOf()

    private fun heInitialization(size: Int): DoubleArray {
        val scale = sqrt(6.0 / (inputSize + size))
        return DoubleArray(size) { Random.nextDouble(-scale, scale) }
    }
    private fun relu(x: Double) = max(0.01, x)

    private fun forwardPass(input: FloatArray): Pair<DoubleArray, DoubleArray> {
        val hidden = DoubleArray(hiddenSize) { i ->
            var sum = 0.0
            for (j in input.indices) {
                sum += input[j] * weightsInputHidden[j][i]
            }
            relu(sum + biasHidden[i])
        }

        val output = DoubleArray(outputSize) { i ->
            var sum = 0.0
            for (j in hidden.indices) {
                sum += hidden[j] * weightsHiddenOutput[j]
            }
            sum + biasOutput[i]
        }

        return Pair(hidden, output)
    }

    fun predict(input: FloatArray): DoubleArray {
        return forwardPass(input).second
    }
    private fun DoubleArray.clipInPlace(min: Double, max: Double) {
        for (i in indices) {
            this[i] = this[i].coerceIn(min, max)
        }
    }

    private fun backpropagation(input: FloatArray, target: DoubleArray, learningRate: Double) {
        val (hidden, output) = forwardPass(input)

        // Gradient de la perte par rapport à la sortie prédite
        val gradLossOutput = DoubleArray(outputSize) { i -> 2.0 * (output[i] - target[i]) }
        val clipValue = 1.0 // This is an example value; adjust as necessary
        gradLossOutput.clipInPlace(-clipValue, clipValue)

        // Mise à jour des poids et biais de la couche de sortie
        for (i in 0 until outputSize) {
            for (j in 0 until hiddenSize) {
                weightsHiddenOutput[j] -= learningRate * gradLossOutput[i] * hidden[j] // Notez l'ordre des indices ici
            }
            biasOutput[i] -= learningRate * gradLossOutput[i]
        }

        // Mise à jour des poids et biais de la couche cachée
        for (i in 0 until inputSize) {
            for (j in 0 until hiddenSize) {
                val gradRelu = if (hidden[j] > 0) 1 else 0.001
                var gradLossHidden = 0.0
                for (k in 0 until outputSize) {
                    gradLossHidden += gradLossOutput[k] * weightsHiddenOutput[j]
                }
                gradLossHidden *= gradRelu

                weightsInputHidden[i][j] -= learningRate * gradLossHidden * input[i]
            }
        }

        // Mise à jour du biais de la couche cachée
        for (j in 0 until hiddenSize) {
            val gradRelu = if (hidden[j] > 0) 1 else 0.001
            var gradLossHidden = 0.0
            for (k in 0 until outputSize) {
                gradLossHidden += gradLossOutput[k] * weightsHiddenOutput[j]
            }
            gradLossHidden *= gradRelu

            biasHidden[j] -= learningRate * gradLossHidden
        }
    }

    /*fun train(inputs: List<FloatArray>, targets: List<DoubleArray>, epochs: Int, learningRate: Double) {
        for (epoch in 1..epochs) {
            var totalLoss = 0.0
            for ((input, target) in inputs.zip(targets)) {
                val output = forwardPass(input).second
                totalLoss += mseLoss(output, target)
                backpropagation(input, target, learningRate)
            }
            println("Epoch $epoch, Loss: ${totalLoss / inputs.size}")
        }
    }*/
    fun train(inputs: List<FloatArray>, targets: List<DoubleArray>, epochs: Int, learningRate: Double) {
        lastTrainingException = null
        trainingLossHistory.clear()

        for (epoch in 1..epochs) {
            var totalLoss = 0.0
            try {
                for ((input, target) in inputs.zip(targets)) {
                    val output = forwardPass(input).second
                    totalLoss += mseLoss(output, target)
                    backpropagation(input, target, learningRate)
                }
                val averageLoss = totalLoss / inputs.size
                trainingLossHistory.add(averageLoss)
                println("Epoch $epoch, Loss: $averageLoss")
            } catch (e: Exception) {
                lastTrainingException = e
                println("Exception during training at epoch $epoch: ${e.message}")
                break // Sortie anticipée de la boucle d'entraînement en cas d'exception
            }
        }
    }

    // Fonction d'erreur quadratique moyenne
    private fun mseLoss(output: DoubleArray, target: DoubleArray): Double {
        return output.zip(target).sumOf { (o, t) -> (o - t).pow(2) } / output.size
    }
}



private operator fun Double.timesAssign(gradRelu: Number) {

}

fun refineSMB(smb: Float, neuralNetwork: aimiNeuralNetwork, input: FloatArray): Float {
    val prediction = neuralNetwork.predict(input)
    return smb + prediction[0].toFloat()
}
