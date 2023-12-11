package app.aaps.plugins.aps.openAPSAIMI

import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

class aimiNeuralNetwork(private val inputSize: Int, private val hiddenSize: Int, private val outputSize: Int) {

    private val weightsInputHidden = Array(inputSize) { heInitialization(hiddenSize) }
    private val biasHidden = DoubleArray(hiddenSize) { 0.01 }// Petite valeur constante
    private val weightsHiddenOutput = Array(hiddenSize) { heInitialization(outputSize) }
    private val biasOutput = DoubleArray(outputSize) { 0.01 }  // Petite valeur constante
    var lastTrainingException: Exception? = null
    var trainingLossHistory: MutableList<Double> = mutableListOf()
    private fun heInitialization(size: Int): DoubleArray {
        val scale = sqrt(6.0 / (inputSize + size))
        return DoubleArray(size) { Random.nextDouble(-scale, scale) }
    }
    private fun applyDropout(layer: DoubleArray, dropoutRate: Double): DoubleArray {
        return layer.map { if (Random.nextDouble() > dropoutRate) it else 0.0 }.toDoubleArray()
    }

    private fun relu(x: Double) = max(0.0, x)

    private fun forwardPass(input: FloatArray): Pair<DoubleArray, DoubleArray> {
        val hidden = DoubleArray(hiddenSize) { i ->
            var sum = 0.0
            for (j in input.indices) {
                sum += input[j] * weightsInputHidden[j][i]
            }
            relu(sum + biasHidden[i])
        }
        // Application du Dropout sur la première couche cachée
        val hiddenWithDropout = applyDropout(hidden, 0.5) // Taux de Dropout de 50%

        // Calcul des activations de la deuxième couche cachée
        val hidden2 = DoubleArray(hiddenSize) { i ->
            var sum = 0.0
            for (j in hiddenWithDropout.indices) {
                sum += hiddenWithDropout[j] * weightsInputHidden[j][i]
            }
            relu(sum + biasHidden[i])
        }

        val output = DoubleArray(outputSize) { i ->
            var sum = 0.0
            for (j in hidden.indices) {
                sum += hidden[j] * weightsHiddenOutput[j][i]
            }
            sum + biasOutput[i]
        }

        return Pair(hidden2, output)
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
        //val clipValue = 1.0 // This is an example value; adjust as necessary
        //gradLossOutput.clipInPlace(-clipValue, clipValue)

        // Mise à jour des poids et biais de la couche de sortie
        for (i in 0 until outputSize) {
            for (j in 0 until hiddenSize) {
                weightsHiddenOutput[j][i] -= learningRate * gradLossOutput[i] * hidden[j] // Notez l'ordre des indices ici
            }
            biasOutput[i] -= learningRate * gradLossOutput[i]
        }

        // Mise à jour des poids et biais de la couche cachée
        for (i in 0 until inputSize) {
            for (j in 0 until hiddenSize) {
                val gradRelu = if (hidden[j] > 0) 1 else 0.001
                var gradLossHidden = 0.0
                for (k in 0 until outputSize) {
                    gradLossHidden += gradLossOutput[k] * weightsHiddenOutput[j][i]
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
                gradLossHidden += gradLossOutput[k] * weightsHiddenOutput[j][k]
            }
            gradLossHidden *= gradRelu

            biasHidden[j] -= learningRate * gradLossHidden
        }
    }
    /*fun train(
        inputs: List<FloatArray>, targets: List<DoubleArray>,
        validationInputs: List<FloatArray>, validationTargets: List<DoubleArray>,
        epochs: Int, initialLearningRate: Double,
        patience: Int = 10, // pour early stopping
        regularizationLambda: Double = 0.01) { // lambda pour régularisation L2
        var learningRate = initialLearningRate
        var bestLoss = Double.MAX_VALUE
        var epochsWithoutImprovement = 0

        for (epoch in 1..epochs) {
            var totalLoss = 0.0
            try {
                for ((input, target) in inputs.zip(targets)) {
                    val output = forwardPass(input).second
                    totalLoss += mseLoss(output, target) + l2Regularization(regularizationLambda)
                    backpropagation(input, target, learningRate)
                }
                val averageLoss = totalLoss / inputs.size
                trainingLossHistory.add(averageLoss)

                // Validation de l'Époque
                val validationLoss = validate(validationInputs, validationTargets)
                println("Epoch $epoch, Training Loss: $averageLoss, Validation Loss: $validationLoss")

                // Early Stopping
                if (validationLoss < bestLoss) {
                    bestLoss = validationLoss
                    epochsWithoutImprovement = 0
                } else {
                    epochsWithoutImprovement++
                    if (epochsWithoutImprovement >= patience) {
                        println("Early stopping triggered at epoch $epoch")
                        break
                    }
                }

                // Ajustement Dynamique du Taux d'Apprentissage
                learningRate = learningRate * 0.95 // Réduction du taux d'apprentissage de 5% par époque, par exemple

            } catch (e: Exception) {
                lastTrainingException = e
                println("Exception during training at epoch $epoch: ${e.message}")
                break // Sortie anticipée de la boucle d'entraînement en cas d'exception
            }
        }
    }*/
    fun train(
        inputs: List<FloatArray>, targets: List<DoubleArray>,
        validationInputs: List<FloatArray>, validationTargets: List<DoubleArray>,
        epochs: Int, initialLearningRate: Double,
        batchSize: Int = 32, // Taille du lot pour l'entraînement par lots
        patience: Int = 10, // pour early stopping
        regularizationLambda: Double = 0.01 // lambda pour régularisation L2
    ) {
        var learningRate = initialLearningRate
        var bestLoss = Double.MAX_VALUE
        var epochsWithoutImprovement = 0

        for (epoch in 1..epochs) {
            var totalLoss = 0.0
            try {
                // Entraînement par lots
                inputs.chunked(batchSize).zip(targets.chunked(batchSize)).forEach { (batchInputs, batchTargets) ->
                    batchInputs.zip(batchTargets).forEach { (input, target) ->
                        val output = forwardPass(input).second
                        totalLoss += mseLoss(output, target) + l2Regularization(regularizationLambda)
                        backpropagation(input, target, learningRate)
                    }
                }

                val averageLoss = totalLoss / inputs.size
                trainingLossHistory.add(averageLoss)

                // Validation de l'Époque
                val validationLoss = validate(validationInputs, validationTargets)
                println("Epoch $epoch, Training Loss: $averageLoss, Validation Loss: $validationLoss")

                // Early Stopping
                if (validationLoss < bestLoss) {
                    bestLoss = validationLoss
                    epochsWithoutImprovement = 0
                } else {
                    epochsWithoutImprovement++
                    if (epochsWithoutImprovement >= patience) {
                        println("Early stopping triggered at epoch $epoch")
                        break
                    }
                }

                // Ajustement Dynamique du Taux d'Apprentissage
                learningRate = learningRate * 0.95 // Réduction du taux d'apprentissage de 5% par époque, par exemple

            } catch (e: Exception) {
                lastTrainingException = e
                println("Exception during training at epoch $epoch: ${e.message}")
                break // Sortie anticipée de la boucle d'entraînement en cas d'exception
            }
        }
    }


    private fun l2Regularization(lambda: Double): Double {
        var regularizationLoss = 0.0

        // Calculer la régularisation pour les poids entre l'entrée et la couche cachée
        for (layerWeights in weightsInputHidden) {
            for (weight in layerWeights) {
                regularizationLoss += weight * weight
            }
        }

        // Calculer la régularisation pour les poids entre la couche cachée et la sortie
        for (layerWeights in weightsHiddenOutput) {
            for (weight in layerWeights) {
                regularizationLoss += weight * weight
            }
        }

        return lambda * regularizationLoss / 2.0
    }

    fun validate(validationInputs: List<FloatArray>, validationTargets: List<DoubleArray>): Double {
        var totalValidationLoss = 0.0
        for ((input, target) in validationInputs.zip(validationTargets)) {
            val output = forwardPass(input).second
            totalValidationLoss += mseLoss(output, target)
        }
        return totalValidationLoss / validationInputs.size
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
