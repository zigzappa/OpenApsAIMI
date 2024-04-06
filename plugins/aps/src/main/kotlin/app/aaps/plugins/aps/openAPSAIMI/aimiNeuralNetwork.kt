package app.aaps.plugins.aps.openAPSAIMI

import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

class aimiNeuralNetwork(
    private val inputSize: Int,
    private val hiddenSize: Int,
    private val outputSize: Int,
    private var learningRate: Double = 0.01,
    private val regularizationLambda: Double = 0.01,
    private val dropoutRate: Double = 0.5
) {

    private val weightsInputHidden = Array(inputSize) { DoubleArray(hiddenSize) { Random.nextDouble(-sqrt(6.0 / (inputSize + hiddenSize)), sqrt(6.0 / (inputSize + hiddenSize))) } }
    private val biasHidden = DoubleArray(hiddenSize) { 0.01 }
    private val weightsHiddenOutput = Array(hiddenSize) { DoubleArray(outputSize) { Random.nextDouble(-sqrt(6.0 / (hiddenSize + outputSize)), sqrt(6.0 / (hiddenSize + outputSize))) } }
    private val biasOutput = DoubleArray(outputSize) { 0.01 }

    var lastTrainingException: Exception? = null
    var trainingLossHistory: MutableList<Double> = mutableListOf()

    companion object {

        fun refineSMB(smb: Float, neuralNetwork: aimiNeuralNetwork, input: FloatArray): Float {
            val prediction = neuralNetwork.predict(input)[0]
            return smb + prediction.toFloat()
        }

        fun refineBasalaimi(basalaimi: Float, neuralNetwork: aimiNeuralNetwork, input: FloatArray): Float {
            val prediction = neuralNetwork.forwardPassBasal(input).second[0]
            return basalaimi + prediction.toFloat()
        }
    }

    private fun applyDropout(layer: DoubleArray): DoubleArray {
        return layer.map { if (Random.nextDouble() > dropoutRate) it else 0.0 }.toDoubleArray()
    }

    private fun relu(x: Double) = max(0.0, x)

    fun createNewFeature(inputData: List<FloatArray>): List<FloatArray> {
        return inputData.map { originalFeatures ->
            val newFeature = originalFeatures[0] * originalFeatures[1]
            originalFeatures + newFeature
        }
    }

    fun trainForBasalaimi(
        inputs: List<FloatArray>,
        basalaimiTargets: List<DoubleArray>,
        epochs: Int
    ) {
        for (epoch in 1..epochs) {
            var totalLoss = 0.0
            for ((input, target) in inputs.zip(basalaimiTargets)) {
                val (hidden, output) = forwardPassBasal(input)
                totalLoss += mseLoss(output, target)
                backpropagationBasal(input, target)
            }
            val averageLoss = totalLoss / inputs.size
            println("Epoch $epoch: Training Loss = $averageLoss")
            trainingLossHistory.add(averageLoss)
        }
    }

    private fun backpropagationBasal(input: FloatArray, target: DoubleArray) {
        val (hidden, output) = forwardPassBasal(input)
        val gradLossOutput = DoubleArray(outputSize) { i -> 2.0 * (output[i] - target[i]) }

        for (i in 0 until outputSize) {
            for (j in 0 until hiddenSize) {
                weightsHiddenOutput[j][i] -= learningRate * gradLossOutput[i] * hidden[j]
            }
            biasOutput[i] -= learningRate * gradLossOutput[i]
        }

        val gradLossHidden = DoubleArray(hiddenSize) { 0.0 }
        for (i in 0 until hiddenSize) {
            for (j in 0 until outputSize) {
                gradLossHidden[i] += gradLossOutput[j] * weightsHiddenOutput[i][j]
            }
            if (hidden[i] <= 0) {
                gradLossHidden[i] = 0.0
            }
        }

        for (i in 0 until inputSize) {
            for (j in 0 until hiddenSize) {
                weightsInputHidden[i][j] -= learningRate * gradLossHidden[j] * input[i]
            }
        }
    }

    private fun forwardPassBasal(input: FloatArray): Pair<DoubleArray, DoubleArray> {
        val hiddenLayer = DoubleArray(hiddenSize) { i ->
            var sum = 0.0
            for (j in input.indices) {
                sum += input[j] * weightsInputHidden[j][i]
            }
            relu(sum + biasHidden[i])
        }

        val outputLayer = DoubleArray(outputSize) { i ->
            var sum = 0.0
            for (j in hiddenLayer.indices) {
                sum += hiddenLayer[j] * weightsHiddenOutput[j][i]
            }
            sum + biasOutput[i]
        }

        return Pair(hiddenLayer, outputLayer)
    }

    private fun zScoreNormalization(inputData: List<FloatArray>): List<FloatArray> {
        if (inputData.isEmpty()) {
            return emptyList()
        }
        val means = FloatArray(inputData.first().size) { 0f }
        val stdDevs = FloatArray(inputData.first().size) { 0f }

        inputData.forEach { features ->
            features.forEachIndexed { index, value ->
                means[index] += value
                stdDevs[index] += value * value
            }
        }

        means.indices.forEach { i ->
            means[i] /= inputData.size.toFloat()
            stdDevs[i] = sqrt(stdDevs[i] / inputData.size - means[i] * means[i])
        }

        return inputData.map { features ->
            FloatArray(features.size) { index ->
                if (stdDevs[index] != 0.0f) {
                    (features[index] - means[index]) / stdDevs[index]
                } else {
                    features[index] - means[index]
                }
            }
        }
    }

    private fun forwardPass(input: FloatArray): Pair<DoubleArray, DoubleArray> {
        val hidden = DoubleArray(hiddenSize) { i ->
            var sum = 0.0
            for (j in input.indices) {
                sum += input[j] * weightsInputHidden[j][i]
            }
            relu(sum + biasHidden[i])
        }
        val hiddenWithDropout = applyDropout(hidden)

        val hidden2 = DoubleArray(hiddenSize) { i ->
            var sum = 0.0
            for (j in hiddenWithDropout.indices) {
                sum += hiddenWithDropout[j] * weightsInputHidden[j][i]
            }
            relu(sum + biasHidden[i])
        }

        val output = DoubleArray(outputSize) { i ->
            var sum = 0.0
            for (j in hidden2.indices) {
                sum += hidden2[j] * weightsHiddenOutput[j][i]
            }
            sum + biasOutput[i]
        }

        return Pair(hidden2, output)
    }

    fun predict(input: FloatArray): DoubleArray {
        return forwardPass(input).second
    }

    private fun mseLoss(output: DoubleArray, target: DoubleArray): Double {
        return output.indices.sumOf { i -> (output[i] - target[i]).pow(2.0) } / output.size
    }

    private fun l2Regularization(): Double {
        var regularizationLoss = 0.0
        weightsInputHidden.forEach { layer ->
            layer.forEach { weight ->
                regularizationLoss += weight.pow(2.0)
            }
        }
        weightsHiddenOutput.forEach { layer ->
            layer.forEach { weight ->
                regularizationLoss += weight.pow(2.0)
            }
        }
        return regularizationLoss * regularizationLambda
    }

    fun train(
        inputs: List<FloatArray>,
        targets: List<DoubleArray>,
        validationRawInputs: List<FloatArray>,
        validationTargets: List<DoubleArray>,
        epochs: Int,
        batchSize: Int = 32,
        patience: Int = 10
    ) {
        var bestLoss = Double.MAX_VALUE
        var epochsWithoutImprovement = 0
        val preparedInputs = createNewFeature(inputs)
        val normalizedInputs = zScoreNormalization(preparedInputs)
        val preparedValidationInputs = createNewFeature(validationRawInputs)
        val normalizedValidationInputs = zScoreNormalization(preparedValidationInputs)

        var m = Array(weightsInputHidden.size) { DoubleArray(weightsInputHidden[0].size) { 0.0 } }
        var v = Array(weightsInputHidden.size) { DoubleArray(weightsInputHidden[0].size) { 0.0 } }
        var mOutput = Array(weightsHiddenOutput.size) { DoubleArray(weightsHiddenOutput[0].size) { 0.0 } }
        var vOutput = Array(weightsHiddenOutput.size) { DoubleArray(weightsHiddenOutput[0].size) { 0.0 } }
        var t = 1

        for (epoch in 1..epochs.toInt()) {
            var totalLoss = 0.0
            try {
                normalizedInputs.chunked(batchSize.toInt()).zip(targets.chunked(batchSize.toInt())).forEach { (batchInputs, batchTargets) ->
                    batchInputs.zip(batchTargets).forEach { (input, target) ->
                        val output = forwardPass(input).second
                        totalLoss += mseLoss(output, target) + l2Regularization()
                        backpropagation(input, target, m, v, mOutput, vOutput)
                    }
                }
                trainForBasalaimi(normalizedInputs, targets, epochs.toInt())
                trainWithAdam(normalizedInputs, targets, normalizedValidationInputs, validationTargets, epochs - epoch + 1)
                val averageLoss = totalLoss / normalizedInputs.size
                trainingLossHistory.add(averageLoss)

                val validationLoss = validate(normalizedValidationInputs, validationTargets)
                println("Epoch $epoch, Training Loss: $averageLoss, Validation Loss: $validationLoss")

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

                learningRate *= 0.95 // Réduction dynamique du taux d'apprentissage
                t++
            } catch (e: Exception) {
                lastTrainingException = e
                println("Exception during training at epoch $epoch: ${e.message}")
                break
            }
        }
    }
    private fun backpropagation(
        input: FloatArray,
        target: DoubleArray,
        m: Array<DoubleArray>,
        v: Array<DoubleArray>,
        mOutput: Array<DoubleArray>,
        vOutput: Array<DoubleArray>
    ) {
        val (hidden, output) = forwardPass(input)
        val gradLossOutput = DoubleArray(outputSize) { i -> 2.0 * (output[i] - target[i]) }
        val gradHiddenOutput = Array(hiddenSize) { DoubleArray(outputSize) { 0.0 } }
        val gradInputHidden = Array(inputSize) { DoubleArray(hiddenSize) { 0.0 } }

        // Mise à jour des poids de la couche de sortie avec Adam
        for (i in 0 until outputSize) {
            for (j in 0 until hiddenSize) {
                gradHiddenOutput[j][i] = gradLossOutput[i] * hidden[j]
            }
        }
        Adam(weightsHiddenOutput, gradHiddenOutput, mOutput, vOutput, 1, learningRate)

        // Mise à jour des biais de la couche de sortie
        for (i in 0 until outputSize) {
            biasOutput[i] -= learningRate * gradLossOutput[i]
        }

        // Calcul des gradients pour la couche cachée
        val gradLossHidden = DoubleArray(hiddenSize) { j ->
            var sum = 0.0
            for (k in 0 until outputSize) {
                sum += gradLossOutput[k] * weightsHiddenOutput[j][k]
            }
            if (hidden[j] <= 0) 0.0 else sum
        }

        // Mise à jour des poids de la couche cachée avec Adam
        for (i in 0 until inputSize) {
            for (j in 0 until hiddenSize) {
                val gradRelu = if (hidden[j] > 0) 1 else 0.001
                gradInputHidden[i][j] = gradLossHidden[j] * input[i] * gradRelu.toDouble()
            }
        }
        Adam(weightsInputHidden, gradInputHidden, m, v, 1, learningRate)

        // Mise à jour des biais de la couche cachée
        for (j in 0 until hiddenSize) {
            val gradRelu = if (hidden[j] > 0) 1 else 0.001
            biasHidden[j] -= learningRate * gradLossHidden[j] * gradRelu.toDouble()
        }

    }

    /*private fun backpropagation(input: FloatArray, target: DoubleArray) {
        val (hidden, output) = forwardPass(input)
        val gradLossOutput = DoubleArray(outputSize) { i -> 2.0 * (output[i] - target[i]) }

        for (i in 0 until outputSize) {
            for (j in 0 until hiddenSize) {
                weightsHiddenOutput[j][i] -= learningRate * (gradLossOutput[i] * hidden[j] + regularizationLambda * weightsHiddenOutput[j][i])
            }
            biasOutput[i] -= learningRate * gradLossOutput[i]
        }

        for (i in 0 until inputSize) {
            for (j in 0 until hiddenSize) {
                val gradRelu = if (hidden[j] > 0) 1 else 0.001
                var gradLossHidden = 0.0
                for (k in 0 until outputSize) {
                    gradLossHidden += gradLossOutput[k] * weightsHiddenOutput[j][k]
                }
                gradLossHidden *= gradRelu.toDouble()
                weightsInputHidden[i][j] -= learningRate * (gradLossHidden * input[i] + regularizationLambda * weightsInputHidden[i][j])
            }
        }

        for (j in 0 until hiddenSize) {
            val gradRelu = if (hidden[j] > 0) 1 else 0.001
            var gradLossHidden = 0.0
            for (k in 0 until outputSize) {
                gradLossHidden += gradLossOutput[k] * weightsHiddenOutput[j][k]
            }
            gradLossHidden *= gradRelu.toDouble()
            biasHidden[j] -= learningRate * gradLossHidden
        }
    }*/

    private fun Adam(
        params: Array<DoubleArray>,
        grads: Array<DoubleArray>,
        m: Array<DoubleArray>,
        v: Array<DoubleArray>,
        t: Int,
        learningRate: Double = 0.001,
        beta1: Double = 0.9,
        beta2: Double = 0.999,
        epsilon: Double = 1e-8
    ) {
        for (i in params.indices) {
            for (j in params[i].indices) {
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * grads[i][j]
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * grads[i][j] * grads[i][j]
                val mHat = m[i][j] / (1 - beta1.pow(t))
                val vHat = v[i][j] / (1 - beta2.pow(t))
                params[i][j] -= learningRate * mHat / (sqrt(vHat) + epsilon)
            }
        }
    }

    fun trainWithAdam(
        inputs: List<FloatArray>,
        targets: List<DoubleArray>,
        validationRawInputs: List<FloatArray>,
        validationTargets: List<DoubleArray>,
        epochs: Int,
        batchSize: Int = 32,
        patience: Int = 10
    ) {
        val preparedInputs = createNewFeature(inputs)
        val normalizedInputs = zScoreNormalization(preparedInputs)
        val preparedValidationInputs = createNewFeature(validationRawInputs)
        val normalizedValidationInputs = zScoreNormalization(preparedValidationInputs)

        var m = Array(weightsInputHidden.size) { DoubleArray(weightsInputHidden[0].size) { 0.0 } }
        var v = Array(weightsInputHidden.size) { DoubleArray(weightsInputHidden[0].size) { 0.0 } }
        var mOutput = Array(weightsHiddenOutput.size) { DoubleArray(weightsHiddenOutput[0].size) { 0.0 } }
        var vOutput = Array(weightsHiddenOutput.size) { DoubleArray(weightsHiddenOutput[0].size) { 0.0 } }
        var t = 1

        var bestLoss = Double.MAX_VALUE
        var epochsWithoutImprovement = 0

        for (epoch in 1..epochs.toInt()) {
            var totalLoss = 0.0
            try {
                normalizedInputs.chunked(batchSize).zip(targets.chunked(batchSize)).forEach { (batchInputs, batchTargets) ->
                    batchInputs.zip(batchTargets).forEach { (input, target) ->
                        val (_, output) = forwardPass(input)
                        val gradOutputs = DoubleArray(outputSize) { i -> 2.0 * (output[i] - target[i]) }
                        val gradInputHidden = backpropagationWithAdam(input, gradOutputs)

                        totalLoss += mseLoss(output, target) + l2Regularization()
                    }
                }

                val averageLoss = totalLoss / normalizedInputs.size
                trainingLossHistory.add(averageLoss)

                val validationLoss = validate(normalizedValidationInputs, validationTargets)
                println("Epoch $epoch, Training Loss: $averageLoss, Validation Loss: $validationLoss")

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

                learningRate *= 0.95 // Réduction dynamique du taux d'apprentissage
                t++
            } catch (e: Exception) {
                lastTrainingException = e
                println("Exception during training at epoch $epoch: ${e.message}")
                break
            }
        }
    }

    private fun backpropagationWithAdam(input: FloatArray, gradOutputs: DoubleArray): Array<DoubleArray> {
        val (hidden, _) = forwardPass(input)
        val gradInputHidden = Array(inputSize) { DoubleArray(hiddenSize) { 0.0 } }
        val gradHiddenOutput = Array(hiddenSize) { DoubleArray(outputSize) { 0.0 } }

        for (i in 0 until outputSize) {
            for (j in 0 until hiddenSize) {
                gradHiddenOutput[j][i] = gradOutputs[i] * hidden[j]
            }
        }

        for (i in 0 until inputSize) {
            for (j in 0 until hiddenSize) {
                val gradRelu = if (hidden[j] > 0) 1 else 0.001
                var gradLossHidden = 0.0
                for (k in 0 until outputSize) {
                    gradLossHidden += gradOutputs[k] * weightsHiddenOutput[j][k]
                }
                gradLossHidden *= gradRelu.toDouble()
                gradInputHidden[i][j] = gradLossHidden * input[i]
            }
        }

        return gradInputHidden
    }

    private fun validate(inputs: List<FloatArray>, targets: List<DoubleArray>): Double {
        var totalLoss = 0.0
        for ((input, target) in inputs.zip(targets)) {
            val output = forwardPass(input).second
            totalLoss += mseLoss(output, target) + l2Regularization()
        }
        return totalLoss / inputs.size
    }

}


/*
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

    // Création d'une nouvelle caractéristique
    fun createNewFeature(inputData: List<FloatArray>): List<FloatArray> {
        return inputData.map { originalFeatures ->
            val newFeature = originalFeatures[0] * originalFeatures[1] // Par exemple, produit de deux caractéristiques
            originalFeatures + newFeature // Ajoute la nouvelle caractéristique à la fin de l'array
        }
    }
    fun trainForBasalaimi(
        inputs: List<FloatArray>,
        basalaimiTargets: List<DoubleArray>,
        epochs: Int,
        learningRate: Double
    ) {
        for (epoch in 1..epochs) {
            var totalLoss = 0.0
            for ((input, target) in inputs.zip(basalaimiTargets)) {
                val (hidden, output) = forwardPassbasal(input)
                totalLoss += mseLoss(output, target)

                backpropagationbasal(input, target, learningRate)
            }
            val averageLoss = totalLoss / inputs.size
            println("Epoch $epoch: Training Loss = $averageLoss")
        }
    }
    private fun backpropagationbasal(input: FloatArray, target: DoubleArray, learningRate: Double) {
        // Passe en avant pour obtenir les activations
        val (hidden, output) = forwardPassbasal(input)

        // Gradient de l'erreur par rapport à la sortie
        val gradLossOutput = DoubleArray(outputSize) { i ->
            2.0 * (output[i] - target[i])
        }

        // Mise à jour des poids et biais de la couche de sortie
        for (i in 0 until outputSize) {
            for (j in 0 until hiddenSize) {
                weightsHiddenOutput[j][i] -= learningRate * gradLossOutput[i] * hidden[j]
            }
            biasOutput[i] -= learningRate * gradLossOutput[i]
        }

        // Gradient de l'erreur par rapport à la couche cachée
        val gradLossHidden = DoubleArray(hiddenSize) { 0.0 }
        for (i in 0 until hiddenSize) {
            for (j in 0 until outputSize) {
                gradLossHidden[i] += gradLossOutput[j] * weightsHiddenOutput[i][j]
            }
            if (hidden[i] <= 0) {
                gradLossHidden[i] = 0.0 // Application de la fonction dérivée de ReLU
            }
        }

        // Mise à jour des poids et biais de la couche d'entrée
        for (i in 0 until inputSize) {
            for (j in 0 until hiddenSize) {
                weightsInputHidden[i][j] -= learningRate * gradLossHidden[j] * input[i]
            }
        }
    }
    private fun forwardPassbasal(input: FloatArray): Pair<DoubleArray, DoubleArray> {
        // Calcul des activations pour la première couche cachée
        val hiddenLayer = DoubleArray(hiddenSize) { i ->
            var sum = 0.0
            for (j in input.indices) {
                sum += input[j] * weightsInputHidden[j][i]
            }
            relu(sum + biasHidden[i]) // Applique la fonction d'activation ReLU
        }

        // Calcul des activations pour la couche de sortie
        val outputLayer = DoubleArray(outputSize) { i ->
            var sum = 0.0
            for (j in hiddenLayer.indices) {
                sum += hiddenLayer[j] * weightsHiddenOutput[j][i]
            }
            sum + biasOutput[i] // Pas de fonction d'activation pour la couche de sortie dans cet exemple
        }

        return Pair(hiddenLayer, outputLayer)
    }

    // Normalisation Z-score
    private fun zScoreNormalization(inputData: List<FloatArray>): List<FloatArray> {
        if (inputData.isEmpty()) {
            // Retourner une liste vide ou gérer le cas d'une liste vide d'une manière appropriée
            return emptyList()
        }
        val means = FloatArray(inputData.first().size) { 0f }
        val stdDevs = FloatArray(inputData.first().size) { 0f }

        // Accumuler la somme et la somme des carrés pour chaque caractéristique
        inputData.forEach { features ->
            features.forEachIndexed { index, value ->
                means[index] += value
                stdDevs[index] += value * value
            }
        }

        // Calcul de la moyenne et de l'écart-type
        means.indices.forEach { i ->
            means[i] /= inputData.size.toFloat()
            stdDevs[i] = sqrt(stdDevs[i] / inputData.size - means[i] * means[i])
        }

        // Application de la normalisation Z-score
        return inputData.map { features ->
            FloatArray(features.size) { index ->
                if (stdDevs[index] != 0.0f) {
                    (features[index] - means[index]) / stdDevs[index]
                } else {
                    features[index] - means[index] // Si l'écart-type est 0, on soustrait simplement la moyenne
                }
            }
        }
    }



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

    private fun backpropagation(input: FloatArray, target: DoubleArray, learningRate: Double, regularizationLambda: Double) {
        val (hidden, output) = forwardPass(input)

        // Gradient de la perte par rapport à la sortie prédite
        val gradLossOutput = DoubleArray(outputSize) { i -> 2.0 * (output[i] - target[i]) }
        //val clipValue = 1.0 // This is an example value; adjust as necessary
        //gradLossOutput.clipInPlace(-clipValue, clipValue)

        // Mise à jour des poids et biais de la couche de sortie
        for (i in 0 until outputSize) {
            for (j in 0 until hiddenSize) {
                weightsHiddenOutput[j][i] -= learningRate * (gradLossOutput[i] * hidden[j] + regularizationLambda * weightsHiddenOutput[j][i])
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

                weightsInputHidden[i][j] -= learningRate * (gradLossHidden * input[i] + regularizationLambda * weightsInputHidden[i][j])
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
    fun train(
        inputs: List<FloatArray>, targets: List<DoubleArray>,
        validationRawInputs: List<FloatArray>, validationTargets: List<DoubleArray>,
        epochs: Double, initialLearningRate: Double,
        batchSize: Int = 32, // Taille du lot pour l'entraînement par lots
        patience: Int = 10, // pour early stopping
        regularizationLambda: Double = 0.01 // lambda pour régularisation L2
    ) {
        var learningRate = initialLearningRate
        var bestLoss = Double.MAX_VALUE
        var epochsWithoutImprovement = 0
        val preparedInputs = createNewFeature(inputs)
        val normalizedInputs = zScoreNormalization(preparedInputs)
        val preparedValidationInputs = createNewFeature(validationRawInputs)
        val normalizedValidationInputs = zScoreNormalization(preparedValidationInputs)


        for (epoch in 1..epochs.toInt()) {
            var totalLoss = 0.0
            try {
                normalizedInputs.chunked(batchSize).zip(targets.chunked(batchSize)).forEach { (batchInputs, batchTargets) ->
                    batchInputs.zip(batchTargets).forEach { (input, target) ->
                        val output = forwardPass(input).second
                        totalLoss += mseLoss(output, target) + l2Regularization(regularizationLambda)
                        backpropagation(input, target, learningRate, regularizationLambda)
                    }
                }

                val averageLoss = totalLoss / normalizedInputs.size
                trainingLossHistory.add(averageLoss)

                // Validation de l'Époque
                val validationLoss = validate(normalizedValidationInputs, validationTargets)
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

private fun mseLoss(output: DoubleArray, target: DoubleArray): Double {
    return output.zip(target).sumOf { (o, t) -> (o - t).pow(2) } / output.size
}



private operator fun Double.timesAssign(gradRelu: Number) {

}

fun refineSMB(smb: Float, neuralNetwork: aimiNeuralNetwork, input: FloatArray): Float {
    val prediction = neuralNetwork.predict(input)[0]
    return smb + prediction.toFloat()
}
fun refineBasalaimi(basalaimi: Float, neuralNetwork: aimiNeuralNetwork, input: FloatArray): Float {
    val prediction = neuralNetwork.predict(input)[0] // Utilise la prédiction du réseau entraîné
    return basalaimi + prediction.toFloat()
}*/
