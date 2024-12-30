package app.aaps.plugins.aps.openAPSAIMI

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.math.tanh
import kotlin.random.Random

// class AimiNeuralNetwork(
//     private val inputSize: Int,
//     private val hiddenSize: Int,
//     private val outputSize: Int,
//     private val config: TrainingConfig = TrainingConfig(),
//     private val regularizationLambda: Double = 0.01,
//     private val dropoutRate: Double = 0.5
// ) {
//
//     //private var weightsInputHidden = Array(inputSize) { DoubleArray(hiddenSize) { Random.nextDouble(-sqrt(6.0 / (inputSize + hiddenSize)), sqrt(6.0 / (inputSize + hiddenSize))) } }
//     private var weightsInputHidden = Array(inputSize) {
//         DoubleArray(hiddenSize) { Random.nextDouble(-sqrt(2.0 / inputSize), sqrt(2.0 / inputSize)) }
//     }
//     private var bestLoss: Double = Double.MAX_VALUE
//     private var biasHidden = DoubleArray(hiddenSize) { 0.01 }
//     private var weightsHiddenOutput = Array(hiddenSize) { DoubleArray(outputSize) { Random.nextDouble(-sqrt(6.0 / (hiddenSize + outputSize)), sqrt(6.0 / (hiddenSize + outputSize))) } }
//     private var biasOutput = DoubleArray(outputSize) { 0.01 }
//
//     private var lastTrainingException: Exception? = null
//     private var trainingLossHistory: MutableList<Double> = mutableListOf()
//
//     companion object {
//
//         fun refineSMB(smb: Float, neuralNetwork: AimiNeuralNetwork, input: DoubleArray?): Float {
//             val floatInput = input?.map { it.toFloat() }?.toFloatArray() ?: return smb
//             val prediction = neuralNetwork.predict(floatInput)[0]
//             return smb + prediction.toFloat()
//         }
//
//     }
//     private fun activation(x: Double, type: String = "ReLU"): Double {
//         return when (type) {
//             "Sigmoid" -> 1.0 / (1.0 + exp(-x))
//             "Tanh" -> tanh(x)
//             "LeakyReLU" -> if (x > 0) x else 0.01 * x
//             else -> max(0.0, x) // ReLU par défaut
//         }
//     }
//
//
//     private fun applyDropout(layer: DoubleArray): DoubleArray {
//         return layer.map { if (Random.nextDouble() > dropoutRate) it else 0.0 }.toDoubleArray()
//     }
//
//     private fun relu(x: Double) = max(0.0, x)
//
//     private fun createNewFeature(inputData: List<FloatArray>): List<FloatArray> {
//         return inputData.map { originalFeatures ->
//             val newFeature = originalFeatures[0] * originalFeatures[1]
//             originalFeatures + newFeature
//         }
//     }
//
//     private fun zScoreNormalization(inputData: List<FloatArray>): List<FloatArray> {
//         if (inputData.isEmpty()) {
//             return emptyList()
//         }
//         val means = FloatArray(inputData.first().size) { 0f }
//         val stdDevs = FloatArray(inputData.first().size) { 0f }
//
//         inputData.forEach { features ->
//             features.forEachIndexed { index, value ->
//                 means[index] += value
//                 stdDevs[index] += value * value
//             }
//         }
//
//         means.indices.forEach { i ->
//             means[i] /= inputData.size.toFloat()
//             stdDevs[i] = sqrt(stdDevs[i] / inputData.size - means[i] * means[i])
//         }
//
//         return inputData.map { features ->
//             FloatArray(features.size) { index ->
//                 if (stdDevs[index] != 0.0f) {
//                     (features[index] - means[index]) / stdDevs[index]
//                 } else {
//                     features[index] - means[index]
//                 }
//             }
//         }
//     }
//     private fun applyDropout(layer: DoubleArray, dropoutRate: Double = 0.5): DoubleArray {
//         return layer.map { if (Random.nextDouble() > dropoutRate) it else 0.0 }.toDoubleArray()
//     }
//     private fun batchNormalization(layer: DoubleArray): DoubleArray {
//         val mean = layer.average()
//         val variance = layer.map { (it - mean).pow(2.0) }.average()
//         return layer.map { (it - mean) / sqrt(variance + 1e-8) }.toDoubleArray()
//     }
//
//
//
//     private fun forwardPass(input: FloatArray): Pair<DoubleArray, DoubleArray> {
//         val hidden = DoubleArray(hiddenSize) { i ->
//             var sum = 0.0
//             for (j in input.indices) {
//                 sum += input[j] * weightsInputHidden[j][i]
//             }
//             //relu(sum + biasHidden[i])
//             sum + biasHidden[i]
//         }
//         val activatedHidden = hidden.map { activation(it, "LeakyReLU") }.toDoubleArray()
//         val normalizedHidden = batchNormalization(activatedHidden)
//         val hiddenWithDropout = applyDropout(normalizedHidden)
//
//         //val hiddenWithDropout = applyDropout(hidden)
//
//         val hidden2 = DoubleArray(hiddenSize) { i ->
//             var sum = 0.0
//             for (j in hiddenWithDropout.indices) {
//                 sum += hiddenWithDropout[j] * weightsInputHidden[j][i]
//             }
//             relu(sum + biasHidden[i])
//         }
//
//         val output = DoubleArray(outputSize) { i ->
//             var sum = 0.0
//             for (j in hidden2.indices) {
//                 sum += hidden2[j] * weightsHiddenOutput[j][i]
//             }
//             sum + biasOutput[i]
//         }
//
//         return Pair(hidden2, output)
//     }
//
//     fun predict(input: FloatArray): DoubleArray {
//         return forwardPassSimplified(input).second
//     }
//
//     private fun mseLoss(output: DoubleArray, target: DoubleArray): Double {
//         return output.indices.sumOf { i -> (output[i] - target[i]).pow(2.0) } / output.size
//     }
//
//     private fun l2Regularization(): Double {
//         var regularizationLoss = 0.0
//         weightsInputHidden.forEach { layer ->
//             layer.forEach { weight ->
//                 regularizationLoss += weight.pow(2.0)
//             }
//         }
//         weightsHiddenOutput.forEach { layer ->
//             layer.forEach { weight ->
//                 regularizationLoss += weight.pow(2.0)
//             }
//         }
//         return regularizationLoss * regularizationLambda
//     }
//
//     private fun backpropagation(input: FloatArray, target: DoubleArray, m: Array<DoubleArray>, v: Array<DoubleArray>, mOutput: Array<DoubleArray>, vOutput: Array<DoubleArray>) {
//         val (hidden, output) = forwardPassSimplified(input)
//         val gradLossOutput = calculateOutputLayerGradient(output, target)
//         updateWeightsAndBiasesForOutputLayer(hidden, gradLossOutput, mOutput, vOutput)
//
//         val gradLossHidden = calculateHiddenLayerGradient(hidden, gradLossOutput)
//         updateWeightsAndBiasesForHiddenLayer(input, hidden, gradLossHidden, m, v)
//     }
//
//     private fun calculateOutputLayerGradient(output: DoubleArray, target: DoubleArray): DoubleArray {
//         return DoubleArray(outputSize) { i -> 2.0 * (output[i] - target[i]) }
//     }
//
//     private fun updateWeightsAndBiasesForOutputLayer(hidden: DoubleArray, gradLossOutput: DoubleArray, mOutput: Array<DoubleArray>, vOutput: Array<DoubleArray>) {
//         val gradHiddenOutput = calculateGradHiddenOutput(hidden, gradLossOutput)
//         adam(weightsHiddenOutput, gradHiddenOutput, mOutput, vOutput, 1, config.learningRate)
//         updateBiases(biasOutput, gradLossOutput)
//     }
//
//     private fun calculateHiddenLayerGradient(hidden: DoubleArray, gradLossOutput: DoubleArray): DoubleArray {
//         return DoubleArray(hiddenSize) { j ->
//             if (hidden[j] <= 0) 0.0
//             else gradLossOutput.indices.sumOf { k -> gradLossOutput[k] * weightsHiddenOutput[j][k] }
//         }
//     }
//
//     private fun updateWeightsAndBiasesForHiddenLayer(input: FloatArray, hidden: DoubleArray, gradLossHidden: DoubleArray, m: Array<DoubleArray>, v: Array<DoubleArray>) {
//         val gradInputHidden = calculateGradInputHidden(input, hidden, gradLossHidden)
//         adam(weightsInputHidden, gradInputHidden, m, v, 1, config.learningRate)
//         updateBiasesForHiddenLayer(hidden, gradLossHidden)
//     }
//
//     private fun calculateGradHiddenOutput(hidden: DoubleArray, gradLossOutput: DoubleArray): Array<DoubleArray> {
//         return Array(hiddenSize) { j ->
//             DoubleArray(outputSize) { i ->
//                 gradLossOutput[i] * hidden[j]
//             }
//         }
//     }
//
//     private fun calculateGradInputHidden(input: FloatArray, hidden: DoubleArray, gradLossHidden: DoubleArray): Array<DoubleArray> {
//         return Array(inputSize) { i ->
//             DoubleArray(hiddenSize) { j ->
//                 val gradRelu = if (hidden[j] > 0) 1.0 else 0.001
//                 gradLossHidden[j] * input[i] * gradRelu
//             }
//         }
//     }
//
//     private fun updateBiases(biases: DoubleArray, gradLoss: DoubleArray) {
//         for (i in biases.indices) {
//             biases[i] -= config.learningRate * gradLoss[i]
//         }
//     }
//
//     private fun updateBiasesForHiddenLayer(hidden: DoubleArray, gradLossHidden: DoubleArray) {
//         for (j in hidden.indices) {
//             val gradRelu = if (hidden[j] > 0) 1.0 else 0.001
//             biasHidden[j] -= config.learningRate * gradLossHidden[j] * gradRelu
//         }
//     }
//
//     private fun adam(
//         params: Array<DoubleArray>,
//         grads: Array<DoubleArray>,
//         m: Array<DoubleArray>,
//         v: Array<DoubleArray>,
//         t: Int,
//         learningRate: Double = 0.001,
//         beta1: Double = 0.9,
//         beta2: Double = 0.999,
//         epsilon: Double = 1e-8
//     ) {
//         for (i in params.indices) {
//             for (j in params[i].indices) {
//                 m[i][j] = beta1 * m[i][j] + (1 - beta1) * grads[i][j]
//                 v[i][j] = beta2 * v[i][j] + (1 - beta2) * grads[i][j] * grads[i][j]
//                 val mHat = m[i][j] / (1 - beta1.pow(t))
//                 val vHat = v[i][j] / (1 - beta2.pow(t))
//                 params[i][j] -= learningRate * mHat / (sqrt(vHat) + epsilon)
//             }
//         }
//     }
//
//     private fun adamSimplified(
//         weights: Array<DoubleArray>,
//         grads: Array<DoubleArray>,
//         m: Array<DoubleArray>,
//         v: Array<DoubleArray>,
//         t: Int,
//         learningRate: Double = 0.001,
//         beta1: Double = 0.9,
//         beta2: Double = 0.999,
//         epsilon: Double = 1e-8
//     ) {
//         for (i in weights.indices) {
//             for (j in weights[i].indices) {
//                 m[i][j] = beta1 * m[i][j] + (1 - beta1) * grads[i][j]
//                 v[i][j] = beta2 * v[i][j] + (1 - beta2) * grads[i][j] * grads[i][j]
//
//                 val mHat = m[i][j] / (1 - beta1.pow(t))
//                 val vHat = v[i][j] / (1 - beta2.pow(t))
//
//                 weights[i][j] -= learningRate * mHat / (sqrt(vHat) + epsilon)
//             }
//         }
//     }
//     private fun forwardPassSimplified(input: FloatArray): Pair<DoubleArray, DoubleArray> {
//         // Calcul d'une couche cachée simple
//         val hidden = DoubleArray(hiddenSize) { i ->
//             input.indices.sumOf { j -> input[j] * weightsInputHidden[j][i] } + biasHidden[i]
//         }
//
//         val activatedHidden = hidden.map { relu(it) }.toDoubleArray()
//
//         // Calcul de la sortie
//         val output = DoubleArray(outputSize) { i ->
//             activatedHidden.indices.sumOf { j -> activatedHidden[j] * weightsHiddenOutput[j][i] } + biasOutput[i]
//         }
//
//         return Pair(activatedHidden, output)
//     }
//     fun trainSimplified(
//         inputs: List<FloatArray>,
//         targets: List<DoubleArray>,
//         epochs: Int,
//         batchSize: Int = 32
//     ) {
//         trainWithAdamSimplified(inputs, targets, epochs, batchSize)
//     }
//     private fun trainWithAdamSimplified(
//         inputs: List<FloatArray>,
//         targets: List<DoubleArray>,
//         epochs: Int,
//         batchSize: Int = 32
//     ) {
//         if (inputs.isEmpty() || targets.isEmpty()) {
//             println("No training data available. Aborting.")
//             return
//         }
//
//         // Normalisation basique
//         val normalizedInputs = zScoreNormalization(inputs)
//         if (normalizedInputs.isEmpty()) {
//             println("Normalization resulted in empty data. Aborting.")
//             return
//         }
//
//         // Initialisation des paramètres Adam
//         val m = Array(weightsInputHidden.size) { DoubleArray(weightsInputHidden[0].size) { 0.0 } }
//         val v = Array(weightsInputHidden.size) { DoubleArray(weightsInputHidden[0].size) { 0.0 } }
//         var t = 0
//
//         for (epoch in 1..epochs) {
//             println("Starting epoch $epoch...")
//             var totalLoss = 0.0
//
//             try {
//                 // Processus par batch
//                 normalizedInputs.chunked(batchSize).zip(targets.chunked(batchSize)).forEach { (batchInputs, batchTargets) ->
//                     batchInputs.zip(batchTargets).forEach { (input, target) ->
//                         val (_, output) = forwardPassSimplified(input)
//                         val gradOutputs = DoubleArray(outputSize) { i -> 2.0 * (output[i] - target[i]) }
//
//                         // Calcul des gradients et mise à jour des poids
//                         val (gradInputHidden, _) = backpropagationWithAdam(input, gradOutputs)
//                         updateWeightsAdam(weightsInputHidden, gradInputHidden, m, v, ++t)
//
//                         totalLoss += mseLoss(output, target)
//                     }
//                 }
//
//                 val averageLoss = totalLoss / normalizedInputs.size
//                 println("Epoch $epoch complete. Training Loss: $averageLoss")
//             } catch (e: Exception) {
//                 println("Exception during epoch $epoch: ${e.message}")
//                 e.printStackTrace()
//                 return
//             }
//         }
//     }
//
//
//     private fun adjustLearningRate(epoch: Int, validationLoss: Double) {
//         if (validationLoss > bestLoss) {
//             config.learningRate *= 0.9 // Réduis si la validation n’améliore pas
//         }
//     }
//
//         private fun updateWeightsAdam(
//             weights: Array<DoubleArray>,
//             grads: Array<DoubleArray>,
//             m: Array<DoubleArray>,
//             v: Array<DoubleArray>, t: Int)
//         {
//             val beta1 = config.beta1
//             val beta2 = config.beta2
//             val epsilon = config.epsilon
//             val weightDecay = config.weightDecay
//
//             for (i in weights.indices) {
//                 for (j in weights[i].indices) {
//                     m[i][j] = beta1 * m[i][j] + (1 - beta1) * grads[i][j]
//                     v[i][j] = beta2 * v[i][j] + (1 - beta2) * grads[i][j] * grads[i][j]
//
//                     val mHat = m[i][j] / (1 - beta1.pow(t.toDouble()))
//                     val vHat = v[i][j] / (1 - beta2.pow(t.toDouble()))
//
//                     //weights[i][j] -= config.learningRate * mHat / (sqrt(vHat) + epsilon)
//                     weights[i][j] -= config.learningRate * mHat / (sqrt(vHat) + epsilon) + weightDecay * weights[i][j]
//                 }
//             }
//         }
//
//         private fun backpropagationWithAdam(input: FloatArray, gradOutputs: DoubleArray): Pair<Array<DoubleArray>, Array<DoubleArray>> {
//             val (hidden, _) = forwardPassSimplified(input)
//             val gradInputHidden = Array(inputSize) { DoubleArray(hiddenSize) { 0.0 } }
//             val gradHiddenOutput = Array(hiddenSize) { DoubleArray(outputSize) { 0.0 } }
//
//             // Calculate gradients for weights between the hidden layer and the output layer
//             for (i in 0 until outputSize) {
//                 for (j in 0 until hiddenSize) {
//                     gradHiddenOutput[j][i] = gradOutputs[i] * hidden[j]
//                 }
//             }
//
//             // Calculate gradients for weights between the input layer and the hidden layer
//             for (i in 0 until inputSize) {
//                 for (j in 0 until hiddenSize) {
//                     var gradLossHidden = 0.0
//                     for (k in 0 until outputSize) {
//                         gradLossHidden += gradOutputs[k] * weightsHiddenOutput[j][k]
//                     }
//                     // Correctly apply the ReLU derivative
//                     val reluDerivative = if (hidden[j] > 0) 1.0 else 0.001
//                     gradInputHidden[i][j] = gradLossHidden * reluDerivative * input[i]
//                 }
//             }
//
//             // Return both sets of gradients
//             return Pair(gradInputHidden, gradHiddenOutput)
//         }
//
//     private fun validate(inputs: List<FloatArray>, targets: List<DoubleArray>): Double {
//             var totalLoss = 0.0
//             for ((input, target) in inputs.zip(targets)) {
//                 val output = forwardPassSimplified(input).second
//                 totalLoss += mseLoss(output, target) + l2Regularization()
//             }
//             return totalLoss / inputs.size
//         }
//
//     }


class AimiNeuralNetwork(
    private val inputSize: Int,
    private val hiddenSize: Int,
    private val outputSize: Int,
    private val config: TrainingConfig = TrainingConfig(),   // Injection de votre config
    private val regularizationLambda: Double = 0.01          // L2 reg (optionnel)
) {

    //----------------------------------------------------------------------------------------------
    // 1) Poids & biais : input->hidden et hidden->output
    //----------------------------------------------------------------------------------------------
    private var weightsInputHidden = Array(inputSize) {
        DoubleArray(hiddenSize) {
            Random.nextDouble(-sqrt(2.0 / inputSize), sqrt(2.0 / inputSize))
        }
    }
    private var biasHidden = DoubleArray(hiddenSize) { 0.01 }

    private var weightsHiddenOutput = Array(hiddenSize) {
        DoubleArray(outputSize) {
            Random.nextDouble(-sqrt(2.0 / hiddenSize), sqrt(2.0 / hiddenSize))
        }
    }
    private var biasOutput = DoubleArray(outputSize) { 0.01 }

    // Historique pour debug ou tracking
    private val trainingLossHistory = mutableListOf<Double>()
    private var bestValLoss = Double.MAX_VALUE  // Pour l'early stopping

    //----------------------------------------------------------------------------------------------
    // 2) Fonctions d'activation & normalisation
    //----------------------------------------------------------------------------------------------

    private fun leakyRelu(x: Double, alpha: Double = config.leakyReluAlpha): Double {
        return if (x >= 0) x else alpha * x
    }

    private fun batchNormalization(values: DoubleArray): DoubleArray {
        val mean = values.average()
        val variance = values.map { (it - mean).pow(2.0) }.average()
        return values.map { (it - mean) / kotlin.math.sqrt(variance + 1e-8) }.toDoubleArray()
    }

    private fun applyDropout(values: DoubleArray, dropoutRate: Double): DoubleArray {
        return values.map { if (Random.nextDouble() < dropoutRate) 0.0 else it }.toDoubleArray()
    }

    //----------------------------------------------------------------------------------------------
    // 3) Forward pass
    //----------------------------------------------------------------------------------------------
    /**
     * @param inferenceMode = true pour la prédiction (pas de dropout)
     */
    private fun forwardPass(
        input: FloatArray,
        inferenceMode: Boolean = false
    ): Pair<DoubleArray, DoubleArray> {
        // Couche cachée
        val hiddenRaw = DoubleArray(hiddenSize) { h ->
            var sum = 0.0
            for (i in input.indices) {
                sum += input[i] * weightsInputHidden[i][h]
            }
            sum + biasHidden[h]
        }

        // Activation LeakyReLU
        val hiddenActivated = hiddenRaw.map { leakyRelu(it) }.toDoubleArray()

        // BatchNorm (optionnel, en mode entraînement uniquement)
        val hiddenNorm = if (!inferenceMode && config.useBatchNorm) {
            batchNormalization(hiddenActivated)
        } else {
            hiddenActivated
        }

        // Dropout (optionnel, en mode entraînement uniquement)
        val hiddenDropped = if (!inferenceMode && config.useDropout) {
            applyDropout(hiddenNorm, config.dropoutRate)
        } else {
            hiddenNorm
        }

        // Couche de sortie
        val output = DoubleArray(outputSize) { o ->
            var sum = 0.0
            for (h in hiddenDropped.indices) {
                sum += hiddenDropped[h] * weightsHiddenOutput[h][o]
            }
            sum + biasOutput[o]
        }

        return hiddenDropped to output
    }

    /**
     * Prévoir/predire la sortie (inférence, donc pas de dropout)
     */
    fun predict(input: FloatArray): DoubleArray {
        return forwardPass(input, inferenceMode = true).second
    }

    //----------------------------------------------------------------------------------------------
    // 4) Loss & régularisation
    //----------------------------------------------------------------------------------------------
    private fun mseLoss(output: DoubleArray, target: DoubleArray): Double {
        val sumSq = output.indices.sumOf { i -> (output[i] - target[i]).pow(2.0) }
        return sumSq / output.size
    }

    private fun l2Regularization(): Double {
        var reg = 0.0
        weightsInputHidden.forEach { row ->
            row.forEach { w -> reg += w.pow(2.0) }
        }
        weightsHiddenOutput.forEach { row ->
            row.forEach { w -> reg += w.pow(2.0) }
        }
        return reg * regularizationLambda
    }

    //----------------------------------------------------------------------------------------------
    // 5) Backpropagation + ADAM
    //----------------------------------------------------------------------------------------------
    /**
     * Calcule les gradients sur weightsInputHidden et weightsHiddenOutput
     */
    private fun backpropagation(input: FloatArray, target: DoubleArray): Pair<Array<DoubleArray>, Array<DoubleArray>> {
        val (hidden, output) = forwardPass(input, inferenceMode = false)

        // dL/dOut
        val gradOutput = DoubleArray(outputSize) { i -> 2.0 * (output[i] - target[i]) }

        // Grad sur la couche hidden->output
        val gradHiddenOutput = Array(hiddenSize) { h ->
            DoubleArray(outputSize) { o ->
                gradOutput[o] * hidden[h]
            }
        }

        // gradHidden : dérivé LeakyReLU
        val gradHidden = DoubleArray(hiddenSize) { h ->
            val sum = gradOutput.indices.sumOf { o -> gradOutput[o] * weightsHiddenOutput[h][o] }
            // LeakyRelu derivative
            if (hidden[h] >= 0) sum else sum * config.leakyReluAlpha
        }

        // Grad sur la couche input->hidden
        val gradInputHidden = Array(inputSize) { i ->
            DoubleArray(hiddenSize) { h ->
                gradHidden[h] * input[i]
            }
        }

        return gradInputHidden to gradHiddenOutput
    }

    // Stockage des moments Adam
    private val mInputHidden = Array(inputSize) { DoubleArray(hiddenSize) { 0.0 } }
    private val vInputHidden = Array(inputSize) { DoubleArray(hiddenSize) { 0.0 } }
    private val mHiddenOutput = Array(hiddenSize) { DoubleArray(outputSize) { 0.0 } }
    private val vHiddenOutput = Array(hiddenSize) { DoubleArray(outputSize) { 0.0 } }
    private var adamStep = 0

    private fun adamUpdate(
        weights: Array<DoubleArray>,
        grads: Array<DoubleArray>,
        m: Array<DoubleArray>,
        v: Array<DoubleArray>
    ) {
        adamStep++
        val beta1 = config.beta1
        val beta2 = config.beta2
        val eps = config.epsilon
        for (i in weights.indices) {
            for (j in weights[i].indices) {
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * grads[i][j]
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * grads[i][j] * grads[i][j]

                val mHat = m[i][j] / (1 - beta1.pow(adamStep.toDouble()))
                val vHat = v[i][j] / (1 - beta2.pow(adamStep.toDouble()))

                weights[i][j] -= config.learningRate * (mHat / (sqrt(vHat) + eps))

                // L2 weight decay (optionnel)
                weights[i][j] -= config.weightDecay * weights[i][j]
            }
        }
    }

    //----------------------------------------------------------------------------------------------
    // 6) Entraînement + early stopping
    //----------------------------------------------------------------------------------------------
    /**
     * Entraîne le réseau en mini-batch, avec validation pour early stopping
     */
    fun trainWithValidation(
        trainInputs: List<FloatArray>,
        trainTargets: List<DoubleArray>,
        valInputs: List<FloatArray>,
        valTargets: List<DoubleArray>
    ) {
        if (trainInputs.isEmpty()) {
            println("No training data - aborting.")
            return
        }

        // Reset de l'historique
        trainingLossHistory.clear()
        bestValLoss = Double.MAX_VALUE
        adamStep = 0

        val totalEpochs = if (config.epochs <= 0) 1000 else config.epochs
        val batchSize = if (config.batchSize <= 0) 32 else config.batchSize
        var epochsWithoutImprovement = 0

        for (epoch in 1..totalEpochs) {
            val indices = trainInputs.indices.shuffled()
            var totalLoss = 0.0

            // mini-batch
            indices.chunked(batchSize).forEach { batchIdx ->
                batchIdx.forEach { idx ->
                    val input = trainInputs[idx]
                    val target = trainTargets[idx]
                    val (gradIH, gradHO) = backpropagation(input, target)

                    // Update input->hidden
                    adamUpdate(weightsInputHidden, gradIH, mInputHidden, vInputHidden)
                    // Update hidden->output
                    adamUpdate(weightsHiddenOutput, gradHO, mHiddenOutput, vHiddenOutput)

                    // recalc pour la loss
                    val out = forwardPass(input, inferenceMode = false).second
                    totalLoss += mseLoss(out, target)
                }
            }

            val avgTrainLoss = totalLoss / trainInputs.size
            trainingLossHistory.add(avgTrainLoss)

            // Validation
            val valLoss = validate(valInputs, valTargets)
            println("Epoch $epoch/$totalEpochs - trainLoss=$avgTrainLoss - valLoss=$valLoss")

            if (valLoss < bestValLoss) {
                bestValLoss = valLoss
                epochsWithoutImprovement = 0
                // Optionnel : sauvegarder les poids
            } else {
                epochsWithoutImprovement++
                if (epochsWithoutImprovement >= config.patience) {
                    println("Early stopping at epoch $epoch (no improvement).")
                    break
                }
            }
        }
    }

    /**
     * Validation = calcule la perte moyenne + L2
     */
    fun validate(valInputs: List<FloatArray>, valTargets: List<DoubleArray>): Double {
        if (valInputs.isEmpty()) return 0.0

        var totalLoss = 0.0
        for (i in valInputs.indices) {
            val out = forwardPass(valInputs[i], inferenceMode = true).second
            totalLoss += mseLoss(out, valTargets[i])
        }
        totalLoss += l2Regularization()
        return totalLoss / valInputs.size
    }

    //----------------------------------------------------------------------------------------------
    // 7) Méthode statique "refineSMB"
    //----------------------------------------------------------------------------------------------
    companion object {
        /**
         * Appelable depuis votre code appelant :
         *   val adjusted = AimiNeuralNetwork.refineSMB(smb, nn, input)
         */
        fun refineSMB(smb: Float, nn: AimiNeuralNetwork, input: DoubleArray?): Float {
            if (input == null) return smb
            val floatInput = input.map { it.toFloat() }.toFloatArray()
            val prediction = nn.predict(floatInput)[0]
            return smb + prediction.toFloat()
        }
    }

    //----------------------------------------------------------------------------------------------
    // 8) Normalisation z-score (optionnelle)
    //----------------------------------------------------------------------------------------------
    fun zScoreNormalization(data: List<FloatArray>): List<FloatArray> {
        if (data.isEmpty()) return emptyList()
        val dim = data.first().size

        val means = DoubleArray(dim) { 0.0 }
        val stdDevs = DoubleArray(dim) { 0.0 }

        data.forEach { row ->
            for (i in row.indices) {
                means[i] += row[i]
                stdDevs[i] += (row[i] * row[i])
            }
        }

        for (i in means.indices) {
            means[i] /= data.size
            stdDevs[i] = kotlin.math.sqrt(stdDevs[i] / data.size - means[i].pow(2.0))
        }

        return data.map { row ->
            FloatArray(dim) { i ->
                if (stdDevs[i] != 0.0) {
                    ((row[i] - means[i]) / stdDevs[i]).toFloat()
                } else {
                    (row[i] - means[i]).toFloat()
                }
            }
        }
    }
}
