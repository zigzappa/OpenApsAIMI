package app.aaps.plugins.aps.openAPSAIMI
import kotlin.math.max
class aimiNeuralNetwork (private val inputSize: Int, private val hiddenSize: Int, private val outputSize: Int) {
    private val weightsInputHidden = Array(inputSize) { DoubleArray(hiddenSize) { Math.random() } }
    private val biasHidden = DoubleArray(hiddenSize) { Math.random() }
    private val weightsHiddenOutput = DoubleArray(hiddenSize) { Math.random() }
    private val biasOutput = DoubleArray(outputSize) { Math.random() }

    private fun relu(x: Double) = max(0.0, x)

    private fun forwardPass(input: FloatArray): DoubleArray {
        // Calcul de la couche cachée
        val hidden = DoubleArray(hiddenSize) { i ->
            var sum = 0.0
            for (j in input.indices) {
                sum += input[j] * weightsInputHidden[j][i]
            }
            relu(sum + biasHidden[i])
        }

        // Calcul de la couche de sortie
        return DoubleArray(outputSize) { i ->
            var sum = 0.0
            for (j in hidden.indices) {
                sum += hidden[j] * weightsHiddenOutput[j]
            }
            sum + biasOutput[i] // Utilisez une fonction d'activation appropriée ici pour la régression ou la classification
        }
    }

    fun predict(input: FloatArray): DoubleArray {
        return forwardPass(input)
    }

}
fun refineSMB(smb: Float, neuralNetwork: aimiNeuralNetwork, input: FloatArray): Float {
    // Obtient la prédiction du réseau de neurones
    val prediction = neuralNetwork.predict(input)

    // Exemple : ajustement additif
    return smb + prediction[0].toFloat()
}