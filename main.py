from numpy import exp, array, random, dot

# Sourced from :
# https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1


class NeuralNetwork():
    def __init__(self):
        # seed the random number generator, so it generates the same numbers every time program runs
        random.seed(1)
        # we model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3x1 matrix with values -1 to 1 and mean 0.
        self.synapticWeights = 2 * random.random((3,1)) - 1

    # The sigmoid function, which describes an S shaped curve.
    # we pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight
    def __sigmoidDerivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time
    def train(self, trainingSetInputs, trainingSetOutputs, numberOfIterations):
        for iteration in range(numberOfIterations):
            # Pass the training set through neural network
            output = self.think(trainingSetInputs)

            # calculate the error (the difference between the desired output and the predicted output)
            error = trainingSetOutputs - output

            # multiply the rror by the input and again by the gradient of the sigmoid curve
            # this means less confident weights are adjusted more
            # this means inputs, which are zero, do not cause changes to the weights
            adjustment = dot(trainingSetInputs.T, error * self.__sigmoidDerivative(output))

            # adjust the weights
            self.synapticWeights += adjustment

    # the neural network thinks
    def think(self, inputs):
        # pass inputs through neural network
        return self.__sigmoid(dot(inputs, self.synapticWeights))


if __name__ == "__main__":

    # initialise the neural network
    neuralNetwork = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neuralNetwork.synapticWeights)

    # The training set. WWe have 4 examples, each consisting of 3 input values and 1 output value
    trainingSetInput = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 0], [1, 1, 0]])
    trainingSetOutput = array([[0, 1, 1, 0, 0, 1]]).T

    # train the neural network using a training set
    # do it 10000 times and make small adjustments each time
    neuralNetwork.train(trainingSetInput, trainingSetOutput, 20000)

    print("New synaptic weights after training: ")
    print(neuralNetwork.synapticWeights)

    # Test the neural network with a new situation
    print("Considering new situation [1, 0, 0] -> ?: ")
    print(neuralNetwork.think(array([1, 0, 0])))



































