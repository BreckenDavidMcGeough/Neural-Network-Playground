import numpy as np 

class HiddenLayer:
    def __init__(self, input_matrix, num_hidden_neurons, activation_function):
        self.input_matrix = input_matrix
        self.num_hidden_neurons = num_hidden_neurons
        self.activation_function = activation_function
        self.weights = np.random.rand(len(input_matrix[0]), num_hidden_neurons)

    def forwardPropagation(self, input_matrix):
        self.z = np.dot(input_matrix, self.weights)
        #a = self.sigmoid(self.z)
        self.a = self.activation_function(self.z)
        return self.a

class NeuralNetworkArchitecture:
    def __init__(self, X, y, num_hidden_layers, num_hidden_neurons, activation_function, epochs, learning_rate):
        self.X = X
        self.y = y
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_neurons = num_hidden_neurons
        self.activation_function = activation_function
        self.hidden_layers = self.generateHiddenArchitecture()
        self.gradients = []
        self.epochs = epochs
        self.learning_rate = learning_rate

    def generateHiddenArchitecture(self):
        print("Generating hidden layers...\n")
        hidden_layers = []
        layer_output = self.X
        prev_layer = "input_layer"
        for i in range(0, self.num_hidden_layers):
            HL = HiddenLayer(layer_output, self.num_hidden_neurons, self.activation_function)
            HL.forwardPropagation(layer_output)
            hidden_layers.append(HL)
            layer_output = HL.a
            prev_layer = "layer " + str(i+1)
        HL = HiddenLayer(layer_output, len(self.y[0]), self.activation_function)
        hidden_layers.append(HL)
        return hidden_layers

    def printWeights(self):
        for i in range(0, len(self.hidden_layers)):
            print("Weights for layer " + str(i+1) + ": \n" + str(self.hidden_layers[i].weights) + "\n")

    def sigmoidDerivative(self, z):
        return np.exp(-z) / ((1 + np.exp(-z))**2)

    def forwardPropagation(self):
        input_layer = self.X
        for i in range(0, len(self.hidden_layers)):
            self.hidden_layers[i].forwardPropagation(input_layer)
            input_layer = self.hidden_layers[i].a
        self.yhat = self.hidden_layers[len(self.hidden_layers)-1].a

    def regressionLoss(self):
        RL = (1/2) * (self.y - self.yhat)**2
        return RL

    def crossEntropyLoss(self):
        CEL = -(self.y * np.log(yhat) + (1 - self.y) * np.log(1 - yhat))
        return CEL

    def gradientDescent(self):
        HL_len = len(self.hidden_layers)-1
        #dCdyhat = self.hidden_layers[HL_len].a - self.y
        yhat = self.hidden_layers[HL_len].a
        dCdyhat = -(self.y - yhat)
        delta = np.multiply(dCdyhat, self.sigmoidDerivative(self.hidden_layers[HL_len].z))
        dCdWL = np.dot(self.hidden_layers[HL_len-1].a.T,delta)
        self.gradients.append(dCdWL)
        for i in range(HL_len-1,0,-1):
            helper = np.dot(delta,self.hidden_layers[i+1].weights.T)
            delta = np.multiply(helper, self.sigmoidDerivative(self.hidden_layers[i].z))
            dCdWL = np.dot(self.hidden_layers[i-1].a.T, delta)
            self.gradients.append(dCdWL)
        helper = np.dot(delta,self.hidden_layers[1].weights.T)
        delta = np.multiply(helper, self.sigmoidDerivative(self.hidden_layers[0].z))
        dCdWL = np.dot(self.X.T, delta)
        self.gradients.append(dCdWL)

    def backPropagation(self):
        for _ in range(0, self.epochs):
            self.forwardPropagation()
            self.gradientDescent()
            for i in range(0, len(self.hidden_layers)):
                self.hidden_layers[i].weights -= self.learning_rate * self.gradients[len(self.gradients)-i-1]
            self.gradients = []

    def predict(self, X):
        input_layer = X
        for i in range(0, len(self.hidden_layers)):
            self.hidden_layers[i].forwardPropagation(input_layer)
            input_layer = self.hidden_layers[i].a
        return self.hidden_layers[len(self.hidden_layers)-1].a

    def meanSquaredError(self,yhat):
        const = 1/len(self.y)
        MSE = const * sum((self.y - yhat)**2)
        return MSE


        






