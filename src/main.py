import numpy as np 

class HiddenLayer: #stores information about a hidden layer L
    def __init__(self, input_matrix, num_hidden_neurons, activation_function):
        self.input_matrix = input_matrix
        self.num_hidden_neurons = num_hidden_neurons
        self.activation_function = activation_function
        self.weights = np.random.rand(len(input_matrix[0]), num_hidden_neurons) #weights matrix is number of features by num_hidden_neurons
        self.bias = np.random.rand(1, num_hidden_neurons) #bias vector is 1 by num_hidden_neurons

    def forwardPropagation(self, input_matrix): #propagate inputs matrix through weights matrix, then pass through activation function and include bias term for layer L
        self.input_matrix = input_matrix
        self.z = np.dot(input_matrix, self.weights) #will be a matrix of size (num_samples, num_hidden_neurons)
        self.z += self.bias #include bias
        #a = self.sigmoid(self.z)
        self.a = self.activation_function(self.z) #pass each element through activation function
        return self.a

class NeuralNetworkArchitecture:
    def __init__(self, X, y, num_hidden_layers, num_hidden_neurons, activation_function, epochs, learning_rate):
        self.X = X #input matrix
        self.y = y #output matrix
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_neurons = num_hidden_neurons
        self.activation_function = activation_function
        self.hidden_layers = self.generateHiddenArchitecture()
        self.gradients = [] #stores gradients for each layer
        self.deltas = [] #stores deltas for each layer
        self.epochs = epochs #number of epochs to train
        self.learning_rate = learning_rate

    def generateHiddenArchitecture(self): #iteratively generate hidden layers, where a layer L has input that is the output of layer L-1
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

    def sigmoidDerivative(self, z): #derivative of sigmoid function
        return np.exp(-z) / ((1 + np.exp(-z))**2)

    def forwardPropagation(self, input_layer): #forward propagate through the entire network for prediction
        #input_layer = self.X
        for i in range(0, len(self.hidden_layers)):
            self.hidden_layers[i].forwardPropagation(input_layer)
            input_layer = self.hidden_layers[i].a
        self.yhat = self.hidden_layers[len(self.hidden_layers)-1].a

    def regressionLoss(self): #regression loss function
        RL = (1/2) * (self.y - self.yhat)**2
        return RL

    def crossEntropyLoss(self): #cross entropy loss function
        CEL = -(self.y * np.log(yhat) + (1 - self.y) * np.log(1 - yhat))
        return CEL

    def gradientDescent(self): #calculate gradients for a given layer L
        HL_len = len(self.hidden_layers)-1
        #dCdyhat = self.hidden_layers[HL_len].a - self.y

        #first, computer gradient for the last layer
        yhat = self.hidden_layers[HL_len].a #prediction
        dCdyhat = -(self.y - yhat) #derivative of loss function
        delta = np.multiply(dCdyhat, self.sigmoidDerivative(self.hidden_layers[HL_len].z)) 
        dCdWL = np.dot(self.hidden_layers[HL_len-1].a.T,delta) #a_{L-1}^T o (dC/dyhat * dsigma_prime/dz) = dC/dWL, where o is the dot product and * is the hadamard product
        self.gradients.append(dCdWL)
        self.deltas.append(delta)

        #computer gradients from rest of layers, going backwards through the network
        for i in range(HL_len-1,0,-1):
            helper = np.dot(delta,self.hidden_layers[i+1].weights.T)
            delta = np.multiply(helper, self.sigmoidDerivative(self.hidden_layers[i].z)) #delta_L = (delta_{L+1} o WL_{L+1}^T) * dsigma_prime/dz
            dCdWL = np.dot(self.hidden_layers[i-1].a.T, delta) #a_{L-1}^T o delta_L = dC/dWL, where o is the dot product and * is the hadamard product
            self.deltas.append(delta)
            self.gradients.append(dCdWL)

        #computer gradient for the first layer
        helper = np.dot(delta,self.hidden_layers[1].weights.T)
        delta = np.multiply(helper, self.sigmoidDerivative(self.hidden_layers[0].z)) #delta_L = (delta_{L+1} o WL_{L+1}^T) * dsigma_prime/dz
        dCdWL = np.dot(self.X.T, delta) #X^T o delta_L = dC/dWL, where o is the dot product and * is the hadamard product
        self.deltas.append(delta)
        self.gradients.append(dCdWL)

    def backPropagation(self): #back propagate through the network to update weights and biases based on gradient calculations at each layer
        for _ in range(0, self.epochs): #iterate through epochs
            self.forwardPropagation(self.X) 
            self.gradientDescent()
            for i in range(0, len(self.hidden_layers)): #update weights and biases at each layer with gradients
                self.hidden_layers[i].weights -= self.learning_rate * self.gradients[len(self.gradients)-i-1] #update weights: W_L = W_L - learning_rate * dC/dWL
                bias_gradient = np.sum(self.deltas[len(self.deltas)-i-1], axis=0) #sum up the deltas for each neuron in the layer (sum of the columns of the delta matrix) to align with bias vector
                self.hidden_layers[i].bias -= self.learning_rate * bias_gradient #update bias: b_L = b_L - learning_rate * dC/dbL
            self.gradients = [] #reset gradients for next epoch

    def train(self): #train the network
        self.backPropagation()

    def predict(self, input_matrix): #predict output for a given input matrix
        self.forwardPropagation(input_matrix)
        return self.yhat

    def meanSquaredError(self): #MSE to calculate loss
        const = 1/len(self.y)
        MSE = const * sum((self.y - self.yhat)**2)
        return MSE


        






