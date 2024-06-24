import sys
import os

sys.path.append('../src')

from main import * 

def getArgs():
    return sys.argv[1:]


X = np.array([[3, 5], 
              [5, 1], 
              [10, 2],
              [0, 5],
              [0, 0]])
y = np.array([[.75], 
              [.83], 
              [.93],
              [.65],
              [.25]])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def RELU(z):
    pass

args = getArgs()

num_hidden_layers = args[1]
num_hidden_neurons = args[2]
epochs = args[3]
learning_rate = args[4]

NN = NeuralNetworkArchitecture(X, y, num_hidden_layers, num_hidden_neurons, sigmoid, epochs, learning_rate)
NN.train()
print(NN.yhat)
print("\n")

print("MSE: " + str(NN.meanSquaredError()) + "\n")

print(NN.predict(np.array([[8,3]])))
print("\n")
print(NN.predict(np.array([[1,1]])))
