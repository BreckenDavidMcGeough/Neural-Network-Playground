# Neural-Network-Playground
## An implementation of an artificial neural network constructor with user-defined architecture from scratch in Python

The user can see how different neural network architectures do on a training dataset. The user could specify the number of hidden layers in their neural network, the number of neurons per hidden layer, and the input and output feature of the dataset. They are also able to input an activation function of their choice (ReLU, sigmoid, softmax, etc.) to be used in forward propagation numeric linear algebra calculations, and are able to specify which loss function they want to utilize (cross entropy for classification, mean squared error for regression) in gradient descent calculations during backpropagation. The user is also able to specify hyperparameters such as the number of epochs during training and the learning rate. All of this allows the user to inspect how different neural network architectures and hyperparamters can influence the learning process and final result of the neural network. 

## A technical breakdown

The code works by generating a specific number of hidden layers (specified by the user), where each layer has an associated weights matrix (who's size is determined by the number of neurons per hidden layer specified by the use). Each layer receives an input matrix from the previous layer (initial is just the training dataset), and propagates it through its layer of weights before applying an activation function to it (the activation function is specified by the user). The output is a new matrix with different dimensions. For the specific linear algebra involved in these matrix calculations, please check out this link: 

* [forward propagation theory](https://www.d2l.ai/chapter_multilayer-perceptrons/backprop.html)

During training, the data is propagated through the network architecture, and then the weights at each layer are adjusted during the process of backpropagation through gradient descent. Starting from the output layer and ending at the input layer, the gradient of the loss function with respect to the weights matrix at a layer L is calculated recursively, as the calculation involves taking the dot product and hadamard product of each previous layer's data from L because of the chain rule. The weights at each layer are then adjusted by the gradient, which is scaled by the learning rate (specified by the user) to control how drastic the weights change. The process of gradient descent is basically analagous to nudging each weight slightly up or down, in a direction that tries to minimize the error term between the prediction and actual output data. This process occurs n times, where n is the number of epochs specified by the user. By the end of this, the weights should be adjusted enough so that propagating the training data through the network with the updated weights yields a prediction that closely matches the training data output. For the specific linear algebra calculations involved to calculate the gradient of the loss function (specified by the user) with respect to a particular layer's weights (and for a derivation of the equations involving the chain rule), please check out this link:

* [gradient descent and backpropagation theory](https://en.wikipedia.org/wiki/Backpropagation#Matrix_multiplication)

* [backpropagation and gradient descent supplemental lecture notes](https://cs231n.stanford.edu/slides/2018/cs231n_2018_ds02.pdf)

Here are some links that give more information on hyperparameters and different activation functions and how they affect the training process:

* [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))

* [activation functions](https://en.wikipedia.org/wiki/Activation_function#Table_of_activation_functions)

* [loss functions](https://towardsdatascience.com/loss-functions-and-their-use-in-neural-networks-a470e703f1e9)

## How to use it for yourself 

I am currently working on a UI in the terminal that will allow the user to input specifics and get back the accuracy and layout of their architecture. For now, just adjust the activation/loss functions (check out the activation and loss functions link for the specific mathematical functions) and change the hyperparamters values.

First, clone the repo. Then, navigate to the 'tests' directory and run the 'test1.py' script. Within the 'test1.py' script, you can modify the training data and the values of the number of hidden layers, number of neurons per hidden layer, number of training epochs, learning rate, and can swap out or add activation functions to fully customize your ANN architecture. Just save and run again.

'test2.py' will receive inputs from the command line in the form of [hidden_layers,hidden_neurons,epochs,learning_rate] when you run the script, and will construct your ANN.

## Results and metrics

Utilizing metrics such as the mean squared error, I was able to construct an architecture and tune hyperparameters sufficiently to get an extremely close prediction matrix to the actual output training data in 'test1.py'. I tested the trained model on some novel inputs, and received predictions that I would deem accurate and solid, taking into account the training data. 

* [MSE](https://en.wikipedia.org/wiki/Mean_squared_error#Predictor)
