"""
    This is a 2 layer neural network 
    with one input layer and one output layer.
"""
# import our dependencies
import numpy as np


# training inputs
X = np.array([[0, 1, 0],
              [0, 0, 1], 
              [1, 1, 1], 
              [1, 0, 0]])

# training ouputs
# the T stands for transpose which means that it will basically rotate the array by 90 degrees so the output corresponds to the correct input
Y = np.array([[0, 0, 1, 1]]).T

# sigmoid activation function
# it will output a number between 0 and 1
def sigmoid(val, deriv=False):
    if(deriv == True):
        return val * (1 - val)
    else:
        return 1 / (1 + np.exp(-val))
# seed the numbers so it is the same numbers every time you run the program
np.random.seed(1)

# model a neuron with 3 inputs and 1 output and initialize weights to a 3 x 1 matrix with a mean of 0
syn0 = 2 * np.random.random((3, 1)) -1

# start training our network
# trains it 10000 times
for train in range(100000):
    
    # forward propagate
    # the first layer is just our data so we assign it to the inputs
    l0 = X
    # now the network will try to predict the output
    # it will multiply the matrices from the training data with our weights from syn0
    # then it will pass through the sigmoid activation function
    l1 = sigmoid(np.dot(l0, syn0))

    # calculate the error of the prediction from the output layer i.e l1
    # desired output - our predicted output
    l1_error = Y - l1

    # multiply the error (4, 1)matrix with the slope (4, 1)matrix element by element
    # we will heavily adjust the weights of the less confident predictions and adjust the weights of the very confident predictions very slightly
    l1_adjustments = l1_error * sigmoid(l1, True)

    # adjust the weights
    syn0 += np.dot(l0.T, l1_adjustments)

print("Output after the training")
print(l1)



