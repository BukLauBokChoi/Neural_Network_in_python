"""
A neural network from scratch using stochastic gradient descent to solve logic or
"""

import numpy as np

input_data = np.array([[[0, 0]],
                       [[0, 1]],
                       [[1, 0]],
                       [[1, 1]]])

output_data = np.array([[[0]],
                        [[1]],
                        [[1]],
                        [[1]]])

# size of the network
in_size = 2
out_size = 1

# intialize parameters
np.random.seed(0)
w1 = np.random.random((in_size, out_size))
b1 = np.random.random((out_size, 1))

# hyper parameters
learning_rate = 0.5
epochs = 750

def sigmoid(val, deriv=False):
    if deriv:
        return val * (1-val)
    return 1/(1+np.exp(-val))


def Train(X, Y, w1, b1, lr, epochs):
    for i in range(epochs):
        for j in range(len(Y)):
            
            # forward
            P = sigmoid(np.dot(X[j], w1) + b1)

            # backward
            # error
            Error = 0.5 * ((P - Y[j])**2)

            # stochastic gradient descent
            delta_w1 = np.dot(X[j].T, (P-Y[j]) * sigmoid(P, True).T)
            delta_b1 = (P-Y[j]) * sigmoid(P, True)

            w1 -= lr * delta_w1
            b1 -= lr * delta_b1

            # print the prediction after training
            if i == 749:
                print("Predicrion after training")
                print(P)


if __name__ == "__main__":

    Train(input_data, output_data, w1, b1, learning_rate, epochs)
