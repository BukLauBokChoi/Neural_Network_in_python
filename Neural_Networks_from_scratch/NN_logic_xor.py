"""
This is a three layer neural network that will predict the xor value of an input, 
using gradient descent and backpropagation
"""

import numpy as np

class NeuralNetwork:


    # relu activation function for hidden layer
    # derivative of relu at a specific point
    def relu(self, val, deriv=False):
        if deriv:
            val[val <= 0] = 0
            val[val > 0] = 1
            return val
        return np.maximum(0, val)


    # sigmoid actiavation function for output layer
    # derivative of sigmoid at a specific point
    def sigmoid(self, val, deriv=False):
        if deriv:
            return val * (1-val)
        return 1/(1+np.exp(-val))

    # start taining
    def Train(self, X, Y, w_ih, b_h, w_ho, b_o, lr, epochs):
        # training loop
        for i in range(epochs):

            # forward propagte
            # weighted sum of the hidden layer
            Z_h = np.dot(X, w_ih) + b_h

            # prediction of the hidden layer
            P_h = self.sigmoid(Z_h, False)

            # weighted sum of output layer
            Z_o = np.dot(P_h, w_ho) + b_o

            # prediction of the outputlayer
            P_o = self.sigmoid(Z_o, False)

            # backpropagation
            # error for prediction
            # MSE - mean squared error
            E_o = (P_o - Y)

            # partial derivative of error with respect to the parameters
            dE_dPo = (P_o - Y) # 4, 1
            dPo_dZo = self.sigmoid(P_o, True) # 4, 1
            dZo_dwho = P_h # 4, 2
            dZo_dPh = w_ho # 2, 1
            dPh_dZh = self.sigmoid(P_h, True) # 4, 2
            dZh_dwih = X # 4, 2

            delta_ho = dE_dPo * dPo_dZo

            # update weights and biases from hidden to output
            delta_w_ho = lr * np.dot(dZo_dwho.T, delta_ho)
            delta_b_o = sum(lr * delta_ho)
            #print(delta_w_ho)

            w_ho -= delta_w_ho
            b_o -= delta_b_o

            # update weights and biases from input to hidden
            delta_w_ih = lr * np.dot((dPh_dZh * dZh_dwih).T, np.dot(delta_ho, dZo_dPh.T))
            delta_b_h = sum(lr * np.dot(dPh_dZh.T, np.dot(delta_ho, dZo_dPh.T)))
            #print(delta_w_ih)

            w_ih -= delta_w_ih
            b_h -= delta_b_h

        print("prediction")
        print(P_o)
        print("MSE for output layer")
        print(E_o)
        print(P_h)



if __name__ == "__main__":

    nn = NeuralNetwork()

    input_data = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])

    output_data = np.array([[0],
                            [1],
                            [1],
                            [0]])

    # number of neurons in each layer
    n_i = 2
    n_h = 2
    n_o = 1

    # initialize weights and biases
    # seed the numbers
    np.random.seed(0)
    # weights and biases for hidden layer
    w_ih = np.random.random((n_i, n_h))
    b_h = np.random.random((n_h))
    # weights and biases for output layer
    w_ho = np.random.random((n_h, n_o))
    b_o = np.random.random(n_o)

    # hyper parameters
    learning_rate = 0.1
    epochs = 1000


    nn.Train(input_data, output_data, w_ih, b_h, w_ho, b_o, learning_rate, epochs)