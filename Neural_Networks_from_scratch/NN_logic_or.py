import numpy as np

class NeuralNetwork:

    # sigmoid activation function
    def sigmoid(self, val):
        return 1/(1+np.exp(-val))

    # derivative of sigmoid at a specific point
    def deriv_sigmoid(self, val):
        return self.sigmoid(val) * (1-self.sigmoid(val))

    # start training
    def Train(self, X, Y, w1, b1, lr, epochs):
        # training loop
        for i in range(epochs):

            # forward propagate
            # weighted sum
            Z = np.dot(X, w1) + b1

            # prediction
            P = self.sigmoid(Z)

            # back propagate
            # calculate the cost using SSE- sum of squared errors
            cost = (P - Y) ** 2

            # calculate the partial derivatives with respect to each parameter
            dcost_w1 = np.dot(X.T, 2*(P-Y) * self.deriv_sigmoid(P))
            dcost_b1 = sum(2*(P-Y) * self.deriv_sigmoid(P))

            # How much we should update the parameters
            w1_update = dcost_w1 * lr
            b1_update = dcost_b1 * lr

            # update weights and biases
            w1 -= w1_update
            b1 -= b1_update

        print("Given the inputs:")
        print(in_data)
        print("The network predicted the outputs:")
        print(P)
        print("And the correct outputs were:")
        print(out_data)
        print(w1)
        print(b1)


if __name__ == "__main__":

    # training data
    in_data = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

    out_data = np.array([[0],
                         [1],
                         [1],
                         [1]])


    # number of neurons in each layer
    n_inputs = 2
    n_outputs = 1
    # weights and biases
    np.random.seed()
    w1 = np.random.random((n_inputs, n_outputs))
    b1 = np.random.random((n_outputs, n_outputs))

    # hyper parameters
    learning_rate = 0.1
    epochs = 10000

    nn = NeuralNetwork()

    nn.Train(in_data, out_data, w1, b1, learning_rate, epochs)


