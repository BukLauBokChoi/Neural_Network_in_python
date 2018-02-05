import numpy as np

class NeuralNetwork:

    # initialize random weights and biases
    def __init__(self):
        np.random.seed()
        self.w1 = np.random.random((2, 1))
        self.b1 = np.random.random((4, 1))

    # sigmoid activation function
    def sigmoid(self, val):
        return 1/(1+np.exp(-val))

    # derivative of sigmoid at a specific point
    def deriv_sigmoid(self, val):
        return self.sigmoid(val) * (1-self.sigmoid(val))

    # start training
    def Train(self, X, Y, lr, epochs):
        # training loop
        for i in range(epochs):

            # forward propagate
            # weighted sum
            Z = np.dot(X, self.w1) + self.b1

            # prediction
            P = self.sigmoid(Z)

            # back propagate
            # calculate the cost using SSE- sum of squared errors
            cost = (P - Y) ** 2

            # calculate the partial derivatives with respect to each parameter
            dcost_w1 = np.dot(X.T, 2*(P-Y) * self.deriv_sigmoid(P))
            dcost_b1 = 2*(P-Y) * self.deriv_sigmoid(P)

            # How much we should update the parameters
            w1_update = dcost_w1 * lr
            b1_update = dcost_b1 * lr

            # update weights and biases
            self.w1 -= w1_update
            self.b1 -= b1_update

        print("Given the inputs:")
        print(in_data)
        print("The network predicted the outputs:")
        print(P)
        print("And the correct outputs were:")
        print(out_data)

if __name__ == "__main__":

    # training data
    in_data = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

    out_data = np.array([[0],
                         [0],
                         [0],
                         [1]])

    # hyper parameters
    learning_rate = 0.03
    epochs = 10000

    nn = NeuralNetwork()

    nn.Train(in_data, out_data, learning_rate, epochs)


