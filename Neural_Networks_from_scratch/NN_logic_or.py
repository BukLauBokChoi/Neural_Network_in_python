import numpy as np

class NeuralNetwork:

    # initialize weights and biases
    def __init__(self):
        np.random.seed(0)
        self.weight = np.random.random((2, 1))
        self.bias = np.random.random((4, 1))

    # sigmoid activation function
    def sigmoid(self, val):
        return 1/(1+np.exp(-val))

    # derivative of sigmoid at a specific point
    def deriv_sigmoid(self, val):
        return self.sigmoid(val) * (1-self.sigmoid(val))

    # start training
    # input data, output data, learning rate, epochs
    def Train(self, X, Y, lr, epochs):
        
        # training loop
        for i in range(epochs):
            
            # forward propagate
            # weighted sum plus the bias
            Z = np.dot(X, self.weight) + self.bias
            # prediction
            P = self.sigmoid(Z)
            

            # back propagate
            # SSE sum of squared error
            cost = (P - Y) ** 2

            # partial derivatives of the cost with respect to the parameters
            dcost_dw = np.dot(X.T, 2*(P-Y) * self.deriv_sigmoid(P))
            dcost_db = 2*(P-Y) * self.deriv_sigmoid(P)

            # how much we will change the parameters
            w_adj = dcost_dw * lr
            bias_adj = dcost_db * lr

            # update weights and biases
            self.weight -= w_adj
            self.bias -= bias_adj

        print("weights before training:")
        print(self.weight)
        print("biases before training:")
        print(self.bias)
        
        print("Given the inputs:")
        print(in_data)
        print("The network predicted the outputs:")
        print(P)
        print("And the correct outputs are:")
        print(out_data)

        print("new weights")
        print(self.weight)
        print("new biases")
        print(self.bias)


if __name__ == "__main__":

    # input data
    in_data = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

    # output data
    out_data = np.array([[0],
                         [1],
                         [1],
                         [1]])

    # hyper parameters
    learning_rate = 0.04
    epochs = 10000

    nn = NeuralNetwork()

    print(nn.Train(in_data, out_data, learning_rate, epochs))
