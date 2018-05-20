"""
A neural network with l2 regularization using tensorflow to solve the XOR problem
"""

# imports
import tensorflow as tf


XOR_input = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_label = [[0], [1], [1], [0]]

# inputs
x = tf.placeholder("float", shape=[4, 2])
# labels
y = tf.placeholder("float", shape=[4, 1])


# hyper parameters
# number of nodes in each layer
in_nodes = 2
h1_nodes = 4
h2_nodes = 10
h3_nodes = 10
out_nodes = 1

# learning rate
learning_rate = 0.01

# number of epochs
num_epochs = 10000

# l2 regularization lambda
lambda_ = 0.01


# the neural network model
def neural_network_model():
    "weights and biases"

    h1_parameters = {"weights": tf.Variable(tf.random_normal([in_nodes, h1_nodes])),
                     "biases": tf.Variable(tf.zeros([h1_nodes]))}

    h2_parameters = {"weights": tf.Variable(tf.random_normal([h1_nodes, h2_nodes])),
                     "biases": tf.Variable(tf.zeros([h2_nodes]))}

    h3_parameters = {"weights": tf.Variable(tf.random_normal([h2_nodes, h3_nodes])),
                     "biases": tf.Variable(tf.zeros([h3_nodes]))}


    out_parameters = {"weights": tf.Variable(tf.random_normal([h2_nodes, out_nodes])),
                      "biases": tf.Variable(tf.zeros([out_nodes]))}

    backprop(x, y, h1_parameters, h2_parameters, h3_parameters, out_parameters)

# forwardpropagation
def forwardprop(x, h1_parameters, h2_parameters, h3_parameters, out_parameters):

    # input -> hidden1
    h1_z = tf.add(tf.matmul(x, h1_parameters["weights"]), h1_parameters["biases"])
    h1_y = tf.nn.leaky_relu(h1_z)

    # hidden1 -> hidden2
    h2_z = tf.add(tf.matmul(h1_y, h2_parameters["weights"]), h2_parameters["biases"])
    h2_y = tf.nn.leaky_relu(h2_z)

    # hidden2 -> hidden3
    h3_z = tf.add(tf.matmul(h2_y, h3_parameters["weights"]), h3_parameters["biases"])
    h3_y = tf.nn.leaky_relu(h3_z)

    # hidden2 -> output
    out_z = tf.add(tf.matmul(h3_y, out_parameters["weights"]), out_parameters["biases"])
    out_y = tf.nn.sigmoid(out_z)

    # return the prediction
    return out_y


# backpropagation
def backprop(x, y, h1_parameters, h2_parameters, h3_parameters, out_parameters):
    
    # feed the parameters of the network through forwardpropagation
    prediction = forwardprop(x, h1_parameters, h2_parameters, h3_parameters, out_parameters)

    # calculate the loss
    # cross entropy
    loss = tf.reduce_mean(0.5 * (prediction - y) ** 2)

    # apply l2 regularization
    regularizer = tf.nn.l2_loss(h2_parameters["weights"]) + tf.nn.l2_loss(out_parameters["weights"])
    loss = tf.reduce_mean(loss + lambda_ * regularizer)

    # minimize the loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


    # session
    with tf.Session() as sess:
        # initlialize all the variables
        sess.run(tf.global_variables_initializer())

        # training loop
        for epoch in range(num_epochs):
            
            # feed the prediction and the labels through the loss function and the optimizer
            sess.run([optimizer, loss], feed_dict={x: XOR_input, y: XOR_label})

        
        print(sess.run(forwardprop(x, h1_parameters, h2_parameters, h3_parameters, out_parameters), feed_dict={x: XOR_input}))




neural_network_model()
