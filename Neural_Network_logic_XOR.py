"""
A Neural Netowrk created to solve and, or, and xor logic gate using tensorflow
"""

# import dependencies
import tensorflow as tf
import tensorflow.contrib.eager as tfe


# logic gates inputs and labels
OR_input = [[0, 0], [0, 1], [1, 0], [1, 1]]
OR_label = [[0], [1], [1], [1]]

AND_input = [[0, 0], [0, 1], [1, 0], [1, 1]]
AND_label = [[1], [0], [0], [1]]

XOR_input = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_label = [[0], [1], [1], [0]]


# input
x = tf.placeholder("float", shape=[4, 2], name="input")
# labels
y = tf.placeholder("float", shape=[4, 1], name="label")

# parameters of the network
# number of nodes in each layer
l1_nodes = 2 # input layer
l2_nodes = 2 # hidden layer
l3_nodes = 1 # output layer

# weights and biases
l2_w = tf.Variable(tf.random_uniform([l1_nodes, l2_nodes], -1, 1), name="l2_weights")
l3_w = tf.Variable(tf.random_uniform([l2_nodes, l3_nodes], -1, 1), name="l3_weights")

l2_b = tf.Variable(tf.zeros([l2_nodes]), name="l2_bias")
l3_b = tf.Variable(tf.zeros([l3_nodes]), name="l3_bias")

# hyperparameters
learning_rate = 0.1
num_epochs = 10000


# forwardpropagation
# input -> hidden
l2_y = tf.nn.relu(tf.matmul(x, l2_w) + l2_b)

# hidden -> output
l3_y = tf.nn.sigmoid(tf.matmul(l2_y, l3_w) + l3_b)


# backpropagation
# loss function
loss = tf.reduce_mean(((y * tf.log(l3_y)) + 
		             ((1 - y) * tf.log(1.0 - l3_y))) * -1)

# optimization
optimize = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)


# training
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # initialize all the variables
    sess.run(init)

    for epochs in range(num_epochs):
        sess.run([loss, optimize], feed_dict={x: XOR_input, y: XOR_label})

    print("prediction after training: {}".format(sess.run(l3_y, feed_dict={x: XOR_input, y:XOR_label})))

