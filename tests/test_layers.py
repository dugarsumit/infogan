from infogan.helpers.mnist import Mnist
from infogan.helpers.layer_params import LayerParams
from infogan.helpers.custom_layers import *
import numpy as np
import time
from progressbar import ETA, Bar, Percentage, ProgressBar

mnist_data_path = "/home/sumit/Documents/repo/datasets/mnist"

layer_params = LayerParams()

def test_2layer_perceptron():
    batch_size = 100
    mnist = Mnist(data_path = mnist_data_path)
    x = tf.placeholder(tf.float32, [batch_size, 784], name = "x")
    y_ = tf.placeholder(tf.int64, [batch_size, ], name = "y_")

    # model definition
    layer_params.layer_input = custom_fully_connected(layer_params, layer_input = x, name = "hidden1", num_neurons = 100)
    layer_params.layer_input = fc_batch_norm(layer_params, name = "bn")
    layer_params.layer_input = custom_relu(layer_params)
    y = custom_fully_connected(layer_params, name = "hidden2", num_neurons = 10)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = y,
        labels = y_,
    )
    train_step = tf.train.AdamOptimizer(learning_rate = 0.001, name="adam").minimize(cross_entropy)

    sess = tf.InteractiveSession()
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    tf.global_variables_initializer().run()
    widgets = [Percentage(), Bar(marker = '='), ETA()]
    pbar = ProgressBar(maxval = 100, widgets = widgets).start()
    for i in range(100):
        pbar.update(i)
        time.sleep(0.2)
        batch_xs, batch_ys = mnist.train.next_random_batch(batch_size)
        sess.run(fetches = [train_step], feed_dict = {x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))
    sess.close()
    writer.close()

def test_flatten_layer():
    a = [[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]]
    print(np.shape(a))
    layer_params.layer_input = a
    layer_output = custom_flatten(layer_params, name="flatten")
    with tf.Session() as sess:
        out = sess.run(layer_output)
        print(out.shape)

def test_variable_scope():
    print("j")
    with tf.variable_scope("scope1"):
        # add a new variable to the graph
        var = tf.get_variable("variable1", [1])
        print(tf.get_variable_scope().name)
    # print the name of variable
    print(var.name)

if __name__ == "__main__":
    test_2layer_perceptron()
    #test_flatten_layer()
    #test_variable_scope()