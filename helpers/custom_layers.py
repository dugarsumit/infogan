import tensorflow as tf
import numpy as np

"""
https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
set axis = -1 while using for conv2d
name = "batch_norm"
"""
def conv_batch_norm(layer_params, **kwargs):
    layer_params.set_layer_params(**kwargs)
    layer_output = tf.layers.batch_normalization(
        inputs = layer_params.layer_input,
        momentum = layer_params.momentum,
        epsilon = layer_params.epsilon,
        name = layer_params.name,
        gamma_initializer = tf.random_normal_initializer(1., 0.02),
        beta_initializer = tf.constant_initializer(0.),
        training = layer_params.training,
        axis = -1
    )
    return layer_output

def fc_batch_norm(layer_params, **kwargs):
    layer_params.set_layer_params(**kwargs)
    layer_output = tf.layers.batch_normalization(
        inputs = layer_params.layer_input,
        momentum = layer_params.momentum,
        epsilon = layer_params.epsilon,
        name = layer_params.name,
        gamma_initializer = tf.random_normal_initializer(1., 0.02),
        beta_initializer = tf.constant_initializer(0.),
        training = layer_params.training,
        axis = -1
    )
    return layer_output


def custom_relu(layer_params, **kwargs):
    layer_params.set_layer_params(**kwargs)
    layer_output = tf.nn.relu(
        features = layer_params.layer_input,
        name = layer_params.name
    )
    return layer_output


def custom_leaky_relu(layer_params, **kwargs):
    layer_params.set_layer_params(**kwargs)
    with tf.name_scope(name = layer_params.name):
        assert layer_params.leakiness <= 1
        x = layer_params.layer_input
        layer_output = tf.maximum(x, layer_params.leakiness * x)
    return layer_output


def custom_tanh(layer_params, **kwargs):
    layer_params.set_layer_params(**kwargs)
    layer_output = tf.nn.tanh(
        x = layer_params.layer_input,
        name = layer_params.name
    )
    return layer_output


def custom_conv2d(layer_params, **kwargs):
    layer_params.set_layer_params(**kwargs)
    layer_output = tf.layers.conv2d(
        inputs = layer_params.layer_input,
        filters = layer_params.output_dim,
        kernel_size = (layer_params.kernel_h, layer_params.kernel_w),
        strides = layer_params.strides,
        padding = layer_params.padding,
        kernel_initializer = tf.truncated_normal_initializer(stddev = layer_params.stddev),
        name = layer_params.name,

    )
    return layer_output


'''
https://www.tensorflow.org/api_docs/python/tf/layers/conv2d_transpose
strides are defined along height and width
'''
def custom_deconv2d(layer_params, **kwargs):
    layer_params.set_layer_params(**kwargs)
    layer_output = tf.layers.conv2d_transpose(
        inputs = layer_params.layer_input,
        filters = layer_params.output_dim,
        kernel_size = (layer_params.kernel_h, layer_params.kernel_w),
        strides = layer_params.strides,
        padding = layer_params.padding,
        kernel_initializer = tf.truncated_normal_initializer(stddev = layer_params.stddev),
        name = layer_params.name
    )
    return layer_output


def custom_fully_connected(layer_params, **kwargs):
    layer_params.set_layer_params(**kwargs)
    shape = layer_params.layer_input.shape
    input_ = layer_params.layer_input
    if len(shape) == 4:
        input_ = tf.reshape(input_, tf.stack([tf.shape(input_)[0], np.prod(shape[1:])]))
        input_.set_shape([None, np.prod(shape[1:])])
        layer_params.layer_input = input_
        #print(layer_params.layer_input.shape)

    layer_output = tf.layers.dense(
        inputs = layer_params.layer_input,
        units = layer_params.num_neurons,
        kernel_initializer = tf.random_normal_initializer(stddev = layer_params.stddev),
        bias_initializer = tf.constant_initializer(0.0),
        name = layer_params.name
    )
    return layer_output

"""
https://stackoverflow.com/questions/36668542/flatten-batch-in-tensorflow
"""
def custom_flatten(layer_params, **kwargs):
    layer_params.set_layer_params(**kwargs)
    layer_output = tf.contrib.layers.flatten(
        inputs = layer_params.layer_input
    )
    return layer_output

"""
def conv_batch_norm1(layer_params, **kwargs):
    layer_params.set_layer_params(**kwargs)
    ema = tf.train.ExponentialMovingAverage(decay = 0.9)
    shape = layer_params.layer_input.get_shape().as_list()
    shp = shape[-1]
    with tf.variable_scope(tf.get_variable_scope(), reuse = False):
        gamma = tf.Variable(tf.random_normal([shp], stddev=0.02), name="gamma", dtype = tf.float32)
        beta = tf.Variable(tf.constant(shape = [shp], value = 0.0, dtype = np.float32), name="beta", dtype = tf.float32)
        mean, variance = tf.nn.moments(layer_params.layer_input, [0, 1, 2])
        # sigh...tf's shape system is so..
        mean.set_shape((shp,))
        variance.set_shape((shp,))
        ema_apply_op = ema.apply([mean, variance])
        if layer_params.training == True:
            with tf.control_dependencies([ema_apply_op]):
                normalized_x = tf.nn.batch_normalization(
                    x = layer_params.layer_input,
                    mean = mean,
                    variance = variance,
                    offset = beta,
                    scale = gamma,
                    variance_epsilon = layer_params.epsilon,
                    name = layer_params.name
                )
        else:
            normalized_x = tf.nn.batch_normalization(
                x = layer_params.layer_input,
                mean = ema.average(mean),
                variance = ema.average(variance),
                offset = beta,
                scale = gamma,
                variance_epsilon = layer_params.epsilon,
                name = layer_params.name
            )
    return normalized_x

def fc_batch_norm1(layer_params, **kwargs):
    layer_params.set_layer_params(**kwargs)
    ori_shape = layer_params.layer_input.get_shape().as_list()
    if ori_shape[0] is None:
        ori_shape[0] = -1
    new_shape = [ori_shape[0], 1, 1, ori_shape[1]]
    layer_params.layer_input = tf.reshape(layer_params.layer_input, new_shape)
    normalized_x = conv_batch_norm(layer_params)
    return tf.reshape(normalized_x, ori_shape)

"""