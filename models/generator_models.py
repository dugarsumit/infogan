import tensorflow as tf
from helpers.layer_params import LayerParams
from helpers.custom_layers import *

class Generator(object):
    def __init__(self, network_type, image_shape):
        self.network_type = network_type
        self.image_shape = image_shape
        self.layer_params = LayerParams()
        self.model = None

    def generate(self, z_var, training, reuse):
        with tf.variable_scope("g_net", reuse = reuse):
        #if self.model is None:
            if self.network_type == "mnist":
                self.model = self.mnist_generator_model(z_var, training)
            elif self.network_type == "svhn":
                self.model = self.svhn_generator_model(z_var, training)
            elif self.network_type == "omniglot":
                self.model = self.mnist_basic_generator_model(z_var, training)
            elif self.network_type == "celebA":
                self.model = self.celebA_generator_model(z_var, training)
            elif self.network_type == "cifar10":
                self.model = self.cifar10_generator_model(z_var, training)
            else:
                raise NotImplementedError
        return self.model

    def cifar10_generator_model(self, z_var, training):
        # input layer, image_size = 28, z_var = (128,74)
        image_size = self.image_shape[0]
        self.layer_params.layer_input = z_var
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l1",
                                                               num_neurons = int(image_size / 16 * image_size / 16 * 448))
        self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l1", training = training)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l1")
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = tf.reshape(self.layer_params.layer_input,
                                                   [-1, int(image_size / 16), int(image_size / 16), 448])

        self.layer_params.layer_input = custom_deconv2d(self.layer_params, name = "dconv_l2", kernel_h = 4,
                                                        kernel_w = 4,
                                                        output_dim = 256)
        self.layer_params.layer_input = conv_batch_norm(self.layer_params, name = "dconv_bn_l2", training = training)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l2")

        self.layer_params.layer_input = custom_deconv2d(self.layer_params, name = "dconv_l3", kernel_h = 4,
                                                        kernel_w = 4,
                                                        output_dim = 128)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l3")

        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = custom_deconv2d(self.layer_params, name="dconv_l4", kernel_h=4, kernel_w=4,
                                                        output_dim=64)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l4")
        print(self.layer_params.layer_input.get_shape())

        self.layer_params.layer_input = custom_deconv2d(self.layer_params, name = "dconv_l5", kernel_h = 4,
                                                        kernel_w = 4, output_dim = 3)
        self.layer_params.layer_input = custom_tanh(self.layer_params, name = "dconv_tanh_l5")
        print(self.layer_params.layer_input.get_shape())
        layer_output = custom_flatten(self.layer_params)
        print(layer_output.get_shape())
        return layer_output

    def celebA_generator_model(self, z_var, training):
        # input layer, image_size = 28, z_var = (128,74)
        image_size = self.image_shape[0]
        self.layer_params.layer_input = z_var
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l1",
                                                               num_neurons = int(image_size / 16 * image_size / 16 * 448))
        self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l1", training = training)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l1")
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = tf.reshape(self.layer_params.layer_input,
                                                   [-1, int(image_size / 16), int(image_size / 16), 448])

        self.layer_params.layer_input = custom_deconv2d(self.layer_params, name = "dconv_l2", kernel_h = 4,
                                                        kernel_w = 4,
                                                        output_dim = 256)
        self.layer_params.layer_input = conv_batch_norm(self.layer_params, name = "dconv_bn_l2", training = training)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l2")

        self.layer_params.layer_input = custom_deconv2d(self.layer_params, name = "dconv_l3", kernel_h = 4,
                                                        kernel_w = 4,
                                                        output_dim = 128)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l3")

        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = custom_deconv2d(self.layer_params, name="dconv_l4", kernel_h=4, kernel_w=4,
                                                        output_dim=64)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l4")
        print(self.layer_params.layer_input.get_shape())

        self.layer_params.layer_input = custom_deconv2d(self.layer_params, name = "dconv_l5", kernel_h = 4,
                                                        kernel_w = 4, output_dim = 3)
        self.layer_params.layer_input = custom_tanh(self.layer_params, name = "dconv_tanh_l5")
        print(self.layer_params.layer_input.get_shape())
        layer_output = custom_flatten(self.layer_params)
        print(layer_output.get_shape())
        return layer_output

    def omniglot_generator_model(self, z_var, training):
        # input layer, image_size = 28, z_var = (128,74)
        image_size = self.image_shape[0]
        self.layer_params.layer_input = z_var
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l1",
                                                               num_neurons = 1024)
        self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l1", training = training)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l1")
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l2",
                                                               num_neurons = int(image_size / 4 * image_size / 4 * 128)) #6272
        self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l2", training = training)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l2")
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = tf.reshape(self.layer_params.layer_input,
                                                   [-1, int(image_size / 4), int(image_size / 4), 128])
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = custom_deconv2d(self.layer_params, name="dconv_l3", kernel_h=4, kernel_w=4,
                                                        output_dim=64)
        self.layer_params.layer_input = conv_batch_norm(self.layer_params, name="dconv_bn_l3", training = training)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l3")
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = custom_deconv2d(self.layer_params, name = "dconv_l4", kernel_h = 4,
                                                        kernel_w = 4, output_dim = 1)
        print(self.layer_params.layer_input.get_shape())
        layer_output = custom_flatten(self.layer_params)
        print(layer_output.get_shape())
        return layer_output

    """
    def omniglot_generator_model(self, z_var, training):
        # input layer, image_size = 28, z_var = (128,74)
        image_size = self.image_shape[0]
        self.layer_params.layer_input = z_var
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l1",
                                                               num_neurons = 1024)
        self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l1", training = training)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l1")
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l2",
                                                               num_neurons = int(image_size / 5 * image_size / 5 * 64)) #6272
        self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l2", training = training)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l2")
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = tf.reshape(self.layer_params.layer_input,
                                                   [-1, int(image_size / 5), int(image_size / 5), 64])
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = custom_deconv2d(self.layer_params, name="dconv_l3", kernel_h=9, kernel_w=9,
                                                        output_dim=32, data_h=1, data_w=1, padding='VALID')
        self.layer_params.layer_input = conv_batch_norm(self.layer_params, name="dconv_bn_l3", training = training)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l3")
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = custom_deconv2d(self.layer_params, name = "dconv_l4", kernel_h = 9,
                                                        kernel_w = 9, output_dim = 1, data_h=1, data_w=1, padding='VALID')
        print(self.layer_params.layer_input.get_shape())
        layer_output = custom_flatten(self.layer_params)
        print(layer_output.get_shape())
        return layer_output
    """

    def svhn_generator_model(self, z_var, training):
        # input layer, image_size = 28, z_var = (128,74)
        image_size = self.image_shape[0]
        self.layer_params.layer_input = z_var
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l1",
                                                               num_neurons = int(image_size / 16 * image_size / 16 * 448))
        self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l1", training = training)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l1")
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = tf.reshape(self.layer_params.layer_input,
                                                   [-1, int(image_size / 16), int(image_size / 16), 448])

        self.layer_params.layer_input = custom_deconv2d(self.layer_params, name = "dconv_l2", kernel_h = 4,
                                                        kernel_w = 4,
                                                        output_dim = 256)
        self.layer_params.layer_input = conv_batch_norm(self.layer_params, name = "dconv_bn_l2", training = training)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l2")

        self.layer_params.layer_input = custom_deconv2d(self.layer_params, name = "dconv_l3", kernel_h = 4,
                                                        kernel_w = 4,
                                                        output_dim = 128)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l3")

        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = custom_deconv2d(self.layer_params, name="dconv_l4", kernel_h=4, kernel_w=4,
                                                        output_dim=64)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l4")
        print(self.layer_params.layer_input.get_shape())

        self.layer_params.layer_input = custom_deconv2d(self.layer_params, name = "dconv_l5", kernel_h = 4,
                                                        kernel_w = 4, output_dim = 3)
        self.layer_params.layer_input = custom_tanh(self.layer_params, name = "dconv_tanh_l5")
        print(self.layer_params.layer_input.get_shape())
        layer_output = custom_flatten(self.layer_params)
        print(layer_output.get_shape())
        return layer_output

    def mnist_generator_model(self, z_var, training):
        # input layer, image_size = 28, z_var = (128,74)
        image_size = self.image_shape[0]
        self.layer_params.layer_input = z_var
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l1",
                                                               num_neurons = 1024)
        self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l1", training = training)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l1")
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l2",
                                                               num_neurons = int(image_size / 4 * image_size / 4 * 128)) #6272
        self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l2", training = training)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l2")
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = tf.reshape(self.layer_params.layer_input,
                                                   [-1, int(image_size / 4), int(image_size / 4), 128])
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = custom_deconv2d(self.layer_params, name="dconv_l3", kernel_h=4, kernel_w=4,
                                                        output_dim=64)
        self.layer_params.layer_input = conv_batch_norm(self.layer_params, name="dconv_bn_l3", training = training)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l3")
        print(self.layer_params.layer_input.get_shape())
        self.layer_params.layer_input = custom_deconv2d(self.layer_params, name = "dconv_l4", kernel_h = 4,
                                                        kernel_w = 4, output_dim = 1)
        print(self.layer_params.layer_input.get_shape())
        layer_output = custom_flatten(self.layer_params)
        print(layer_output.get_shape())
        return layer_output

    def mnist_basic_generator_model(self, z_var, training):
        # input layer, image_size = 28, z_var = (128,74)
        image_size = self.image_shape[0]
        self.layer_params.layer_input = z_var
        self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l1", num_neurons = 128)
        self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l1", training = training)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l1")
        self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l2", num_neurons = 256)
        self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l2", training = training)
        self.layer_params.layer_input = custom_relu(self.layer_params, name = "relu_l2")
        layer_output = custom_fully_connected(self.layer_params, name = "fc_l3", num_neurons = image_size * image_size)
        print(layer_output.get_shape())
        return layer_output