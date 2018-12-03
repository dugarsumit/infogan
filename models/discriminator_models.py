import tensorflow as tf
from ..helpers.layer_params import LayerParams
from ..helpers.custom_layers import *


class Discriminator(object):
    def __init__(self, network_type, image_shape):
        self.network_type = network_type
        self.image_shape = image_shape
        self.layer_params = LayerParams()
        self.model = None
        self.encoder_model = None
        self.shared_model = None

    def discriminate(self, x_var, reuse):
        #with tf.variable_scope(name_or_scope = scope, reuse = reuse):
        #if self.model is None:
        if self.network_type == "mnist":
            self.model = self.mnist_discriminator_model(x_var, reuse)
        elif self.network_type == "svhn":
            self.model = self.svhn_discriminator_model(x_var, reuse)
        elif self.network_type == "omniglot":
            self.model = self.mnist_basic_discriminator_model(x_var, reuse)
        elif self.network_type == "celebA":
            self.model = self.celebA_discriminator_model(x_var, reuse)
        elif self.network_type == "cifar10":
            self.model = self.cifar10_discriminator_model(x_var, reuse)
        else:
            raise NotImplementedError
        return self.model

    #final_output_dim is equal to the dim of reg latent codes
    def encode(self, x_var, final_output_dim, reuse):
        #with tf.variable_scope(name_or_scope = scope, reuse = reuse):
            #if self.encoder_model is None:
        if self.network_type == "mnist":
            self.encoder_model = self.mnist_encoder_model(x_var, final_output_dim)
        elif self.network_type == "svhn":
            self.encoder_model = self.svhn_encoder_model(x_var, final_output_dim)
        elif self.network_type == "omniglot":
            self.encoder_model = self.mnist_basic_encoder_model(x_var, final_output_dim)
        elif self.network_type == "celebA":
            self.encoder_model = self.celebA_encoder_model(x_var, final_output_dim)
        elif self.network_type == "cifar10":
            self.encoder_model = self.cifar10_encoder_model(x_var, final_output_dim)
        else:
            raise NotImplementedError
        return self.encoder_model
#----------------------------------------------cifar10-------------------------------------------------------------

    def cifar10_shared_model(self, x_var, reuse):
        # input layer
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            self.layer_params.layer_input = tf.reshape(x_var, [-1] + list(self.image_shape))
            self.layer_params.layer_input = custom_conv2d(self.layer_params,
                                                          name = "conv_l1", output_dim = 64, kernel_h = 4, kernel_w = 4)
            self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l1")
            self.layer_params.layer_input = custom_conv2d(self.layer_params,
                                                          name = "conv_l2", output_dim = 128, kernel_h = 4, kernel_w = 4)
            self.layer_params.layer_input = conv_batch_norm(self.layer_params, name = "conv_bn_l2")
            self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l2")
            self.layer_params.layer_input = custom_conv2d(self.layer_params,
                                                          name = "conv_l3", output_dim = 256, kernel_h = 4,
                                                          kernel_w = 4)
            self.layer_params.layer_input = conv_batch_norm(self.layer_params, name = "conv_bn_l3")
            layer_output = custom_leaky_relu(self.layer_params, name = "lrelu_l3")
        return layer_output

    def cifar10_discriminator_model(self, x_var, reuse):
        self.shared_model = self.cifar10_shared_model(x_var, reuse = reuse)
        self.layer_params.layer_input = self.shared_model
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            layer_output = custom_fully_connected(self.layer_params, name="fc_l4", num_neurons=1)
        return layer_output

    def cifar10_encoder_model(self, x_var, final_output_dim):
        self.layer_params.layer_input = self.cifar10_shared_model(x_var, reuse = True)
        self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l5", num_neurons = 128)
        self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l5")
        self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l5")
        layer_output = custom_fully_connected(self.layer_params, name = "fc_l6", num_neurons = final_output_dim)
        return layer_output

#---------------------------------------------celebA----------------------------------------------------------------

    def celebA_shared_model(self, x_var, reuse):
        # input layer
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            self.layer_params.layer_input = tf.reshape(x_var, [-1] + list(self.image_shape))
            self.layer_params.layer_input = custom_conv2d(self.layer_params,
                                                          name = "conv_l1", output_dim = 64, kernel_h = 4, kernel_w = 4)
            self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l1")
            self.layer_params.layer_input = custom_conv2d(self.layer_params,
                                                          name = "conv_l2", output_dim = 128, kernel_h = 4, kernel_w = 4)
            self.layer_params.layer_input = conv_batch_norm(self.layer_params, name = "conv_bn_l2")
            self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l2")
            self.layer_params.layer_input = custom_conv2d(self.layer_params,
                                                          name = "conv_l3", output_dim = 256, kernel_h = 4,
                                                          kernel_w = 4)
            self.layer_params.layer_input = conv_batch_norm(self.layer_params, name = "conv_bn_l3")
            layer_output = custom_leaky_relu(self.layer_params, name = "lrelu_l3")
        return layer_output

    def celebA_discriminator_model(self, x_var, reuse):
        self.shared_model = self.celebA_shared_model(x_var, reuse = reuse)
        self.layer_params.layer_input = self.shared_model
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            layer_output = custom_fully_connected(self.layer_params, name="fc_l4", num_neurons=1)
        return layer_output

    def celebA_encoder_model(self, x_var, final_output_dim):
        self.layer_params.layer_input = self.celebA_shared_model(x_var, reuse = True)
        self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l5", num_neurons = 128)
        self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l5")
        self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l5")
        layer_output = custom_fully_connected(self.layer_params, name = "fc_l6", num_neurons = final_output_dim)
        return layer_output



#---------------------------------------------omniglot--------------------------------------------------------------

    def omniglot_shared_model(self, x_var, reuse):
        # input layer
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            self.layer_params.layer_input = tf.reshape(x_var, [-1] + list(self.image_shape))
            self.layer_params.layer_input = custom_conv2d(self.layer_params,
                                                          name = "conv_l1", output_dim = 64, kernel_h = 4, kernel_w = 4)
            self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l1")
            self.layer_params.layer_input = custom_conv2d(self.layer_params,
                                                          name = "conv_l2", output_dim = 128, kernel_h = 4, kernel_w = 4)
            self.layer_params.layer_input = conv_batch_norm(self.layer_params, name = "conv_bn_l2")
            self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l2")
            self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l3", num_neurons = 1024)
            self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l3")
            layer_output = custom_leaky_relu(self.layer_params, name = "lrelu_l3")
        return layer_output

    def omniglot_discriminator_model(self, x_var, reuse):
        self.shared_model = self.mnist_shared_model(x_var, reuse = reuse)
        self.layer_params.layer_input = self.shared_model
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            layer_output = custom_fully_connected(self.layer_params, name="fc_l4", num_neurons=1)
        return layer_output

    def omniglot_encoder_model(self, x_var, final_output_dim):
        # input to encoder
        self.layer_params.layer_input = self.mnist_shared_model(x_var, reuse = True)
        #self.layer_params.layer_input = self.shared_model
        #with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
        #with tf.variable_scope("e_net", reuse = self.reuse):
        self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l5", num_neurons = 128)
        self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l5")
        self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l5")
        layer_output = custom_fully_connected(self.layer_params, name = "fc_l6", num_neurons = final_output_dim)
        return layer_output

    """
    def omniglot_shared_model(self, x_var, reuse):
        # input layer
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            self.layer_params.layer_input = tf.reshape(x_var, [-1] + list(self.image_shape))
            self.layer_params.layer_input = custom_conv2d(self.layer_params,
                                                          name = "conv_l1", output_dim = 64, kernel_h = 4, kernel_w = 4)
            self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l1")
            self.layer_params.layer_input = custom_conv2d(self.layer_params,
                                                          name = "conv_l2", output_dim = 128, kernel_h = 4, kernel_w = 4)
            self.layer_params.layer_input = conv_batch_norm(self.layer_params, name = "conv_bn_l2")
            self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l2")
            self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l3", num_neurons = 1024)
            self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l3")
            layer_output = custom_leaky_relu(self.layer_params, name = "lrelu_l3")
        return layer_output

    def omniglot_discriminator_model(self, x_var, reuse):
        self.shared_model = self.omniglot_shared_model(x_var, reuse = reuse)
        self.layer_params.layer_input = self.shared_model
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            layer_output = custom_fully_connected(self.layer_params, name="fc_l4", num_neurons=1)
        return layer_output

    def omniglot_encoder_model(self, x_var, final_output_dim):
        # input to encoder
        self.layer_params.layer_input = self.omniglot_shared_model(x_var, reuse = True)
        #self.layer_params.layer_input = self.shared_model
        #with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
        #with tf.variable_scope("e_net", reuse = self.reuse):
        self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l5", num_neurons = 128)
        self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l5")
        self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l5")
        layer_output = custom_fully_connected(self.layer_params, name = "fc_l6", num_neurons = final_output_dim)
        return layer_output
    """

#----------------------------------------------svhn-------------------------------------------------------------------

    def svhn_shared_model(self, x_var, reuse):
        # input layer
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            self.layer_params.layer_input = tf.reshape(x_var, [-1] + list(self.image_shape))
            self.layer_params.layer_input = custom_conv2d(self.layer_params,
                                                          name = "conv_l1", output_dim = 64, kernel_h = 4, kernel_w = 4)
            self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l1")
            self.layer_params.layer_input = custom_conv2d(self.layer_params,
                                                          name = "conv_l2", output_dim = 128, kernel_h = 4, kernel_w = 4)
            self.layer_params.layer_input = conv_batch_norm(self.layer_params, name = "conv_bn_l2")
            self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l2")
            self.layer_params.layer_input = custom_conv2d(self.layer_params,
                                                          name = "conv_l3", output_dim = 256, kernel_h = 4,
                                                          kernel_w = 4)
            self.layer_params.layer_input = conv_batch_norm(self.layer_params, name = "conv_bn_l3")
            layer_output = custom_leaky_relu(self.layer_params, name = "lrelu_l3")
        return layer_output

    def svhn_discriminator_model(self, x_var, reuse):
        self.shared_model = self.svhn_shared_model(x_var, reuse = reuse)
        self.layer_params.layer_input = self.shared_model
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            layer_output = custom_fully_connected(self.layer_params, name="fc_l4", num_neurons=1)
        return layer_output

    def svhn_encoder_model(self, x_var, final_output_dim):
        self.layer_params.layer_input = self.svhn_shared_model(x_var, reuse = True)
        self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l5", num_neurons = 128)
        self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l5")
        self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l5")
        layer_output = custom_fully_connected(self.layer_params, name = "fc_l6", num_neurons = final_output_dim)
        return layer_output


#--------------------------------------------Mnist---------------------------------------------------------------------

    def mnist_shared_model(self, x_var, reuse):
        # input layer
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            self.layer_params.layer_input = tf.reshape(x_var, [-1] + list(self.image_shape))
            self.layer_params.layer_input = custom_conv2d(self.layer_params,
                                                          name = "conv_l1", output_dim = 64, kernel_h = 4, kernel_w = 4)
            self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l1")
            self.layer_params.layer_input = custom_conv2d(self.layer_params,
                                                          name = "conv_l2", output_dim = 128, kernel_h = 4, kernel_w = 4)
            self.layer_params.layer_input = conv_batch_norm(self.layer_params, name = "conv_bn_l2")
            self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l2")
            self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l3", num_neurons = 1024)
            self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l3")
            layer_output = custom_leaky_relu(self.layer_params, name = "lrelu_l3")
        return layer_output

    def mnist_discriminator_model(self, x_var, reuse):
        self.shared_model = self.mnist_shared_model(x_var, reuse = reuse)
        self.layer_params.layer_input = self.shared_model
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            layer_output = custom_fully_connected(self.layer_params, name="fc_l4", num_neurons=1)
        return layer_output

    def mnist_encoder_model(self, x_var, final_output_dim):
        # input to encoder
        self.layer_params.layer_input = self.mnist_shared_model(x_var, reuse = True)
        #self.layer_params.layer_input = self.shared_model
        #with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
        #with tf.variable_scope("e_net", reuse = self.reuse):
        self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l5", num_neurons = 128)
        self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l5")
        self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l5")
        layer_output = custom_fully_connected(self.layer_params, name = "fc_l6", num_neurons = final_output_dim)
        return layer_output

    def mnist_basic_shared_model(self, x_var, reuse):
        # input layer
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            self.layer_params.layer_input = tf.reshape(x_var, [-1] + list(self.image_shape))
            self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l1", num_neurons = 256)
            self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l1")
            self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l2", num_neurons = 128)
            self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l2")
            layer_output = custom_leaky_relu(self.layer_params, name = "lrelu_l2")
        return layer_output

    def mnist_basic_discriminator_model(self, x_var, reuse):
        self.shared_model = self.mnist_basic_shared_model(x_var, reuse = reuse)
        self.layer_params.layer_input = self.shared_model
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            layer_output = custom_fully_connected(self.layer_params, name="fc_l3", num_neurons=1)
        return layer_output

    def mnist_basic_encoder_model(self, x_var, final_output_dim):
        # input to encoder
        self.layer_params.layer_input = self.mnist_basic_shared_model(x_var, reuse = True)
        self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "fc_l4", num_neurons = 64)
        self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "fc_bn_l4")
        self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "lrelu_l4")
        layer_output = custom_fully_connected(self.layer_params, name = "fc_l5", num_neurons = final_output_dim)
        return layer_output


    """
        def mnist_common_model(self, x_var, final_output_dim, reuse):
        # input to encoder
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            self.layer_params.layer_input = tf.reshape(x_var, [-1] + list(self.image_shape))
            self.layer_params.layer_input = custom_conv2d(self.layer_params,
                                                          name = "e_conv_l1", output_dim = 64, kernel_h = 4, kernel_w = 4)
            self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "e_lrelu_l1")
            self.layer_params.layer_input = custom_conv2d(self.layer_params,
                                                          name = "e_conv_l2", output_dim = 128, kernel_h = 4, kernel_w = 4)
            self.layer_params.layer_input = conv_batch_norm(self.layer_params, name = "e_conv_bn_l2")
            self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "e_lrelu_l2")
            self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "e_fc_l3", num_neurons = 1024)
            self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "e_fc_bn_l3")
            self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "e_lrelu_l3")

            self.layer_params.layer_input = custom_fully_connected(self.layer_params, name = "e_fc_l4",
                                                                   num_neurons = 128)
            self.layer_params.layer_input = fc_batch_norm(self.layer_params, name = "e_fc_bn_l4")
            self.layer_params.layer_input = custom_leaky_relu(self.layer_params, name = "e_lrelu_l4")
            layer_output = custom_fully_connected(self.layer_params, name = "e_fc_l5", num_neurons = final_output_dim)
        return layer_output
    """
