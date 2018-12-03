import tensorflow as tf
from ..helpers.distributions import Product, Distribution, Gaussian, Categorical, Bernoulli
from ..models.generator_models import Generator
from ..models.discriminator_models import Discriminator

class InfoGAN(object):
    """
    :type output_dist: Distribution
    :type latent_spec: list[(Distribution, bool)]
    :type batch_size: int
    :type network_type: string
    """
    def __init__(self, output_dist, latent_spec, batch_size, image_shape, network_type):
        self.output_dist = output_dist
        #entire latent specification
        self.latent_spec = latent_spec
        #entire latent distributions
        self.latent_dist = Product([x for x, _ in latent_spec])
        #regularized latent distribution for regularized latent codes
        self.reg_latent_dist = Product([x for x, reg in latent_spec if reg])
        #non refularized latent distributions for noise latent codes
        self.nonreg_latent_dist = Product([x for x, reg in latent_spec if not reg])
        self.batch_size = batch_size
        #defines type of data this network is created for
        self.network_type = network_type
        self.image_shape = image_shape


        #regularized continuous and discrete distributions
        self.reg_cont_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, Gaussian)])
        self.reg_disc_latent_dist = Product(
            [x for x in self.reg_latent_dist.dists if isinstance(x, (Categorical, Bernoulli))])

        self.generator = Generator(network_type = self.network_type, image_shape = self.image_shape)
        self.discriminator = Discriminator(network_type = self.network_type, image_shape = self.image_shape)

    """
    z_var = (z,c)
    1) generates fake sample generating distribution(P(x|z,c)) using generator model.
    2) apply activation on the generated discribution.(Its like applying activation layer)
    3) generate fake samples using P(x|z,c) distribution.
    """
    def generate(self, z_var, reuse = None, training = True):
        x_dist_flat = self.generator.generate(z_var = z_var, training = training, reuse = reuse)
        x_dist_info = self.output_dist.activate_dist(x_dist_flat)
        fake_x = self.output_dist.sample(x_dist_info)
        return fake_x, x_dist_info

    """
    1) runs the discriminator net on the input data samples and gives the truth value for each
        sample whether its real or fake.
    2) apply sigmoid activation to the output of discriminator.
    3) calculate Q(c|x) using encoder, here c represents regularized latent codes
    4) apply activation on the encoder output
    """
    def discriminate(self, x_var, reuse = None):
        with tf.variable_scope("d_net", reuse = reuse):
            d_out = self.discriminator.discriminate(x_var = x_var, reuse = reuse)
            #reg_dist_flat = self.discriminator.encode(x_var = x_var, final_output_dim = self.reg_latent_dist.dist_flat_dim,
                                                     # reuse = reuse)
        d = tf.nn.sigmoid(d_out[:, 0])
        #reg_dist_info = self.reg_latent_dist.activate_dist(reg_dist_flat)
        return d #, self.reg_latent_dist.sample(reg_dist_info), reg_dist_info, reg_dist_flat

    def encode(self, x_var, reuse = None):
        with tf.variable_scope("d_net", reuse = reuse):
            reg_dist_flat = self.discriminator.encode(x_var = x_var, final_output_dim = self.reg_latent_dist.dist_flat_dim,
                                                      reuse = reuse)
            reg_dist_info = self.reg_latent_dist.activate_dist(reg_dist_flat)
            return reg_dist_info

    '''
    This function separates regularized latent codes from the noise vector.
    zip : it stitches elements into tuples
    self.latent_spec : its a tuple of 2 elements (distribution, boolean)
    '''
    def reg_z(self, z_var):
        ret = []
        for (_, reg_i), z_i in zip(self.latent_spec, self.latent_dist.split_var(z_var)):
            if reg_i:
                ret.append(z_i)
        return self.reg_latent_dist.join_vars(ret)

    """
    Seperates out discontinuous regularized latent codes from regularized latent codes
    """
    def disc_reg_z(self, reg_z_var):
        ret = []
        for dist_i, z_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_var(reg_z_var)):
            if isinstance(dist_i, (Categorical, Bernoulli)):
                ret.append(z_i)
        return self.reg_disc_latent_dist.join_vars(ret)

    """
    Seperates out discontinuous regularized distri info from regularized distri info
    """
    def disc_reg_dist_info(self, reg_dist_info):
        ret = []
        for dist_i, dist_info_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_dist_info(reg_dist_info)):
            if isinstance(dist_i, (Categorical, Bernoulli)):
                ret.append(dist_info_i)
        return self.reg_disc_latent_dist.join_dist_infos(ret)

    """
    Seperates out continuous regularized latent codes from regularized latent codes
    """
    def cont_reg_z(self, reg_z_var):
        ret = []
        for dist_i, z_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_var(reg_z_var)):
            if isinstance(dist_i, Gaussian):
                ret.append(z_i)
        return self.reg_cont_latent_dist.join_vars(ret)

    """
    Seperates out continuous regularized distri info from regularized distri info
    """
    def cont_reg_dist_info(self, reg_dist_info):
        ret = []
        for dist_i, dist_info_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_dist_info(reg_dist_info)):
            if isinstance(dist_i, Gaussian):
                ret.append(dist_info_i)
        return self.reg_cont_latent_dist.join_dist_infos(ret)