from tensorflow.contrib import distributions
from infogan.helpers.distributions import Categorical
import tensorflow as tf
import numpy as np
from infogan.helpers.distributions import Uniform, Categorical, Gaussian, MeanBernoulli
import os
from infogan.helpers.mnist import Mnist
from infogan.models.infogan_model import InfoGAN
from infogan.trainers.infogan_trainer import InfoGANTrainer
import dateutil.tz
import datetime


if __name__ == "__main__":
    mnist_data_path = "/home/sumit/Documents/repo/datasets/mnist"
    batch_size = 128
    dataset = Mnist(data_path = mnist_data_path)

    latent_spec = [
        (Uniform(62), False),
        (Categorical(10), True),
        (Uniform(1, fix_std = True), True),
        (Uniform(1, fix_std = True), True)
    ]

    model = InfoGAN(
        output_dist = MeanBernoulli(dataset.image_dim),
        latent_spec = latent_spec,
        batch_size = batch_size,
        image_shape = dataset.image_shape,
        network_type = "mnist"
    )
    TINY = 1e-8
    input_tensor = input_tensor = tf.placeholder(tf.float32,
                                                      [batch_size, dataset.image_dim], name = "input_tensor")
    z_var = model.latent_dist.sample_prior(batch_size)
    fake_x, _ = model.generate(z_var)
    real_d = model.discriminate(input_tensor)
    fake_d = model.discriminate(fake_x, reuse = True)
    fake_reg_z_dist_info = model.encode(fake_x)
    reg_z = model.reg_z(z_var)
    discriminator_loss = - tf.reduce_mean(tf.log(real_d + TINY) + tf.log(1. - fake_d + TINY))
    generator_loss = - tf.reduce_mean(tf.log(fake_d + TINY))
    mi_est = tf.constant(0.)
    cross_ent = tf.constant(0.)

    if len(model.reg_disc_latent_dist.dists) > 0:
        disc_reg_z = model.disc_reg_z(reg_z)
        disc_reg_dist_info = model.disc_reg_dist_info(fake_reg_z_dist_info)
        disc_log_q_c_given_x = model.reg_disc_latent_dist.logli(disc_reg_z, disc_reg_dist_info)
        disc_log_q_c = model.reg_disc_latent_dist.logli_prior(disc_reg_z)
        disc_cross_ent = tf.reduce_mean(-disc_log_q_c_given_x)
        disc_ent = tf.reduce_mean(-disc_log_q_c)
        disc_mi_est = disc_ent - disc_cross_ent
        disc_mi_est = tf.Print(disc_mi_est, [disc_mi_est])
        mi_est += disc_mi_est
        cross_ent += disc_cross_ent
        discriminator_loss -= disc_mi_est
        generator_loss -= disc_mi_est

    if len(model.reg_cont_latent_dist.dists) > 0:
        cont_reg_z = model.cont_reg_z(reg_z)
        cont_reg_dist_info = model.cont_reg_dist_info(fake_reg_z_dist_info)
        cont_log_q_c_given_x = model.reg_cont_latent_dist.logli(cont_reg_z, cont_reg_dist_info)
        cont_log_q_c = model.reg_cont_latent_dist.logli_prior(cont_reg_z)
        cont_cross_ent = tf.reduce_mean(-cont_log_q_c_given_x)
        cont_ent = tf.reduce_mean(-cont_log_q_c)
        cont_mi_est = cont_ent - cont_cross_ent
        mi_est += cont_mi_est
        cross_ent += cont_cross_ent
        discriminator_loss -= cont_mi_est
        generator_loss -= cont_mi_est

    all_vars = tf.trainable_variables()
    d_vars = [var for var in all_vars if var.name.startswith('d_')]
    g_vars = [var for var in all_vars if var.name.startswith('g_')]

    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate = 2e-4, beta1 = 0.5, name = "adam_d")
    discriminator_gradients = discriminator_optimizer.compute_gradients(loss = discriminator_loss, var_list = d_vars)
    #with tf.control_dependencies([tf.assert_non_negative(disc_mi_est)]):
    discriminator_trainer = discriminator_optimizer.apply_gradients(grads_and_vars = discriminator_gradients)
    #discriminator_gradients = tf.Print(discriminator_gradients, [discriminator_gradients])
    #print(discriminator_gradients)

    generator_optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3, beta1 = 0.5, name = "adam_g")
    generator_gradients = generator_optimizer.compute_gradients(loss = generator_loss, var_list = g_vars)
    generator_trainer = generator_optimizer.apply_gradients(grads_and_vars = generator_gradients)
    #generator_gradients = tf.Print(generator_gradients, [generator_gradients])
    #print(discriminator_gradients)


    """
    discriminator_trainer = tf.train.AdamOptimizer(learning_rate = 2e-4, beta1 = 0.5,
                                                        name = "adam_d").minimize(discriminator_loss,
                                                                                    var_list = d_vars)

    generator_trainer = tf.train.AdamOptimizer(learning_rate = 1e-3, beta1 = 0.5,
                                                    name = "adam_g").minimize(generator_loss,
                                                                              var_list = g_vars)
    """

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(2):
            for i in range(5):
                x, _ = dataset.train.next_random_batch(batch_size)
                feed_dict = {input_tensor: x}
                print(sess.run(fetches = [discriminator_trainer]+[disc_mi_est, cont_mi_est], feed_dict = feed_dict))
                sess.run(generator_trainer, feed_dict = feed_dict)
                #print(sess.run([discriminator_loss, generator_loss, disc_mi_est, cont_mi_est]))


