import tensorflow as tf
import numpy as np

from ..helpers.distributions import Bernoulli, Gaussian, Categorical
from progressbar import ETA, Bar, Percentage, ProgressBar
import collections

TINY = 1e-8

class InfoGANTrainer(object):
    def __init__(self,
                 model,
                 batch_size,
                 dataset = None,
                 exp_name = "experiment",
                 log_dir = "logs",
                 checkpoint_dir = "ckt",
                 max_epoch = 100,
                 updates_per_epoch = 100,
                 snapshot_interval = 10000,
                 info_reg_coeff = 1.0,
                 discriminator_learning_rate = 2e-4,
                 generator_learning_rate = 2e-4,
                 load_checkpoint_path = None
                 ):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.snapshot_interval = snapshot_interval
        self.updates_per_epoch = updates_per_epoch
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.info_reg_coeff = info_reg_coeff
        self.discriminator_trainer = None
        self.generator_trainer = None
        self.input_tensor = None
        self.log_vars = []
        self.load_checkpoint_path = load_checkpoint_path


    def init_opt(self):
        #This would be a 2D tensor with dim [128,784]. This is the placeholder for storing input
        self.input_tensor = input_tensor = tf.placeholder(tf.float32,
                                                          [self.batch_size, self.dataset.image_dim], name = "input_x")
        #z_var dimension = [batch_size, dimension of latent codes], We are trying to sample prior noise or latent codes
        z_var = self.model.latent_dist.sample_prior(self.batch_size)
        #generate fake samples using noise vector
        with tf.name_scope("g_fake_x"):
            fake_x, _ = self.model.generate(z_var)
        #apply discriminator net on real samples, real_d is the measure of realness
        #of samples. High value of real_d means it is more real according to D
        with tf.name_scope("d_real_x"):
            real_d = self.model.discriminate(input_tensor)
        #apply discriminator net on the fake samples, fake_d represent the value of
        #realness for each sample. More is its value more real a sample is according to D
        with tf.name_scope("d_fake_x"):
            fake_d = self.model.discriminate(fake_x, reuse = True)
        with tf.name_scope("e_fake_x"):
            fake_reg_z_dist_info = self.model.encode(fake_x)
        # split z into two parts (noise, latent codes)=(z,c)=(nonreg_z,reg_z)
        reg_z = self.model.reg_z(z_var)
        #D wants to reduce its loss by marking fake samples as fake i.e less score to fake_d
        #G wants to reduce its loss by making fake samples that can be marked as real by D. i.e high score for fake_d
        discriminator_loss = - tf.reduce_mean(tf.log(real_d + TINY) + tf.log(1. - fake_d + TINY))
        generator_loss = - tf.reduce_mean(tf.log(fake_d + TINY))
        self.log_vars.append(("discriminator_loss", discriminator_loss))
        self.log_vars.append(("generator_loss", generator_loss))

        #mutual information estimation
        mi_est = tf.constant(0.)
        #cross entropy
        cross_ent = tf.constant(0.)

        # compute regularization
        if len(self.model.reg_disc_latent_dist.dists) > 0:
            # Q(c)
            disc_reg_z = self.model.disc_reg_z(reg_z)
            # Q(c|x)
            disc_reg_dist_info = self.model.disc_reg_dist_info(fake_reg_z_dist_info)
            # Q(c|x).log(Q(c|x)) = logli(c,Q(c|x))
            disc_log_q_c_given_x = self.model.reg_disc_latent_dist.logli(disc_reg_z, disc_reg_dist_info)
            # Q(c).log(Q(c)) = logli_prior(Q(c))
            disc_log_q_c = self.model.reg_disc_latent_dist.logli_prior(disc_reg_z)
            # H(c|x~G(z,c)) = E[-log(Q(c|x))] = -sum_c[Q(c|x).log(Q(c|x))]
            disc_cross_ent = tf.reduce_mean(-disc_log_q_c_given_x)
            # H(c) = E[-log(Q(c))] = -sum_c[Q(c).log(Q(c))]
            disc_ent = tf.reduce_mean(-disc_log_q_c)
            # I(c;x~G(z,c)) = H(c) - H(c|x)
            disc_mi_est = disc_ent - disc_cross_ent
            #disc_mi_est = tf.Print(disc_mi_est, [disc_mi_est])
            #with tf.control_dependencies([tf.assert_non_negative(disc_mi_est)]):
            mi_est += disc_mi_est
            cross_ent += disc_cross_ent
            self.log_vars.append(("MI_disc", disc_mi_est))
            self.log_vars.append(("CrossEnt_disc", disc_cross_ent))
            # regularized losses
            discriminator_loss -= self.info_reg_coeff * disc_mi_est
            generator_loss -= self.info_reg_coeff * disc_mi_est

        if len(self.model.reg_cont_latent_dist.dists) > 0:
            cont_reg_z = self.model.cont_reg_z(reg_z)
            cont_reg_dist_info = self.model.cont_reg_dist_info(fake_reg_z_dist_info)
            cont_log_q_c_given_x = self.model.reg_cont_latent_dist.logli(cont_reg_z, cont_reg_dist_info)
            cont_log_q_c = self.model.reg_cont_latent_dist.logli_prior(cont_reg_z)
            cont_cross_ent = tf.reduce_mean(-cont_log_q_c_given_x)
            cont_ent = tf.reduce_mean(-cont_log_q_c)
            cont_mi_est = cont_ent - cont_cross_ent
            mi_est += cont_mi_est
            cross_ent += cont_cross_ent
            self.log_vars.append(("MI_cont", cont_mi_est))
            self.log_vars.append(("CrossEnt_cont", cont_cross_ent))
            discriminator_loss -= self.info_reg_coeff * cont_mi_est
            generator_loss -= self.info_reg_coeff * cont_mi_est

        for idx, dist_info in enumerate(self.model.reg_latent_dist.split_dist_info(fake_reg_z_dist_info)):
            if "stddev" in dist_info:
                self.log_vars.append(("max_std_%d" % idx, tf.reduce_max(dist_info["stddev"])))
                self.log_vars.append(("min_std_%d" % idx, tf.reduce_min(dist_info["stddev"])))

        self.log_vars.append(("MI", mi_est))
        self.log_vars.append(("CrossEnt", cross_ent))
        self.log_vars.append(("max_real_d", tf.reduce_max(real_d)))
        self.log_vars.append(("min_real_d", tf.reduce_min(real_d)))
        self.log_vars.append(("max_fake_d", tf.reduce_max(fake_d)))
        self.log_vars.append(("min_fake_d", tf.reduce_min(fake_d)))

        all_vars = tf.trainable_variables()
        d_vars = [var for var in all_vars if var.name.startswith('d_')]
        g_vars = [var for var in all_vars if var.name.startswith('g_')]

        self.discriminator_trainer = tf.train.AdamOptimizer(self.discriminator_learning_rate, beta1 = 0.5,
                                                            name = "adam_d", ).minimize(discriminator_loss,
                                                                                        var_list = d_vars)

        self.generator_trainer = tf.train.AdamOptimizer(self.generator_learning_rate, beta1 = 0.5,
                                                        name = "adam_g").minimize(generator_loss,
                                                                                  var_list = g_vars)

        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

        #with tf.variable_scope("model") as scope:
        self.visualize_all_factors()

    def visualize_all_factors(self):
        with tf.Session():
            #dim = 2 if self.exp_name.startswith('omniglot') else 10
            dim = 10
            # This will concatenate the two arrays. dim = (100+batch_size-100,62)
            fixed_noncat = np.concatenate([
                # This will tile the (10,62) in 10 layers making the new shape (10*10,62)
                np.tile(
                    # It will sample 10 samples from non regularized latent distribution. Each sample
                    # will have dimension which was given earlier to non-reg distri (62 in case of
                    # )
                    self.model.nonreg_latent_dist.sample_prior(10).eval(),
                    [10, 1]
                ),
                self.model.nonreg_latent_dist.sample_prior(self.batch_size - 100).eval(),
            ], axis=0)
            # (100+batch_size-100,12)
            fixed_cat = np.concatenate([
                # (100,12)
                np.tile(
                    # (10,12)
                    self.model.reg_latent_dist.sample_prior(10).eval(),
                    [10, 1]
                ),
                self.model.reg_latent_dist.sample_prior(self.batch_size - 100).eval(),
            ], axis=0)

        offset = 0
        for dist_idx, dist in enumerate(self.model.reg_latent_dist.dists):
            if isinstance(dist, Gaussian):
                assert dist.dim == 1, "Only dim=1 is currently supported"
                c_vals = []
                for idx in range(10):
                    # extend will add one element at a time to c_vals. A list of 10
                    # elements will be generated
                    c_vals.extend([-1.0 + idx * 2.0 / 9] * 10)
                # It means 0 will be repeated batch_soze-100 times and added to the list.
                c_vals.extend([0.] * (self.batch_size - 100))
                # at this point c_val will have 128 elements
                # vary_cat = (128,1)
                vary_cat = np.asarray(c_vals, dtype=np.float32).reshape((-1, 1))
                # creates a deep copy of fixed_cat (128,12)
                cur_cat = np.copy(fixed_cat)
                # cur_cat=(128,12)
                cur_cat[:, offset:offset+1] = vary_cat
                offset += 1
            elif isinstance(dist, Categorical):
                # This is done for having all the variety of categorical latent variable
                # in the noise tensor so we can create all types of digits.
                # lookup is (10,10) identity matrix
                lookup = np.eye(dist.dim, dtype=np.float32)
                cat_ids = []
                for idx in range(dim):
                    cat_ids.extend([idx] * int(100/dim))
                cat_ids.extend([0] * (self.batch_size - 100))
                # cat_ids = (128,)
                cur_cat = np.copy(fixed_cat)
                # cur_cat=(128,12)
                # lookup[cat_ids] will give be a matrix of (128,10)
                print(cur_cat.shape)
                print(lookup.shape)
                cur_cat[:, offset:offset+dist.dim] = lookup[cat_ids]
                offset += dist.dim
            elif isinstance(dist, Bernoulli):
                assert dist.dim == 1, "Only dim=1 is currently supported"
                cat_ids = []
                for idx in range(10):
                    cat_ids.extend([int(idx / 5)] * 10)
                cat_ids.extend([0] * (self.batch_size - 100))
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset+dist.dim] = np.expand_dims(np.array(cat_ids), axis=-1)
                offset += dist.dim
            else:
                raise NotImplementedError
            # fixed_noncat=(128,62), cur_cat=(128,12), z_var=(128,74)
            z_var = tf.constant(np.concatenate([fixed_noncat, cur_cat], axis=1))

            #reuse = True if dist_idx == 0 else True
            _, x_dist_info = self.model.generate(z_var, reuse = True, training = False)

            # just take the mean image
            if isinstance(self.model.output_dist, Bernoulli):
                img_var = x_dist_info["p"]
            elif isinstance(self.model.output_dist, Gaussian):
                img_var = x_dist_info["mean"]
            else:
                raise NotImplementedError
            #img_var = self.dataset.inverse_transform(img_var)
            rows = 10
            img_var = tf.reshape(img_var, [self.batch_size] + list(self.dataset.image_shape))
            # It is just keeping first 100 samples out of 128 samples
            img_var = img_var[:rows * rows, :, :, :]
            # rearranging 100 samples in rows n cols of 10 by 10
            imgs = tf.reshape(img_var, [rows, rows] + list(self.dataset.image_shape))
            stacked_img = []
            for row in range(rows):
                row_img = []
                for col in range(rows):
                    row_img.append(imgs[row, col, :, :, :])
                stacked_img.append(tf.concat(axis = 1, values = row_img))
            imgs = tf.concat(axis = 0, values = stacked_img)
            imgs = tf.expand_dims(imgs, 0)
            # adding generated image to tf image summary
            tf.summary.image("image_%d_%s" % (dist_idx, dist.__class__.__name__), imgs)

    def get_pretraining_data(self):
        x = []
        x_real, _ = self.dataset.train.next_random_batch(self.batch_size)
        x.append(x_real)
        z_var = self.model.latent_dist.sample_prior(self.batch_size)
        fake_x, _ = self.model.generate(z_var)
        x.append(fake_x)
        return np.array(x)

    def train(self):
        self.init_opt()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            saver = tf.train.Saver()
            if self.load_checkpoint_path is not None:
                saver.restore(sess, self.load_checkpoint_path)
            else:
                sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            counter = 0
            log_vars = [x for _, x in self.log_vars]
            log_keys = [x for x, _ in self.log_vars]
            for epoch in range(self.max_epoch):
                widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                pbar = ProgressBar(maxval = self.updates_per_epoch, widgets = widgets).start()
                all_log_vals = []
                for i in range(self.updates_per_epoch):
                    pbar.update(i)
                    x, _ = self.dataset.train.next_random_batch(self.batch_size)
                    feed_dict = {self.input_tensor: x}
                    '''
                    log_vars is list of tensor variables
                    self.discriminator_trainer is an tf.Operation so the corresponding fetched value will be None
                    log_vals will store the values of log_vars after evaluation
                    '''

                    log_vals = sess.run(fetches = [self.discriminator_trainer] + log_vars, feed_dict = feed_dict)[1:]
                    if i % 2 == 0:
                        sess.run(self.generator_trainer, feed_dict = feed_dict)
                    all_log_vals.append(log_vals)
                    counter += 1
                    # This saves a snapshot of the model
                    if counter % self.snapshot_interval == 0:
                        snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                        fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                        print("Model saved in file: %s" % fn)

                x, _ = self.dataset.train.next_random_batch(self.batch_size)
                summary_str = sess.run(summary_op, {self.input_tensor: x})
                summary_writer.add_summary(summary_str, counter)

                avg_log_vals = np.mean(np.array(all_log_vals), axis = 0)
                log_dict = dict(zip(log_keys, avg_log_vals))
                log_dict = collections.OrderedDict(sorted(log_dict.items()))

                log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in log_dict.items())
                print("Epoch %d | " % (epoch) + log_line)
                if np.any(np.isnan(avg_log_vals)):
                    raise ValueError("NaN detected!")
