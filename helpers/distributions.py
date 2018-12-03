import tensorflow as tf
import numpy as np
import itertools


TINY = 1e-8

floatX = np.float32

class Distribution(object):
    @property
    def dist_flat_dim(self):
        """
        :rtype: int
        """
        raise NotImplementedError

    @property
    def dim(self):
        """
        :rtype: int
        """
        raise NotImplementedError

    @property
    def effective_dim(self):
        """
        The effective dimension when used for rescaling quantities. This can be different from the
        actual dimension when the actual values are using redundant representations (e.g. for categorical
        distributions we encode it in onehot representation)
        :rtype: int
        """
        raise NotImplementedError

    # I dont think this is used anywhere. May be its overwritten version in its base class
    # is used some where.
    def kl_prior(self, dist_info):
        return self.kl(dist_info, self.prior_dist_info(dist_info.values()[0].get_shape()[0]))

    def logli(self, x_var, dist_info):
        """
        :param x_var:
        :param dist_info:
        :return: log likelihood of the data
        """
        raise NotImplementedError

    def logli_prior(self, x_var):
        return self.logli(x_var, self.prior_dist_info(x_var.get_shape()[0]))

    def nonreparam_logli(self, x_var, dist_info):
        """
        :param x_var:
        :param dist_info:
        :return: the non-reparameterizable part of the log likelihood
        """
        raise NotImplementedError

    def activate_dist(self, flat_dist):
        """
        :param flat_dist: flattened dist info without applying nonlinearity yet
        :return: a dictionary of dist infos
        """
        raise NotImplementedError

    @property
    def dist_info_keys(self):
        """
        :rtype: list[str]
        """
        raise NotImplementedError

    def entropy(self, dist_info):
        """
        :return: entropy for each minibatch entry
        """
        raise NotImplementedError

    def marginal_entropy(self, dist_info):
        """
        :return: the entropy of the mixture distribution averaged over all minibatch entries. Will return in the same
        shape as calling `:code:Distribution.entropy`
        """
        raise NotImplementedError

    def marginal_logli(self, x_var, dist_info):
        """
        :return: the log likelihood of the given variable under the mixture distribution averaged over all minibatch
        entries.
        """
        raise NotImplementedError

    def sample(self, dist_info):
        raise NotImplementedError

    def sample_prior(self, batch_size):
        return self.sample(self.prior_dist_info(batch_size))

    def prior_dist_info(self, batch_size):
        """
        :return: a dictionary containing distribution information about the standard prior distribution, the shape
                 of which is jointly decided by batch_size and self.dim
        """
        raise NotImplementedError


class Categorical(Distribution):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self.dim

    @property
    def effective_dim(self):
        return 1

    """
    dist_info : It gives pmf for a sample space, i.e prob of each data point belonging to each category
    x_var : actual categorical labels for each data point
    This method returns the log likelihood
    """
    def logli(self, x_var, dist_info):
        prob = dist_info["prob"]
        # its a vector with dimensions equal to the number of data points.
        return tf.reduce_sum(tf.log(prob + TINY) * x_var, reduction_indices=1)

    """
    Generates a prior categorical distribution by assigning
    equal prob to each class.
    Returns a categorical pmf of dimension (batch_size,dim)
    """
    def prior_dist_info(self, batch_size):
        prob = tf.ones([batch_size, self.dim]) * floatX(1.0 / self.dim)
        return dict(prob = prob)

    """
    We are trying to averageout log likelihood here
    tf.tile : It repeates a given input tensor multiple times to create a new tensor
    https://www.tensorflow.org/api_docs/python/tf/tile
    tf.reduce_mean : computes mean of a tensor along a given axis
    tf.stack : stacks tensors along a given axis
    """
    def marginal_logli(self, x_var, dist_info):
        prob = dist_info["prob"]
        avg_prob = tf.tile(
            input = tf.reduce_mean(prob, axis = 0, keep_dims=True),
            multiples = tf.stack([tf.shape(prob)[0], 1], axis = 0)
        )
        return self.logli(x_var, dict(prob=avg_prob))

    def nonreparam_logli(self, x_var, dist_info):
        return self.logli(x_var, dist_info)

    """
    :param p: left dist info
    :param q: right dist info
    :return: KL(p||q) : its a scalar
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    returns a vector of dimension dim that is sum of KL of all samples
    across different classes.
    """
    def kl(self, p, q):
        p_prob = p["prob"]
        q_prob = q["prob"]
        return tf.reduce_sum(
            p_prob * (tf.log(p_prob + TINY) - tf.log(q_prob + TINY)),
            reduction_indices=1
        )

    '''
    This function basically samples lables based on the logit matrix of the data.
    tf.multinomial : 2nd dimension of logit passed to multinomial defines
    the classes in multinomial. So here multinomial will generate one sample(class ids)
    for each row of logit matrix such that each sample will be distributed to
    one of the classes with probabilities defined in that row of the logit matrix.
    ids : its a vector of class ids. Dimension of ids = prob.shape[0]
    np.eye : identity matrix
    tf.nn.embedding_lookup : http://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do
    Dimension of retured tensor = prob.shape
    '''
    def sample(self, dist_info):
        prob = dist_info["prob"]
        ids = tf.multinomial(tf.log(prob + TINY), num_samples=1)[:, 0]
        onehot = tf.constant(np.eye(self.dim, dtype=np.float32))
        #embedding_lookup function retrieves rows of the onehot tensor.
        #The behavior is similar to using indexing with arrays in numpy
        return tf.nn.embedding_lookup(onehot, ids)

    """
    Generates softmax activations
    softmax = exp(logits) / reduce_sum(exp(logits), axis=1)
    """
    def activate_dist(self, flat_dist):
        return dict(prob=tf.nn.softmax(flat_dist))

    """
    It will return a vector of dimension = prob.shape[0] = number of rows of prob
    """
    def entropy(self, dist_info):
        prob = dist_info["prob"]
        return -tf.reduce_sum(prob * tf.log(prob + TINY), axis=1)

    """
    Similar concept as marginal logli
    """
    def marginal_entropy(self, dist_info):
        prob = dist_info["prob"]
        avg_prob = tf.tile(
            tf.reduce_mean(prob, reduction_indices=0, keep_dims=True),
            tf.pack([tf.shape(prob)[0], 1])
        )
        return self.entropy(dict(prob=avg_prob))

    @property
    def dist_info_keys(self):
        return ["prob"]


class Gaussian(Distribution):
    def __init__(self, dim, fix_std=False):
        self._dim = dim
        self._fix_std = fix_std

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self._dim * 2

    @property
    def effective_dim(self):
        return self._dim

    '''
    Will this also work for multivariate gaussian?
    '''
    def logli(self, x_var, dist_info):
        mean = dist_info["mean"]
        stddev = dist_info["stddev"]
        epsilon = (x_var - mean) / (stddev + TINY)
        return tf.reduce_sum(
            - 0.5 * np.log(2 * np.pi) - tf.log(stddev + TINY) - 0.5 * tf.square(epsilon),
            reduction_indices=1,
        )

    def prior_dist_info(self, batch_size):
        mean = tf.zeros([batch_size, self.dim])
        stddev = tf.ones([batch_size, self.dim]) # shouldn't the covariance matrix be a square matrix?
        return dict(mean=mean, stddev=stddev)

    def nonreparam_logli(self, x_var, dist_info):
        return tf.zeros_like(x_var[:, 0])

    '''
    https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    I think this will only work for univariate gaussians
    '''
    def kl(self, p, q):
        p_mean = p["mean"]
        p_stddev = p["stddev"]
        q_mean = q["mean"]
        q_stddev = q["stddev"]
        # means: (N*D)
        # std: (N*D)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) + ln(\sigma_2/\sigma_1)
        numerator = tf.square(p_mean - q_mean) + tf.square(p_stddev) - tf.square(q_stddev)
        denominator = 2. * tf.square(q_stddev)
        return tf.reduce_sum(
            numerator / (denominator + TINY) + tf.log(q_stddev + TINY) - tf.log(p_stddev + TINY),
            reduction_indices=1
        )

    def sample(self, dist_info):
        mean = dist_info["mean"]
        stddev = dist_info["stddev"]
        '''
        random_normal : It will return samples from a standard gaussian.
        Shape of the output tensor is given as input.
        '''
        epsilon = tf.random_normal(tf.shape(mean))
        return mean + epsilon * stddev

    @property
    def dist_info_keys(self):
        return ["mean", "stddev"]

    def activate_dist(self, flat_dist):
        mean = flat_dist[:, :self.dim]
        if self._fix_std:
            stddev = tf.ones_like(mean)
        else:
            stddev = tf.sqrt(tf.exp(flat_dist[:, self.dim:]))
        return dict(mean=mean, stddev=stddev)


class Uniform(Gaussian):
    """
    This distribution will sample prior data from a uniform distribution, but
    the prior and posterior are still modeled as a Gaussian
    """

    def kl_prior(self):
        raise NotImplementedError

    # def prior_dist_info(self, batch_size):
    #     raise NotImplementedError

    # def logli_prior(self, x_var):
    #     #
    #     raise NotImplementedError

    def sample_prior(self, batch_size):
        return tf.random_uniform([batch_size, self.dim], minval=-1., maxval=1.)


class Bernoulli(Distribution):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self._dim

    @property
    def effective_dim(self):
        return self._dim

    @property
    def dist_info_keys(self):
        return ["p"]

    def logli(self, x_var, dist_info):
        p = dist_info["p"]
        # its a vector with dimensions equal to the number of data points.
        return tf.reduce_sum(
            x_var * tf.log(p + TINY) + (1.0 - x_var) * tf.log(1.0 - p + TINY),
            reduction_indices=1
        )

    def nonreparam_logli(self, x_var, dist_info):
        return self.logli(x_var, dist_info)

    '''
    Returns distribution info map after applying sigmoid activation.
    '''
    def activate_dist(self, flat_dist):
        return dict(p=tf.nn.sigmoid(flat_dist))

    def sample(self, dist_info):
        p = dist_info["p"]
        '''
        less : returns truth value of (first element < second element)
        random_uniform : outputs random values from a uniform distribution [0,1)
        '''
        return tf.cast(tf.less(tf.random_uniform(p.get_shape()), p), tf.float32)

    def prior_dist_info(self, batch_size):
        return dict(p=0.5 * tf.ones([batch_size, self.dim]))

'''
I don't understand this class
'''
class MeanBernoulli(Bernoulli):
    """
    Behaves almost the same as the usual Bernoulli distribution, except that when sampling from it, directly
    return the mean instead of sampling binary values
    """

    def sample(self, dist_info):
        return dist_info["p"]

    def nonreparam_logli(self, x_var, dist_info):
        return tf.zeros_like(x_var[:, 0])


# class MeanCenteredUniform(MeanBernoulli):
#     """
#     Behaves almost the same as the usual Bernoulli distribution, except that when sampling from it, directly
#     return the mean instead of sampling binary values
#     """


class Product(Distribution):
    def __init__(self, dists):
        """
        :type dists: list[Distribution]
        """
        self._dists = dists

    @property
    def dists(self):
        return list(self._dists)

    @property
    def dim(self):
        return sum(x.dim for x in self.dists)

    @property
    def effective_dim(self):
        return sum(x.effective_dim for x in self.dists)

    @property
    def dims(self):
        return [x.dim for x in self.dists]

    """
    A vector of flat distribution values from all distributions
    """
    @property
    def dist_flat_dims(self):
        return [x.dist_flat_dim for x in self.dists]

    """
    Sum of all flat distributions. It returns a scalar
    """
    @property
    def dist_flat_dim(self):
        return sum(x.dist_flat_dim for x in self.dists)

    '''
    returns a list of distribution info keys.
    '''
    @property
    def dist_info_keys(self):
        ret = []
        for idx, dist in enumerate(self.dists):
            for k in dist.dist_info_keys:
                ret.append("id_%d_%s" % (idx, k))
        return ret

    """
    Splits distribution info into a list.
    """
    def split_dist_info(self, dist_info):
        ret = []
        for idx, dist in enumerate(self.dists):
            cur_dist_info = dict()
            for k in dist.dist_info_keys:
                cur_dist_info[k] = dist_info["id_%d_%s" % (idx, k)]
            ret.append(cur_dist_info)
        return ret

    def join_dist_infos(self, dist_infos):
        ret = dict()
        for idx, dist, dist_info_i in zip(itertools.count(), self.dists, dist_infos):
            for k in dist.dist_info_keys:
                ret["id_%d_%s" % (idx, k)] = dist_info_i[k]
        return ret

    """
    Split the tensor variable or value into per component.
    [0] + cum_dims : it means 0 gets added at the start of the list.
    Outputs a list with each element telling about number of latent
    variables from each distribution
    """
    def split_var(self, x):
        cum_dims = list(np.cumsum(self.dims))
        out = []
        for slice_from, slice_to, dist in zip([0] + cum_dims, cum_dims, self.dists):
            sliced = x[:, slice_from:slice_to]
            out.append(sliced)
        return out

    def join_vars(self, xs):
        """
        Join the per component tensor variables into a whole tensor
        """
        return tf.concat(axis = 1, values = xs)

    def split_dist_flat(self, dist_flat):
        """
        Split flat dist info into per component
        """
        cum_dims = list(np.cumsum(self.dist_flat_dims))
        out = []
        for slice_from, slice_to, dist in zip([0] + cum_dims, cum_dims, self.dists):
            sliced = dist_flat[:, slice_from:slice_to]
            out.append(sliced)
        return out

    def prior_dist_info(self, batch_size):
        ret = []
        for dist_i in self.dists:
            ret.append(dist_i.prior_dist_info(batch_size))
        return self.join_dist_infos(ret)

    def kl(self, p, q):
        ret = tf.constant(0.)
        for p_i, q_i, dist_i in zip(self.split_dist_info(p), self.split_dist_info(q), self.dists):
            ret += dist_i.kl(p_i, q_i)
        return ret

    def activate_dist(self, dist_flat):
        ret = dict()
        for idx, dist_flat_i, dist_i in zip(itertools.count(), self.split_dist_flat(dist_flat), self.dists):
            dist_info_i = dist_i.activate_dist(dist_flat_i)
            for k, v in dist_info_i.items():
                ret["id_%d_%s" % (idx, k)] = v
        return ret

    def sample(self, dist_info):
        ret = []
        for dist_info_i, dist_i in zip(self.split_dist_info(dist_info), self.dists):
            ret.append(tf.cast(dist_i.sample(dist_info_i), tf.float32))
        return tf.concat(axis = 1, values = ret)

    '''
    Creates prior samples from all the distributions that are used to define
    latent codes in a particular model.
    '''
    def sample_prior(self, batch_size):
        ret = []
        for dist_i in self.dists:
            ret.append(tf.cast(dist_i.sample_prior(batch_size), tf.float32))
        '''
        This will concat latent samples from different distributions to create
        one list of latent samples corresponding to each batch sample. So, output
        dimension would be somewhat [batch_size, dist_1.dim + dist_2.dim + ...]
        '''
        return tf.concat(axis = 1, values = ret)

    def logli(self, x_var, dist_info):
        ret = tf.constant(0.)
        for x_i, dist_info_i, dist_i in zip(self.split_var(x_var), self.split_dist_info(dist_info), self.dists):
            ret += dist_i.logli(x_i, dist_info_i)
        return ret

    def marginal_logli(self, x_var, dist_info):
        ret = tf.constant(0.)
        for x_i, dist_info_i, dist_i in zip(self.split_var(x_var), self.split_dist_info(dist_info), self.dists):
            ret += dist_i.marginal_logli(x_i, dist_info_i)
        return ret

    def entropy(self, dist_info):
        ret = tf.constant(0.)
        for dist_info_i, dist_i in zip(self.split_dist_info(dist_info), self.dists):
            ret += dist_i.entropy(dist_info_i)
        return ret

    def marginal_entropy(self, dist_info):
        ret = tf.constant(0.)
        for dist_info_i, dist_i in zip(self.split_dist_info(dist_info), self.dists):
            ret += dist_i.marginal_entropy(dist_info_i)
        return ret

    def nonreparam_logli(self, x_var, dist_info):
        ret = tf.constant(0.)
        for x_i, dist_info_i, dist_i in zip(self.split_var(x_var), self.split_dist_info(dist_info), self.dists):
            ret += dist_i.nonreparam_logli(x_i, dist_info_i)
        return ret