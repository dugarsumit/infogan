from helpers.dataset import Dataset
from tensorflow.examples.tutorials import mnist
import os
import numpy as np

class Mnist(object):

    def __init__(self, data_path=None):
        self._data_path = data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        data_from_tf = mnist.input_data.read_data_sets(self.data_path)
        self._train = Dataset(data_from_tf.train.images, data_from_tf.train.labels)
        self._test = Dataset(data_from_tf.test.images, data_from_tf.test.labels)
        self._validation = Dataset(data_from_tf.validation.images, data_from_tf.validation.labels)
        self._image_dim = 28 * 28
        self._image_shape = (28, 28, 1)

    @property
    def data_path(self):
        return self._data_path

    @property
    def test(self):
        return self._test

    @property
    def validation(self):
        return self._validation

    @property
    def train(self):
        return self._train

    @property
    def image_dim(self):
        return self._image_dim

    @property
    def image_shape(self):
        return self._image_shape

    def inverse_transform(self, img):
        return (img+1.)/2.