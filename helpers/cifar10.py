from helpers.dataset import Dataset
from tensorflow.examples.tutorials import mnist
import os
import numpy as np
import pickle
import sys


class Cifar10(object):

    def __init__(self, data_path=None):
        self._data_path = data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        cifar10_data = self.read_cifar10_data(self.data_path)
        self._train = Dataset(cifar10_data['x_train'], cifar10_data['y_train'])
        self._test = Dataset(cifar10_data['x_test'], cifar10_data['y_test'])
        self._validation = Dataset(cifar10_data['x_val'], cifar10_data['y_val'])
        self._image_dim = 32 * 32 * 3
        self._image_shape = (32, 32, 3)

    def read_cifar10_data(self, data_path):
        num_training = 48000
        num_validation = 1000
        num_test = 1000
        print(data_path)
        print(sys.getdefaultencoding())
        with open(data_path, 'rb') as f:
            datadict = pickle.load(f, encoding='bytes')
            for key, value in datadict.items():
                print(key)
            print(b'labels' in datadict)
            X = np.array(datadict[b'data'])
            y = np.array(datadict[b'labels'])
            X = X.reshape(-1, 3, 32, 32).astype(np.float32)
            X = X.transpose(0, 2, 3, 1)

        X /= 255.0
        X = np.reshape(X, (X.shape[0], -1))
        # Normalize the data: subtract the mean image
        #mean_image = np.mean(X, axis = 0)
        #X -= mean_image
        # Subsample the data
        mask = range(num_training)
        x_train = X[mask]
        y_train = y[mask]
        mask = range(num_training, num_training + num_validation)
        x_val = X[mask]
        y_val = y[mask]
        mask = range(num_training + num_validation,
                     num_training + num_validation + num_test)
        x_test = X[mask]
        y_test = y[mask]
        return {'x_train': x_train,
                'y_train': y_train,
                'x_test': x_test,
                'y_test': y_test,
                'x_val': x_val,
                'y_val': y_val}

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