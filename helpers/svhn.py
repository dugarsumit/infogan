from helpers.dataset import Dataset
import os
import urllib
import scipy.io as sio
import numpy as np

train_data_url = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
test_data_url = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
train_file_name = "train_32x32.mat"
test_file_name = "test_32x32.mat"

class Svhn(object):
    def __init__(self, data_path=None):
        self._data_path = data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        svhn_data = self.read_svhn_data(self.data_path)
        self._train = Dataset(svhn_data['x_train'], svhn_data['y_train'])
        self._test = Dataset(svhn_data['x_test'], svhn_data['y_test'])
        self._validation = Dataset(svhn_data['x_val'], svhn_data['y_val'])
        self._image_dim = 32 * 32 * 3
        self._image_shape = (32, 32, 3)

    def read_svhn_data(self, data_path):
        train_data_path = os.path.join(data_path, train_file_name)
        test_data_path = os.path.join(data_path, test_file_name)
        if not os.path.isfile(train_data_path):
            train_file = urllib.URLopener()
            train_file.retrieve(train_data_url, train_file_name)
        if not os.path.isfile(test_data_path):
            test_file = urllib.URLopener()
            test_file.retrieve(test_data_url, test_file_name)
        test_data = sio.loadmat(test_data_path)
        train_data = sio.loadmat(train_data_path)

        x_train_total = train_data['X']
        x_train_total = np.array(x_train_total).T
        x_train_total = np.array(x_train_total).swapaxes(1, 3)
        x_train_total = np.reshape(x_train_total, (x_train_total.shape[0], -1))
        x_train_total = x_train_total/255
        #mean_image = np.sum(x_train_total, axis = 0)/x_train_total.shape[0]
        #x_train_total = x_train_total - mean_image
        x_train = x_train_total[:60000,:]

        y_train_total = train_data['y']
        y_train_total[y_train_total == 10] = 0
        y_train = y_train_total[: 60000,:]

        x_val = x_train_total[60000:,:]
        y_val = y_train_total[60000:,:]

        x_test = test_data['X']
        x_test = np.array(x_test).T
        x_test = np.array(x_test).swapaxes(1, 3)
        x_test = np.reshape(x_test, (x_test.shape[0], -1))

        y_test = test_data['y']
        y_test[y_test == 10] = 0

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
