from helpers.dataset import Dataset
import os
import urllib
import scipy.io as sio
import numpy as np
from scipy.misc import imread, imresize

train_data = '/home/sumit/Documents/repo/datasets/celebA/celeba_train.npy'
val_data = '/home/sumit/Documents/repo/datasets/celebA/celeba_val.npy'
test_data = '/home/sumit/Documents/repo/datasets/celebA/celeba_test.npy'

class CelebA(object):
    def __init__(self, data_path=None):
        self._data_path = data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        celebA_data = self.read_celebA_data(self.data_path)
        self._train = Dataset(celebA_data['x_train'], celebA_data['y_train'])
        self._test = Dataset(celebA_data['x_test'], celebA_data['y_test'])
        self._validation = Dataset(celebA_data['x_val'], celebA_data['y_val'])
        self._image_dim = 32 * 32 * 3
        self._image_shape = (32, 32, 3)

    def read_celebA_data(self, data_path):
        if os.path.isfile(train_data) and os.path.isfile(val_data) and os.path.isfile(test_data):
            x_train = np.load(train_data)
            x_val = np.load(val_data)
            x_test = np.load(test_data)
        else:
            imgs = os.listdir(data_path)
            celebs = []
            for n, img in enumerate(imgs):
                if n < 40000:
                    img_path = os.path.join(data_path, img)
                    celeb = imread(img_path)
                    celeb = imresize(celeb, (32, 32, 3))
                    celebs.append(celeb)
                    print(n)

            x_train_total = np.array(celebs)
            x_train_total = np.reshape(x_train_total, (x_train_total.shape[0], -1))
            x_train_total = x_train_total/255
            #mean_image = np.sum(x_train_total, axis = 0)/x_train_total.shape[0]
            #x_train_total = x_train_total - mean_image
            x_train = x_train_total[:38000,:]
            x_val = x_train_total[38000:39000,:]
            x_test = x_train_total[39000:40000,:]
        return {'x_train': x_train,
                'y_train': None,
                'x_test': x_test,
                'y_test': None,
                'x_val': x_val,
                'y_val': None}

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
