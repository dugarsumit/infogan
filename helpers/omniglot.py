from ..helpers.dataset import Dataset
import os
import scipy.io as sio
from scipy.misc import imread, imresize
import numpy as np

languages = ['Greek', 'N_Ko', 'Arcadian', 'Blackfoot_(Canadian_Aboriginal_Syllabics)',
             'Sanskrit', 'Mkhedruli_(Georgian)', 'Japanese_(katakana)', 'Malay_(Jawi_-_Arabic)',
             'Burmese_(Myanmar)', 'Braille', 'Anglo-Saxon_Futhorc', 'Korean',
             'Asomtavruli_(Georgian)', 'Grantha', 'Tagalog', 'Japanese_(hiragana)',
             'Gujarati', 'Balinese', 'Bengali', 'Armenian', 'Tifinagh',
             'Ojibwe_(Canadian_Aboriginal_Syllabics)', 'Syriac_(Estrangelo)',
             'Cyrillic', 'Hebrew', 'Latin', 'Futurama', 'Alphabet_of_the_Magi',
             'Early_Aramaic', 'Inuktitut_(Canadian_Aboriginal_Syllabics)']

include_languages = ['Greek', 'Sanskrit']


class Omniglot(object):
    def __init__(self, data_path=None):
        self._data_path = data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        omniglot_data = self.read_omniglot_data(self.data_path)
        self._train = Dataset(omniglot_data['x_train'], None)
        self._test = Dataset(None, None)
        self._validation = Dataset(None, None)
        self._image_dim =  28 * 28 * 1
        self._image_shape = (28, 28, 1)

    def read_omniglot_data(self, data_path):
        folders = os.listdir(data_path)
        characters = []
        for folder in folders:
            language = folder
            if language in include_languages:
                alphabets_folder_path = os.path.join(data_path, language)
                alphabet_folders = os.listdir(alphabets_folder_path)
                for alphabet_folder in alphabet_folders:
                    alphabet_path = os.path.join(alphabets_folder_path,alphabet_folder)
                    alphabet_imgs_path = os.listdir(alphabet_path)
                    for alphabet_img in alphabet_imgs_path:
                        alphabet_img_path = os.path.join(alphabet_path, alphabet_img)
                        alphabet = imread(alphabet_img_path)
                        alphabet = imresize(alphabet, (28, 28, 1))
                        characters.append(alphabet)

        x_train = np.array(characters)
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
        x_train = x_train/255
        mean_image = np.sum(x_train, axis = 0)/x_train.shape[0]
        x_train = x_train - mean_image

        return {'x_train': x_train}

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