from infogan.helpers.mnist import Mnist
from infogan.helpers.svhn import Svhn
from infogan.helpers.omniglot import Omniglot
from infogan.helpers.celebA import CelebA
from infogan.helpers.cifar10 import Cifar10
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread

mnist_data_path = "/home/sumit/Documents/repo/datasets/mnist"
svhn_data_path = "/home/sumit/Documents/repo/datasets/svhn"
omniglot_data_path = "/home/sumit/Documents/repo/datasets/omniglot/images_background"
celebA_data_path = "/home/sumit/Documents/repo/datasets/celebA/img_align_celeba"
cifar10_data_path = '/home/sumit/Documents/repo/datasets/cifar10/cifar10_train.p'


def load_mnist():
    mnist = Mnist(data_path = mnist_data_path)
    print("mnist train size ", mnist.train.num_samples)
    print("mnist validation size ", mnist.validation.num_samples)
    print("mnist test size ", mnist.test.num_samples)
    print("mnist image dim ", mnist.image_dim)
    print("mnist image shape ", mnist.image_shape)
    x, _ = mnist.train.next_random_batch(1)
    x = x.reshape(mnist.image_shape)
    plt.imshow(x.squeeze())
    plt.show()
    print("mnist sample dim", x.shape)


def load_svhn():
    svhn = Svhn(data_path = svhn_data_path)
    print("svhn train size ", svhn.train.num_samples)
    print("svhn validation size ", svhn.validation.num_samples)
    print("svhn test size ", svhn.test.num_samples)
    x, _ = svhn.train.next_random_batch(1)
    x = x.reshape(svhn.image_shape)
    plt.imshow(x)
    plt.show()
    print("svhn sample dim", x.shape)
    print(np.max(x))


def load_omniglot():
    omniglot = Omniglot(data_path = omniglot_data_path)
    print("svhn train size ", omniglot.train.num_samples)
    x, _ = omniglot.train.next_random_batch(1)
    x = x.reshape(omniglot.image_shape)
    print("svhn sample dim", x.shape)
    plt.imshow(x)
    plt.show()
    print(np.max(x))


def load_celebA():
    celeba = CelebA(data_path = celebA_data_path)
    print("celeba train size ", celeba.train.num_samples)
    print("celeba validation size ", celeba.validation.num_samples)
    print("celeba test size ", celeba.test.num_samples)
    train = celeba.train.images
    val = celeba.validation.images
    test = celeba.test.images
    np.save('/home/sumit/Documents/repo/datasets/celebA/celeba_train', train)
    np.save('/home/sumit/Documents/repo/datasets/celebA/celeba_val', val)
    np.save('/home/sumit/Documents/repo/datasets/celebA/celeba_test', test)
    x, _ = celeba.train.next_random_batch(1)
    x = x.reshape(celeba.image_shape)
    #x = imread('/home/sumit/Documents/repo/datasets/celebA/img_align_celeba/113633.jpg')
    print("celeba sample dim", x.shape)
    plt.imshow(x)
    plt.show()
    print(np.max(x))

def read_binary_data_file():
    val = np.load('/home/sumit/Documents/repo/datasets/celebA/celeba_val.npy')
    print(len(val))
    train = np.load('/home/sumit/Documents/repo/datasets/celebA/celeba_train.npy')
    print(len(train))


def load_cifar10():
    cifar10 = Cifar10(data_path = cifar10_data_path)
    print("cifar10 train size ", cifar10.train.num_samples)
    print("cifar10 validation size ", cifar10.validation.num_samples)
    print("cifar10 test size ", cifar10.test.num_samples)
    x, _ = cifar10.train.next_random_batch(1)
    x = x.reshape(cifar10.image_shape)
    plt.imshow(x, aspect = 'normal')
    plt.show()
    print("cifar10 sample dim", x.shape)
    print(np.max(x))


if __name__ == "__main__":
    #load_mnist()
    #load_svhn()
    #load_omniglot()
    #load_celebA()
    #read_binary_data_file()
    load_cifar10()

