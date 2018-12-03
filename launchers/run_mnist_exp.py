from infogan.helpers.distributions import Uniform, Categorical, Gaussian, MeanBernoulli
import os
from infogan.helpers.mnist import Mnist
from infogan.models.infogan_model import InfoGAN
from infogan.trainers.infogan_trainer import InfoGANTrainer
import dateutil.tz
import datetime

if __name__ == "__main__":

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M')
    mnist_data_path = "/home/sumit/Documents/repo/datasets/mnist"
    root_log_dir = "../logs/mnist"
    root_checkpoint_dir = "../ckt/mnist"
    batch_size = 128
    updates_per_epoch = 400
    max_epoch = 50
    snapshot_interval = 1000

    exp_name = "mnist_%s" % timestamp

    log_dir = os.path.join(root_log_dir, exp_name)

    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)
    load_checkpoint_path = os.path.join(root_checkpoint_dir,
                                        'mnist_2017_06_12_13_16/mnist_2017_06_12_13_16_1000.ckpt')

    os.makedirs(log_dir, exist_ok = True)
    os.makedirs(checkpoint_dir, exist_ok = True)

    dataset = Mnist(data_path = mnist_data_path)

    '''
    Here we are trying to create a latent specification by defining
    distributions for discrete and continuous latent variables. Also
    True/False indicate whether that distribution is regularized or not.
    Regularized latent variables are those latent variables against which
    we will find mutual information.
    '''
    latent_spec = [
        (Uniform(62), False),
        (Categorical(10), True),
        (Uniform(1, fix_std=True), True),
        (Uniform(1, fix_std=True), True)
    ]

    '''
    Here we are trying to define our model parameters
    '''
    model = InfoGAN(
        output_dist=MeanBernoulli(dataset.image_dim),
        latent_spec=latent_spec,
        batch_size=batch_size,
        image_shape=dataset.image_shape,
        network_type="mnist"
    )

    '''
    Here we are defining our trainer
    '''
    algo = InfoGANTrainer(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        exp_name=exp_name,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        max_epoch=max_epoch,
        updates_per_epoch=updates_per_epoch,
        info_reg_coeff=1.0,
        generator_learning_rate=1e-3,
        discriminator_learning_rate=2e-4,
        snapshot_interval = snapshot_interval,
        load_checkpoint_path = None
    )

    algo.train()
