from infogan.helpers.distributions import Uniform, Categorical, Gaussian, MeanBernoulli
import os
from infogan.helpers.celebA import CelebA
from infogan.models.infogan_model import InfoGAN
from infogan.trainers.infogan_trainer import InfoGANTrainer
import dateutil.tz
import datetime

if __name__ == "__main__":

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M')
    celebA_data_path = "/home/sumit/Documents/repo/datasets/celebA/img_align_celeba"
    root_log_dir = "../logs/celebA"
    root_checkpoint_dir = "../ckt/celebA"
    batch_size = 128
    updates_per_epoch = 100
    max_epoch = 10
    snapshot_interval = 1000

    exp_name = "celebA_%s" % timestamp

    log_dir = os.path.join(root_log_dir, exp_name)

    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

    os.makedirs(log_dir, exist_ok = True)
    os.makedirs(checkpoint_dir, exist_ok = True)

    dataset = CelebA(data_path = celebA_data_path)

    '''
    Here we are trying to create a latent specification by defining
    distributions for discrete and continuous latent variables. Also
    True/False indicate whether that distribution is regularized or not.
    Regularized latent variables are those latent variables against which
    we will find mutual information.
    '''
    latent_spec = [
        (Uniform(128), False),
        (Categorical(10), True),
        (Categorical(10), True),
        (Categorical(10), True),
        (Categorical(10), True),
        (Categorical(10), True),
        (Categorical(10), True),
        (Categorical(10), True),
        (Categorical(10), True),
        (Categorical(10), True),
        (Categorical(10), True)
    ]

    '''
    Here we are trying to define our model parameters
    '''
    model = InfoGAN(
        output_dist=MeanBernoulli(dataset.image_dim),
        latent_spec=latent_spec,
        batch_size=batch_size,
        image_shape=dataset.image_shape,
        network_type="celebA"
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
        snapshot_interval = snapshot_interval
    )

    algo.train()
