# Reimplementation of InfoGAN in Tensorflow
The code in this repository was implemented on top the official [repository](https://github.com/openai/InfoGAN) of the paper -  [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets.](https://arxiv.org/abs/1606.03657)

The experiments from this work are summarized in the following poster.  
![](https://github.com/dugarsumit/infogan/blob/master/poster.png)

## Dependencies
* python==3.5.2
* tensorflow==1.0.1
Other dependencies can be installed using the requirements.txt file
```
user@host:infogan$ pip install -r requirements.txt
```

## Running Experiments
Experiments on different datasets(celebA, cifar10, mnist, omniglot, svhn) can be launched by using their respective launcher scripts. Make sure to set the correct data location path and other training parameters in the launcher scripts.
```
user@host:infogan$ python launchers/run_mnist_exp.py
```

