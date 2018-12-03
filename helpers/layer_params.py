class LayerParams(object):
    def __init__(self,
                 layer_input = None,
                 in_dim = None,
                 output_dim = None,
                 phase = None,
                 num_neurons = None,
                 epsilon = 1e-5,
                 momentum = 0.1,
                 name = None,
                 kernel_h = 5,
                 kernel_w = 5,
                 data_h = 2,
                 data_w = 2,
                 stddev = 0.02,
                 padding = 'SAME',
                 training = True,
                 leakiness = 0.1
                 ):
        self._layer_input = layer_input
        self._epsilon = epsilon
        self._momentum = momentum
        self._name = name
        self._in_dim = in_dim
        self._output_dim = output_dim
        self._kernel_h = kernel_h
        self._kernel_w = kernel_w
        self._data_h = data_h
        self._data_w = data_w
        self._stddev = stddev
        self._padding = padding
        self._strides = [self.data_h, self.data_w]
        self._training = training
        self._phase = phase
        self._num_neurons = num_neurons
        self._leakiness = leakiness

    def set_layer_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def layer_input(self):
        return self._layer_input

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def momentum(self):
        return self._momentum

    @property
    def name(self):
        return self._name

    @property
    def in_dim(self):
        return self._in_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def kernel_h(self):
        return self._kernel_h

    @property
    def kernel_w(self):
        return self._kernel_w

    @property
    def data_h(self):
        return self._data_h

    @property
    def data_w(self):
        return self._data_w

    @property
    def stddev(self):
        return self._stddev

    @property
    def padding(self):
        return self._padding

    @property
    def strides(self):
        return self._strides

    @property
    def training(self):
        return self._training

    @property
    def phase(self):
        return self._phase

    @property
    def num_neurons(self):
        return self._num_neurons

    @property
    def leakiness(self):
        return self._leakiness


    '''
    Setters
    '''

    @layer_input.setter
    def layer_input(self, layer_input):
        self._layer_input = layer_input

    @epsilon.setter
    def epsilon(self, epsilon):
        self._epsilon = epsilon

    @momentum.setter
    def momentum(self, momentum):
        self._momentum = momentum

    @name.setter
    def name(self, name):
        self._name = name

    @in_dim.setter
    def in_dim(self, in_dim):
        self._in_dim = in_dim

    @output_dim.setter
    def output_dim(self, output_dim):
        self._output_dim = output_dim

    @kernel_h.setter
    def kernel_h(self, kernel_h):
        self._kernel_h = kernel_h

    @kernel_w.setter
    def kernel_w(self, kernel_w):
        self._kernel_w = kernel_w

    @data_h.setter
    def data_h(self, data_h):
        self._data_h = data_h

    @data_w.setter
    def data_w(self, data_w):
        self._data_w = data_w

    @stddev.setter
    def stddev(self, stddev):
        self._stddev = stddev

    @padding.setter
    def padding(self, padding):
        self._padding = padding

    @strides.setter
    def strides(self, strides):
        self._strides = strides

    @training.setter
    def training(self, training):
        self._training = training

    @phase.setter
    def phase(self, phase):
        self._phase = phase

    @num_neurons.setter
    def num_neurons(self, num_neurons):
        self._num_neurons = num_neurons

    @leakiness.setter
    def leakiness(self, leakiness):
        self._leakiness = leakiness