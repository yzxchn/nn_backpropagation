import numpy as np
import random

class Layer:
    def __init__(self, neuron_count, act_func, bias):
        """
        neuron_count: number of neurons in this layer
        act_func: the active function used in the neurons in this layer
        bias: a bias term applied to all neurons in this layer
        """
        self.weights = self._init_weight(neuron_count)
        self.act_func = act_func
        self.bias = bias if bias else random.random()

    def _init_weights(self, neuron_count, init_func=None):
        """
        Given neuron count of N, randomly initialize a N x N matrix of weights.
        init_func: a function that initializes a weight matrix, should take one 
                    argument.
        """
        if init_func:
            weights = init_func(neuron_count)
            assert weights.shape == (neuron_count, neuron_count)
        else:
            weights = np.random.randn(neuron_count, neuron_count)
        return weights
