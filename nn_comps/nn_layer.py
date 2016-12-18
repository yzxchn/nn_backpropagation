import numpy as np
import random

class NNLayer:
    def __init__(self, input_count, neuron_count, act_func, bias):
        """
        neuron_count: number of neurons in this layer
        act_func: the active function used in the neurons in this layer
        bias: a bias term applied to all neurons in this layer
        """
        self.weights = self._init_weights(input_count, neuron_count)
        self.act_func = act_func
        self.bias = bias if bias else random.random()
        self.outputs = None

    def _init_weights(self, input_count, neuron_count, init_func=None):
        """
        Given input count M and neuron count of N, randomly initialize a M x N 
        matrix of weights.
        init_func: a function that initializes a weight matrix, should take two
                    argument.
        """
        if init_func:
            weights = init_func(input_count, neuron_count)
            assert weights.shape == (input_count, neuron_count)
        else:
            weights = np.random.randn(input_count, neuron_count)
        return weights

    def forward_pass(self, inputs):
        """
        Get the forward-pass outputs of this layer.
        Equivalent to g(Wz + b), where g is the activation function of this 
        layer, z is the inputs to this layer, and b is the bias
        inputs: a numpy array

        returns: a numpy array
        """
        # input size must be equal to the number of rows in the weights matrix
        assert inputs.shape == (len(self.weights),)
        self.outputs = self.act_func(np.dot(inputs, self.weights) + self.bias)
        return self.outputs


if __name__ == "__main__":
    from functions import identity
    l = NNLayer(2, 2, identity, 1)
    print(l.weights)
    inputs = np.array([1, 2])
    print(l.forward_pass(inputs))
