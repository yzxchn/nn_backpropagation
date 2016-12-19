import numpy as np
import random

class NNLayer(object):
    def __init__(self, input_count, neuron_count, act_func, act_func_derv, 
                weights=None, bias=random.random()):
        """
        input_count: number of inputs from the previous layer
        neuron_count: number of neurons in this layer
        act_func: the activation function used in the neurons in this layer
        act_func_derv: the derivative of the activation function
        error_func_derv: the derivative of the error function
        bias: a bias term applied to all neurons in this layer
        """
        self.weights = weights
        if weights is None:
            self.weights = self._init_weights(input_count, neuron_count)
        self.act_func = act_func
        self.act_func_derv = act_func_derv
        self.bias = bias
        self.deltas = None
        self.inputs = None
        self.outputs = None

    def _init_weights(self, input_count, neuron_count, init_weights=None):
        """
        Given input count M and neuron count of N, randomly initialize a M x N 
        matrix of weights.
        """
        #if init_func:
        #    weights = init_func(input_count, neuron_count)
        #    assert weights.shape == (input_count, neuron_count)
        #else:
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
        self.inputs = inputs
        self.outputs = self.act_func(np.dot(inputs, self.weights) + self.bias)
        return self.outputs

    def output_wrt_inputs(self):
        """
        Calculate the partial derivative of the output with respect to the 
        inputs to this layer.
        equivalant to calculating g'(a)
        """
        return self.act_func_derv(np.dot(self.inputs, self.weights) + self.bias)

    def get_deltas(self):
        """
        return the delta values associated with neurons in this layer
        """
        return self.deltas

    def get_weights(self):
        """
        return the weights on edges coming to this layer
        """
        return self.weights


if __name__ == "__main__":
    from functions import identity
    l = NNLayer(2, 2, identity, 1)
    print(l.weights)
    inputs = np.array([1, 2])
    print(l.forward_pass(inputs))
