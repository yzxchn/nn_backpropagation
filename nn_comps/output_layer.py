from nn_layer import NNLayer
import random
import numpy as np

class OutputLayer(NNLayer):
    def __init__(self, input_count, neuron_count, act_func, act_func_derv, 
                error_func, error_derv, weights=None, bias=random.random()):
        """
        error_derv: the partial derivative of the error function
        """
        super(OutputLayer, self).__init__(input_count, neuron_count, act_func, 
                                    act_func_derv, weights, bias)
        self.error_func = error_func
        self.error_derv = error_derv

    def update_weight(self, target, learning_rate=0.1):
        self.deltas = self.calculate_delta(target)
        gradient = np.tile(self.deltas, (len(self.inputs), 1))*\
                   self.inputs[:, np.newaxis]
        assert self.weights.shape == gradient.shape
        self.weights -= gradient*learning_rate

    def calculate_delta(self, target):
        """
        Calculate the delta values for this layer
        Equivalent to g'(a)*(d(E)/d(y))
        """
        return self.output_wrt_inputs()*self.error_wrt_output(target)


    def error_wrt_output(self, target):
        """
        Calculate the partial derivative of the total error with respect to 
        the output of this layer.
        d(E)/d(y)
        """
        return self.error_derv(target, self.outputs)

    def get_error(self, target):
        """
        Calculate the error between the output and the target output, using 
        error function provided during initialization
        """
        return self.error_func(target, self.outputs) 

if __name__ == "__main__":
    from functions import *
    ol = OutputLayer(2, 2, sigmoid_vectorized, sigmoid_derv_vectorized, error_derv, 0)
    inputs = np.array([0.1, 0.7])
    for i in range(1000):
        ol.forward_pass(inputs)
        print(ol.outputs)
        ol.update_weight(np.array([0.5, -0.5]), 0.5)
