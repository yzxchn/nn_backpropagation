from nn_layer import NNLayer
import random
import numpy as np

class HiddenLayer(NNLayer):
    def __init__(self, input_count, neuron_count, act_func, act_func_derv, 
                 weights=None, bias=random.random()):
        super(HiddenLayer, self).__init__(input_count, neuron_count, act_func,
                                        act_func_derv, weights, bias)
        self.next_layer = None

    def set_next_layer(self, layer):
        self.next_layer = layer

    def update_weight(self, learning_rate=0.1):
        self.deltas = self.calculate_delta()
        gradient = np.tile(self.deltas, (len(self.inputs), 1))*\
                    self.inputs[:, np.newaxis]
        assert self.weights.shape == gradient.shape
        self.weights -= gradient*learning_rate

    def calculate_delta(self):
        """
        Calculate the delta values associated with neurons in this hidden layer.
        equivalent to g'(a(j))*sum(delta(k)w(kj))
        """
        next_deltas = self.next_layer.get_deltas()
        next_weights = self.next_layer.get_weights()
        next_errors = np.sum(np.tile(next_deltas, (len(next_weights), 1))*\
                             next_weights, axis=1)
        return self.output_wrt_inputs()*next_errors

if __name__ == "__main__":
    from output_layer import OutputLayer
    from functions import *
    hl = HiddenLayer(2, 2, sigmoid_vectorized, sigmoid_derv_vectorized, 0)
    ol = OutputLayer(2, 2, sigmoid_vectorized, sigmoid_derv_vectorized, error_derv, 0)
    hl.set_next_layer(ol)
    inputs = np.array([-0.7, 0.7])
    targets = np.array([0.0, 0.5])
    for i in range(10000):
        print(hl.bias)
        hl.forward_pass(inputs)
        ol.forward_pass(hl.outputs)
        ol.update_weight(targets, 0.5)
        hl.update_weight(0.4)
        print(ol.outputs)
