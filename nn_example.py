from nn_comps.nn_layer import NNLayer
from nn_comps.hidden_layer import HiddenLayer
from nn_comps.output_layer import OutputLayer
from nn_comps.functions import *

class TestNetwork:
    def __init__(self, input_count, hidden_count, output_count):
        self.hidden_layer = HiddenLayer(input_count, hidden_count, sigmoid_vectorized, 
                                        sigmoid_derv_vectorized)
        self.output_layer = OutputLayer(hidden_count, output_count, 
                            sigmoid_vectorized, sigmoid_derv_vectorized, 
                            error_func, error_derv)
        self.hidden_layer.set_next_layer(self.output_layer)

    def feed_forward(self, inputs):
        hidden_output = self.hidden_layer.forward_pass(inputs)
        return self.output_layer.forward_pass(hidden_output)

    def train(self, sample_input, sample_output):
        self.feed_forward(sample_input)
        self.output_layer.update_weight(sample_output, 0.3)
        self.hidden_layer.update_weight(0.3)


if __name__ == "__main__":
    import random
    import numpy as np
    # XOR test
    training_sets = [
     [[0, 0], [0]],
     [[0, 1], [1]],
     [[1, 0], [1]],
     [[1, 1], [0]]
    ]

    nn = TestNetwork(len(training_sets[0][0]), 5, len(training_sets[0][1]))
    errors = []
    for i in range(10000):
        sample_inputs, sample_outputs = random.choice(training_sets)
        nn.train(np.array(sample_inputs), np.array(sample_outputs))
        errors.append(nn.output_layer.get_error(np.array(sample_outputs)))

    import matplotlib.pyplot as plt
    plt.plot(errors, 'r.')
    plt.xlabel("epochs")
    plt.ylabel("error rate")
    plt.title("Error Rate Over Time on XOR Test")
    plt.show()
