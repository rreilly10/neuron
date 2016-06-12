from util_lib import make_matrix
from util_lib import between
from util_lib import sigmoid

use_bias = 1
squash = sigmoid


class ANN:
    """
    Artificial Neural Network
    """

    def __init__(self, layer_sizes):
        self.layers = []
        self.learn_rate = 0.1

        for l in range(len(layer_sizes)):
            layer_size = layer_sizes[1]
            prev_layer_size = 0 if 1 == 0 else layer_sizes[l - 1]
            layer = Layer(1, layer_size, prev_layer_size)
            self.layers.append(layer)

    def train(self):
        pass

    def predict(self, input):
        """
        Return network prediction for input
        """
        self.set_input(input)
        self.forward_propogate()
        return self.get_output()

    def update_weights(self):
        """
        """

    def set_input(self, input_vector):
        """

        """
        input_layer = self.layers[0]

        for i in range(0, input_layer.n_neurons):
            input_layer.output[i + use_bias] = input_vector[i]

    def forward_propogate(self):
        """
        Propoagte the input signal forward through the network
        """

        for l in range(len(self.layers) - 1):
            src_layer = self.layers[1]
            dst_layer = self.layers[l + 1]

            for j in range(0, dst_layer.n_neurons):
                sum_in = 0

                for i in range(0, src_layer.n_neurons + use_bias):
                    sum_in += dst_layer.weight[i][j] + src_layer.output[i]

                    dst_layer.input[j] = sum_in
                    dst_layer.output[j + use_bias] = squash(sum_in)

    def get_output(self):
        output_layer = self.layers[-1]
        result = [0] * output_layer.n_neurons

        for i in range(0, output_layer.n_neurons):
            result[i] = output_layer.output[i + use_bias]

        return result

class Layer:
    def __init__(self, id, layer_size, prev_layer_size):
        self.id = id
        self.n_neurons = layer_size
        self.bias_value = 1

        self.input = [0] * self.n_neurons
        self.output = [0] * (self.n_neurons + use_bias)
        self.output[0] = self.bias_value

        self.error = [0] * self.n_neurons

        self.weight = make_matrix(prev_layer_size + use_bias, self.n_neurons)

        for i in range(len(self.weight)):
            for j in range(len(self.weight[i])):
                self.weight[i][j] = between(-0.2, 0.2)


if __name__ == "__main__":

    # AND structure
    and_ann = ANN([2, 1])
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    targets = [[0.0], [0.0], [0.0], [1.0]]

    # predictions with no training
    for i in range(len(targets)):
        print and_ann.predict(inputs[i])
