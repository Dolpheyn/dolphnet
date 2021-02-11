from random import random
from .utils import dot, sigmoid

class BPNN:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        """
        Initialize a network with random weights and a single bias for 
        hidden and output layers.

        Parameters
        ----------
        n_inputs : int
            The number of neurons in the input layer
        n_hidden : int
            The number of neurons in the hidden layer
        n_outputs : int
            The number of neurons in the output layer
        """
        self.network = list()

        hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        self.network.append(hidden_layer)

        output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        self.network.append(output_layer)

    def __forward(self, row):
        for layer in self.network:
            prev_layer_outputs = []
            for neuron in layer:
                neuron['output'] = sigmoid(dot(row, neuron['weights']))
                prev_layer_outputs.append(neuron['output'])
            row = prev_layer_outputs

    def __calculate_output_layer_deltas(self, layer):
        for j, neuron in enumerate(layer):
            t = self.y_train[j]
            y = neuron['output']

            neuron['delta'] = (t - y) * y * (1 - y)

    def __calculate_non_output_layer_deltas(self, layer, prev_layer):
        deltas = list()

        for i_neuron, neuron in enumerate(layer):
            prev_to_cur_neuron_weights = [prev_neuron['weights'][i_neuron] for prev_neuron in prev_layer]
            prev_neuron_deltas = [prev_neuron['delta'] for prev_neuron in prev_layer]

            delta_in = dot(prev_to_cur_neuron_weights, prev_neuron_deltas)

            neuron['delta'] = delta_in * neuron['output'] * (1 - neuron['output'])

    def __backpropagate_errors(self):
        for i, layer in reversed(enumerate(self.network)):
            # output layer is index 0 because we're
            # enumerating the network in reversed order.
            n_layers = len(self.network)
            is_output_layer = i == (n_layers - 1)

            if is_output_layer:
                self.__calculate_output_layer_deltas(layer)
            else:
                prev_layer = self.network[i + 1]
                self.__calculate_non_output_layer_deltas(layer, prev_layer)

    def __update_weights(self, row):
        for i, layer in enumerate(self.network):
            is_input_layer = i == 0

            inputs = row

            if not is_input_layer:
                prev_layer = self.network[i - 1]
                inputs = [neuron['output'] for neuron in prev_layer]

            for neuron in layer:
                for j, input_val in enumerate(inputs):
                    weight_change = neuron['delta'] * self.ALPHA * input_val
                    neuron['weights'][j] += weight_change

    def fit(self, X_train, y_train, learning_rate, n_epoch):
        self.X_train = X_train
        self.y_train = y_train
        self.ALPHA = learning_rate

        self.sum_squared_errors = list()
        for epoch in range(n_epoch):
            sum_squared_error = 0.0
            for row in X_train:
                self.__forward(row)
                self.__backpropagate_errors()
                self.__update_weights(row)

                output_layer = self.network[-1]
                outputs = [neuron['output'] for neuron in output_layer]
                sum_squared_error += sum([(self.y_train[i] - outputs[i])**2 for i in range(len(self.y_train))])

            self.sum_squared_errors.append(sum_squared_error)

    def predict(self, inputs):
        self.__forward(inputs)
        output_layer = self.network[-1]
        return [neuron['output'] for neuron in output_layer]
