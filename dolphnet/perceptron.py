from .utils import dot

class Perceptron:
    def __init__(self, treshold, learning_rate):
        self.THETA = treshold
        self.ALPHA = learning_rate

    def add_inputs(self, inputs):
        self.inputs = inputs

    def add_weights(self, weights):
        self.weights = weights

    def step(self, y_in):
        if y_in > self.THETA:
            return 1
        elif -self.THETA <= y_in <= self.THETA:
            return 0
        else:
            return -1

    def calc_weight_delta(self, target, x):
        delta_w = [0, 0, 0]

        for i in range(len(delta_w)):
            delta_w[i] = self.ALPHA * target * x[i]

        return delta_w

    def weights_is_all_zero(self, matrix):
        weights = [w for row in matrix for w in row]
        return not any(n !=0 for n in weights)

    def train(self, verbose_output=False):
        epoch = 1
        done = False

        while True:
            print(f'Starting epoch {epoch} with weights {self.weights}...')
            print('')

            weight_delta_vec = []

            if verbose_output:
                print('| x1 | x2 | b | y-in          | y  | target | Δw1 | Δw2 | Δbi | w1(0) | w2(0) | b(0) |')
                print('|----|----|---|---------------|----|:------:|:---:|:---:|:---:|:-----:|:-----:|:----:|')

            for inp in self.inputs:
                x = inp.get("data")
                target = inp.get("target")

                y_in = dot(x, self.weights)
                y = self.step(y_in)

                weight_deltas = [0, 0, 0]
                if y != target:
                    weight_deltas = self.calc_weight_delta(target, x)

                    for i in range(len(self.weights)):
                        self.weights[i] += weight_deltas[i]

                weight_delta_vec.append(weight_deltas)

                to_print = f'| {x[0]} | {x[1]} | {x[2]} | {y_in} | {y} | {target} | {weight_deltas[0]} |'
                to_print += f'{weight_deltas[1]} | {weight_deltas[2]} | {self.weights[0]} | {self.weights[1]} |'
                to_print += f'{self.weights[2]} |'

                if verbose_output:
                    print(to_print)

            print('')
            done = self.weights_is_all_zero(weight_delta_vec)

            if done:
                print(f'Training done with epoch count: {epoch}\n')
                print(f'Training done with weights: {self.weights}\n')
                return

            epoch += 1
            
    def predict(self, inputs):
        y_in = dot(inputs, self.weights)
        return self.step(y_in)


