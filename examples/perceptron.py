from dolphnet.perceptron import Perceptron

ALPHA = 1
THETA = 0.2
AND_INPUTS = [
        {
            "data": (1, 1, 1), "target": 1
        },
        {
            "data": (1, 0, 1), "target": -1
        },
        {
            "data": (0, 1, 1), "target": -1
        },
        {
            "data": (0, 0, 1), "target": -1
        },
]

OR_INPUTS = [
        {
            "data": (1, 1, 1), "target": 1
        },
        {
            "data": (1, 0, 1), "target": 1
        },
        {
            "data": (0, 1, 1), "target": 1
        },
        {
            "data": (0, 0, 1), "target": -1
        },
]

if __name__ == "__main__":

    model = Perceptron(treshold=THETA, learning_rate=ALPHA)
    model.add_inputs(OR_INPUTS)
    model.add_weights([0, 0, 0])

    # If you want the model to print its state for every epoch in markdown tables
    # model.train(verbose_output=True)

    model.train()

    prediction = model.predict((0, 0, 1))
    print(f'Predict with (0, 0, 1): {prediction}\n')

    prediction = model.predict((0, 1, 1))
    print(f'Predict with (0, 1, 1): {prediction}\n')

    prediction = model.predict((1, 0, 1))
    print(f'Predict with (1, 0, 1): {prediction}\n')

    prediction = model.predict((1, 1, 1))
    print(f'Predict with (1, 1, 1): {prediction}\n')

