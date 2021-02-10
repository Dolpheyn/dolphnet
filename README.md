# Dolphnet :dolphin:

Some ML models' implementation for learning purposes.

I plan to make each model output its state verbosely for observablility
reasons. I wanted to understand how the weights and biases transforms in the
training process.

If you want the verbose output, set the parameter `verbose_output=True` when
calling the train method on any model.

## Usage

* Perceptron

```bash
$ python3 -m examples.perceptron
```

Send it to a file or clipboard and paste to [dillinger.io](https://dillinger.io/) to see the rendered
markdown.

```bash
$ python3 -m examples.perceptron > output.md
OR
$ python3 -m examples.perceptron > xclip -selection c
```

### Example Rendered Outputs

1. [OR operator - Binary Input, Bipolar Target](examples/perceptron_or.md)
1. [AND operator - Binary Input, Bipolar Target](examples/perceptron_and.md)
