def main():
    from collections import namedtuple
    from tensor import Scalar
    import random as ran
    import math

    def he_normal():
        """ Instantiate a set of scalars sampled randomly from a normal distribution. He is the preferred weight initialization when using relu activations. """
        return lambda size: [Scalar(ran.gauss(mu=0.0, sigma=math.sqrt(2 / size))) for _ in range(size)]

    def xavier_uniform():
        """ Instantiate a set of scalars sampled randomly from a uniform distribution. Xavier is the preferred weight initialization when using tanh and/or sigmoid activations. """
        return lambda size: [Scalar(ran.uniform(-1 / math.sqrt(size), 1 / math.sqrt(size))) for _ in range(size)]

    def relu():
        """ Applies the relu function to each input. """
        return lambda inputs: [input.relu() for input in inputs]

    def tanh():
        """ Applies the tanh function to each input. """
        return lambda inputs: [input.tanh() for input in inputs]

    def sigmoid():
        """ Applies the sigmoid function to each input. """
        return lambda inputs: [input.sigmoid() for input in inputs]

    def dense_layer(num_inputs, num_outputs, activation=lambda inputs: inputs, initializer=None):
        """ Creates a dense layer for the network. Each output is calculated by multiplying the input values by the layer's weights via dot product, summing, adding a bias, and applying an activation. """
        initializer = xavier_uniform() if initializer is None else initializer
        weights, biases = [initializer(num_inputs) for _ in range(num_outputs)], initializer(num_outputs)
        forward_fn = lambda inputs: activation([sum(input * weight for input, weight in zip(inputs, subWeights)) + bias for subWeights, bias in zip(weights, biases)])
        forward_fn.parameters = lambda: [weight for subWeights in weights for weight in subWeights] + biases

        return forward_fn

    def forward_pass(inputs, layers):
        """ Perform a forward pass over the net by providing the inputs as well as the parameters. Along with computing a forward pass, this will also construct a graph of the net that can be used in back propagation. """
        outputs = [Scalar(input) for input in inputs]

        for layer in layers:
            outputs = layer(outputs)

        return outputs

    def sgd_optimizer(parameters, learning_rate):
        """
        Optimizes the loss using stochastic gradient descent. After each backward pass, step is called which multiplies the gradient for each parameter by the learning rate.
        After each step optimization, the gradients need to be zeroed before another backward pass is performed.
        """
        def step():
            for param in parameters:
                param.data += -param.grad * learning_rate

        def zero():
            for param in parameters:
                param.grad = 0.0

        optimizer = namedtuple('SGD', ['step', 'zero'])
        optimizer.step = step
        optimizer.zero = zero
        return optimizer

    def binary_cross_entropy_loss(target, predicted):
        """ Common loss function for binary classification. Expects target values and predicted values to be in the range 0-1. """
        return -sum((t * p.log() + (1 - t) * (1 - p).log()) / len(predicted) for t, p in zip(target, predicted))

    def mse_loss(target, predicted):
        """ Measures the mean squared error (squared L2 norm) between each target and predicted element """
        return sum((t - p) ** 2 / len(predicted) for t, p in zip(target, predicted))

    """ Define the size of the neural net. Here we have an input of 2 neurons, followed by a hidden layer of 8 neurons, and then an output of 1 neuron(s). """
    num_inputs, hidden_layer_size, num_outputs = 2, 4, 1
    layers = [
        dense_layer(num_inputs, hidden_layer_size, activation=relu(), initializer=he_normal()),
        dense_layer(hidden_layer_size, hidden_layer_size, activation=tanh(), initializer=xavier_uniform()),
        dense_layer(hidden_layer_size, num_outputs, activation=sigmoid(), initializer=xavier_uniform()),
    ]
    parameters = [param for layer in layers for param in layer.parameters()]

    optimizer = sgd_optimizer(parameters, learning_rate=0.1)

    # Target function is the function that we are trying to create a function estimator for. Usually you would not have a literal function but instead would have data to sample from.
    # In this example target_fn, we want to determine if a point is within a circle with a radius of 2.0.
    target_fn = lambda x, y: float(x ** 2 + y ** 2 <= 2.0)

    num_steps = 5_000
    batch_size = 32

    # Perform n number of steps to optimize the network.
    for step in range(num_steps):

        # For each step, perform a forward pass with the input sample data.
        X = [[ran.uniform(-2.0, 2.0) for _ in range(num_inputs)] for _ in range(batch_size)]
        Y = [[target_fn(*inputs)] for inputs in X]
        predicted = [forward_pass(inputs, layers) for inputs in X]

        # Calculate the loss between the target values and the predicted values. Use either mse_loss or binary_cross_entropy_loss here.
        losses = [binary_cross_entropy_loss(target, predicted) / batch_size for target, predicted in zip(Y, predicted)]

        for loss in losses:
            loss.backward()

        optimizer.step()
        optimizer.zero()

        if step % 100 == 0:
            print(f'Step={step}, Loss={sum(loss.data for loss in losses): .04f}')

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    main()
