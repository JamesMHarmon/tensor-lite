def main():
    from tensor import Scalar
    import random as ran

    def instantiate_scalars_normal(size):
        """ Instantiate a set of scalars sampled randomly from a normal distribution. """
        return [Scalar(ran.gauss(mu=0.0, sigma=1.0)) for _ in range(size)]

    def relu():
        """ Applies the relu function to each input. """
        return lambda inputs: [input.relu() for input in inputs]

    def tanh():
        """ Applies the tanh function to each input. """
        return lambda inputs: [input.tanh() for input in inputs]

    def sigmoid():
        """ Applies the sigmoid function to each input. """
        return lambda inputs: [input.sigmoid() for input in inputs]

    def dense_layer(num_inputs, num_outputs, activation=lambda inputs: inputs):
        """ Creates a dense layer for the network. Each output is calculated by multiplying the input values by the layer's weights, summing, adding a bias, and applying an activation. """
        weights, biases = [instantiate_scalars_normal(num_inputs) for _ in range(num_outputs)], instantiate_scalars_normal(num_outputs)
        forward_fn = lambda inputs: activation([sum((input * weight for input in inputs for weight in subWeights)) + bias for subWeights, bias in zip(weights, biases)])
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
        Gradients need to then be zerod and another backward pass performed.
        """
        def step():
            for param in parameters:
                param.data += -param.grad * learning_rate

        def zero():
            for param in parameters:
                param.grad = 0.0

        optimizer = lambda x: x
        optimizer.step = step
        optimizer.zero = zero
        return optimizer

    """ Define the size of the neural net. Here we have an input of 2 neurons, followed by a hidden layer of 8 neurons, and then an output of 1 neuron(s). """
    num_inputs, hidden_layer_size, num_outputs = 2, 8, 1
    layers = [
        dense_layer(num_inputs, hidden_layer_size, activation=tanh()),
        dense_layer(hidden_layer_size, hidden_layer_size, activation=tanh()),
        dense_layer(hidden_layer_size, num_outputs, activation=sigmoid()),
    ]
    parameters = [param for layer in layers for param in layer.parameters()]

    optimizer = sgd_optimizer(parameters, learning_rate=0.1)

    # Target function is the function that we are trying to create a function estimator for. Usually you would not have a literal function but instead would have data to sample from.
    # In this example target_fn, we want to determine if two points are within a circle with a radius of 2.0.
    target_fn = lambda x, y: float(x ** 2 + y ** 2 <= 2.0)

    num_steps = 500
    # Perform n number of steps to optimize the network.
    for step in range(num_steps):

        # For each step, perform a forward pass with the input sample data.
        inputs = [ran.uniform(-2.0, 2.0) for _ in range(num_inputs)]
        outputs = forward_pass(inputs, layers)

        # Calculate the loss between the actual values and the predicated values. The loss function used is MSE (Mean Squared Error) in this example.
        actuals = [target_fn(*inputs)]
        loss = sum((target - predicated) ** 2 for target, predicated in zip(actuals, outputs)) / len(outputs)
        print(f'Step={step + 1}, Loss={loss.data}')
        loss.backward()
        optimizer.step()
        optimizer.zero()

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    main()