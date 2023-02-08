def main():
    from tensor import Scalar
    import numpy as np

    def dense_layer(num_inputs, num_outputs, activation=lambda x: x):
        """ Creates a dense layer for the network. Each output is calculated by multiplying the input values by the layer's weights, summing, adding a bias, and applying an activation. """
        # Instantiate the parameters for the net by providing a list of input and output sizes. This will create the initial weights and bias values by sampling randomly from a normal distribution.
        weights, biases = np.random.normal(size=(num_inputs, num_outputs)), np.random.normal(size=(num_outputs))

        return lambda inputs: [activation(sum((input * weight for input in inputs for weight in subWeights)) + bias) for subWeights, bias in zip(weights, biases)]

    def forward_pass(inputs, layers):
        """ Perform a forward pass over the net by providing the inputs as well as the parameters. Along with computing a forward pass, this will also construct a graph of the net that can be used in back propagation. """
        outputs = [Scalar(input) for input in inputs]

        for layer in layers:
            outputs = layer(outputs)

        return outputs

    """ Define the size of the neural net. Here we have an input of 2 neurons, followed by a hidden layer of 8 neurons, and then an output of 2 neurons. """
    num_inputs, hidden_layer_size, num_outputs = 2, 8, 2
    layers = [
        dense_layer(num_inputs, hidden_layer_size, activation=lambda x: x.relu()),
        dense_layer(hidden_layer_size, hidden_layer_size, activation=lambda x: x.relu()),
        dense_layer(hidden_layer_size, num_outputs, activation=lambda x: x.sigmoid())
    ]

    # Target function is the function that we are trying to create a function estimator for. Usually you would not have a literal function but instead would have data to sample from.
    target_fn = lambda x, y, b: x * y + b

    num_steps = 100
    # Perform n number of steps to optimize the network.
    for step in range(num_steps):

        # For each step, perform a forward pass with the input sample data.
        outputs = forward_pass(inputs, layers)

        # Calculate the loss between the actual values and the predicated values. The loss function used is MSE (Mean Squared Error) in this example.
        loss = sum((target - predicated) ** 2 for target, predicated in zip(actuals, outputs)) / len(outputs)
        print(f'Step={step}, Loss={loss.data}')
        loss.backward()

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    main()
