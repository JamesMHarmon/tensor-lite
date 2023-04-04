def main():
    from tensor import Scalar, SGD
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
        """Create a dense layer for the network. Each output is calculated by performing a dot product between inputs and weights, adding a bias term, and applying an activation function."""
        initializer = xavier_uniform() if initializer is None else initializer
        weights = [initializer(num_inputs) for _ in range(num_outputs)]
        biases = [Scalar(0.0) for _ in range(num_outputs)]

        def forward_fn(inputs):
            matmul = lambda inputs, weights: (input * weight for input, weight in zip(inputs, weights))
            outputs = [sum(matmul(inputs, connections)) + bias for connections, bias in zip(weights, biases)]
            return activation(outputs)

        forward_fn.parameters = lambda: [param for weight in weights for param in weight] + biases

        return forward_fn

    def forward_pass(inputs, layers):
        """ Perform a forward pass through using the provided inputs and parameters. This also constructs a computation graph for backpropagation. """
        outputs = [Scalar(input) for input in inputs]

        for layer in layers:
            outputs = layer(outputs)

        return outputs

    def binary_cross_entropy_loss(target, predicted):
        """Common loss function for binary classification. Expects target values and predicted values to be in the range [0, 1]."""
        loss = sum(-(t * p.log() + (1 - t) * (1 - p).log()) for t, p in zip(target, predicted))

        return loss / len(predicted)

    def regularization_loss(reg_term, parameters):
        """ Calculates the L2 regularization loss for the provided set of neural network parameters. """
        return reg_term * sum(parameter ** 2 for parameter in parameters)

    def calculate_accuracy(size):
        X = [[ran.uniform(-2.0, 2.0) for _ in range(num_inputs)] for _ in range(size)]
        Y = [target_fn(*inputs) for inputs in X]

        # Convert the predictions to binary values (0 or 1) using a threshold of 0.5
        binary_predictions = [1.0 if forward_pass(inputs, layers)[0].data >= 0.5 else 0.0 for inputs in X]

        # Calculate the accuracy
        num_correct = sum(1 for binary_prediction, target in zip(binary_predictions, Y) if target == binary_prediction)
        accuracy = num_correct / len(Y)
        return accuracy

    """ Define the size of the neural net. Here we have an input of 2 neurons, followed by a hidden layer of 8 neurons, and then an output of 1 neuron(s). """
    num_inputs, hidden_layer_size, num_outputs = 2, 4, 1
    layers = [
        dense_layer(num_inputs, hidden_layer_size, activation=relu(), initializer=he_normal()),
        dense_layer(hidden_layer_size, hidden_layer_size, activation=tanh(), initializer=xavier_uniform()),
        dense_layer(hidden_layer_size, num_outputs, activation=sigmoid(), initializer=xavier_uniform()),
    ]
    parameters = [param for layer in layers for param in layer.parameters()]


    # Target function is the function that we are trying to create a function estimator for. Usually you would not have a literal function but instead would have data to sample from.
    # In this example target_fn, we want to determine if a point is within a circle with a radius of 1.414.
    target_fn = lambda x, y: float(x ** 2 + y ** 2 <= 2.0)

    num_steps = 5_000
    batch_size = 32
    learning_rate = 0.1
    regularization_term = 2e-3

    optimizer = SGD(parameters, learning_rate=learning_rate, momentum=0.5)

    # Perform n number of steps to optimize the network.
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        # For each step, perform a forward pass with the input sample data.
        X = [[ran.uniform(-2.0, 2.0) for _ in range(num_inputs)] for _ in range(batch_size)]
        Y = [[target_fn(*inputs)] for inputs in X]
        predicted = [forward_pass(inputs, layers) for inputs in X]

        # Calculate the loss between the target values and the predicted values. Use either mse_loss or binary_cross_entropy_loss here.
        losses = [binary_cross_entropy_loss(target, predicted) for target, predicted in zip(Y, predicted)]
        reg_loss = regularization_loss(regularization_term, parameters)
        total_loss = sum(losses) / batch_size + reg_loss
        total_loss.backward()
        optimizer.step()

        if step in [1, num_steps] or step % 100 == 0:
            accuracy = calculate_accuracy(100)
            print(f'Step={step}, Loss={total_loss.data - reg_loss.data:.04f}, RegLoss={reg_loss.data:0.4f}, Accuracy={accuracy:0.1%}')

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    main()
