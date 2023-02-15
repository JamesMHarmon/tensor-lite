def main():
    from tensor import Scalar

    def mean(vals):
        return sum(vals) / len(vals)

    def linear_regression(theta, x):
        w, b = theta
        return [x * w + b for x in x]

    def mse_loss(y_targets, y_predictions):
        return mean([(y_prediction - y_target) ** 2 for y_target, y_prediction in zip(y_targets, y_predictions)])

    # Generate some sample data emulating the function y = 2x + 1
    x = [1, 2, 3, 4, 5]
    y = [3, 5, 7, 9, 11]

    # Initialize the parameters
    theta = [Scalar(0.0), Scalar(0.0)]

    # Set the learning rate and the number of iterations
    learning_rate = 0.05
    num_iterations = 1000

    # Train the model using gradient descent
    for _ in range(num_iterations):
        # Construct the computational graph of the linear regression model and predict the output values using the learned parameters.
        y_predictions = linear_regression(theta, x)
        loss = mse_loss(y, y_predictions)
        # Compute the gradient of the loss function with respect to the parameters
        loss.backward()
        # Update the learnable parameters
        for param in theta:
            param.data -= learning_rate * param.grad
            param.grad = 0.0

    # Print the learned parameters.
    print("Learned parameters: ", theta)

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    main()
