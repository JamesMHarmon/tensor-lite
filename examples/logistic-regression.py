def main():
    from tensor import Scalar

    def logistic_regression(theta, x):
        w, b = theta
        z = x * w + b
        return z.sigmoid()

    def binary_cross_entropy_loss(y_target, y_prediction):
        return -(y_target * y_prediction.log() + (1 - y_target) * (1 - y_prediction).log())

    # Generate some sample data
    x = [1, 2, 3, 4, 5]
    y = [0, 0, 1, 1, 1]

    # Initialize the parameters
    theta = [Scalar(0.0), Scalar(0.0)]

    # Set the learning rate and the number of iterations
    learning_rate = 0.01
    num_iterations = 1000

    # Train the model using gradient descent
    for i in range(num_iterations):
        # Construct the computational graph of the logistic regression model and predict the output values using the learned parameters.
        y_predictions = [logistic_regression(theta, x) for x in x]
        loss = [binary_cross_entropy_loss(y, y_pred) for y, y_pred in zip(y, y_predictions)]
        total_loss = sum(loss) / len(loss)
        # Compute the gradient of the loss function with respect to the parameters
        total_loss.backward()
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
