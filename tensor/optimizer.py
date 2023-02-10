import math

class SGD():
    """
    Optimizes the loss using stochastic gradient descent. After each backward pass, step is called which multiplies the gradient for each parameter by the learning rate.
    After each step optimization, the gradients need to be zeroed before another backward pass is performed.
    """
    def __init__(self, parameters, learning_rate=0.001, momentum=0.9):
        self.parameters = parameters
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = [0.0 for _ in range(len(parameters))]

    def step(self):
        for i, param in enumerate(self.parameters):
            self.velocities[i] = self.velocities[i] * self.momentum - self.lr * param.grad
            param.data += self.velocities[i]

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0.0

class Adam(SGD):
    """ Adam optimization algorithm, a variant of stochastic gradient descent (SGD).

    Adam is an optimization algorithm that balances the need for fast convergence with the stability of the optimization process. It uses moving averages of the gradient and squared gradient to perform adaptive updates to the parameters.
    
    Parameters:
    parameters (list): A list of model parameters to be optimized.
    learning_rate (float): The learning rate for the optimization algorithm.
    beta1 (float, optional): The decay rate for the moving average of the gradient. Defaults to 0.9.
    beta2 (float, optional): The decay rate for the moving average of the squared gradient. Defaults to 0.999.
    eps (float, optional): A small value used to avoid division by zero. Defaults to 1e-8.
    
    Attributes:
    beta1 (float): The decay rate for the moving average of the gradient.
    beta2 (float): The decay rate for the moving average of the squared gradient.
    eps (float): A small value used to avoid division by zero.
    step_num (int): The current iteration number.
    means (list): A list of moving averages of the gradient for each parameter.
    variances (list): A list of moving averages of the squared gradient for each parameter.
    parameters (list): A list of model parameters to be optimized.
    learning_rate (float): The learning rate for the optimization algorithm.
    
    """
    def __init__(self, parameters, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_num = 0

        self.means = [0.0 for _ in range(len(parameters))]
        self.variances = [0.0 for _ in range(len(parameters))]
        
        super().__init__(parameters, learning_rate)

    def step(self):
        for i, param in enumerate(self.parameters):
            self.means[i] = self.beta1 * self.means[i] + (1.0 - self.beta1) * param.grad
            self.variances[i] = self.beta2 * self.variances[i] + (1.0 - self.beta2) * param.grad ** 2

            corrected_mean = self.means[i] / (1.0 - self.beta1 ** (self.step_num + 1))
            corrected_variance = self.variances[i] / (1.0 - self.beta2 ** (self.step_num + 1))

            param.data -= self.lr * corrected_mean / (math.sqrt(corrected_variance) + self.eps)

        self.step_num += 1
