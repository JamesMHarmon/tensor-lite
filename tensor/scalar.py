import math
from typing import Union

from tensor.grad import grad

Scalarable = Union['Scalar', float]

class Scalar:
    """
    Represents a single scalar value in the tensor graph.

    Attributes:
        data (float): The value of the scalar.
        grad (float): The gradient of the scalar with respect to its inputs.
    """
    def __init__(self, data: float) -> None:
        """
        Initialize a new scalar value.

        Args:
            data (float): The value of the scalar.
        """
        self.data = data
        self.grad = 0.0

    def parents(self):
        """ Returns the parent nodes of this node in the tensor graph. """
        return ()

    def backward(self):
        """ Computes the gradient of this scalar with respect to its inputs. The gradient is accumulated in the `grad` attribute. """
        return grad(self)

    def _backward(self, grad):
        """ Computes the gradients to be passed backward to the parent nodes. """
        return ()

    def sigmoid(self) -> 'Scalar':
        return Sigmoid(self)

    def __add__(self, other: Scalarable) -> 'Scalar':
        return Add(self, other)

    def __mul__(self, other: Scalarable) -> 'Scalar':
        return Multiply(self, other)

    def __pow__(self,  exponent: float) -> 'Scalar':
        return Pow(self, exponent)

    def __radd__(self, other: Scalarable) -> 'Scalar':
        return self + other

    def __sub__(self, other: Scalarable) -> 'Scalar':
        return self + -other

    def __rsub__(self, other: Scalarable) -> 'Scalar':
        return -self + other

    def __neg__(self) -> 'Scalar':
        return -1.0 * self

    def __rmul__(self, other: Scalarable) -> 'Scalar':
        return self * other

    def __truediv__(self, other: Scalarable) -> 'Scalar':
        return self * other ** -1

    def __rtruediv__(self, other: Scalarable) -> 'Scalar':
        return other * self ** -1

    def _as_scalar(self, data: Scalarable) -> 'Scalar':
        return data if isinstance(data, Scalar) else Scalar(data)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(data={self.data!r}, grad={self.grad!r})'

class Add(Scalar):
    def __init__(self, left: Scalarable, right: Scalarable):
        self._left = self._as_scalar(left)
        self._right = self._as_scalar(right)
        sum = self._left.data + self._right.data

        super().__init__(sum)

    def _backward(self, grad: float):
        return (grad, grad)

    def parents(self):
        return (self._left, self._right)

class Multiply(Scalar):
    def __init__(self, left: Scalarable, right: Scalarable):
        self._left = self._as_scalar(left)
        self._right = self._as_scalar(right)
        product = self._left.data * self._right.data

        super().__init__(product)

    def _backward(self, grad: float):
        return (self._right.data * grad, self._left.data * grad)

    def parents(self):
        return (self._left, self._right)

class Pow(Scalar):
    def __init__(self, base: Scalarable, exponent: Scalarable):
        self._base = self._as_scalar(base)
        self._exp = self._as_scalar(exponent)

        pow = self._base.data ** self._exp.data

        super().__init__(pow)

    def _backward(self, grad: float):
        return (
            grad * (self._exp.data * self._base.data ** (self._exp.data - 1)),
            grad * (self._base.data ** self._exp.data * math.log(self._base.data))
        )

    def parents(self):
        return (self._base, self._exp)

class Sigmoid(Scalar):
    def __init__(self, value):
        self._value = self._as_scalar(value)
        sigmoid = 1 / (1 + math.exp(-self._value.data))

        super().__init__(sigmoid)

    def _backward(self, grad: float):
        return (grad * (self.data * (1 - self.data)), )

    def parents(self):
        return (self._value, )