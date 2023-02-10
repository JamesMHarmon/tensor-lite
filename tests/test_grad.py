import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensor import Scalar, grad

def test_grad():
    a = Scalar(2.0)
    b = Scalar(3.0)
    c = a * b
    grad(c)

    assert a.grad == 3.0
    assert b.grad == 2.0

def test_grad_reused_scalars():
    a = Scalar(2.0)
    b = Scalar(3.0)
    c = (a + a + a) * b
    grad(c)

    assert a.grad == 9.0
    assert b.grad == 6.0