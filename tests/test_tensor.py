import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensor import Scalar, grad

def test_grad_add():
    a = Scalar(2.0)
    b = Scalar(3.0)
    c = (a + b) * 2.0
    c.backward()

    assert a.grad == 2.0
    assert b.grad == 2.0

def test_grad_add_non_scalar_rhs():
    a = Scalar(2.0)
    c = (a + 3.0) * 2.0
    c.backward()

    assert a.grad == 2.0

def test_grad_add_non_scalar_lhs():
    a = Scalar(2.0)
    c = (3.0 + a) * 2.0
    c.backward()

    assert a.grad == 2.0

def test_grad_subtract():
    a = Scalar(2.0)
    c = (a - 3.0) * 2.0
    c.backward()

    assert a.grad == 2.0

def test_grad_pow():
    a = Scalar(2.0)
    b = a ** 2
    b.backward()

    assert a.grad == 4.0

def test_grad_exp():
    a = Scalar(2.0)
    b = Scalar(3.0117) ** a
    b.backward()

    assert_appx_eq(a.grad, 10.0, 4)

def test_grad_log():
    a = Scalar(2.0)
    b = a.log()
    b.backward()

    assert_appx_eq(b.data, 0.6931, 4)
    assert_appx_eq(a.grad, 0.5, 4)

def test_grad_sigmoid():
    a = Scalar(0.5)
    b = a.sigmoid()
    b.backward()

    assert_appx_eq(b.data, 0.6225, 4)
    assert_appx_eq(a.grad, 0.2350, 4)

def test_grad_sigmoid_2():
    a = Scalar(0.0)
    b = a.sigmoid()
    b.backward()

    assert_appx_eq(b.data, 0.5, 4)
    assert_appx_eq(a.grad, 0.25, 4)

def test_grad_sigmoid_3():
    a = Scalar(3.0)
    b = a.sigmoid()
    b.backward()

    assert_appx_eq(b.data, 0.9526, 4)
    assert_appx_eq(a.grad, 0.04518, 4)

def test_grad_tanh():
    a = Scalar(0.0)
    b = a.tanh()
    b.backward()

    assert_appx_eq(b.data, 0.0, 4)
    assert_appx_eq(a.grad, 1.0, 4)

def test_grad_tanh_2():
    a = Scalar(1.0)
    b = a.tanh()
    b.backward()

    assert_appx_eq(b.data, 0.7615, 3)
    assert_appx_eq(a.grad, 0.4199, 3)

def test_grad_relu():
    a = Scalar(1.0)
    b = a.relu() * 2.0
    b.backward()

    assert_appx_eq(b.data, 2.0, 3)
    assert_appx_eq(a.grad, 2.0, 3)

def test_grad_relu_neg():
    a = Scalar(-1.0)
    b = a.relu() * 2.0
    b.backward()

    assert_appx_eq(b.data, 0.0, 3)
    assert_appx_eq(a.grad, 0.0, 3)

def assert_appx_eq(a, b, digits):
    epsilon = 10 ** -digits
    diff = abs(a) if b == 0 else abs(a / b - 1)
    assert diff < epsilon