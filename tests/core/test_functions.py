import numpy as np

from core import Sigmoid, Softmax
from core.functions import CrossEntropyLoss


def test_sigmoid_function():
    x = np.array([-0.34, -0.66, 1.757, -3.909, 1.046])

    actual = Sigmoid.apply(x)
    expected = np.array([0.4158, 0.3407, 0.8528, 0.01967, 0.734])
    np.testing.assert_allclose(actual, expected, rtol=1e-2)


def test_sigmoid_derivatives():
    sig = Sigmoid()
    x = np.array([-0.34, -0.66, 1.757, -3.909, 1.046])
    expected = np.array([0.2429, 0.2246, 0.1255, 0.01928, 0.1924])
    actual = sig.derivative(x)
    np.testing.assert_allclose(actual, expected, rtol=1e-2)


def test_softmax():
    x = np.array([1, 2, 3])
    expected = np.array([0.09003057317038, 0.2447284710548, 0.66524095577482])
    np.testing.assert_allclose(Softmax.apply(x), expected)


def test_softmax_derivatives():
    softmax = Softmax()
    x = np.array([0.26894142, 0.73105858])
    expected = np.array(
        np.array([[0.19661193, -0.19661193], [-0.19661193, 0.19661193]])
    )

    np.testing.assert_allclose(softmax.derivative(x), expected)
