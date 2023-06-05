import numpy as np
import pytest

from core import Dropout, LinearLayer, Sigmoid


def test_linear_layer_z_a():
    layer = LinearLayer(in_dimension=2, out_dimension=3, activation_fn=Sigmoid())
    assert layer.shape() == (3, 2)
    assert layer.weights.shape == (3, 3)

    x = np.array([1, 2])
    x_with_bias = np.concatenate((np.array([1]), x))

    expected = (layer.weights @ x_with_bias.T).T
    output = layer.z(x)
    np.testing.assert_allclose(output, expected)

    np.testing.assert_allclose(layer.forward(x), Sigmoid.apply(expected))


def test_linear_layer_forward_example():
    input_vector = np.array([1, -2])

    layer = LinearLayer(in_dimension=2, out_dimension=2, activation_fn=Sigmoid)
    layer.weights = np.array([2, -1, 0.67, -3, 1, -0.67]).reshape(2, 3)

    output = layer.forward(input_vector)
    ground_truth = np.array([0.4158, 0.3407])

    np.testing.assert_allclose(output, ground_truth, rtol=1e-3)


def test_linear_layer_backward_example_d():
    input_vector = np.array([0.8528, 0.01967])
    gradient_vector = np.array([3.846])

    layer = LinearLayer(in_dimension=2, out_dimension=1, activation_fn=Sigmoid)
    layer.weights = np.array([0.5, 0.67, -1.3]).reshape(1, 3)

    layer.backward(input_vector=input_vector, backward_vector=gradient_vector)

    ground_truth = np.array([0.74, 0.631, 0.01455]).reshape(1, -1)
    np.testing.assert_allclose(layer.gradient, ground_truth, rtol=1e-3)


def test_linear_layer_backward_example_c():
    input_vector = np.array([0.4158, 0.3407])
    gradient_vector = np.array([0.67, -1.3]) * 3.846 * 0.1924

    layer = LinearLayer(in_dimension=2, out_dimension=2, activation_fn=Sigmoid)
    layer.weights = np.array([1, 1, 1, -4, -0.33, 0.67]).reshape(2, 3)

    layer.backward(input_vector=input_vector, backward_vector=gradient_vector)
    ground_truth = np.array(
        [[0.06224, 0.02588, 0.02121], [-0.01855, -0.007712, -0.00632]]
    )
    np.testing.assert_allclose(layer.gradient, ground_truth, rtol=1e-3)

    # # Batch size 2 example


def test_linear_layer_backward_example_b():
    input_vector = np.array([1, -2])
    gradient_vector = np.array([0.06836706, 0.04982015])

    layer = LinearLayer(in_dimension=2, out_dimension=2, activation_fn=Sigmoid())
    layer.weights = np.array([2, -1, 0.67, -3, 1, -0.67]).reshape(2, 3)

    layer.backward(input_vector=input_vector, backward_vector=gradient_vector)
    ground_truth = np.array(
        [[0.01661, 0.01661, -0.03321], [0.01119, 0.01119, -0.02238]]
    )
    np.testing.assert_allclose(layer.gradient, ground_truth, atol=1e-2)


def test_dropout_layer_forward():
    dropout = Dropout(0.5)
    input_vector = np.random.rand(5).reshape(1, -1)
    output_vector = dropout.forward(input_vector, train=True)
    np.testing.assert_equal(output_vector, input_vector * dropout.mask)

    input_vector = np.random.rand(10)
    output_vector = dropout.forward(input_vector, train=False)
    np.testing.assert_equal(output_vector, input_vector * 0.5)


def test_dropout_layer_backward():
    dropout = Dropout(0.5)
    input_vector = np.random.rand(5).reshape(1, -1)
    gradient_vector = np.random.rand(5)
    with pytest.raises(ValueError):
        dropout.backward(input_vector, gradient_vector)

    with pytest.raises(ValueError):
        dropout.forward(input_vector, train=False)
        dropout.backward(input_vector, gradient_vector)

    dropout.forward(input_vector, train=True)
    new_gradient_vector = dropout.backward(input_vector, gradient_vector)
    np.testing.assert_equal(new_gradient_vector, gradient_vector * dropout.mask)
