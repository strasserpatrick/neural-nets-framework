import numpy as np
import pytest

from core import BinaryCrossEntropyLoss, Dropout, LinearLayer, Sequential, Sigmoid


@pytest.fixture
def model():
    model = Sequential(
        loss_fn=BinaryCrossEntropyLoss(),
        layers=[
            LinearLayer(in_dimension=2, out_dimension=2, activation_fn=Sigmoid()),
            LinearLayer(in_dimension=2, out_dimension=2, activation_fn=Sigmoid()),
            LinearLayer(in_dimension=2, out_dimension=1, activation_fn=Sigmoid()),
        ],
    )

    model.layers[0].weights = np.array([[2, -1, 0.67], [-3, 1, -0.67]])
    model.layers[1].weights = np.array([[1, 1, 1], [-4, -0.33, 0.67]])
    model.layers[2].weights = np.array([[0.5, 0.67, -1.3]])

    return model


@pytest.fixture
def model_with_dropout():
    model = Sequential(
        loss_fn=BinaryCrossEntropyLoss(),
        layers=[
            LinearLayer(in_dimension=2, out_dimension=2, activation_fn=Sigmoid()),
            Dropout(0.5),
            LinearLayer(in_dimension=2, out_dimension=1, activation_fn=Sigmoid()),
        ],
    )
    return model


def test_model_forward(model):

    x = np.array([1, -2])
    yhat = model.forward(x)
    np.testing.assert_allclose(yhat, np.array([0.74]), atol=1e-3)

    loss = model.loss(yhat, np.array([0]))
    np.testing.assert_allclose(loss, np.array([1.34696]), atol=1e-3)


def test_model_backward(model):
    x = np.array([1, -2])
    y = np.array([0])
    model.backward(x, y)

    np.testing.assert_allclose(
        model.layers[0].gradient,
        np.array([[0.01661, 0.01661, -0.03321], [0.01119, 0.01119, -0.02238]]),
        atol=1e-3,
    )
    np.testing.assert_allclose(
        model.layers[1].gradient,
        np.array([[0.06224, 0.02588, 0.02121], [-0.01855, -0.007712, -0.00632]]),
        atol=1e-3,
    )
    np.testing.assert_allclose(
        model.layers[2].gradient,
        np.array([[0.74, 0.631, 0.01455]]),
        atol=1e-2,
    )


def test_save_and_load_weights(tmp_path, model):

    tmp_file = tmp_path / "weights.json"

    model.save_weights(tmp_file)

    new_model = Sequential(
        loss_fn=BinaryCrossEntropyLoss(),
        layers=[
            LinearLayer(in_dimension=2, out_dimension=2, activation_fn=Sigmoid()),
            Dropout(0.5),
            LinearLayer(in_dimension=2, out_dimension=1, activation_fn=Sigmoid()),
        ],
    )

    new_model.load_weights(tmp_file)

    for layer, new_layer in zip(model.layers, new_model.layers):
        np.testing.assert_allclose(layer.weights, new_layer.weights)
