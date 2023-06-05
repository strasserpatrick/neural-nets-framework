from abc import ABC, abstractmethod
from typing import List

import numpy as np
from pydantic import BaseSettings

from core.layers import AbstractLayer


class AbstractModel(ABC):
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def loss(self, yhat, y):
        return np.sum(self.loss_fn.apply(yhat, y))

    @abstractmethod
    def forward(self, x):
        ...

    @abstractmethod
    def backward(self, x, y):
        ...

    @abstractmethod
    def gradient_zero(self):
        ...

    @abstractmethod
    def gradient_update(self, gradient_update_function, **kwargs):
        ...

    @abstractmethod
    def load_weights(self, path):
        ...

    @abstractmethod
    def save_weights(self, path):
        ...


class Sequential(AbstractModel):
    def __init__(self, loss_fn, layers: List[AbstractLayer]):
        super().__init__(loss_fn=loss_fn)
        self.layers = layers

    def __str__(self) -> str:
        return "Sequential Model: \n" + "\n".join(
            [f"Layer {idx + 1}: {layer}" for idx, layer in enumerate(self.layers)]
        )

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x

    def backward(self, x, y):

        # initial input vector list = data input
        input_vector_list = [x]

        # run forward pass and save all input vectors
        for layer in self.layers[:-1]:
            x = layer(x, train=True)
            input_vector_list.append(x)

        # initial gradient_vector = loss derivation
        yhat = self.layers[-1](x, train=True)
        backward_vector = self.loss_fn.derivative(yhat, y)

        # run backwards pass
        for idx in range(len(self.layers) - 1, -1, -1):
            backward_vector = self.layers[idx].backward(
                input_vector=input_vector_list[idx],
                backward_vector=backward_vector,
            )

        return backward_vector

    def gradient_zero(self):
        old_gradients = [layer.gradient for layer in self.layers]
        for layer in self.layers:
            layer.gradient_zero()
        return old_gradients

    def gradient_update(self, gradient_update_function, gradient_update_args=None):
        for idx, layer in enumerate(self.layers):
            if gradient_update_args:
                layer.gradient_update(
                    gradient_update_function,
                    gradient_update_arg=gradient_update_args[idx],
                )
            else:
                layer.gradient_update(gradient_update_function)

    def load_weights(self, path):
        model_weights = ModelWeights.parse_file(path)
        for layer, weights in zip(self.layers, model_weights.layer_weights):
            layer.weights = np.array(weights)

    def save_weights(self, path):
        model_weights = ModelWeights(
            layer_weights=[layer.weights.tolist() for layer in self.layers]
        )
        import json

        with open(path, "w") as f:
            json.dump(model_weights.dict(), f)


# Loading and saving weights is very basic.
# It is necessary to build a model with the same shapes and activation functions as the saved model.
# Only the weights are saved and loaded.


class ModelWeights(BaseSettings):
    layer_weights: List[List[List[float]]]
