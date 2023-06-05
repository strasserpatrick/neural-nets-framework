from abc import ABC, abstractmethod

import numpy as np


class AbstractLayer(ABC):
    def __init__(self, in_dimension: int, out_dimension: int, activation_fn):
        self.in_dimension = in_dimension

        # num of out = num of neurons in layer
        self.out_dimension = out_dimension
        self.activation_fn = activation_fn

        self.weights = np.random.uniform(
            low=-1, high=1, size=(out_dimension, in_dimension + 1)
        )
        self.gradient = np.zeros_like(self.weights)

    def shape(self):
        return self.out_dimension, self.in_dimension  # bias weight is abstracted

    def gradient_zero(self):
        self.gradient = np.zeros_like(self.weights)

    def gradient_update(self, gradient_update_function, gradient_update_arg=None):
        if gradient_update_arg is None:
            self.weights += gradient_update_function(self.gradient)
        else:
            self.weights += gradient_update_function(self.gradient, gradient_update_arg)

    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        ...

    @abstractmethod
    def backward(self, input_vector, backward_vector):
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...


class LinearLayer(AbstractLayer):
    def __init__(self, in_dimension, out_dimension, activation_fn):
        super().__init__(
            in_dimension=in_dimension,
            out_dimension=out_dimension,
            activation_fn=activation_fn,
        )

    def z(self, x):
        # Dimension 0 is batch_size
        if not x.shape[0] == self.in_dimension:
            raise ValueError(
                f"x.shape must be the same as in_dimension, but shape is {x.shape[0]}"
            )

        x = np.concatenate((np.array([1]), x))

        # return x @ self.weights.T
        # this is equivalent to the above, but faster
        # return (self.weights @ x.T).T
        return x @ self.weights.T

    def a(self, z):
        return self.activation_fn.apply(z)

    def forward(self, x, *args, **kwargs):
        return self.a(self.z(x))

    def backward(self, input_vector, backward_vector):

        # forward pass
        z = self.z(input_vector)

        # derive a to z
        da_dz = self.activation_fn.derivative(z)

        input_vector_bias = np.concatenate((np.array([1]), input_vector))

        # repeat input vector for each neuron in layer
        input_vector_bias_repeated = np.repeat(
            input_vector_bias.reshape(1, -1), self.out_dimension, axis=0
        )

        # update weights
        prefactor = backward_vector * da_dz
        self.gradient = np.diag(prefactor) @ input_vector_bias_repeated

        # calculate gradient product for next layer
        gradient_product = self.weights[:, 1:].T @ prefactor

        return gradient_product

    def __str__(self) -> str:
        return f"LinearLayer(in_dimension={self.in_dimension}, out_dimension={self.out_dimension})"


class Dropout(AbstractLayer):
    def __init__(self, dropout_rate: int):
        super().__init__(in_dimension=0, out_dimension=0, activation_fn=None)
        self.dropout_rate = dropout_rate

    def forward(self, x, *args, **kwargs):
        if "train" in kwargs and kwargs["train"]:
            self.mask = np.random.binomial(1, self.dropout_rate, size=x.shape)
            return x * self.mask
        else:
            # during test: scale output by dropout rate
            # https://leimao.github.io/blog/Dropout-Explained/
            return x * self.dropout_rate

    def backward(self, input_vector, backward_vector):
        # forward pass for initializing mask is called by model abstraction
        if not hasattr(self, "mask"):
            raise ValueError(
                "forward pass for initializing mask needs to be called by model abstraction"
            )
        return backward_vector * self.mask

    def __str__(self) -> str:
        return f"Dropout(dropout_rate={self.dropout_rate})"
