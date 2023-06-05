from abc import ABC, abstractmethod

from core.model import AbstractModel


class AbstractOptimizer(ABC):
    def __init__(self, model: AbstractModel, learning_rate: float):
        self.model = model
        self.learning_rate = learning_rate

    @abstractmethod
    def backpropagation(self, x, y):
        ...


class GradientDescent(AbstractOptimizer):
    def __init__(self, model: AbstractModel, learning_rate: float):
        super().__init__(model=model, learning_rate=learning_rate)

        self.update_function = lambda grad: -self.learning_rate * grad

    def backpropagation(self, x, y):
        self.model.gradient_zero()
        self.model.backward(x=x, y=y)
        self.model.gradient_update(self.update_function)


class GradienDescentMomentum(AbstractOptimizer):
    def __init__(self, model: AbstractModel, learning_rate: float, momentum: float):
        super().__init__(model=model, learning_rate=learning_rate)
        self.momentum = momentum

        self.update_function = (
            lambda grad, old_grad: -self.learning_rate * grad
            + self.learning_rate * self.momentum * old_grad
        )

    def backpropagation(self, x, y):
        old_gradients = self.model.gradient_zero()
        self.model.backward(x=x, y=y)
        self.model.gradient_update(
            gradient_update_function=self.update_function,
            gradient_update_args=old_gradients,
        )
