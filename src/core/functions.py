from abc import ABC, abstractmethod

import numpy as np


class AbstractFunction(ABC):
    @staticmethod
    @abstractmethod
    def apply(x):
        ...

    @staticmethod
    @abstractmethod
    def derivative(result):
        ...


class Sigmoid(AbstractFunction):
    @staticmethod
    def apply(x):
        # # clip to avoid overflow
        x = np.clip(x, -100, 10)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        return np.array(Sigmoid.apply(x) * (1 - Sigmoid.apply(x)))


class ReLU(AbstractFunction):
    @staticmethod
    def apply(x):
        return np.maximum(0, x)

    @staticmethod
    def derivative(x):
        return np.where(x > 0, 1, 0)


class Linear(AbstractFunction):
    @staticmethod
    def apply(x):
        return x

    @staticmethod
    def derivative(x):
        return np.ones_like(x)


class BinaryCrossEntropyLoss(AbstractFunction):
    @staticmethod
    def apply(yhat, y):
        yhat = yhat[0]
        loss = -y * np.log(yhat) - (1 - y) * np.log(1 - yhat)
        return loss

    @staticmethod
    def derivative(yhat, y):

        derivative = -y / yhat + (1 - y) / (1 - yhat)

        return derivative


class CrossEntropyLoss(AbstractFunction):
    @staticmethod
    def apply(yhat, y):
        s = Softmax.apply(yhat)

        if (s == 0).any():
            print("Warning: Softmax output is 0")

        cel = -np.sum(y * np.log(np.maximum(s, 1e-50)))
        return cel

    @staticmethod
    def derivative(yhat, y):
        s = Softmax.apply(yhat)
        return s - y


class Softmax(AbstractFunction):
    @staticmethod
    def apply(x):
        # Clip to avoid overflow and underflow
        xx = np.clip(x, -350, 350)
        return np.exp(xx) / np.sum(np.exp(xx), axis=0)

        # # shift to avoid overflow
        # shiftx = np.exp(x - np.max(x) + 1e-50)
        # return shiftx / np.sum(shiftx)

    @staticmethod
    def derivative(s):
        jacobian_m = np.diag(s)

        # fix read-only error
        jacobian_m.setflags(write=1)

        for i in range(len(jacobian_m)):
            for j in range(len(jacobian_m)):
                if i == j:
                    jacobian_m[i][j] = s[i] * (1 - s[i])
                else:
                    jacobian_m[i][j] = -s[i] * s[j]
        return jacobian_m
