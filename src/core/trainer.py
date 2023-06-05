import numpy as np
from matplotlib import pyplot as plt

from core.dataloader import Dataloader
from core.functions import Softmax
from core.metrics import accuracy, f1_score
from core.model import AbstractModel
from core.optimizer import AbstractOptimizer, GradientDescent


class Trainer:
    def __init__(
        self,
        model: AbstractModel,
        epochs: int,
        optimizer: AbstractOptimizer,
        train_dataloader: Dataloader = None,
        test_dataloader: Dataloader = None,
    ):

        self.model = model

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.epochs = epochs
        self.optimizer = optimizer

    def sanity_check(self):
        for layer in self.model.layers:
            weights = layer.weights
            if np.isnan(weights).any() or np.isinf(weights).any():
                print(f"Nan value found in layer {layer}")

    def train_epoch(self):
        epoch_loss = 0
        epoch_accuracy = 0

        self.train_dataloader.reset()

        self.sanity_check()

        for x, y in self.train_dataloader:
            yhat = self.model(x, train=True)

            ypred = np.argmax(Softmax.apply(yhat))
            if np.equal(ypred, np.argmax(y)):
                epoch_accuracy += 1

            # current loss
            loss = self.model.loss_fn.apply(yhat, y)
            epoch_loss += loss

            # optimize
            self.optimizer.backpropagation(x, y)

        return epoch_loss, round(epoch_accuracy / len(self.train_dataloader), 2)

    def train(self):
        train_loss_history = []
        test_f1_history = []
        train_accuracy_history = []
        test_accuracy_history = []

        for idx in range(1, self.epochs + 1):
            epoch_loss, epoch_accuracy = self.train_epoch()
            train_loss_history.append(epoch_loss)

            test_f1, test_acc, confusion_matrix = self.test(4, verbose=False)

            train_accuracy_history.append(epoch_accuracy)

            test_acc = accuracy(confusion_matrix)
            test_accuracy_history.append(test_acc)

            test_f1_history.append(test_f1)

            print(f"Epoch: {idx}\t Loss: {epoch_loss:.2f}\t Test F1: {test_f1:.2f}\t Test Acc: {test_acc:.2f}")

        return train_loss_history, test_f1_history, train_accuracy_history, test_accuracy_history

    def test(self, number_of_classes, verbose=True):
        sum_loss = 0
        correctly_classified = 0
        incorrectly_classified = 0

        confussion_matrix = np.zeros((number_of_classes, number_of_classes))

        self.test_dataloader.reset()

        for x, y in self.test_dataloader:
            y = np.argmax(y)
            yhat = self.model(x, train=False)

            # current loss
            loss = self.model.loss_fn.apply(yhat, y)
            sum_loss += loss

            yhat_logits = np.argmax(Softmax.apply(yhat))
            if np.equal(yhat_logits, y):
                correctly_classified += 1
                if verbose and y != 0:
                    print("gesture correctly classified")
            else:
                incorrectly_classified += 1

            confussion_matrix[y][yhat_logits] += 1

        if verbose:
            print(f"Correctly classified: {correctly_classified}")
            print(f"Incorrectly classified: {incorrectly_classified}")

        return (
            round(np.mean([f1_score(confussion_matrix, label=i) for i in range(1, 4)]), 2),
            accuracy(confussion_matrix),
            confussion_matrix,
        )
