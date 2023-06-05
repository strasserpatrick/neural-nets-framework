import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import core
from core.dataloader import Dataloader as CoreDataloader

from core import BaseDataset, Sequential, LinearLayer
from core.functions import CrossEntropyLoss, ReLU, Linear
from core.optimizer import GradientDescent


@pytest.fixture
def pytorch_model():
    class PytorchModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(64, 64)
            self.layer2 = torch.nn.Linear(64, 32)
            self.layer3 = torch.nn.Linear(32, 4)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_pred = self.forward(x)

            loss = F.cross_entropy(input=y, target=y_pred)
            return loss

        def forward(self, x):
            x = self.layer1(x)
            x = torch.relu(x)
            x = self.layer2(x)
            x = torch.relu(x)
            x = self.layer3(x)
            x = torch.softmax(x, dim=1)
            return x

        def configure_optimizers(self):
            params = self.parameters()
            optimizer = optim.SGD(params=params, lr=0.001)
            return optimizer

    model = PytorchModel()

    return model


@pytest.fixture
def input_data():
    torch.manual_seed(
        1234
    )  # for insuring that using this fixture multiple times leads to same random numbers
    x = torch.randn(100, 64)

    y = torch.randn(100, 4)
    argmax = torch.argmax(y, dim=1)

    identity_matrix = np.eye(4)
    one_hot_y = torch.Tensor(identity_matrix[argmax])

    return x, one_hot_y


@pytest.fixture
def pl_dataloader(input_data):
    x, y = input_data
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset)

    return dataloader


@pytest.fixture
def core_dataloader(input_data):
    # convert to numpy values
    x, y = input_data
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    class CoreTensorDataset(BaseDataset):
        def __init__(self, x, y):
            super().__init__()
            self.x = x
            self.y = y

            self.len = self.x.shape[0]

        def __len__(self):
            return self.len

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

        def split(self, split_list):
            raise NotImplementedError("Splitting is not implemented for CSV datasets")

    core_dataset = CoreTensorDataset(x, y)
    core_dataloader = CoreDataloader(core_dataset)

    return core_dataloader


@pytest.fixture
def core_model():
    loss_fn = CrossEntropyLoss()
    model = Sequential(
        loss_fn=loss_fn,
        layers=[
            LinearLayer(in_dimension=64, out_dimension=64, activation_fn=ReLU),
            LinearLayer(in_dimension=64, out_dimension=32, activation_fn=ReLU),
            LinearLayer(in_dimension=32, out_dimension=4, activation_fn=Linear),
        ],
    )

    return model


def test_core_against_pytorch(
    core_model, pytorch_model, core_dataloader, pl_dataloader
):

    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(model=pytorch_model, train_dataloaders=pl_dataloader)

    state_dict = pytorch_model.state_dict()

    layer1_weights = state_dict["layer1.weight"].cpu().detach().numpy().T
    layer1_bias = state_dict["layer1.bias"].cpu().detach().numpy().T
    layer2_weights = state_dict["layer2.weight"].cpu().detach().numpy().T
    layer2_bias = state_dict["layer2.bias"].cpu().detach().numpy().T
    layer3_weights = state_dict["layer3.weight"].cpu().detach().numpy().T
    layer3_bias = state_dict["layer3.bias"].cpu().detach().numpy().T

    pl_learned_model_weights_layer1 = np.concatenate(
        [layer1_weights, layer1_bias.reshape(1, -1)], axis=0
    )
    pl_learned_model_weights_layer2 = np.concatenate(
        [layer2_weights, layer2_bias.reshape(1, -1)], axis=0
    )
    pl_learned_model_weights_layer3 = np.concatenate(
        [layer3_weights, layer3_bias.reshape(1, -1)], axis=0
    )

    pl_learned_model_weights_list = [
        pl_learned_model_weights_layer1.T,
        pl_learned_model_weights_layer2.T,
        pl_learned_model_weights_layer3.T,
    ]

    # CORE MODULE
    core_model.layers[0].weights = pl_learned_model_weights_layer1.T
    core_model.layers[1].weights = pl_learned_model_weights_layer2.T
    core_model.layers[2].weights = pl_learned_model_weights_layer3.T

    optimizer = GradientDescent(core_model, learning_rate=0.01)

    trainer = core.Trainer(
        train_dataloader=core_dataloader,
        test_dataloader=core_dataloader,  # testing files for cross validation are ignored
        model=core_model,
        optimizer=optimizer,
        epochs=50,
    )

    (
        _,
        _,
        _,
        _,
    ) = trainer.train()  # return value of trainer.train() is not used in this test

    core_model_weights = [layer.weights for layer in core_model.layers]

    np.testing.assert_equal(pl_learned_model_weights_list, core_model_weights)
