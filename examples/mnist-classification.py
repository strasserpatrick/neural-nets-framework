from matplotlib import pyplot as plt

from core import Dataloader
from core.dataset import CsvDataset
from core.functions import CrossEntropyLoss, Linear, ReLU
from core.layers import LinearLayer
from core.model import Sequential
from core.optimizer import GradienDescentMomentum
from core.trainer import Trainer
from core.transforms import MinMaxScaler, OneHotEncoder, Standardizer

mnist_train_dataset = CsvDataset(
    path="data/mnist/train/mnist_train.csv", transform_y=OneHotEncoder(10)
)
mnist_test_dataset = CsvDataset(
    path="data/mnist/test/mnist_test.csv", transform_y=OneHotEncoder(10)
)

scaler = Standardizer()
mnist_train_dataset.x = scaler.fit_transform(mnist_train_dataset.x)
mnist_test_dataset.x = scaler.transform(mnist_test_dataset.x)

mnist_train_dataloader = Dataloader(mnist_train_dataset)
mnist_test_dataloader = Dataloader(mnist_test_dataset)

loss_fn = CrossEntropyLoss()

model = Sequential(
    loss_fn=loss_fn,
    layers=[
        LinearLayer(in_dimension=784, out_dimension=128, activation_fn=ReLU),
        LinearLayer(in_dimension=128, out_dimension=10, activation_fn=Linear),
    ],
)

optim = GradienDescentMomentum(model, learning_rate=0.01, momentum=0.9)

trainer = Trainer(
    train_dataloader=mnist_train_dataloader,
    test_dataloader=mnist_test_dataloader,
    model=model,
    epochs=100,
    optimizer=optim,
)

loss_history = trainer.train()
[print(f"Epoch {i + 1}: {loss:.2f}") for i, loss in enumerate(loss_history)]
plt.plot(loss_history)
plt.show()


trainer.test()
