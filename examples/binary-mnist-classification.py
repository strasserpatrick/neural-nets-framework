from core import Dataloader
from core.dataset import CsvDataset
from core.functions import BinaryCrossEntropyLoss, ReLU, Sigmoid
from core.layers import LinearLayer
from core.model import Sequential
from core.trainer import Trainer
from core.transforms import Binarizer, Standardizer

binarizer = Binarizer(target=5)

mnist_train_dataset = CsvDataset(
    path="data/mnist_small/train/mnist_train.csv", transform_y=binarizer
)
mnist_test_dataset = CsvDataset(
    path="data/mnist_small/test/mnist_test.csv", transform_y=binarizer
)

minmax = Standardizer()
mnist_train_dataset.x = minmax.fit_transform(mnist_train_dataset.x)
mnist_test_dataset.x = minmax.transform(mnist_test_dataset.x)

mnist_train_dataloader = Dataloader(mnist_train_dataset)
mnist_test_dataloader = Dataloader(mnist_test_dataset)

loss_fn = BinaryCrossEntropyLoss()

model = Sequential(
    loss_fn=loss_fn,
    layers=[
        LinearLayer(in_dimension=784, out_dimension=32, activation_fn=ReLU),
        LinearLayer(in_dimension=32, out_dimension=1, activation_fn=Sigmoid),
    ],
)

trainer = Trainer(
    train_dataloader=mnist_train_dataloader,
    test_dataloader=mnist_test_dataloader,
    model=model,
    epochs=30,
    learning_rate=0.01,
)

loss_history = trainer.train()
