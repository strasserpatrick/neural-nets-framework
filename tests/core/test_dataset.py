import numpy as np
import pytest

from core import BaseDataset


@pytest.fixture
def simple_dataset():
    class SimpleDataset(BaseDataset):
        def __init__(self):
            self.data = np.random.rand(300, 20)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

        def split(self, split_list):
            ...

    return SimpleDataset()


def test_dataset(simple_dataset):
    assert len(simple_dataset) == 300
    assert simple_dataset[0].shape == (20,)


def test_dataset_split(simple_dataset):
    ...
