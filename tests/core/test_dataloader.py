import numpy as np
import pytest

from core import BaseDataset, Dataloader


@pytest.fixture
def mock_dataset():
    class MockDataset(BaseDataset):
        def __init__(self):
            self.data = np.random.rand(300, 20)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

        def split(self, split_list):
            ...

    return MockDataset()


@pytest.fixture
def simple_dataloader(mock_dataset):
    return Dataloader(dataset=mock_dataset)


def test_dataloader(simple_dataloader):
    assert len(simple_dataloader) == 300
    iterable = iter(simple_dataloader)
    assert next(iterable).shape == (20,)
    assert next(iterable).shape == (20,)
    assert next(iterable).shape == (20,)
