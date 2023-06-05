import random
from abc import ABC

import numpy as np

from core.dataset import BaseDataset


class Dataloader(ABC):
    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset

        self.sample_number = len(self.dataset)
        self.reset()

    def __iter__(self):
        return self

    def __len__(self):
        return self.sample_number

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        else:
            item = self.dataset[self.indices[self.index]]
        self.index += 1
        return item

    def reset(self):
        self.indices = list(range(len(self.dataset)))
        random.shuffle(self.indices)
        self.index = 0
