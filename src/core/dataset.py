from abc import ABC, abstractmethod

import pandas as pd


class BaseDataset(ABC):
    def __init__(self, transform_X=None, transform_y=None, transform=None):
        self.transform_X = transform_X
        self.transform_y = transform_y
        self.transform = transform

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def __getitem__(self, idx):
        # returns a tuple of (x, y)
        ...

    @abstractmethod
    def split(self, split_list):
        ...


class CsvDataset(BaseDataset):
    def __init__(self, path, transform_X=None, transform_y=None):
        super().__init__(transform_X=transform_X, transform_y=transform_y)

        self.data = pd.read_csv(path)

        self.x = self.data.iloc[:, 1:].values
        self.y = self.data.iloc[:, 0].values

        if self.transform_X:
            self.x = self.transform_X(self.x)

        if self.transform_y:
            self.y = self.transform_y(self.y)

        self.len = self.x.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def split(self, split_list):
        raise NotImplementedError("Splitting is not implemented for CSV datasets")
