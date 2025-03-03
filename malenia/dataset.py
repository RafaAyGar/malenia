import os

import numpy as np
import pandas as pd
from joblib import load


class Dataset:
    def __init__(self, name, dataset_type, path):
        self.name = name
        self.type = dataset_type
        self.path = path


class TSDataset:

    def __init__(
        self,
        path,
        name,
    ):
        self.path = path
        self.name = name

    def load(self, fold):
        from aeon.datasets import load_from_tsfile

        X_train, y_train = load_from_tsfile(os.path.join(self.path, self.name) + f"/{self.name}_TRAIN.ts")
        X_test, y_test = load_from_tsfile(os.path.join(self.path, self.name) + f"/{self.name}_TEST.ts")
        return X_train, y_train, X_test, y_test


class OrcaPythonDataset:
    """Represent a dataset in Orca Python format on the hard-drive."""

    def __init__(
        self,
        path,
        name,
    ):
        self.path = path
        self.name = name

    def load(self, fold):
        train = pd.read_csv(os.path.join(self.path, f"train_{self.name}.0"), sep=" ", header=None)
        test = pd.read_csv(os.path.join(self.path, f"test_{self.name}.0"), sep=" ", header=None)
        X_train = np.array(train.iloc[:, :-1])
        y_train = train.iloc[:, -1]
        X_test = np.array(test.iloc[:, :-1])
        y_test = test.iloc[:, -1]

        return X_train, y_train, X_test, y_test


class TabularDataset:
    """Represent a dataset in Orca Python format on the hard-drive."""

    def __init__(
        self,
        path,
        name,
    ):
        self.path = path
        self.name = name

    def load(self, fold):
        train = pd.read_csv(os.path.join(self.path, f"train_{self.name}.0"), sep=" ", header=None)
        test = pd.read_csv(os.path.join(self.path, f"test_{self.name}.0"), sep=" ", header=None)
        X_train = train.drop(columns=["target"])
        y_train = train["target"]
        X_test = test.drop(columns=["target"])
        y_test = test["target"]

        return X_train, y_train, X_test, y_test


class TabularDataset:
    """Represent a dataset in UEA/UCR format on the hard-drive."""

    def __init__(
        self,
        path,
        name,
    ):
        self.path = path
        self.name = name

    def load(self, fold):
        """Load dataset."""
        fold_path = os.path.join(self.path, self.name, f"train_fold_{fold}.pkl")
        print(fold_path)
        print("fold path exists:", os.path.exists(fold_path))
        with open(fold_path, "rb") as dataset_binary:
            dataset = load(dataset_binary)
        return dataset
