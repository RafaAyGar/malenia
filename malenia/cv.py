import os

import pandas as pd

from aeon.utils.sampling import stratified_resample


class OrcaPythonCV:
    def __init__(self, total_folds):
        self.total_folds = total_folds

    def get_fold_from_disk(self, dataset, fold):
        train = pd.read_csv(
            os.path.join(dataset.path, f"train_{dataset.name}.{str(fold)}"),
            sep=" ",
            header=None,
        )
        test = pd.read_csv(
            os.path.join(dataset.path, f"test_{dataset.name}.{str(fold)}"),
            sep=" ",
            header=None,
        )
        X_train = train.iloc[:, :-1]
        X_test = test.iloc[:, :-1]
        y_train = train.iloc[:, -1]
        y_test = test.iloc[:, -1]
        return X_train, y_train, X_test, y_test

    def get_n_splits(self):
        return self.total_folds


class StratifiedCV:
    def __init__(self, total_folds):
        self.total_folds = total_folds

    def apply(self, X_train, y_train, X_test, y_test, fold):
        if fold != 0:
            X_train, y_train, X_test, y_test = stratified_resample(
                X_train, y_train, X_test, y_test, fold
            )
            return X_train, y_train, X_test, y_test
        else:
            return X_train, y_train, X_test, y_test

    def get_n_splits(self):
        return self.total_folds
