import os

import pandas as pd

from itertools import chain

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

class OrcaPythonCV:
    def __init__(self, total_folds=30):
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
    def __init__(self, total_folds=30):
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



def stratified_resample(X_train, y_train, X_test, y_test, random_state=None):
    random_state = check_random_state(random_state)
    all_labels = np.concatenate([y_train, y_test], axis=0)
    if isinstance(X_train, pd.DataFrame):
        all_data = pd.concat([X_train, X_test], ignore_index=True)
    elif isinstance(X_train, list):
        all_data = list(x for x in chain(X_train, X_test))
    else:  # 3D or 2D numpy
        all_data = np.concatenate([X_train, X_test], axis=0)

    # count class occurrences
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test = np.unique(y_test)

    # haven't built functionality to deal with classes that exist in
    # test but not in train
    assert set(unique_train) == set(unique_test)

    new_train_indices = []
    new_test_indices = []
    for label, count_train in zip(unique_train, counts_train):
        class_indexes = np.argwhere(all_labels == label).ravel()

        # randomizes the order and partition into train and test
        random_state.shuffle(class_indexes)
        new_train_indices.extend(class_indexes[:count_train])
        new_test_indices.extend(class_indexes[count_train:])

    if isinstance(X_train, pd.DataFrame):
        new_X_train = all_data.iloc[new_train_indices]
        new_X_test = all_data.iloc[new_test_indices]
        new_X_train = new_X_train.reset_index(drop=True)
        new_X_test = new_X_test.reset_index(drop=True)
    elif isinstance(X_train, list):
        new_X_train = list(all_data[i] for i in new_train_indices)
        new_X_test = list(all_data[i] for i in new_test_indices)
    else:  # 3D or 2D numpy
        new_X_train = all_data[new_train_indices]
        new_X_test = all_data[new_test_indices]

    new_y_train = all_labels[new_train_indices]
    new_y_test = all_labels[new_test_indices]

    return new_X_train, new_y_train, new_X_test, new_y_test

