import os
from time import time

import numpy as np
import sklearn
import sktime
from sklearn.metrics import mean_absolute_error
from sktime.benchmarking.data import UEADataset
from sktime.series_as_features.model_selection import StratifiedCV

from malenia.methods.my_stc.my_stc import ShapeletTransformClassifier

print("sklearn loaded from: ", sklearn.__path__)
print("sktime loaded from: ", sktime.__path__)

N_THREADS = 1

data_path = "/home/rayllon/DATA/tsoc/"
databases = os.listdir(data_path)
databases.sort()
datasets = []
for i, db in enumerate(databases):
    if (
        db
        == "MiddlePhalanxTW"
        # or db == "WindTurbinePower_5b"
    ):
        datasets.append(UEADataset(path=data_path, name=db))

my_stc = ShapeletTransformClassifier()

for dataset in datasets:
    X_train, y_train, X_test, y_test = dataset.load_crude()

    y_train = y_train.astype(np.int)
    y_test = y_test.astype(np.int)

    my_stc.fit(X_train, y_train)
