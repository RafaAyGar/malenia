import os
import sys
from logging import StreamHandler, getLogger

import numpy as np
from joblib import load
from pandas import Timestamp

from malenia.clustering.clustering_aux import (save_final_cluster_distances,
                                               save_predictions)

## Uncomment when using sktime-dl methods
# sys.path.append("/home/rayllon/GitHub/sktime-dl")
# import sktime_dl


console = StreamHandler()
log = getLogger()
log.addHandler(console)


## Read console arguments
#
dataset_path = str(sys.argv[1]).strip()
method_path = str(sys.argv[2]).strip()
fold = int(sys.argv[3].strip())
overwrite_predictions = eval(sys.argv[4])
do_save_final_cluster_distances = eval(sys.argv[5])
job_info = str(sys.argv[6])
results_path = str(sys.argv[7])
dataset_name = str(sys.argv[8]).strip()


### Load Dataset
##
#
with open(dataset_path, "rb") as dataset_binary:
    dataset = load(dataset_binary)
#
# Check if dataset has a load_crude() method
#
if not hasattr(dataset, "load_crude"):
    raise ValueError(f"Dataset {dataset.name} must implement a load_crude() method")
#
###


### Load Method
##
#
with open(method_path, "rb") as method_binary:
    method = load(method_binary)
    if type(method) is tuple:
        method_clus, method_posthoc = method
    else:
        method_clus = method
        method_posthoc = None
del method
if hasattr(method_clus, "random_state"):
    assert method_clus.random_state == fold
###


### Load data
##
#
# TSOC datasets
X_train, y_train, X_test, y_test = dataset.load_crude()
del dataset
# Orreview datasets
# X_train, y_train, X_test, y_test = cv.get_fold_from_disk(dataset, fold)
# X_train = X_train.to_numpy()
# X_test = X_test.to_numpy()
#
if hasattr(X_train, "reset_index"):
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
#
# Join X_train and X_test (numpy arrays)
X = np.concatenate((X_train, X_test))
del X_train, X_test
y = np.concatenate((y_train, y_test))
del y_train, y_test

###

## Set number of clusters
#
method_clus.n_clusters = len(np.unique(y))


### Fit and Predict on test
##
#
fit_estimator_start_time = Timestamp.now()
y_pred = method_clus.fit_predict(X)
fit_estimator_end_time = Timestamp.now()
del X


### Post-hoc label the clusters with given method
##
#
if not method_posthoc is None:
    y_pred, y_pred_reverse = method_posthoc(y_pred, method_clus.final_cluster_dist)
else:
    y_pred_reverse = None

### Save predictions
##
#
try:
    save_predictions(
        y_true=y,
        y_pred=y_pred,
        y_pred_reverse=y_pred_reverse,
        fit_estimator_start_time=fit_estimator_start_time,
        fit_estimator_end_time=fit_estimator_end_time,
        job_info=job_info,
        results_path=results_path,
    )
    log.warning(f"Done! - Fit {job_info} saved!")
except Exception as e:
    log.warning(
        f"** Could not save predictions - "
        f"Fit - {job_info} - "
        f"* EXCEPTION: \n{e}"
    )
#
###

if do_save_final_cluster_distances:
    try:
        save_final_cluster_distances(
            final_cluster_distances=method_clus.final_cluster_dist,
            job_info=job_info,
            results_path=results_path,
        )
        log.warning(f"Done! - Final cluster distances {job_info} saved!")
    except Exception as e:
        log.warning(
            f"** Could not save final cluster distances - "
            f"Fit - {job_info} - "
            f"* EXCEPTION: \n{e}"
        )
