import os
import sys
from logging import StreamHandler, getLogger

import numpy as np
import pandas as pd
from joblib import load
from malenia.clustering.clustering_aux import save_final_cluster_distances, save_predictions
from pandas import Timestamp

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
y = y.astype(int)
del y_train, y_test

###

## Set number of clusters
#


if type(method_clus) is str:
    ### Load clusters and distances from disk
    ##
    #
    saved_clusters_fold = 0
    # Find the fold of the saved clusters (usually is zero as clustering methods
    # doesn't have a stochastic component)
    if os.path.exists(os.path.join(method_clus, dataset_name, f"seed_{fold}__clusters.csv")):
        saved_clusters_fold = fold
    else:
        saved_clusters_fold = 0
    # Load clusters and distances
    clusters_path = os.path.join(
        method_clus, dataset_name, f"seed_{saved_clusters_fold}__clusters.csv"
    )
    distances_path = os.path.join(
        method_clus,
        dataset_name,
        f"seed_{saved_clusters_fold}__final_clusters_dist.npy",
    )
    clusters_results = pd.read_csv(clusters_path)
    clusters = clusters_results["y_pred"]
    final_cluster_dist = np.load(distances_path)
    clustering_start_time = clusters_results["clustering_start_time"]
    clustering_end_time = clusters_results["clustering_end_time"]
else:
    ### Fit and Predict on test and get clusters and distances
    ##
    #
    method_clus.n_clusters = len(np.unique(y))
    clustering_start_time = Timestamp.now()
    clusters = method_clus.fit_predict(X)
    clustering_end_time = Timestamp.now()
    final_cluster_dist = method_clus.final_cluster_dist
    del X, method_clus


### Post-hoc label the clusters with given method (if required)
##
#
if method_posthoc is None:
    clusters_reverse = None
    posthoc_end_time = Timestamp.now()
    posthoc_start_time = Timestamp.now()
else:
    posthoc_start_time = Timestamp.now()
    clusters, clusters_reverse = method_posthoc(clusters, final_cluster_dist, y)
    posthoc_end_time = Timestamp.now()

### Save predictions
##
#
try:
    save_predictions(
        y_true=y,
        y_pred=clusters,
        y_pred_reverse=clusters_reverse,
        clustering_start_time=clustering_start_time,
        clustering_end_time=clustering_end_time,
        posthoc_start_time=posthoc_start_time,
        posthoc_end_time=posthoc_end_time,
        job_info=job_info,
        results_path=results_path,
    )
    log.warning(f"Done! - Fit {job_info} saved!")
except Exception as e:
    log.warning(
        f"** Could not save predictions - " f"Fit - {job_info} - " f"* EXCEPTION: \n{e}"
    )
#
###


### Save final cluster distances
##
#
if do_save_final_cluster_distances:
    try:
        save_final_cluster_distances(
            final_cluster_distances=final_cluster_dist,
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
