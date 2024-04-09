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
if dataset.type == "time_series":
    from aeon.datasets import load_from_tsfile

    X_train, y_train = load_from_tsfile(
        os.path.join(dataset.path, dataset.name) + f"/{dataset.name}_TRAIN.ts"
    )
    X_test, y_test = load_from_tsfile(
        os.path.join(dataset.path, dataset.name) + f"/{dataset.name}_TEST.ts"
    )
elif dataset.type == "orreview":
    import pandas as pd

    train = pd.read_csv(
        os.path.join(dataset.path, dataset.name, f"train_{dataset.name}.{str(0)}"),
        sep=" ",
        header=None,
    )
    test = pd.read_csv(
        os.path.join(dataset.path, dataset.name, f"test_{dataset.name}.{str(0)}"),
        sep=" ",
        header=None,
    )
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    del train
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]
    del test
elif dataset.type == "tabular_tsoc":
    import pandas as pd

    train = pd.read_csv(
        os.path.join(dataset.path, dataset.name, f"{dataset.name}_TRAIN.csv"),
    )
    test = pd.read_csv(
        os.path.join(dataset.path, dataset.name, f"{dataset.name}_TEST.csv"),
    )
    X_train = train.drop(columns=["target"])
    y_train = train["target"]
    del train
    X_test = test.drop(columns=["target"])
    y_test = test["target"]
    del test

del dataset
#
if hasattr(X_train, "reset_index"):
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
if hasattr(y_train, "reset_index"):
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
#
# Join X_train and X_test (numpy arrays)
X = np.concatenate((X_train, X_test))
del X_train, X_test
y = np.concatenate((y_train, y_test))
y = y.astype(int)
del y_train, y_test
# check if data has nans
if np.count_nonzero(np.isnan(X)) or np.count_nonzero(np.isnan(y)):
    raise ValueError("Training data has nan values!")
#
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
    if os.path.exists(os.path.join(method_clus, dataset_name, f"seed_{fold}_clusters.csv")):
        saved_clusters_fold = fold
    else:
        saved_clusters_fold = 0
    # Load clusters and distances
    clusters_path = os.path.join(
        method_clus, dataset_name, f"seed_{saved_clusters_fold}_clusters.csv"
    )
    distances_path = os.path.join(
        method_clus,
        dataset_name,
        f"seed_{saved_clusters_fold}_final_clusters_dist.npy",
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


### As in clustering we don't know the output label, we can't know if the ordering of the
### clusters is the correct one. So we will compare the results with the reverse ordering.
##
#
if clusters_reverse is not None:
    from malenia.metrics import amae

    print("Comparing clusters and clusters_reverse")
    if amae(y, clusters_reverse) < amae(y, clusters):
        print("Reverse clusters was selected")
        clusters = clusters_reverse
        clusters_reverse = None

### Save predictions
##
#
try:
    save_predictions(
        y_true=y,
        y_pred=clusters,
        # y_pred_reverse=clusters_reverse,
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
