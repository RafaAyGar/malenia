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


### Read console arguments
##
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
    from malenia.cv import StratifiedCV

    cv = StratifiedCV()

    X_train, y_train = load_from_tsfile(
        os.path.join(dataset.path, dataset.name) + f"/{dataset.name}_TRAIN.ts"
    )
    X_test, y_test = load_from_tsfile(
        os.path.join(dataset.path, dataset.name) + f"/{dataset.name}_TEST.ts"
    )
    X_train, y_train, X_test, y_test = cv.apply(X_train, y_train, X_test, y_test, fold)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

elif dataset.type == "orreview":
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.concat(
        [
            pd.read_csv(
                os.path.join(dataset.path, dataset.name, f"train_{dataset.name}.0"),
                header=None,
                sep=" ",
            ),
            pd.read_csv(
                os.path.join(dataset.path, dataset.name, f"test_{dataset.name}.0"),
                header=None,
                sep=" ",
            ),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        data.iloc[:, :-1], data.iloc[:, -1], test_size=0.33, random_state=fold
    )
    del data
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

del dataset

y_train = y_train.astype(int)
y_test = y_test.astype(int)

# check if data has nans
if np.count_nonzero(np.isnan(X_train)) or np.count_nonzero(np.isnan(y_train)):
    raise ValueError("Training data has nan values!")
elif np.count_nonzero(np.isnan(X_test)) or np.count_nonzero(np.isnan(y_test)):
    raise ValueError("Test data has nan values!")
#
###


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
    clusters_train = clusters_results["y_pred"].to_numpy()
    train_clusters_dist = np.load(distances_path)
    clustering_train_start_time = clusters_results["clustering_start_time"]
    clustering_train_end_time = clusters_results["clustering_end_time"]
else:
    ### Fit and Predict on test and get clusters and distances
    ##
    #
    method_clus.n_clusters = len(np.unique(y_train))
    clustering_train_start_time = Timestamp.now()
    clusters_train = method_clus.fit_predict(X_train)
    del X_train
    clustering_train_end_time = Timestamp.now()

    clustering_test_start_time = Timestamp.now()
    clusters_test = method_clus.predict(X_test)
    del X_test
    clustering_test_end_time = Timestamp.now()

    if hasattr(method_clus, "get_final_cluster_dist"):
        train_clusters_dist = method_clus.get_final_cluster_dist()
    else:
        from sklearn.metrics import pairwise_distances

        train_clusters_dist = pairwise_distances(method_clus.cluster_centers_)
        train_clusters_dist[np.where(train_clusters_dist == 0.0)] = np.inf
    del method_clus


### Post-hoc label the clusters with given method (if required)
##
#
if method_posthoc is None:
    clusters_train_reverse = None
    posthoc_train_start_time = Timestamp.now()
    posthoc_train_end_time = Timestamp.now()
    posthoc_test_start_time = None
    posthoc_test_end_time = None
else:
    posthoc_train_start_time = Timestamp.now()
    method_posthoc = method_posthoc.find_OLO(train_clusters_dist)
    clusters_train, clusters_train_reverse = method_posthoc.apply_OLO(clusters_train)
    posthoc_train_end_time = Timestamp.now()

    posthoc_test_start_time = Timestamp.now()
    clusters_test, clusters_test_reverse = method_posthoc.apply_OLO(clusters_test)
    posthoc_test_end_time = Timestamp.now()


### As in clustering we don't know the output label, we can't know if the ordering of the
### clusters is the correct one. So we will compare the results with the reverse ordering.
##
#
if clusters_train_reverse is not None:
    from malenia.metrics import amae

    print("Comparing clusters and clusters_reverse")
    if amae(y_train, clusters_train_reverse) < amae(y_train, clusters_train):
        print("Reverse clusters was selected")
        clusters_train = clusters_train_reverse
        clusters_test = clusters_test_reverse
        clusters_train_reverse = None
        clusters_test_reverse = None

### Save train and test predictions
##
#
try:
    save_predictions(
        y_true=y_train,
        y_pred=clusters_train,
        # y_pred_reverse=clusters_reverse,
        clustering_start_time=clustering_train_start_time,
        clustering_end_time=clustering_train_end_time,
        posthoc_start_time=posthoc_train_start_time,
        posthoc_end_time=posthoc_train_end_time,
        job_info=job_info,
        results_path=results_path,
        train_or_test="train",
    )
    save_predictions(
        y_true=y_test,
        y_pred=clusters_test,
        # y_pred_reverse=clusters_reverse,
        clustering_start_time=clustering_test_start_time,
        clustering_end_time=clustering_test_end_time,
        posthoc_start_time=posthoc_test_start_time,
        posthoc_end_time=posthoc_test_end_time,
        job_info=job_info,
        results_path=results_path,
        train_or_test="test",
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
            final_cluster_distances=train_clusters_dist,
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
