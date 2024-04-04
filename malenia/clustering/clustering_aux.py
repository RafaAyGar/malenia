import os

import numpy as np
import pandas as pd
from joblib import dump
from malenia.save_and_load import get_job_path


def save_predictions(
    y_true,
    y_pred,
    clustering_start_time,
    clustering_end_time,
    posthoc_start_time,
    posthoc_end_time,
    job_info,
    results_path,
    y_pred_reverse=None,
):
    path = get_job_path(job_info, results_path) + "clusters.csv"

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    if y_pred_reverse is None:
        y_pred_reverse = np.ones(y_pred.shape) * -1

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    preds = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_pred_reverse": y_pred_reverse,
            "clustering_start_time": clustering_start_time,
            "clustering_end_time": clustering_end_time,
            "posthoc_start_time": posthoc_start_time,
            "posthoc_end_time": posthoc_end_time,
        }
    )
    preds.to_csv(path, index=False)


def save_final_cluster_distances(
    final_cluster_distances,
    job_info,
    results_path,
):
    path = get_job_path(job_info, results_path) + "final_clusters_dist"

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    np.save(path, final_cluster_distances)
