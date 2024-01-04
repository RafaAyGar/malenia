import os

import numpy as np
import pandas as pd
from joblib import dump


def get_job_path(job_info, results_path):
    job_info = job_info.split("__")
    for j in job_info:
        print("****", j)
    path = os.path.join(
        results_path, job_info[0], job_info[1], job_info[2], "seed_" + job_info[3] + "_"
    )
    return path


def save_method(method, job_info, results_path):
    path = get_job_path(job_info, results_path) + "trained_method.joblib"
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    dump(method, path)


def save_predictions(
    y_true,
    y_pred,
    y_proba,
    oob_probas,
    fit_estimator_start_time,
    fit_estimator_end_time,
    predict_estimator_start_time,
    predict_estimator_end_time,
    train_or_test,
    job_info,
    results_path,
):
    path = get_job_path(job_info, results_path) + train_or_test + ".csv"

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    if oob_probas is not None:
        oob_probas = np.asarray(oob_probas)
    else:
        oob_probas = np.zeros(y_proba.shape)
    preds = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": pd.Series(list(y_proba)),
            "oob_probas": pd.Series(list(oob_probas)),
            "fit_estimator_start_time": fit_estimator_start_time,
            "fit_estimator_end_time": fit_estimator_end_time,
            "predict_estimator_start_time": predict_estimator_start_time,
            "predict_estimator_end_time": predict_estimator_end_time,
        }
    )
    preds.to_csv(path, index=False)


def save_cv_results(cv_results, job_info, results_path):
    path = get_job_path(job_info, results_path) + "cv_results.csv"
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    pd.DataFrame(cv_results).to_csv(path, index=False)
