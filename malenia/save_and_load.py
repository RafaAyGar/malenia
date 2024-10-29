import os

import numpy as np
import pandas as pd
from joblib import dump


def get_job_path(job_info, results_path):
    job_info = job_info.split("___")
    for j in job_info:
        print("****", j)
    path = os.path.join(results_path, job_info[0], job_info[1], job_info[2], "seed_" + job_info[3] + "_")
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
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
    else:
        y_proba = np.ones(y_true.shape) * -1
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


def save_predictions_with_fit_time(
    y_true,
    y_pred,
    y_proba,
    fit_time,
    train_or_test,
    job_info,
    results_path,
):
    path = get_job_path(job_info, results_path) + train_or_test + ".csv"

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
    else:
        y_proba = np.ones(y_true.shape) * -1
    preds = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": pd.Series(list(y_proba)),
            "fit_time": fit_time,
        }
    )
    preds.to_csv(path, index=False)


def save_predictions_multitask_multifarm(
    y_true,
    y_pred,
    tasks,
    farms,
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

    y_true_pred_per_farm_task = {}
    for farm_name, farm_index in farms.items():
        for task_name, task_index in tasks.items():
            y_true_name = f"y_true_{farm_name}_{task_name}"
            y_pred_name = f"y_pred_{farm_name}_{task_name}"
            if len(farms) == 1:
                y_true_pred_per_farm_task[y_true_name] = y_true[:, task_index]
                y_true_pred_per_farm_task[y_pred_name] = y_pred[:, task_index]
            else:
                y_true_pred_per_farm_task[y_true_name] = y_true[farm_index, :, task_index]
                y_true_pred_per_farm_task[y_pred_name] = y_pred[farm_index, :, task_index]

    preds = pd.DataFrame(
        {
            **y_true_pred_per_farm_task,
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
