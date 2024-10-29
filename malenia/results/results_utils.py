import pandas as pd
import numpy as np


def _extract_global_and_specific_method_name(method_name):
    method_name_global = method_name.split("_")[0]
    method_name_specif = ""
    for part in method_name.split("_")[1:]:
        method_name_specif += part + "_"
    method_name_specif = method_name_specif[:-1]  # remove last "_"

    return method_name_global, method_name_specif


def _get_metric_results(metric, y_true, y_pred, y_proba):
    if metric.work_with_probas:
        return metric.compute(y_true, y_proba)
    else:
        return metric.compute(y_true, y_pred)


def _print_results_info(results):
    print("····································")
    print(f"→ There are {len(results['method'].unique())} methods.")
    print(f"→ There are {len(results['dataset'].unique())} datasets.")
    print(f"→ There are {len(results['seed'].unique())} seeds.")
    print("················")
    for dataset in results["dataset"].unique():
        for method in results["method"].unique():
            n_seeds = results[(results["dataset"] == dataset) & (results["method"] == method)].shape[0]
            print(f"→ {method} on {dataset} has {n_seeds} seeds.")
    print("····································")


def get_rankings(res, greaterIsBetter, metric_name, round_decimals=4):
    s = res.copy()
    s = s.reset_index(drop=True)
    if greaterIsBetter:
        s = s * (-1)

    methods = res.columns.values

    N, k = s.shape
    S = np.sort(s.T, axis=0)
    r = np.argsort(s.T, axis=0)
    idx = k * np.tile(np.arange(N), (k, 1)).T + r.T
    R = np.tile(np.arange(1, k + 1), (N, 1))
    S = np.transpose(S)
    for i in range(N):
        for j in range(k):
            index = S[i, j] == S[i, :]
            R[i, index] = np.mean(R[i, index])

    r_flat = r.flatten()
    R_flat = R.flatten()
    idx_flat = idx.flatten()
    for i, i_val in enumerate(idx_flat):
        r_flat[i_val] = R_flat[i]
    r_flat = r_flat.reshape(r.shape, order="F")

    r = r_flat
    r = np.transpose(r)

    r = np.mean(r, axis=0).round(round_decimals)

    ranks_df = pd.DataFrame()
    for i in range(len(methods)):
        ranks_df = pd.concat(
            [ranks_df, pd.DataFrame({"method": [methods[i]], f"rank_{metric_name}": [r[i]]})], ignore_index=True
        )

    return ranks_df
