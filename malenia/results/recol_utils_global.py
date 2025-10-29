import pandas as pd


def filter_results_by_min_n_classes(results, min_n_classes):
    if min_n_classes is not None:
        n_classes = results["dataset"].str.split("_", expand=True)[0]
        n_classes = n_classes.str.replace("dr", "").str.replace("oc", "")
        results["n_classes"] = n_classes.astype(int)
        results = results[results["n_classes"] >= min_n_classes].copy()
    elif min_n_classes is not None and not isinstance(min_n_classes, int):
        raise ValueError(
            "min_n_classes should be an int or None. " f"Got {type(min_n_classes)} instead."
        )
    return results


def rankings_avgseeds(
    results, metrics_with_gib, datasets_min_n_classes=None, method="avg_seeds"
):

    results = filter_results_by_min_n_classes(results, datasets_min_n_classes)

    if method == "avg_seeds":
        ranking_along_metrics = pd.DataFrame()
        for metric, gib in metrics_with_gib.items():
            res = (
                results.groupby(["method", "dataset"])
                .mean()
                .reset_index()
                .pivot(index="dataset", columns="method", values=metric)
            )
            metric_ranks_series = res.T.rank(method="average", ascending=not gib).mean(axis=1)
            metric_ranks = pd.DataFrame(columns=["method"])
            metric_ranks["method"] = metric_ranks_series.index
            metric_ranks[f"ranks_{metric}"] = metric_ranks_series.values
            metric_ranks = metric_ranks.set_index("method")
            ranking_along_metrics = pd.concat([ranking_along_metrics, metric_ranks], axis=1)
        ranking_along_metrics = ranking_along_metrics.sort_values(by="method")
    elif method == "by_seed":
        results["dataset_seed"] = results["dataset"] + "_" + results["seed"].astype(str)
        ranking_along_metrics = pd.DataFrame()
        for metric, gib in metrics_with_gib.items():
            metric_ranks_series = (
                results.pivot(index="dataset_seed", columns="method", values=metric)
                .T.rank(method="average", ascending=not gib)
                .mean(axis=1)
            )
            metric_ranks = pd.DataFrame(columns=["method"])
            metric_ranks["method"] = metric_ranks_series.index
            metric_ranks[f"ranks_{metric}"] = metric_ranks_series.values
            metric_ranks = metric_ranks.set_index("method")
            ranking_along_metrics = pd.concat([ranking_along_metrics, metric_ranks], axis=1)
        ranking_along_metrics = ranking_along_metrics.sort_values(by="method")
    else:
        raise ValueError(
            "method should be 'avg_seeds' or 'by_seed'. " f"Got {method} instead."
        )

    return ranking_along_metrics


def means_with_stds(results, metrics, datasets_min_n_classes=None):

    results = filter_results_by_min_n_classes(results, datasets_min_n_classes)

    results_by_method = results.copy()
    results_by_method = results_by_method.groupby(["dataset", "method"]).mean(
        numeric_only=True
    )
    results_by_method = results_by_method.drop(columns=["seed"])
    results_by_method = results_by_method.reset_index()
    results_by_method = results_by_method.groupby(["method"]).mean(numeric_only=True)
    results_by_method = results_by_method[metrics].sort_values(by="method")
    results_by_method = results_by_method.reset_index()
    results_by_method.set_index("method", inplace=True)

    results_by_method_std = results.copy()
    results_by_method_std = results_by_method_std.groupby(["dataset", "method"]).std(
        numeric_only=True
    )
    results_by_method_std = results_by_method_std.drop(columns=["seed"])
    results_by_method_std = results_by_method_std.reset_index()
    results_by_method_std = results_by_method_std.groupby(["method"]).mean(numeric_only=True)
    results_by_method_std = results_by_method_std[metrics].sort_values(by="method")
    results_by_method_std = results_by_method_std.reset_index()
    results_by_method_std.set_index("method", inplace=True)

    results = pd.concat(
        [
            results_by_method.add_suffix(" (mean)"),
            results_by_method_std.add_suffix(" (std)"),
        ],
        axis=1,
    )
    sorted_columns = []
    for metric in metrics:
        sorted_columns.append(f"{metric} (mean)")
        sorted_columns.append(f"{metric} (std)")
    results = results[sorted_columns]

    return results
