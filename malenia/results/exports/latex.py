import numpy as np
import pandas as pd
from malenia.results.plots.cdd import get_rankings


def export_mean_and_std_dataframes_to_latex(
    df_means,
    df_stds,
    greaterIsBetter=True,
    rounding_decimals=3,
    filename=None,
):
    """
    Received df_means and df_stds must have same shape and represent a result table with
    methods as columns and datasets as rows.
    Example of df_means:
    >>> Dataset     Method_1    Method_2    Method_3
    >>> Car         0.1         0.2         0.3
    >>> Titanic     0.2         0.3         0.4
    >>> Iris        0.3         0.4         0.5
    """

    assert df_means.shape == df_stds.shape
    data_means = df_means.round(rounding_decimals).copy()
    data_stds = df_stds.round(rounding_decimals).copy()
    latex = pd.DataFrame(
        np.empty((len(data_means.index), len(data_means.columns)), dtype=str),
        columns=data_means.columns,
        index=data_means.index,
    )

    for dataset in data_means.index:
        # Apply formatting to the dataframe
        # Format the accuracy values as mean with standard deviation
        for method in data_means.columns:
            mean = str(data_means.loc[dataset, method]).ljust(4, "0")
            std = str(data_stds.loc[dataset, method]).ljust(4, "0")
            latex.loc[dataset, method] = f"${mean}_{'{' + std + '}'}$"

        values_to_extract_best = data_means.loc[dataset, :]

        if greaterIsBetter and dataset != "Rankings":
            best_method = values_to_extract_best.idxmax()
            second_best_method = values_to_extract_best.drop(best_method).idxmax()
        else:
            best_method = values_to_extract_best.idxmin()
            second_best_method = values_to_extract_best.drop(best_method).idxmin()

        best_mean = str(data_means.loc[dataset, best_method]).ljust(4, "0")
        sec_best_mean = str(data_means.loc[dataset, second_best_method]).ljust(4, "0")

        if dataset != "Rankings":
            best_std = str(data_stds.loc[dataset, best_method]).ljust(4, "0")
            sec_best_std = str(data_stds.loc[dataset, second_best_method]).ljust(4, "0")
            latex.loc[dataset, best_method] = f"$\\mathbf{'{'}{best_mean}_{'{' + best_std + '}'}{'}'}$"
            latex.loc[dataset, second_best_method] = (
                f"$\\mathit{'{'}{sec_best_mean}_{'{' + sec_best_std + '}'}{'}'}$"
            )
        else:
            latex.loc[dataset, best_method] = f"$\\mathbf{'{'}{best_mean}{'}'}$"
            latex.loc[dataset, second_best_method] = f"$\\mathit{'{'}{sec_best_mean}{'}'}$"

    latex = latex.T

    # Write the formatted dataframe to a LaTeX table file
    if filename is not None:
        with open(filename, "w") as f:
            f.write("\\newcolumntype{M}[1]{>{\\arraybackslash}m{#1}}\n")
            f.write(latex.style.to_latex(column_format="@{}M{3.9cm}ccccc@{}"))

    return latex


def export_dataframe_to_latex(
    df,
    greaterIsBetter=True,
    rounding_decimals=3,
    filename=None,
):
    data = df.round(rounding_decimals).copy()
    latex = pd.DataFrame(
        np.empty((len(data.index), len(data.columns)), dtype=str),
        columns=data.columns,
        index=data.index,
    )

    # Apply formatting to the dataframe
    for dataset in data.index:
        # Format the accuracy values as mean with standard deviation
        for method in data.columns:
            mean = str(data.loc[dataset, method]).ljust(4, "0")
            latex.loc[dataset, method] = f"${mean}$"

        # Find the best and second-best methods for the current dataset
        if greaterIsBetter and dataset != "Rankings":
            best_method = data.loc[dataset].idxmax()
            second_best_method = data.loc[dataset].drop(best_method).idxmax()
        else:
            best_method = data.loc[dataset].idxmin()
            second_best_method = data.loc[dataset].drop(best_method).idxmin()

        best_mean = str(data.loc[dataset, best_method]).ljust(4, "0")
        sec_best_mean = str(data.loc[dataset, second_best_method]).ljust(4, "0")
        if dataset != "Rankings":
            # best_std = str(stds.loc[dataset, best_method]).ljust(
            #     4, "0"
            # )
            # sec_best_std = str(stds.loc[dataset, second_best_method]).ljust(
            #     4, "0"
            # )
            latex.loc[dataset, best_method] = f"$\\mathbf{'{'}{best_mean}{'}'}$"
            latex.loc[dataset, second_best_method] = f"$\\mathit{'{'}{sec_best_mean}{'}'}$"
        else:
            latex.loc[dataset, best_method] = f"$\\mathbf{'{'}{best_mean}{'}'}$"
            latex.loc[dataset, second_best_method] = f"$\\mathit{'{'}{sec_best_mean}{'}'}$"

    latex = latex.T

    # Write the formatted dataframe to a LaTeX table file
    if not filename is None:
        with open(filename, "w") as f:
            f.write("\\newcolumntype{M}[1]{>{\\arraybackslash}m{#1}}\n")
            f.write(latex.style.to_latex(column_format="@{}M{3.9cm}ccccc@{}"))

    return latex


def export_results_method_to_latex(
    results,
    metrics="all",
    greaterIsBetter=True,
    means=None,
    filename=None,
):
    if metrics == "all":
        metrics = list(results.metrics.keys())

    means = (
        results.results.drop(columns=["dataset", "seed", "runtime"])
        .groupby(["method"])
        .mean()
        .reset_index()
        .round(results.rounding_decimals)
    )
    stds = (
        results.results.drop(columns=["dataset", "seed", "runtime"])
        .groupby(["method"])
        .std()
        .reset_index()
        .round(results.rounding_decimals)
    )

    latex = pd.DataFrame(
        np.empty((len(means.index), len(means.columns)), dtype=str),
        columns=means.columns,
        index=means.index,
    )

    for metric in metrics:
        for method in means["method"].values:
            means[means["method"] == method][metric]
            mean = str(means[means["method"] == method][metric].values[0]).ljust(
                results.rounding_decimals + 2, "0"
            )
            std = str(stds[stds["method"] == method][metric].values[0]).ljust(results.rounding_decimals + 2, "0")
            latex.loc[stds["method"] == method, metric] = f"${mean}_{'{' + std + '}'}$"

        # Find the best and second-best methods for the current dataset

        if greaterIsBetter:
            best_method = means[metric].argmax()
            second_best_method = means.drop(best_method)[metric].idxmax()
        else:
            best_method = means[metric].argmin()
            second_best_method = means.drop(best_method)[metric].idxmin()

        best_mean = str(means[metric].iloc[best_method]).ljust(results.rounding_decimals + 2, "0")
        sec_best_mean = str(means[metric].iloc[second_best_method]).ljust(results.rounding_decimals + 2, "0")

        best_std = str(stds[metric].iloc[best_method]).ljust(results.rounding_decimals + 2, "0")
        sec_best_std = str(stds[metric].iloc[second_best_method]).ljust(results.rounding_decimals + 2, "0")

        latex[metric].iloc[best_method] = f"$\\mathbf{'{'}{best_mean}_{'{' + best_std + '}'}{'}'}$"
        latex[metric].iloc[second_best_method] = f"$\\mathit{'{'}{sec_best_mean}_{'{' + sec_best_std + '}'}{'}'}$"

    if not filename is None:
        with open(filename, "w") as f:
            f.write("\\newcolumntype{M}[1]{>{\\arraybackslash}m{#1}}\n")
            f.write(latex.style.to_latex(column_format="@{}M{3.9cm}ccccc@{}"))

    return latex


def export_results_by_dataset_metric_to_latex(
    results,
    metric_name,
    greaterIsBetter=True,
    means=None,
    stds_added=None,
    filename=None,
):
    results.results_by_method_dataset_std = results.results.groupby(["dataset", "method"]).std()
    results.results_by_method_dataset_std = results.results_by_method_dataset_std.drop(columns=["seed"])
    results.results_by_method_dataset_std = results.results_by_method_dataset_std.reset_index()
    results.results_by_method_dataset_std = results.results_by_method_dataset_std.round(results.rounding_decimals)
    stds = results.get_results_by_dataset_metric(metric_name, results.results_by_method_dataset_std)

    if means is None and stds_added is None:
        means = results.get_results_by_dataset_metric(metric_name)
    else:
        means = means.round(results.rounding_decimals)
        stds_added = stds_added.round(results.rounding_decimals)
        stds = pd.concat([stds, stds_added], axis=1)

    results.stds = stds

    rankings = get_rankings(results, metric_name, greaterIsBetter, s=means)
    a = pd.DataFrame([list(rankings)], columns=means.columns, index=["Rankings"])
    means = pd.concat([means, a], ignore_index=False)

    latex = pd.DataFrame(
        np.empty((len(means.index), len(means.columns)), dtype=str),
        columns=means.columns,
        index=means.index,
    )

    results.means = means
    results.latex = latex

    # Apply formatting to the dataframe
    for dataset in means.index:
        # Format the accuracy values as mean with standard deviation
        for method in means.columns:
            mean = str(means.loc[dataset, method]).ljust(results.rounding_decimals + 2, "0")
            if dataset != "Rankings":
                std = str(stds.loc[dataset, method]).ljust(results.rounding_decimals + 2, "0")
                latex.loc[dataset, method] = f"${mean}_{'{' + std + '}'}$"
            else:
                latex.loc[dataset, method] = f"${mean}$"

        # Find the best and second-best methods for the current dataset
        if greaterIsBetter and dataset != "Rankings":
            best_method = means.loc[dataset].idxmax()
            second_best_method = means.loc[dataset].drop(best_method).idxmax()
        else:
            best_method = means.loc[dataset].idxmin()
            second_best_method = means.loc[dataset].drop(best_method).idxmin()

        best_mean = str(means.loc[dataset, best_method]).ljust(results.rounding_decimals + 2, "0")
        sec_best_mean = str(means.loc[dataset, second_best_method]).ljust(results.rounding_decimals + 2, "0")
        if dataset != "Rankings":
            best_std = str(stds.loc[dataset, best_method]).ljust(results.rounding_decimals + 2, "0")
            sec_best_std = str(stds.loc[dataset, second_best_method]).ljust(results.rounding_decimals + 2, "0")
            latex.loc[dataset, best_method] = f"$\\mathbf{'{'}{best_mean}_{'{' + best_std + '}'}{'}'}$"
            latex.loc[dataset, second_best_method] = (
                f"$\\mathit{'{'}{sec_best_mean}_{'{' + sec_best_std + '}'}{'}'}$"
            )
        else:
            latex.loc[dataset, best_method] = f"$\\mathbf{'{'}{best_mean}{'}'}$"
            latex.loc[dataset, second_best_method] = f"$\\mathit{'{'}{sec_best_mean}{'}'}$"

    # Write the formatted dataframe to a LaTeX table file
    if not filename is None:
        with open(filename, "w") as f:
            f.write("\\newcolumntype{M}[1]{>{\\arraybackslash}m{#1}}\n")
            f.write(latex.style.to_latex(column_format="@{}M{3.9cm}ccccc@{}"))

    return latex


def export_wilcoxon_table_to_latex(wilcoxon_dataframe, alpha=0.01, filename=None):
    for i in range(wilcoxon_dataframe.shape[0]):
        for j in range(wilcoxon_dataframe.shape[1]):
            value = wilcoxon_dataframe.iloc[i, j]
            if value < alpha:
                wilcoxon_dataframe.iloc[i, j] = f"$\\mathbf{'{< 0.01}'}$"

    # Write the formatted dataframe to a LaTeX table file
    if not filename is None:
        with open(filename, "w") as f:
            # f.write("\\newcolumntype{M}[1]{>{\\arraybackslash}m{#1}}\n")
            n_cols = wilcoxon_dataframe.shape[1]
            cols_string = ""
            for i in range(n_cols):
                cols_string += "c"
            f.write(wilcoxon_dataframe.style.to_latex(column_format="@{}" + cols_string + "@{}"))

    return wilcoxon_dataframe
