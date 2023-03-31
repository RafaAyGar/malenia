import numpy as np
import pandas as pd



def export_results_by_dataset_metric_to_latex(results, metric_name, greaterIsBetter = True, means = None, stds_added = None, filename = None):

    results.results_by_method_dataset_std = results.results.groupby(["dataset", "method"]).std()
    results.results_by_method_dataset_std = results.results_by_method_dataset_std.drop(columns = ["seed"])
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

    rankings = results.get_rankings(metric_name, greaterIsBetter, s=means)
    a = pd.DataFrame([list(rankings)], columns=means.columns, index=["Rankings"])
    means = pd.concat([means, a], ignore_index=False)
    
    latex = pd.DataFrame(np.empty((len(means.index), len(means.columns)), dtype=str), columns=means.columns, index=means.index)
    
    results.means = means
    results.latex = latex

    # Apply formatting to the dataframe
    for dataset in means.index:
    
        # Format the accuracy values as mean with standard deviation
        for method in means.columns:
            mean = str(means.loc[dataset, method]).ljust(results.rounding_decimals + 2, '0')
            if dataset != "Rankings":
                std = str(stds.loc[dataset, method]).ljust(results.rounding_decimals + 2, '0')
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

        best_mean = str(means.loc[dataset, best_method]).ljust(results.rounding_decimals + 2, '0')
        sec_best_mean = str(means.loc[dataset, second_best_method]).ljust(results.rounding_decimals + 2, '0')
        if dataset != "Rankings":
            best_std = str(stds.loc[dataset, best_method]).ljust(results.rounding_decimals + 2, '0')
            sec_best_std = str(stds.loc[dataset, second_best_method]).ljust(results.rounding_decimals + 2, '0')
            latex.loc[dataset, best_method] = f"$\\mathbf{'{'}{best_mean}_{'{' + best_std + '}'}{'}'}$"
            latex.loc[dataset, second_best_method] = f"$\\mathit{'{'}{sec_best_mean}_{'{' + sec_best_std + '}'}{'}'}$"
        else:
            latex.loc[dataset, best_method] = f"$\\mathbf{'{'}{best_mean}{'}'}$"
            latex.loc[dataset, second_best_method] = f"$\\mathit{'{'}{sec_best_mean}{'}'}$"

    # Write the formatted dataframe to a LaTeX table file
    if not filename is None:
        with open(filename, 'w') as f:
            f.write("\\newcolumntype{M}[1]{>{\\arraybackslash}m{#1}}\n")
            f.write(latex.style.to_latex(column_format="@{}M{3.9cm}ccccc@{}"))
    
    return latex