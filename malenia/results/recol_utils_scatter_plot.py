import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import AnchoredText

def plot_pairwise_scatter(
    r_results,
    method_a_name,
    method_b_name,
    method_a_name_pretty=None,
    method_b_name_pretty=None,
    metric="accuracy",
    metric_name_pretty=None,
    lower_better=False,
    statistic_tests=True,
    title=None,
    figsize=(8, 8),
    fontsize=16,
    color_palette=["#5E4AE3", "red", "#F3A712"],
    savefile=None
):
    """Plot a scatter that compares datasets' results achieved by two methods.

    Parameters
    ----------
    results_a : np.array
        Scores (either accuracies or errors) per dataset for the first approach.
    results_b : np.array
        Scores (either accuracies or errors) per dataset for the second approach.
    method_a : str
        Method name of the first approach.
    method_b : str
        Method name of the second approach.
    metric : str, default = "accuracy"
        Metric to be used for the comparison.
    lower_better : bool, default = False
        If True, lower values are considered better, i.e. errors.
    statistic_tests : bool, default = True
        If True, paired ttest and wilcoxon p-values are shown in the bottom of the plot.
    title : str, default = None
        Title to be shown in the top of the plot.
    figsize : tuple, default = (10, 6)
        Size of the figure.
    color_palette : str, default = "tab10"
        Color palette to be used for the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> from aeon.visualisation import plot_pairwise_scatter
    >>> from aeon.benchmarking.results_loaders import get_estimator_results_as_array
    >>> methods = ["InceptionTimeClassifier", "WEASEL-Dilation"]
    >>> results = get_estimator_results_as_array(estimators=methods)  # doctest: +SKIP
    >>> plot = plot_pairwise_scatter(  # doctest: +SKIP
    ...     results[0], methods[0], methods[1])  # doctest: +SKIP
    >>> plot.show()  # doctest: +SKIP
    >>> plot.savefig("scatterplot.pdf")  # doctest: +SKIP
    """

    palette = color_palette

    res = r_results.groupby(["dataset", "method"]).mean().reset_index().pivot(index="dataset", columns="method", values=metric)
    results_a = res[method_a_name].to_numpy()
    results_b = res[method_b_name].to_numpy()

    if method_a_name_pretty is not None:
        method_a_name = method_a_name_pretty
    if method_b_name_pretty is not None:
        method_b_name = method_b_name_pretty
    if metric_name_pretty is not None:
        metric = metric_name_pretty

    if len(results_a.shape) != 1:
        raise ValueError("results_a must be a 1D array.")
    if len(results_b.shape) != 1:
        raise ValueError("results_b must be a 1D array.")

    if statistic_tests:
        fig, ax = plt.subplots(figsize=figsize, gridspec_kw=dict(bottom=0.2))
    else:
        fig, ax = plt.subplots(figsize=figsize)

    results_all = np.concatenate((results_a, results_b))
    min_value = results_all.min() * 0.97
    max_value = results_all.max() * 1.03

    if not lower_better:
        max_value = min(max_value, 1.001)

    x, y = [min_value, max_value], [min_value, max_value]
    ax.plot(x, y, color="black", alpha=0.5, zorder=1)

    # Choose the appropriate order for the methods. Best method is shown in the y-axis.
    # if (results_a.mean() <= results_b.mean() and not lower_better) or (
    #     results_a.mean() >= results_b.mean() and lower_better
    # ):
    top_res = results_a
    top_method = method_a_name
    right_res = results_b
    right_method = method_b_name

    differences = [
        0 if i - j == 0 else (1 if i - j > 0 else -1) for i, j in zip(right_res, top_res)
    ]
    # This line helps displaying ties on top of losses and wins, as in general there
    # are less number of ties than wins/losses.
    differences, right_res, top_res = map(
        np.array,
        zip(*sorted(zip(differences, right_res, top_res), key=lambda x: -abs(x[0]))),
    )

    first_median = np.median(right_res)
    second_median = np.median(top_res)

    plot = sns.scatterplot(
        x=top_res,
        y=right_res,
        hue=differences,
        hue_order=[1, 0, -1] if lower_better else [-1, 0, 1],
        palette=palette,
        zorder=2,
    )

    # Draw the median value per method as a dashed line from 0 to the median value.
    ax.plot(
        [first_median, min_value] if not lower_better else [first_median, max_value],
        [first_median, first_median],
        linestyle="--",
        color=palette[2],
        zorder=3,
    )

    ax.plot(
        [second_median, second_median],
        [second_median, min_value] if not lower_better else [second_median, max_value],
        linestyle="--",
        color=palette[0],
        zorder=3,
    )

    # legend_median = AnchoredText(
    #     "*Dashed lines represent the median",
    #     loc="lower right" if lower_better else "upper right",
    #     prop=dict(size=8),
    #     bbox_to_anchor=(1.01, 1.07 if lower_better else -0.07),
    #     bbox_transform=ax.transAxes,
    # )
    # ax.add_artist(legend_median)

    # Compute the W, T, and L per methods
    if lower_better:
        differences = [-i for i in differences]
        ax = plt.gca()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.spines["top"].set_visible(True)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.spines["right"].set_visible(True)

    # Setting labels for x and y axis
    plot.set_ylabel(f"{right_method} {metric}\n(mean: {right_res.mean():.3f})", fontsize=fontsize)
    plot.set_xlabel(
        f"{top_method} {metric}\n(mean: {top_res.mean():.3f})", fontsize=fontsize
    )

    wins_A = losses_B = sum(i == 1 for i in differences)
    ties_A = ties_B = sum(i == 0 for i in differences)
    losses_A = wins_B = sum(i == -1 for i in differences)

    # Setting x and y limits
    plot.set_ylim(min_value, max_value)
    plot.set_xlim(min_value, max_value)

    # Remove legend
    plot.get_legend().remove()

    # Setting text with W, T and L for each method
    anc = AnchoredText(
        f"{right_method} wins here\n[{wins_A}W, {ties_A}T, {losses_A}L]",
        loc="upper left" if not lower_better else "lower right",
        frameon=True,
        prop=dict(
            color="#363636",
            # fontweight="bold",
            fontsize=fontsize,
            ha="center",
        ),
    )
    anc.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    anc.patch.set_color("wheat")
    anc.patch.set_edgecolor("black")
    anc.patch.set_alpha(0.5)
    ax.add_artist(anc)

    anc = AnchoredText(
        f"{top_method} wins here\n[{wins_B}W, {ties_B}T, {losses_B}L]",
        loc="lower right" if not lower_better else "upper left",
        frameon=True,
        prop=dict(
            # color=palette[0],
            color="#363636",
            # fontweight="bold",
            fontsize=fontsize,
            ha="center",
        ),
    )
    anc.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    anc.patch.set_color("wheat")
    anc.patch.set_edgecolor("black")
    anc.patch.set_alpha(0.5)
    ax.add_artist(anc)

    # Setting title if provided.
    if title is not None:
        plot.set_title(rf"{title}", fontsize=fontsize)

    if savefile is not None:
        plt.savefig(savefile, bbox_inches="tight", dpi=300)

    return fig, ax
