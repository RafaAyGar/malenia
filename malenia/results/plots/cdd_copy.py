import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from malenia.results.plots.cdd_aux import (_check_friedman, bonferroni_cliques,
                                           holm_cliques, nemenyi_cliques)


def plot_CDD(
    results,
    metric_name,
    greaterIsBetter=True,
    methods="all",
    s=None,
    alpha=0.05,
    cliques=None,
    clique_method="nemenyi",
    savefile_format="pdf",
    savefile=None,
):
    if s is None:
        s = results.get_results_by_dataset_metric(metric_name)
        s = s.reset_index(drop=True)

    if methods != "all":
        s = s[methods]

    if greaterIsBetter:
        s = s * (-1)

    labels = list(s.columns)

    n_datasets, n_methods = s.shape

    round_decimals = 4 if results is None else results.rounding_decimals
    r = get_rankings(s, round_decimals)

    # compute average rankss
    avg_r = np.mean(r, axis=0)
    idx = np.argsort(avg_r, axis=0)

    is_significant = _check_friedman(n_methods, n_datasets, r, alpha)
    # Step 4: If Friedman test is significant find cliques
    if is_significant:
        if cliques is None:
            if clique_method == "nemenyi":
                cliques = nemenyi_cliques(n_methods, n_datasets, avg_r, alpha)
            elif clique_method == "bonferroni":
                cliques = bonferroni_cliques(n_methods, n_datasets, avg_r, alpha)
            elif clique_method == "holm":
                cliques = holm_cliques(s, labels, avg_r, alpha)
            else:
                raise ValueError("clique methods available are only nemenyi, bonferroni and holm.")

    avg_r = np.sort(avg_r, axis=0)
    avg_r = avg_r.T

    # set up plot
    plt.clf()
    fig, ax = plt.subplots(1)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 140)
    ax.set_axis_off()

    tics = np.tile(np.array(np.arange(n_methods)) / (n_methods - 1), (3, 1))
    plt.plot(
        tics.flatten("F"),
        np.tile([100, 105, 100], (1, n_methods)).flatten(),
        linewidth=2,
        color="black",
    )
    tics = np.tile(
        (np.array(range(0, n_methods - 1)) / (n_methods - 1)) + 0.5 / (n_methods - 1),
        (3, 1),
    )
    plt.plot(
        tics.flatten("F"),
        np.tile([100, 102.5, 100], (1, n_methods - 1)).flatten(),
        linewidth=1,
        color="black",
    )

    for i in range(n_methods):
        plt.text(
            i / (n_methods - 1),
            110,
            str(n_methods - i),
            fontsize=12,
            horizontalalignment="center",
        )

    n = cliques.shape[0]

    # labels displayed on the right
    for i in range(int(math.ceil(n_methods / 2))):
        plt.plot(
            [
                (n_methods - avg_r[i]) / (n_methods - 1),
                ((n_methods - avg_r[i]) / (n_methods - 1)),
                1.2,
            ],
            [100, 100 - 5 * (n + 1) - 10 * (i + 1), 100 - 5 * (n + 1) - 10 * (i + 1)],
            color="black",
        )
        plt.text(
            1.2,
            100 - 5 * (n + 1) - 10 * (i + 1) + 2,
            "%.2f" % avg_r[i],
            fontsize=10,
            horizontalalignment="right",
        )
        plt.text(
            1.25,
            100 - 5 * (n + 1) - 10 * (i + 1),
            labels[idx[i]],
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="left",
        )

    # labels displayed on the left
    for i in range(int(np.ceil(n_methods / 2)), n_methods):
        plt.plot(
            [
                (n_methods - avg_r[i]) / (n_methods - 1),
                (n_methods - avg_r[i]) / (n_methods - 1),
                -0.2,
            ],
            [
                100,
                100 - 5 * (n + 1) - 10 * (n_methods - i),
                100 - 5 * (n + 1) - 10 * (n_methods - i),
            ],
            color="black",
        )
        plt.text(
            -0.2,
            100 - 5 * (n + 1) - 10 * (n_methods - i) + 2,
            "%.2f" % avg_r[i],
            fontsize=10,
            horizontalalignment="left",
        )
        plt.text(
            -0.25,
            100 - 5 * (n + 1) - 10 * (n_methods - i),
            labels[idx[i]],
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="right",
        )

    # group cliques of statistically similar classifiers
    for i in range(np.size(cliques, 0)):
        R = avg_r[cliques[i, :]]
        plt.plot(
            [
                ((n_methods - np.min(R)) / (n_methods - 1)) + 0.015,
                ((n_methods - np.max(R)) / (n_methods - 1)) - 0.015,
            ],
            [100 - 5 * (i + 1), 100 - 5 * (i + 1)],
            linewidth=6,
            color="black",
        )

    fig.dpi = 100

    if savefile is None:
        alpha_str = str(alpha).replace(".", "_")
        savefile = f"./results_files/cdd__{metric_name.upper()}__{alpha_str}.{savefile_format}"
    fig.savefig(savefile, pad_inches=0, bbox_inches="tight")

    plt.show()
    plt.clf()
    plt.close()


def get_rankings(s, rounding_decimals):
    ranks = rankdata(s, axis=1)

    ranks.round(rounding_decimals)

    return ranks


def find_sames(p_vals, idx, methods, alpha):
    same = pd.DataFrame(index=methods, columns=methods)
    for i in idx:
        for j in idx:
            method_A = methods[i]
            method_B = methods[j]
            if method_A == method_B:
                same.loc[method_A, method_B] = True
                continue

            if method_A + "_" + method_B in p_vals.index:
                p_val = p_vals.loc[method_A + "_" + method_B]["p_val"]
            elif method_B + "_" + method_A in p_vals.index:
                p_val = p_vals.loc[method_B + "_" + method_A]["p_val"]
            else:
                raise Exception("No p-value for {} and {}".format(method_A, method_B))

            if p_val < alpha:
                same.loc[method_A, method_B] = False
                same.loc[method_B, method_A] = False
            else:
                same.loc[method_A, method_B] = True
                same.loc[method_B, method_A] = True
    return same


def findCliques(same):
    cliques = []
    prevEndOfClique = 0

    for i in range(len(same)):
        clique = [i]
        growClique(same, clique)

        if len(clique) > 1:
            endOfClique = clique[-1]
            if endOfClique > prevEndOfClique:
                cliques.append(clique)
                prevEndOfClique = endOfClique

    finalCliques = [[False] * len(same) for _ in range(len(cliques))]
    for i in range(len(cliques)):
        for j in range(len(cliques[i])):
            finalCliques[i][cliques[i][j]] = True

    return finalCliques


def growClique(same, clique):
    prevVal = clique[-1]
    if prevVal == len(same) - 1:
        return

    cliqueStart = clique[0]
    nextVal = prevVal + 1

    for col in range(cliqueStart, nextVal):
        if not same[nextVal][col]:
            return

    clique.append(nextVal)
    growClique(same, clique)


def testNewCliques():
    same = [
        [True, True, True, False, False, False],
        [True, True, True, True, False, False],
        [True, True, True, False, True, True],
        [False, True, False, True, True, True],
        [False, False, True, True, True, True],
        [False, False, True, True, True, True],
    ]

    noDifference = same
