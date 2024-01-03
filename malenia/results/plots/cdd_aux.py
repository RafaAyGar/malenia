import operator

import numpy as np
from scipy.stats import distributions, find_repeats, wilcoxon


def get_qalpha(alpha: float):
    """Get the alpha value for post hoc Nemenyi."""
    if alpha == 0.01:
        qalpha = [
            0.000,
            2.576,
            2.913,
            3.113,
            3.255,
            3.364,
            3.452,
            3.526,
            3.590,
            3.646,
            3.696,
            3.741,
            3.781,
            3.818,
            3.853,
            3.884,
            3.914,
            3.941,
            3.967,
            3.992,
            4.015,
            4.037,
            4.057,
            4.077,
            4.096,
            4.114,
            4.132,
            4.148,
            4.164,
            4.179,
            4.194,
            4.208,
            4.222,
            4.236,
            4.249,
            4.261,
            4.273,
            4.285,
            4.296,
            4.307,
            4.318,
            4.329,
            4.339,
            4.349,
            4.359,
            4.368,
            4.378,
            4.387,
            4.395,
            4.404,
            4.412,
            4.420,
            4.428,
            4.435,
            4.442,
            4.449,
            4.456,
        ]
    elif alpha == 0.05:
        qalpha = [
            0.000,
            1.960,
            2.344,
            2.569,
            2.728,
            2.850,
            2.948,
            3.031,
            3.102,
            3.164,
            3.219,
            3.268,
            3.313,
            3.354,
            3.391,
            3.426,
            3.458,
            3.489,
            3.517,
            3.544,
            3.569,
            3.593,
            3.616,
            3.637,
            3.658,
            3.678,
            3.696,
            3.714,
            3.732,
            3.749,
            3.765,
            3.780,
            3.795,
            3.810,
            3.824,
            3.837,
            3.850,
            3.863,
            3.876,
            3.888,
            3.899,
            3.911,
            3.922,
            3.933,
            3.943,
            3.954,
            3.964,
            3.973,
            3.983,
            3.992,
            4.001,
            4.009,
            4.017,
            4.025,
            4.032,
            4.040,
            4.046,
        ]
    elif alpha == 0.1:
        qalpha = [
            0.000,
            1.645,
            2.052,
            2.291,
            2.460,
            2.589,
            2.693,
            2.780,
            2.855,
            2.920,
            2.978,
            3.030,
            3.077,
            3.120,
            3.159,
            3.196,
            3.230,
            3.261,
            3.291,
            3.319,
            3.346,
            3.371,
            3.394,
            3.417,
            3.439,
            3.459,
            3.479,
            3.498,
            3.516,
            3.533,
            3.550,
            3.567,
            3.582,
            3.597,
            3.612,
            3.626,
            3.640,
            3.653,
            3.666,
            3.679,
            3.691,
            3.703,
            3.714,
            3.726,
            3.737,
            3.747,
            3.758,
            3.768,
            3.778,
            3.788,
            3.797,
            3.806,
            3.814,
            3.823,
            3.831,
            3.838,
            3.846,
        ]
        #
    else:
        raise Exception("alpha must be 0.01, 0.05 or 0.1")
    return qalpha


def _check_friedman(n_estimators, n_datasets, ranked_data, alpha):
    """
    Check whether Friedman test is significant.

    Larger parts of code copied from scipy.

    Arguments
    ---------
    n_estimators : int
      number of strategies to evaluate
    n_datasets : int
      number of datasets classified per strategy
    ranked_data : np.array (shape: n_estimators * n_datasets)
      rank of strategy on dataset

    Returns
    -------
    is_significant : bool
      Indicates whether strategies differ significantly in terms of performance
      (according to Friedman test).
    """
    if n_estimators < 3:
        raise ValueError(
            "At least 3 sets of measurements must be given for Friedmann test, "
            f"got {n_estimators}."
        )

    # calculate c to correct chisq for ties:
    ties = 0
    for i in range(n_datasets):
        replist, repnum = find_repeats(ranked_data[i])
        for t in repnum:
            ties += t * (t * t - 1)
    c = 1 - ties / (n_estimators * (n_estimators * n_estimators - 1) * n_datasets)

    ssbn = np.sum(ranked_data.sum(axis=0) ** 2)
    chisq = (
        12.0 / (n_estimators * n_datasets * (n_estimators + 1)) * ssbn
        - 3 * n_datasets * (n_estimators + 1)
    ) / c
    p = distributions.chi2.sf(chisq, n_estimators - 1)
    if p < alpha:
        is_significant = True
    else:
        is_significant = False
    return is_significant


def nemenyi_cliques(n_estimators, n_datasets, avranks, alpha):
    """Find cliques using post hoc Nemenyi test."""
    # Get critical value, there is an exact way now
    qalpha = get_qalpha(alpha)
    # calculate critical difference with Nemenyi
    cd = qalpha[n_estimators] * np.sqrt(n_estimators * (n_estimators + 1) / (6 * n_datasets))
    # compute statistically similar cliques
    cliques = np.tile(avranks, (n_estimators, 1)) - np.tile(np.vstack(avranks.T), (1, n_estimators))
    cliques[cliques < 0] = np.inf
    cliques = cliques < cd
    for i in range(n_estimators - 1, 0, -1):
        if np.all(cliques[i - 1, cliques[i, :]] == cliques[i, cliques[i, :]]):
            cliques[i, :] = 0

    n = np.sum(cliques, 1)
    cliques = cliques[n > 1, :]
    return cliques


def bonferroni_cliques(n_estimators, n_datasets, avranks, alpha):
    """Find cliques using post hoc Bonferroni test."""
    # Get critical value, there is an exact way now
    qalpha = get_qalpha(alpha)
    # calculate critical difference with Nemenyi
    cd = qalpha[n_estimators] * np.sqrt(n_estimators * (n_estimators + 1) / (6 * n_datasets))

    # compute statistically similar cliques
    cliques = np.tile(avranks, (n_estimators, 1)) - np.tile(np.vstack(avranks.T), (1, n_estimators))

    cliques[cliques < 0] = np.inf
    cliques = cliques < cd

    for i in range(n_estimators - 1, 0, -1):
        if np.all(cliques[i - 1, cliques[i, :]] == cliques[i, cliques[i, :]]):
            cliques[i, :] = 0

    n = np.sum(cliques, 1)
    cliques = cliques[n > 1, :]
    return cliques


def holm_cliques(results, labels, avranks, alpha):
    """Find cliques using Wilcoxon and post hoc Holm test."""
    # get number of strategies:
    results = np.array(results)
    n_estimators = results.shape[1]
    # init array that contains the p-values calculated by the Wilcoxon signed rank test
    p_values = []
    # loop through the algorithms to compare pairwise
    for i in range(n_estimators - 1):
        # get the name of classifier one
        classifier_1 = labels[i]
        # get the performance of classifier one
        perf_1 = np.array(results[:, i])

        for j in range(i + 1, n_estimators):
            # get the name of the second classifier
            classifier_2 = labels[j]
            # get the performance of classifier two
            perf_2 = np.array(results[:, j])
            # calculate the p_value
            p_value = wilcoxon(perf_1, perf_2, zero_method="wilcox")[1]
            # append to the list
            p_values.append((classifier_1, classifier_2, p_value, False))

    # get the number of hypothesis
    n_hypothesis = len(p_values)

    # sort the list in ascending manner of p-value
    p_values.sort(key=operator.itemgetter(2))

    # correct alpha with holm
    new_alpha = float(alpha / (n_estimators - 1))

    # print(new_alpha)

    ordered_labels = [i for _, i in sorted(zip(avranks, labels))]
    same = np.eye(len(ordered_labels), dtype=bool)

    # loop through the hypothesis
    for i in range(n_hypothesis):
        # test if significant after holm's correction of alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            idx_0 = np.where(np.array(ordered_labels) == p_values[i][0])[0][0]
            idx_1 = np.where(np.array(ordered_labels) == p_values[i][1])[0][0]
            same[idx_0][idx_1] = True
            same[idx_1][idx_0] = True

    # print(p_values)
    # print(ordered_labels)
    # print(same)

    # maybe this can be simplified.
    for i in range(n_estimators):
        for j in range(n_estimators):
            if i > j:
                same[i, j] = 0

    for i in range(n_estimators - 1, 0, -1):
        for j in range(i, 0, -1):
            if np.all(same[j - 1, same[i, :]] == same[i, same[i, :]]):
                same[i, :] = 0

    # maybe remove it.
    for i in range(n_estimators):
        for j in range(i, n_estimators - 1):
            if np.all(same[j + 1, same[i, :]] == same[i, same[i, :]]):
                same[i, :] = 0

    n = np.sum(same, 1)
    cliques = same[n > 1, :]

    # print(cliques)
    return cliques
