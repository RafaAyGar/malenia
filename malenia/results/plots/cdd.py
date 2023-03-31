import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from malenia.results.stats import wilcoxon_test

def plot_CDD(
    results,
    metric_name,
    greaterIsBetter=True,
    methods="all",
    s = None,
    alpha = 0.05,
    clique = None,
    savefile=None
): 
    
    if s is None:
        s = results.get_results_by_dataset_metric(metric_name)
        s = s.reset_index(drop=True)

    if methods != "all":
        s = s[methods]

    if greaterIsBetter:
        s = s * (-1)

    labels = list(s.columns)

    N,k = s.shape
    S = np.sort(s.T, axis=0)
    r = np.argsort(s.T, axis=0)
    idx = k * np.tile(np.arange(N), (k, 1)).T + r.T
    R = np.tile(np.arange(1, k+1), (N, 1))
    S = np.transpose(S)
    for i in range(N):
        for j in range(k):
            index = S[i,j] == S[i,:]
            R[i,index] = np.mean(R[i,index])


    r_flat = r.to_numpy().flatten()
    R_flat = R.flatten()
    idx_flat = idx.to_numpy().flatten()
    for i, i_val in enumerate(idx_flat):
        r_flat[i_val] = R_flat[i]
    r_flat = r_flat.reshape(r.shape, order='F')

    r = r_flat
    r = np.transpose(r)
    
    if alpha == 0.01:
        qalpha = np.array([0.0,2.576,2.913,3.113,3.255,3.364,3.452,3.526,3.59,3.646,3.696,3.741,3.781,3.818,3.853,3.884,3.914,3.941,3.967,3.992,4.015,4.037,4.057,4.077,4.096,4.114,4.132,4.148,4.164,4.179,4.194,4.208,4.222,4.236,4.249,4.261,4.273,4.285,4.296,4.307,4.318,4.329,4.339,4.349,4.359,4.368,4.378,4.387,4.395,4.404,4.412,4.42,4.428,4.435,4.442,4.449,4.456])
    elif alpha == 0.05:
        qalpha = np.array([0.0,1.96,2.344,2.569,2.728,2.85,2.948,3.031,3.102,3.164,3.219,3.268,3.313,3.354,3.391,3.426,3.458,3.489,3.517,3.544,3.569,3.593,3.616,3.637,3.658,3.678,3.696,3.714,3.732,3.749,3.765,3.78,3.795,3.81,3.824,3.837,3.85,3.863,3.876,3.888,3.899,3.911,3.922,3.933,3.943,3.954,3.964,3.973,3.983,3.992,4.001,4.009,4.017,4.025,4.032,4.04,4.046])
    elif alpha == 0.1:
        qalpha = np.array([0.0,1.645,2.052,2.291,2.46,2.589,2.693,2.78,2.855,2.92,2.978,3.03,3.077,3.12,3.159,3.196,3.23,3.261,3.291,3.319,3.346,3.371,3.394,3.417,3.439,3.459,3.479,3.498,3.516,3.533,3.55,3.567,3.582,3.597,3.612,3.626,3.64,3.653,3.666,3.679,3.691,3.703,3.714,3.726,3.737,3.747,3.758,3.768,3.778,3.788,3.797,3.806,3.814,3.823,3.831,3.838,3.846])
    else:
        raise Exception('alpha must be 0.01, 0.05 or 0.1')
    

    # compute critical difference
    cd = qalpha[k - 1] * np.sqrt(
        k * (k + 1) / (6 * N)
    )

    # set up plot
    plt.clf()
    fig, ax = plt.subplots(1)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 140)
    ax.set_axis_off()

    tics = np.tile(np.array(np.arange(k)) / (k - 1), (3, 1))
    plt.plot(
        tics.flatten("F"),
        np.tile([100, 105, 100], (1, k)).flatten(),
        linewidth=2,
        color="black",
    )
    tics = np.tile(
        (np.array(range(0, k - 1)) / (k - 1))
        + 0.5 / (k - 1),
        (3, 1),
    )
    plt.plot(
        tics.flatten("F"),
        np.tile([100, 102.5, 100], (1, k - 1)).flatten(),
        linewidth=1,
        color="black",
    )
    # plt.plot(
    #     [
    #         0,
    #         0,
    #         0,
    #         cd / (k - 1),
    #         cd / (k - 1),
    #         cd / (k - 1),
    #     ],
    #     [127, 123, 125, 125, 123, 127],
    #     linewidth=1,
    #     color="black",
    # )
    # plt.text(
    #     0.5 * cd / (k - 1),
    #     130,
    #     "CD",
    #     fontsize=12,
    #     horizontalalignment="center",
    # )

    for i in range(k):
        plt.text(
            i / (k - 1),
            110,
            str(k - i),
            fontsize=12,
            horizontalalignment="center",
        )

    # compute average rankss
    r = np.mean(r, axis=0)
    idx = np.argsort(r, axis=0)
    results.ranks_idx = idx
    r = np.sort(r, axis=0)

    # compute statistically similar cliques
    if clique is None:
        p_vals = wilcoxon_test(s)
        p_vals_index = p_vals["estimator_1"] + "_" + p_vals["estimator_2"]
        p_vals.set_index(p_vals_index, inplace=True)
        results.p_vals = p_vals
        sames = find_sames(p_vals, idx, s.columns.values, alpha)
        results.sames = sames
        clique = findCliques(np.array(sames))
        results.cliques = clique
    else:
        if clique.size > 0:
            clique = clique[:, idx] > 0
            
    clique = np.array(clique)
    n = clique.shape[0]

    # labels displayed on the right
    for i in range(int(math.ceil(k/2))):
        plt.plot(
            [
                (k - r[i]) / (k - 1),
                ((k - r[i])/(k - 1)),
                1.2
            ],
            [
                100,
                100 - 5 * (n + 1) - 10 * (i + 1),
                100 - 5 * (n + 1) - 10 * (i + 1)
            ],
            color='black'
        )
        plt.text(
            1.2,
            100 - 5 * (n + 1) - 10 * (i + 1) + 2, "%.2f" % r[i], fontsize=10, horizontalalignment='right')
        plt.text(1.25, 100 - 5*(n+1) - 10*(i+1), labels[idx[i]], fontsize=12, verticalalignment='center', horizontalalignment='left')


    # labels displayed on the left
    for i in range(int(np.ceil(k / 2)), k):
        plt.plot(
            [
                (k - r[i]) / (k - 1),
                (k - r[i]) / (k - 1),
                -0.2,
            ],
            [
                100,
                100 - 5 * (n + 1) - 10 * (k - i),
                100 - 5 * (n + 1) - 10 * (k - i),
            ],
            color="black",
        )
        plt.text(
            -0.2,
            100 - 5 * (n + 1) - 10 * (k - i) + 2,
            "%.2f" % r[i],
            fontsize=10,
            horizontalalignment="left",
        )
        plt.text(
            -0.25,
            100 - 5 * (n + 1) - 10 * (k - i),
            labels[idx[i]],
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="right",
        )

    # group cliques of statistically similar classifiers
    for i in range(np.size(clique, 0)):
        R = r[clique[i, :]]
        plt.plot(
            [
                ((k - np.min(R)) / (k - 1)) + 0.015,
                ((k - np.max(R)) / (k - 1)) - 0.015,
            ],
            [100 - 5 * (i + 1), 100 - 5 * (i + 1)],
            linewidth=6,
            color="black",
        )

    fig.dpi = 100

    if not savefile is None:
        fig.savefig(savefile, pad_inches = 0, bbox_inches='tight')

    plt.show()
    plt.clf()
    plt.close()


def get_rankings(results, metric, greaterIsBetter, s = None):
    if s is None:
        s = results.get_results_by_dataset_metric(metric)
        s = s.reset_index(drop=True)

    if greaterIsBetter:
        s = s * (-1)

    N,k = s.shape
    S = np.sort(s.T, axis=0)
    r = np.argsort(s.T, axis=0)
    idx = k * np.tile(np.arange(N), (k, 1)).T + r.T
    R = np.tile(np.arange(1, k+1), (N, 1))
    S = np.transpose(S)
    for i in range(N):
        for j in range(k):
            index = S[i,j] == S[i,:]
            R[i,index] = np.mean(R[i,index])

    r_flat = r.to_numpy().flatten()
    R_flat = R.flatten()
    idx_flat = idx.to_numpy().flatten()
    for i, i_val in enumerate(idx_flat):
        r_flat[i_val] = R_flat[i]
    r_flat = r_flat.reshape(r.shape, order='F')

    r = r_flat
    r = np.transpose(r)
    
    r = np.mean(r, axis=0)

    return r.round(results.rounding_decimals)


def find_sames(p_vals, idx, methods, alpha):
    same = pd.DataFrame(index=methods, columns=methods)
    for i in idx:
        for j in idx:
            method_A = methods[i]
            method_B = methods[j]
            if method_A == method_B:
                same.loc[method_A, method_B] = True
                continue

            if (method_A + "_" + method_B in p_vals.index):
                p_val = p_vals.loc[method_A + "_" + method_B]["p_val"]
            elif (method_B + "_" + method_A in p_vals.index):
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
    
    finalCliques = [[False]*len(same) for _ in range(len(cliques))]
    for i in range(len(cliques)):
        for j in range(len(cliques[i])):
            finalCliques[i][cliques[i][j]] = True
    
    return finalCliques


def growClique(same, clique):
    prevVal = clique[-1]
    if prevVal == len(same)-1:
        return

    cliqueStart = clique[0]
    nextVal = prevVal+1

    for col in range(cliqueStart, nextVal):
        if not same[nextVal][col]:
            return
    
    clique.append(nextVal)
    growClique(same, clique)


def testNewCliques():
    same = [
        [True,  True,  True,  False, False, False],
        [True,  True,  True,  True,  False, False],
        [True,  True,  True,  False, True,  True],
        [False, True,  False, True,  True,  True],
        [False, False, True,  True,  True,  True],
        [False, False, True,  True,  True,  True],
    ]
    
    noDifference = same