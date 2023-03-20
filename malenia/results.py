import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import mean
from malenia.results_utils import findCliques

class Results:
    def __init__(
        self,
        datasets,
        methods,
        metrics,
        results_path,
        seeds = 30,
        rounding_decimals = 4
    ):
        self.datasets = datasets
        self.methods = methods
        self.metrics = metrics
        self.results_path = results_path
        self.seeds = seeds
        self.rounding_decimals = rounding_decimals

        self.results = {
            "dataset" : [],
            "method" : [],
            "seed" : []
        }

        if type(self.datasets) == str:
            self.datasets = os.listdir(self.datasets)

    def _extract_global_and_specific_method_name(self, method_name):
        method_name_global = method_name.split("_")[0]
        method_name_specif = ""
        for part in method_name.split("_")[1:]:
            method_name_specif += part + "_"
        method_name_specif = method_name_specif[:-1] # remove last "_"

        return method_name_global, method_name_specif
    
    def _method_results_info_generator(self, method, train_or_test):
        global_method_name, specif_method_name = self._extract_global_and_specific_method_name(method)
        for specified_dataset in self.datasets:
            for seed in range(self.seeds):
                path = os.path.join(
                    self.results_path,
                    global_method_name,
                    specif_method_name,
                    specified_dataset,
                    "seed_" + str(seed) + "_" + train_or_test + ".csv"
                )
                yield (
                    specified_dataset,
                    seed,
                    path
                )

    def _get_metric_results(self, metric, results_path):
        df = pd.read_csv(results_path)
        y_true = df["y_true"]
        y_pred = df["y_pred"]
        return metric.compute(y_true, y_pred)


    def evaluate(self, group_separated_dims = True, verbose = True):
        self.methods_real_names = []
        for method_pretty, method_info in self.methods.items():
            method_real, isMultivariate = method_info
            self.methods_real_names.append(method_real)
            for dataset, seed, results_path in self._method_results_info_generator(method_real, "test"):
                if isMultivariate and "dim" in dataset:
                    continue
                if os.path.exists(results_path) == False:
                    if verbose:
                        print("* WARNING: results file not found: ", results_path)
                    continue
                if group_separated_dims:
                    dataset = dataset.split("_dim")[0]
                self.results["dataset"].append(dataset)
                self.results["method"].append(method_pretty)
                self.results["seed"].append(seed)
                for metric_name, metric in self.metrics.items():
                    if not metric_name in self.results:
                        self.results[metric_name] = []
                    self.results[metric_name].append(self._get_metric_results(metric, results_path))
        self.results = pd.DataFrame(self.results)
        self.results = self.results.round(self.rounding_decimals)
        self.results_by_method_dataset = self.results.groupby(["dataset", "method"]).mean()
        self.results_by_method_dataset = self.results_by_method_dataset.drop(columns = ["seed"])
        self.results_by_method_dataset = self.results_by_method_dataset.reset_index()
        self.results_by_method_dataset = self.results_by_method_dataset.round(self.rounding_decimals)


    def get_wins_ties_losses(self, metric_name):
        win_tie_losses = pd.DataFrame()
        results_by_method = self.get_results_by_method()
        for strat1 in results_by_method['method']:
            col = "VS " + strat1
            s1 = results_by_method[results_by_method['method']==strat1][metric_name].values
            win_losses = []
            for strat2 in results_by_method['method']:
                if strat1 == strat2:
                    win_losses.append("---")
                    continue
                s2 = results_by_method[results_by_method['method']==strat2][metric_name].values
                w = 0
                l = 0
                t = 0
                for r1, r2 in zip(s1, s2):
                    if r1 > r2:
                        w += 1
                    elif r1 < r2:
                        l += 1
                    else:
                        t += 1
                wlt_string = str(w) + "W | " + str(l) + "L | " + str(t) + "T"
                win_losses.append(wlt_string)
            win_tie_losses[col] = win_losses
        return win_tie_losses


    def get_results_by_method(self):
        self.results_by_method = self.results_by_method_dataset.groupby(["method"]).mean()
        self.results_by_method = self.results_by_method.sort_values(by = list(self.metrics.keys()), ascending = False)
        self.results_by_method = self.results_by_method.reset_index()
        self.results_by_method.set_index("method", inplace = True)
        self.results_by_method = self.results_by_method.round(self.rounding_decimals)
        return self.results_by_method
    

    def get_results_by_dataset_metric(self, metric):
        df = self.results_by_method_dataset.sort_values(["dataset", "method"])
        results_by_methods_dataset_metric = df[df['method'] == self.methods_real_names[0]].copy()
        for pretty_method_name in self.methods.keys():
            method_results = list(df[df['method'] == pretty_method_name][metric])
            results_by_methods_dataset_metric[pretty_method_name] = method_results
            results_by_methods_dataset_metric[pretty_method_name].round(self.rounding_decimals)
            results_by_methods_dataset_metric['dataset'] = df['dataset'].unique()
        final_results_by_methods_dataset_metric = results_by_methods_dataset_metric[
            ["dataset"] + list(self.methods.keys())
        ].copy()
        final_results_by_methods_dataset_metric = final_results_by_methods_dataset_metric.reset_index(drop=True)
        final_results_by_methods_dataset_metric.sort_values(by=["dataset"])
        final_results_by_methods_dataset_metric.set_index("dataset", inplace=True)
        final_results_by_methods_dataset_metric = final_results_by_methods_dataset_metric.round(self.rounding_decimals)
        return final_results_by_methods_dataset_metric  
    
    def plot_Boxplot(self, metric_name, figsize=(16, 8)):
        plt.style.use("seaborn")
        plt.rcParams["figure.figsize"] = figsize
        plt.rcParams["figure.autolayout"] = True

        title_font = {'color':'black', 'weight':'bold',
                    'verticalalignment':'bottom'}

        metric_results_by_dataset = self.get_results_by_dataset_metric(metric_name)
        metric_results_by_dataset.plot(kind='box', title= "Comparison in terms of " + metric_name + " metric")
        plt.title("Comparison in terms of " + metric_name, **title_font)
        plt.ylim(0.0, metric_results_by_dataset.to_numpy().max() + 0.5)
        plt.show()


    def cd(self, metric_name, greaterIsBetter=True, methods="all", name = "CCD Plot", s = None, alpha = 0.1, clique = None, savefile=None): 
        
        if s is None:
            s = self.get_results_by_dataset_metric(metric_name)
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
                R[i,index] = mean(R[i,index])


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
        cd = qalpha[2 - 1] * np.sqrt(
            2 * (2 + 1) / (6 * N)
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
        plt.plot(
            [
                0,
                0,
                0,
                cd / (k - 1),
                cd / (k - 1),
                cd / (k - 1),
            ],
            [127, 123, 125, 125, 123, 127],
            linewidth=1,
            color="black",
        )
        plt.text(
            0.5 * cd / (k - 1),
            130,
            "CD",
            fontsize=12,
            horizontalalignment="center",
        )

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
        r = np.sort(r, axis=0)

        # compute statistically similar cliques
        if clique is None:
            clique = np.tile(r, (k, 1)) - np.tile(r.reshape(-1, 1), (1, k))
            clique[clique < 0] = np.finfo(float).max
            clique = clique < cd
            clique = findCliques(clique.T)
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
        # return fig, ax



    ## OLD SKTIME CDD PLOT
    # 
    # def plot_CDD(self, metric_name=None, alpha=0.1, data="default", save_file=None):
    #     """Plot critical difference diagrams.

    #     References
    #     ----------
    #     original implementation by Aaron Bostrom, modified by Markus Löning.
    #     """
    #     if isinstance(data, str) and data == "default":
    #         dataset = self.results_by_method_dataset
    #     else:
    #         dataset = data

    #     data = (
    #         dataset.copy()
    #         .loc[:, ["dataset", "method", metric_name]]
    #         .pivot(index="method", columns="dataset", values=metric_name)
    #         # .sort_values(by=["strategy"], ascending=False)
    #         .values
    #     )

    #     if "mae" in metric_name:
    #         data = data * (-1)
    #     n_datasets, n_methods = data.shape  # [n_datasets,n_methods] = size(s); correct
    #     labels = list(dataset['method'])

    #     r = np.argsort(data, axis=0)[::-1]
    #     S = np.sort(data, axis=0)[::-1]
    #     idx = n_datasets * np.tile(np.arange(n_methods), (n_datasets, 1)).T + r.T
    #     R = np.asfarray(np.tile(np.arange(n_datasets) + 1, (n_methods, 1)))
    #     S = S.T

    #     for i in range(n_methods):
    #         for j in range(n_datasets):
    #             index = S[i, j] == S[i, :]
    #             R[i, index] = np.mean(R[i, index], dtype=np.float64)

    #     r = np.asfarray(r)
    #     r.T.flat[idx] = R
    #     r = r.T

    #     if alpha == 0.01:
    #         # fmt: off
    #         qalpha = [0.000, 2.576, 2.913, 3.113, 3.255, 3.364, 3.452, 3.526,
    #                   3.590, 3.646, 3.696, 3.741, 3.781, 3.818,
    #                   3.853, 3.884, 3.914, 3.941, 3.967, 3.992, 4.015, 4.037,
    #                   4.057, 4.077, 4.096, 4.114, 4.132, 4.148,
    #                   4.164, 4.179, 4.194, 4.208, 4.222, 4.236, 4.249, 4.261,
    #                   4.273, 4.285, 4.296, 4.307, 4.318, 4.329,
    #                   4.339, 4.349, 4.359, 4.368, 4.378, 4.387, 4.395, 4.404,
    #                   4.412, 4.420, 4.428, 4.435, 4.442, 4.449,
    #                   4.456]
    #         # fmt: on
    #     elif alpha == 0.05:
    #         # fmt: off
    #         qalpha = [0.000, 1.960, 2.344, 2.569, 2.728, 2.850, 2.948, 3.031,
    #                   3.102, 3.164, 3.219, 3.268, 3.313, 3.354,
    #                   3.391, 3.426, 3.458, 3.489, 3.517, 3.544, 3.569, 3.593,
    #                   3.616, 3.637, 3.658, 3.678, 3.696, 3.714,
    #                   3.732, 3.749, 3.765, 3.780, 3.795, 3.810, 3.824, 3.837,
    #                   3.850, 3.863, 3.876, 3.888, 3.899, 3.911,
    #                   3.922, 3.933, 3.943, 3.954, 3.964, 3.973, 3.983, 3.992,
    #                   4.001, 4.009, 4.017, 4.025, 4.032, 4.040,
    #                   4.046]
    #         # fmt: on
    #     elif alpha == 0.1:
    #         # fmt: off
    #         qalpha = [0.000, 1.645, 2.052, 2.291, 2.460, 2.589, 2.693, 2.780,
    #                   2.855, 2.920, 2.978, 3.030, 3.077, 3.120,
    #                   3.159, 3.196, 3.230, 3.261, 3.291, 3.319, 3.346, 3.371,
    #                   3.394, 3.417, 3.439, 3.459, 3.479, 3.498,
    #                   3.516, 3.533, 3.550, 3.567, 3.582, 3.597, 3.612, 3.626,
    #                   3.640, 3.653, 3.666, 3.679, 3.691, 3.703,
    #                   3.714, 3.726, 3.737, 3.747, 3.758, 3.768, 3.778, 3.788,
    #                   3.797, 3.806, 3.814, 3.823, 3.831, 3.838,
    #                   3.846]
    #         # fmt: on
    #     else:
    #         raise Exception("alpha must be 0.01, 0.05 or 0.1")

    #     cd = qalpha[n_datasets - 1] * np.sqrt(
    #         n_datasets * (n_datasets + 1) / (6 * n_methods)
    #     )

    #     # set up plot
    #     fig, ax = plt.subplots(1)
    #     ax.set_xlim(-0.5, 1.5)
    #     ax.set_ylim(0, 140)
    #     ax.set_axis_off()

    #     tics = np.tile(np.array(np.arange(n_datasets)) / (n_datasets - 1), (3, 1))
    #     plt.plot(
    #         tics.flatten("F"),
    #         np.tile([100, 105, 100], (1, n_datasets)).flatten(),
    #         linewidth=2,
    #         color="black",
    #     )
    #     tics = np.tile(
    #         (np.array(range(0, n_datasets - 1)) / (n_datasets - 1))
    #         + 0.5 / (n_datasets - 1),
    #         (3, 1),
    #     )
    #     plt.plot(
    #         tics.flatten("F"),
    #         np.tile([100, 102.5, 100], (1, n_datasets - 1)).flatten(),
    #         linewidth=1,
    #         color="black",
    #     )
    #     plt.plot(
    #         [
    #             0,
    #             0,
    #             0,
    #             cd / (n_datasets - 1),
    #             cd / (n_datasets - 1),
    #             cd / (n_datasets - 1),
    #         ],
    #         [127, 123, 125, 125, 123, 127],
    #         linewidth=1,
    #         color="black",
    #     )
    #     plt.text(
    #         0.5 * cd / (n_datasets - 1),
    #         130,
    #         "CD",
    #         fontsize=12,
    #         horizontalalignment="center",
    #     )

    #     for i in range(n_datasets):
    #         plt.text(
    #             i / (n_datasets - 1),
    #             110,
    #             str(n_datasets - i),
    #             fontsize=12,
    #             horizontalalignment="center",
    #         )

    #     # compute average rankss
    #     # print(r)
    #     r = np.mean(r, axis=0)
    #     idx = np.argsort(r, axis=0)
    #     r = np.sort(r, axis=0)
    #     # print(r)

    #     # compute statistically similar cliques
    #     clique = np.tile(r, (n_datasets, 1)) - np.tile(
    #         np.vstack(r.T), (1, n_datasets)
    #     )
    #     clique[clique < 0] = np.inf
    #     clique = clique < cd

    #     for i in range(n_datasets - 1, 0, -1):
    #         if np.all(clique[i - 1, clique[i, :]] == clique[i, clique[i, :]]):
    #             clique[i, :] = 0

    #     n = np.sum(clique, 1)
    #     clique = clique[n > 1, :]
    #     n = np.size(clique, 0)

    #     for i in range(int(np.ceil(n_datasets / 2))):
    #         plt.plot(
    #             [
    #                 (n_datasets - r[i]) / (n_datasets - 1),
    #                 (n_datasets - r[i]) / (n_datasets - 1),
    #                 1.2,
    #             ],
    #             [
    #                 100,
    #                 100 - 5 * (n + 1) - 10 * (i + 1),
    #                 100 - 5 * (n + 1) - 10 * (i + 1),
    #             ],
    #             color="black",
    #         )
    #         plt.text(
    #             1.2,
    #             100 - 5 * (n + 1) - 10 * (i + 1) + 2,
    #             "%.2f" % r[i],
    #             fontsize=10,
    #             horizontalalignment="right",
    #         )
    #         plt.text(
    #             1.25,
    #             100 - 5 * (n + 1) - 10 * (i + 1),
    #             labels[idx[i]],
    #             fontsize=12,
    #             verticalalignment="center",
    #             horizontalalignment="left",
    #         )

    #     # labels displayed on the left
    #     for i in range(int(np.ceil(n_datasets / 2)), n_datasets):
    #         plt.plot(
    #             [
    #                 (n_datasets - r[i]) / (n_datasets - 1),
    #                 (n_datasets - r[i]) / (n_datasets - 1),
    #                 -0.2,
    #             ],
    #             [
    #                 100,
    #                 100 - 5 * (n + 1) - 10 * (n_datasets - i),
    #                 100 - 5 * (n + 1) - 10 * (n_datasets - i),
    #             ],
    #             color="black",
    #         )
    #         plt.text(
    #             -0.2,
    #             100 - 5 * (n + 1) - 10 * (n_datasets - i) + 2,
    #             "%.2f" % r[i],
    #             fontsize=10,
    #             horizontalalignment="left",
    #         )
    #         plt.text(
    #             -0.25,
    #             100 - 5 * (n + 1) - 10 * (n_datasets - i),
    #             labels[idx[i]],
    #             fontsize=12,
    #             verticalalignment="center",
    #             horizontalalignment="right",
    #         )

    #     # group cliques of statistically similar classifiers
    #     for i in range(np.size(clique, 0)):
    #         R = r[clique[i, :]]
    #         plt.plot(
    #             [
    #                 ((n_datasets - np.min(R)) / (n_datasets - 1)) + 0.015,
    #                 ((n_datasets - np.max(R)) / (n_datasets - 1)) - 0.015,
    #             ],
    #             [100 - 5 * (i + 1), 100 - 5 * (i + 1)],
    #             linewidth=6,
    #             color="black",
    #         )

    #     fig.dpi = 600

    #     if not save_file is None:
    #         fig.savefig(save_file, pad_inches = 0, bbox_inches='tight')
        

    #     plt.show()

    #     return fig, ax