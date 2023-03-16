import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import builtins

class Results:
    def __init__(
        self,
        datasets,
        methods,
        metrics,
        results_path,
        seeds = 30
    ):
        self.datasets = datasets
        self.methods = methods
        self.metrics = metrics
        self.results_path = results_path
        self.seeds = seeds

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


    def evaluate(self, group_separated_dims = True):
        for method_pretty, method_real in self.methods.items():
            for dataset, seed, results_path in self._method_results_info_generator(method_real, "test"):
                if "MULTI" in method_pretty and "dim" in dataset:
                    continue
                if os.path.exists(results_path) == False:
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
        print("Results:")
        print(self.results)
        self.results_by_method_dataset = self.results.groupby(["dataset", "method"]).mean()
        self.results_by_method_dataset = self.results_by_method_dataset.drop(columns = ["seed"])
        self.results_by_method_dataset = self.results_by_method_dataset.reset_index()


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
        self.results_by_method = self.results.groupby(["method"]).mean()
        self.results_by_method = self.results_by_method.drop(columns = ["seed"])
        self.results_by_method = self.results_by_method.sort_values(by = list(self.metrics.keys()), ascending = False)
        self.results_by_method = self.results_by_method.reset_index()
        return self.results_by_method
    
    def get_results_by_dataset_metric(self, metric):
        df = self.results_by_method_dataset.sort_values(["dataset", "method"])
        rocket_fullMAE = df[df['method'] == list(self.methods.values())[0]].copy()
        for pretty_method_name in self.methods.keys():
            method_results = list(df[df['method'] == pretty_method_name][metric])
            rocket_fullMAE[pretty_method_name] = method_results
            rocket_fullMAE[pretty_method_name].round(4)
            rocket_fullMAE['dataset'] = df['dataset'].unique()
        final_rocket_fullMAE = rocket_fullMAE[
            ["dataset"] + list(self.methods.keys())
        ].copy()
        final_rocket_fullMAE = final_rocket_fullMAE.reset_index(drop=True)
        final_rocket_fullMAE.sort_values(by=["dataset"])
        final_rocket_fullMAE.set_index("dataset", inplace=True)
        return final_rocket_fullMAE  


    
    # def cd(self, metric_name, s=None, labels = None, name = None, alpha = 0.1, clique = None):     
    #     # convert scores into ranks

    #     if s is None:
    #         s = self.results_by_method_dataset

    #     s = (
    #         s.copy()
    #         .loc[:, ["dataset", "method", metric_name]]
    #         .pivot(index="method", columns="dataset", values=metric_name)
    #         # .sort_values(by=["strategy"], ascending=False)
    #         .values
    #     )

    #     if "mae" in metric_name:
    #         s = s * (-1)

    #     N,k = s.shape
    #     print("N: ", N)
    #     print("k: ", k)
    #     print("·······························")
    #     print(s)
    #     print("·······························")
    #     print("-----------------------")
    #     print(np.sort(s, axis=1))
    #     S,r = np.sort(s, axis=1)
    #     print("S: ", S)
    #     idx = k * np.tile(np.arange(N), (k, 1)).T + r
    #     R = np.tile(np.arange(1, k+1), (N, 1))
    #     S = np.transpose(S)
    #     for i in range(N):
    #         for j in range(k):
    #             index = S[i,j] == S[i,:]
    #             R[i,index] = np.mean(R(i,index))
        
    #     r[idx] = R
    #     r = np.transpose(r)
    #     # compute critical difference
        
    #     if alpha == 0.01:
    #         qalpha = np.array([0.0,2.576,2.913,3.113,3.255,3.364,3.452,3.526,3.59,3.646,3.696,3.741,3.781,3.818,3.853,3.884,3.914,3.941,3.967,3.992,4.015,4.037,4.057,4.077,4.096,4.114,4.132,4.148,4.164,4.179,4.194,4.208,4.222,4.236,4.249,4.261,4.273,4.285,4.296,4.307,4.318,4.329,4.339,4.349,4.359,4.368,4.378,4.387,4.395,4.404,4.412,4.42,4.428,4.435,4.442,4.449,4.456])
    #     else:
    #         if alpha == 0.05:
    #             qalpha = np.array([0.0,1.96,2.344,2.569,2.728,2.85,2.948,3.031,3.102,3.164,3.219,3.268,3.313,3.354,3.391,3.426,3.458,3.489,3.517,3.544,3.569,3.593,3.616,3.637,3.658,3.678,3.696,3.714,3.732,3.749,3.765,3.78,3.795,3.81,3.824,3.837,3.85,3.863,3.876,3.888,3.899,3.911,3.922,3.933,3.943,3.954,3.964,3.973,3.983,3.992,4.001,4.009,4.017,4.025,4.032,4.04,4.046])
    #         else:
    #             if alpha == 0.1:
    #                 qalpha = np.array([0.0,1.645,2.052,2.291,2.46,2.589,2.693,2.78,2.855,2.92,2.978,3.03,3.077,3.12,3.159,3.196,3.23,3.261,3.291,3.319,3.346,3.371,3.394,3.417,3.439,3.459,3.479,3.498,3.516,3.533,3.55,3.567,3.582,3.597,3.612,3.626,3.64,3.653,3.666,3.679,3.691,3.703,3.714,3.726,3.737,3.747,3.758,3.768,3.778,3.788,3.797,3.806,3.814,3.823,3.831,3.838,3.846])
    #             else:
    #                 raise Exception('alpha must be 0.01, 0.05 or 0.1')
        
    #     cd = qalpha(k) * np.sqrt(k * (k + 1) / (6 * N))
    #     f = plt.figure('Name',name,'visible','off')
    #     set(f,'Units','normalized')
    #     set(f,'Position',np.array([0,0,0.7,0.5]))
    #     plt.clf()
    #     plt.axis('off')
    #     plt.axis(np.array([- 0.5,1.5,0,140]))
    #     plt.axis('xy')
    #     tics = np.matlib.repmat((np.arange(0,(k - 1)+1)) / (k - 1),3,1)
    #     plt.plot(tics,np.matlib.repmat(np.array([100,105,100]),1,k),'LineWidth',2,'Color','k')
    #     tics = np.matlib.repmat(((np.arange(0,(k - 2)+1)) / (k - 1)) + 0.5 / (k - 1),3,1)
    #     plt.plot(tics,np.matlib.repmat(np.array([100,102.5,100]),1,k - 1),'LineWidth',1,'Color','k')
    #     #line([0 0 0 cd/(k-1) cd/(k-1) cd/(k-1)], [127 123 125 125 123 127], 'LineWidth', 1, 'Color', 'k');
    # #h = text(0.5*cd/(k-1), 130, 'CD', 'FontSize', 12, 'HorizontalAlignment', 'center');
        
    #     for i in np.arange(1,k+1).reshape(-1):
    #         plt.text((i - 1) / (k - 1),110, str(k - i + 1),'FontSize',18,'HorizontalAlignment','center')
        
    #     # compute average ranks
        
    #     r = np.mean(r)
    #     r,idx = builtins.sorted(r)
    #     # compute statistically similar cliques
        
    #     if clique is None:
    #         clique = np.matlib.repmat(r,k,1) - np.matlib.repmat(np.transpose(r),1,k)
    #         clique[clique < 0] = sys.float_info.max
    #         clique = clique < cd
    #         for i in np.arange(k,2+- 1,- 1).reshape(-1):
    #             if np.all(clique[i - 1,clique[i,:]] == clique[i,clique[i,:]]):
    #                 clique[i,:] = 0
    #         n = np.sum(clique, 2-1)
    #         clique = clique[n > 1,:]
    #     else:
    #         if (len(clique) > 0):
    #             clique = clique[:,idx] > 0
        
    #     n = clique.shape[1-1]
    #     # labels displayed on the right
        
    #     for i in np.arange(1,np.ceil(k / 2)+1).reshape(-1):
    #         plt.plot(np.array([(k - r(i)) / (k - 1),(k - r(i)) / (k - 1),1.2]),np.array([100,100 - 5 * (n + 1) - 10 * i,100 - 5 * (n + 1) - 10 * i]),'Color','k')
    #         h = plt.text(1.2,100 - 5 * (n + 1) - 10 * i + 5, str(r(i)),'FontSize',24,'HorizontalAlignment','right')
    #         plt.text(1.25,100 - 5 * (n + 1) - 10 * i + 4,labels[idx(i)],'FontSize',28,'VerticalAlignment','middle','HorizontalAlignment','left')
        
    #     # labels displayed on the left
        
    #     for i in np.arange(np.ceil(k / 2) + 1,k+1).reshape(-1):
    #         plt.plot(np.array([(k - r(i)) / (k - 1),(k - r(i)) / (k - 1),- 0.2]),np.array([100,100 - 5 * (n + 1) - 10 * (k - i + 1),100 - 5 * (n + 1) - 10 * (k - i + 1)]),'Color','k')
    #         plt.text(- 0.2,100 - 5 * (n + 1) - 10 * (k - i + 1) + 5, str(r(i)),'FontSize',24,'HorizontalAlignment','left')
    #         plt.text(- 0.25,100 - 5 * (n + 1) - 10 * (k - i + 1) + 4,labels[idx(i)],'FontSize',28,'VerticalAlignment','middle','HorizontalAlignment','right')
        
    #     # group cliques of statistically similar classifiers
        
    #     for i in np.arange(1,clique.shape[1-1]+1).reshape(-1):
    #         R = r[clique[i,:]]
    #         plt.plot(np.array([((k - np.amin(R)) / (k - 1)) + 0.015((k - np.amax(R)) / (k - 1)) - 0.015]),np.array([100 - 5 * i,100 - 5 * i]),'LineWidth',6,'Color','k')
        
    #     # set(f,'CreateFcn','set(gcbo,'Visible','on')')
    #     setattr(f, 'CreateFcn', lambda x: setattr('gcbo', 'Visible', 'on'))
    #     plt.savefig(f,name)
    #     # all done...
    #     return cd,f

    def plot_CDD(self, metric_name=None, alpha=0.1, data="default", save_file=None):
        """Plot critical difference diagrams.

        References
        ----------
        original implementation by Aaron Bostrom, modified by Markus Löning.
        """
        if isinstance(data, str) and data == "default":
            dataset = self.results_by_method_dataset
        else:
            dataset = data

        data = (
            dataset.copy()
            .loc[:, ["dataset", "method", metric_name]]
            .pivot(index="method", columns="dataset", values=metric_name)
            # .sort_values(by=["strategy"], ascending=False)
            .values
        )

        if "mae" in metric_name:
            data = data * (-1)
        n_datasets, n_methods = data.shape  # [n_datasets,n_methods] = size(s); correct
        labels = list(dataset['method'])

        r = np.argsort(data, axis=0)[::-1]
        S = np.sort(data, axis=0)[::-1]
        idx = n_datasets * np.tile(np.arange(n_methods), (n_datasets, 1)).T + r.T
        R = np.asfarray(np.tile(np.arange(n_datasets) + 1, (n_methods, 1)))
        S = S.T

        for i in range(n_methods):
            for j in range(n_datasets):
                index = S[i, j] == S[i, :]
                R[i, index] = np.mean(R[i, index], dtype=np.float64)

        r = np.asfarray(r)
        r.T.flat[idx] = R
        r = r.T

        if alpha == 0.01:
            # fmt: off
            qalpha = [0.000, 2.576, 2.913, 3.113, 3.255, 3.364, 3.452, 3.526,
                      3.590, 3.646, 3.696, 3.741, 3.781, 3.818,
                      3.853, 3.884, 3.914, 3.941, 3.967, 3.992, 4.015, 4.037,
                      4.057, 4.077, 4.096, 4.114, 4.132, 4.148,
                      4.164, 4.179, 4.194, 4.208, 4.222, 4.236, 4.249, 4.261,
                      4.273, 4.285, 4.296, 4.307, 4.318, 4.329,
                      4.339, 4.349, 4.359, 4.368, 4.378, 4.387, 4.395, 4.404,
                      4.412, 4.420, 4.428, 4.435, 4.442, 4.449,
                      4.456]
            # fmt: on
        elif alpha == 0.05:
            # fmt: off
            qalpha = [0.000, 1.960, 2.344, 2.569, 2.728, 2.850, 2.948, 3.031,
                      3.102, 3.164, 3.219, 3.268, 3.313, 3.354,
                      3.391, 3.426, 3.458, 3.489, 3.517, 3.544, 3.569, 3.593,
                      3.616, 3.637, 3.658, 3.678, 3.696, 3.714,
                      3.732, 3.749, 3.765, 3.780, 3.795, 3.810, 3.824, 3.837,
                      3.850, 3.863, 3.876, 3.888, 3.899, 3.911,
                      3.922, 3.933, 3.943, 3.954, 3.964, 3.973, 3.983, 3.992,
                      4.001, 4.009, 4.017, 4.025, 4.032, 4.040,
                      4.046]
            # fmt: on
        elif alpha == 0.1:
            # fmt: off
            qalpha = [0.000, 1.645, 2.052, 2.291, 2.460, 2.589, 2.693, 2.780,
                      2.855, 2.920, 2.978, 3.030, 3.077, 3.120,
                      3.159, 3.196, 3.230, 3.261, 3.291, 3.319, 3.346, 3.371,
                      3.394, 3.417, 3.439, 3.459, 3.479, 3.498,
                      3.516, 3.533, 3.550, 3.567, 3.582, 3.597, 3.612, 3.626,
                      3.640, 3.653, 3.666, 3.679, 3.691, 3.703,
                      3.714, 3.726, 3.737, 3.747, 3.758, 3.768, 3.778, 3.788,
                      3.797, 3.806, 3.814, 3.823, 3.831, 3.838,
                      3.846]
            # fmt: on
        else:
            raise Exception("alpha must be 0.01, 0.05 or 0.1")

        cd = qalpha[n_datasets - 1] * np.sqrt(
            n_datasets * (n_datasets + 1) / (6 * n_methods)
        )

        # set up plot
        fig, ax = plt.subplots(1)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, 140)
        ax.set_axis_off()

        tics = np.tile(np.array(np.arange(n_datasets)) / (n_datasets - 1), (3, 1))
        plt.plot(
            tics.flatten("F"),
            np.tile([100, 105, 100], (1, n_datasets)).flatten(),
            linewidth=2,
            color="black",
        )
        tics = np.tile(
            (np.array(range(0, n_datasets - 1)) / (n_datasets - 1))
            + 0.5 / (n_datasets - 1),
            (3, 1),
        )
        plt.plot(
            tics.flatten("F"),
            np.tile([100, 102.5, 100], (1, n_datasets - 1)).flatten(),
            linewidth=1,
            color="black",
        )
        plt.plot(
            [
                0,
                0,
                0,
                cd / (n_datasets - 1),
                cd / (n_datasets - 1),
                cd / (n_datasets - 1),
            ],
            [127, 123, 125, 125, 123, 127],
            linewidth=1,
            color="black",
        )
        plt.text(
            0.5 * cd / (n_datasets - 1),
            130,
            "CD",
            fontsize=12,
            horizontalalignment="center",
        )

        for i in range(n_datasets):
            plt.text(
                i / (n_datasets - 1),
                110,
                str(n_datasets - i),
                fontsize=12,
                horizontalalignment="center",
            )

        # compute average rankss
        # print(r)
        r = np.mean(r, axis=0)
        idx = np.argsort(r, axis=0)
        r = np.sort(r, axis=0)
        # print(r)

        # compute statistically similar cliques
        clique = np.tile(r, (n_datasets, 1)) - np.tile(
            np.vstack(r.T), (1, n_datasets)
        )
        clique[clique < 0] = np.inf
        clique = clique < cd

        for i in range(n_datasets - 1, 0, -1):
            if np.all(clique[i - 1, clique[i, :]] == clique[i, clique[i, :]]):
                clique[i, :] = 0

        n = np.sum(clique, 1)
        clique = clique[n > 1, :]
        n = np.size(clique, 0)

        for i in range(int(np.ceil(n_datasets / 2))):
            plt.plot(
                [
                    (n_datasets - r[i]) / (n_datasets - 1),
                    (n_datasets - r[i]) / (n_datasets - 1),
                    1.2,
                ],
                [
                    100,
                    100 - 5 * (n + 1) - 10 * (i + 1),
                    100 - 5 * (n + 1) - 10 * (i + 1),
                ],
                color="black",
            )
            plt.text(
                1.2,
                100 - 5 * (n + 1) - 10 * (i + 1) + 2,
                "%.2f" % r[i],
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
        for i in range(int(np.ceil(n_datasets / 2)), n_datasets):
            plt.plot(
                [
                    (n_datasets - r[i]) / (n_datasets - 1),
                    (n_datasets - r[i]) / (n_datasets - 1),
                    -0.2,
                ],
                [
                    100,
                    100 - 5 * (n + 1) - 10 * (n_datasets - i),
                    100 - 5 * (n + 1) - 10 * (n_datasets - i),
                ],
                color="black",
            )
            plt.text(
                -0.2,
                100 - 5 * (n + 1) - 10 * (n_datasets - i) + 2,
                "%.2f" % r[i],
                fontsize=10,
                horizontalalignment="left",
            )
            plt.text(
                -0.25,
                100 - 5 * (n + 1) - 10 * (n_datasets - i),
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
                    ((n_datasets - np.min(R)) / (n_datasets - 1)) + 0.015,
                    ((n_datasets - np.max(R)) / (n_datasets - 1)) - 0.015,
                ],
                [100 - 5 * (i + 1), 100 - 5 * (i + 1)],
                linewidth=6,
                color="black",
            )

        fig.dpi = 600

        if not save_file is None:
            fig.savefig(save_file, pad_inches = 0, bbox_inches='tight')

        plt.show()
        return fig, ax

    def plot_Boxplot(self, metric_name, figsize=(16, 8)):
        plt.style.use("seaborn")
        plt.rcParams["figure.figsize"] = figsize
        plt.rcParams["figure.autolayout"] = True

        title_font = {'color':'black', 'weight':'bold',
                    'verticalalignment':'bottom'}

        metric_results_by_dataset = self.get_results_by_dataset_metric(metric_name)
        metric_results_by_dataset.plot(kind='box', title= "Comparison in terms of " + metric_name + " metric")
        # for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        #     # label.set_fontname('Arial')
        #     # label.set_fontsize(20)
        plt.title("Comparison in terms of " + metric_name, **title_font)
        plt.ylim(0.0, metric_results_by_dataset.to_numpy().max() + 0.5)
        plt.show()
    

    # def cd(self, metric_name, data=None, alpha=0.1, clique=None, fig_name="cdd"):

    #     if data is None:
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

    #     labels = list(dataset['method'])

    #     n_methods, n_datasets = data.shape
    #     print("n_datasets: ", n_datasets)
    #     print("n_methods: ", n_methods)

    #     S = np.sort(data.T)
    #     r = np.argsort(data.T)
    #     idx = n_datasets * np.tile(np.arange(n_methods).reshape(n_methods, 1), (1, n_datasets)).T + r
    #     R = np.tile(np.arange(1, n_datasets+1), (n_methods, 1))
    #     S = S.T

    #     for i in range(n_methods):
    #         for j in range(n_datasets):
    #             index = S[i,j] == S[i,:]
    #             R[i, np.ix_(index)] = np.mean(R[i, np.ix_(index)], axis=1)

    #     print(r)
    #     r = r.ravel()
    #     idx = idx.ravel()
    #     r[idx] = R.T.ravel()
    #     r = r.reshape((n_datasets, n_methods)).T

    #     # compute critical difference
    #     if alpha == 0.01:
    #         qalpha = [0.000, 2.576, 2.913, 3.113, 3.255, 3.364, 3.452, 3.526, 3.590, 3.646, 
    #             3.696, 3.741, 3.781, 3.818, 3.853, 3.884, 3.914, 3.941, 3.967, 3.992, 
    #             4.015, 4.037, 4.057, 4.077, 4.096, 4.114, 4.132, 4.148, 4.164, 4.179, 
    #             4.194, 4.208, 4.222, 4.236, 4.249, 4.261, 4.273, 4.285, 4.296, 4.307, 
    #             4.318, 4.329, 4.339, 4.349, 4.359, 4.368, 4.378, 4.387, 4.395, 4.404, 
    #             4.412, 4.420, 4.428, 4.435, 4.442, 4.449, 4.456]
    #     elif alpha == 0.05:
    #         qalpha = [0.000, 1.960, 2.344, 2.569, 2.728, 2.850, 2.948, 3.031, 3.102, 3.164,
    #             3.219, 3.268, 3.313, 3.354, 3.391, 3.426, 3.458, 3.489, 3.517, 3.544,
    #             3.569, 3.593, 3.616, 3.637, 3.658, 3.678, 3.696, 3.714, 3.732, 3.749,
    #             3.765, 3.780, 3.795, 3.810, 3.824, 3.837, 3.850, 3.863, 3.876, 3.888,
    #             3.899, 3.911, 3.922, 3.933, 3.943, 3.954, 3.964, 3.973, 3.983, 3.992,
    #             4.001, 4.009, 4.017, 4.025, 4.032, 4.040, 4.046]
    #     elif alpha == 0.1:
    #         qalpha = [0.000, 1.645, 2.052, 2.291, 2.460, 2.589, 2.693, 2.780, 2.855, 2.920,
    #             2.978, 3.030, 3.077, 3.120, 3.159, 3.196, 3.230, 3.261, 3.291, 3.319,
    #             3.346, 3.371, 3.394, 3.417, 3.439, 3.459, 3.479, 3.498, 3.516, 3.533,
    #             3.550, 3.567, 3.582, 3.597, 3.612, 3.626, 3.640, 3.653, 3.666, 3.679,
    #             3.691, 3.703, 3.714, 3.726, 3.737, 3.747, 3.758, 3.768, 3.778, 3.788,
    #             3.797, 3.806, 3.814, 3.823, 3.831, 3.838, 3.846]
    #     else:
    #         raise ValueError('alpha must be 0.01, 0.05 or 0.1')
            
    #     cd = qalpha[n_methods] * np.sqrt(n_methods * (n_methods+1) / (6*n_datasets))

    #     f = plt.figure(fig_name, figsize=(0.7, 0.5), dpi=100, facecolor='w', edgecolor='k')
    #     f.patch.set_visible(False)

    #     plt.clf()
    #     plt.axis('off')
    #     plt.axis([-0.5, 1.5, 0, 140])
    #     plt.axis('scaled')
    #     n_methods = len(r)
    #     tics = np.tile(np.arange(n_methods) / (n_methods-1), (3,1))
    #     plt.plot(tics.flatten(), np.tile([100, 105, 100], n_methods), 'k-', linewidth=2)
    #     tics = np.tile((np.arange(n_methods-1)+0.5)/(n_methods-1), (3,1))
    #     plt.plot(tics.flatten(), np.tile([100, 102.5, 100], n_methods-1), 'k-', linewidth=1)

    #     for i in range(n_methods):
    #         plt.text((i)/(n_methods-1), 110, str(n_methods-i), fontsize=18, horizontalalignment='center')

    #     # compute average ranks
    #     # sorted_idx = np.argsort(np.mean(r, axis=0))
    #     # sorted_data = data[sorted_idx,:]
    #     # sorted_labels = [labels[i] for i in sorted_idx]
    #     # r = np.mean(r)
    #     # print("----------------------")
    #     # print(r)
    #     # idx = np.argsort(r)
    #     # print("idx: ", idx)
    #     # r = r[idx]
    #     # r = np.mean(r)
    #     # print(r)
    #     r = np.mean(r, axis=0)
    #     idx = np.argsort(r, axis=0)
    #     r = np.sort(r, axis=0)
    #     # print(r)

    #     # compute statistically similar cliques
    #     if clique is None:
    #         clique = np.tile(r, (n_methods, 1)) - np.tile(r[:, np.newaxis], (1, n_methods)).T
    #         clique[clique<0] = np.finfo(np.float32).max
    #         clique = clique < cd

    #         for i in range(n_methods, 1, -1):
    #             if np.all(clique[i-2, clique[i-1, :]] == clique[i-1, clique[i-1, :]]):
    #                 clique[i-1, :] = 0
            
    #         n = np.sum(clique, axis=1)
    #         clique = clique[n>1, :]
    #     else:
    #         if len(clique) > 0:
    #             clique = clique[:, idx] > 0
        
    #     n = clique.shape[0]

    #     for i in range(1, int(np.ceil(n_methods/2))+1):
    #         plt.plot([(n_methods-r[i-1])/(n_methods-1), (n_methods-r[i-1])/(n_methods-1), 1.2], [100, 100-5*(n+1)-10*i, 100-5*(n+1)-10*i], color='black')
    #         plt.text(1.2, 100-5*(n+1)-10*i+5, str(r[i-1]), fontsize=24, horizontalalignment='right')
    #         plt.text(1.25, 100-5*(n+1)-10*i+4, labels[idx[i-1]], fontsize=28, verticalalignment='center', horizontalalignment='left')

    #     for i in range(int(np.ceil(n_methods/2))+1, n_methods+1):
    #         plt.plot([(n_methods-r[i-1])/(n_methods-1), (n_methods-r[i-1])/(n_methods-1), -0.2], [100, 100-5*(n+1)-10*(n_methods-i+1), 100-5*(n+1)-10*(n_methods-i+1)], color='black')
    #         plt.text(-0.2, 100-5*(n+1)-10*(n_methods-i+1)+5, str(r[i-1]), fontsize=24, horizontalalignment='left')
    #         plt.text(-0.25, 100-5*(n+1)-10*(n_methods-i+1)+4, labels[idx[i-1]], fontsize=28, verticalalignment='center', horizontalalignment='right')

    #     for i in range(clique.shape[0]):
    #         R = r[clique[i, :]]
    #         plt.plot([((n_methods-min(R))/(n_methods-1)) + 0.015, ((n_methods - max(R))/(n_methods-1)) - 0.015], [100-5*i, 100-5*i], linewidth=6, color='black')

    #     f.set_visible(True)
    #     plt.savefig(fig_name)







