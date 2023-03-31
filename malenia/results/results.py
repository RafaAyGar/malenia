import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from joblib import dump, load



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

    def _is_equal_to(self, other):
        is_equal = (
            self.datasets == other.datasets and
            self.methods == other.methods and
            self.metrics.keys() == other.metrics.keys() and
            self.seeds == other.seeds and
            self.rounding_decimals == other.rounding_decimals
        )
        return is_equal

    def evaluate(self, verbose = True, evaluations_binaries_folder = None):
        
        trying_to_load_or_save_evaluated_results_binaries = not evaluations_binaries_folder is None

        if trying_to_load_or_save_evaluated_results_binaries:
            if not os.path.exists(evaluations_binaries_folder):
                os.mkdir(evaluations_binaries_folder)
            
            evaluation_binaries = os.listdir(evaluations_binaries_folder)
            if len(evaluation_binaries) == 0:
                last_id = 0
            else:
                evaluation_binaries.sort()
                last_id = int(str(evaluation_binaries[-1]).split("_")[-1].split(".")[0])

                for evaluation in evaluation_binaries:
                    evaluation = load(os.path.join(evaluations_binaries_folder, evaluation))
                    if self._is_equal_to(evaluation):
                        print("Results already evaluated. Loading from binary file.")
                        return evaluation
                    
        print("Evaluating results...")
        self.methods_real_names = []
        for method_pretty, method_info in self.methods.items():
            method_real = method_info
            self.methods_real_names.append(method_real)
            for dataset, seed, results_path in self._method_results_info_generator(method_real, "test"):
                if os.path.exists(results_path) == False:
                    if verbose:
                        print("* WARNING: results file not found: ", results_path)
                    continue
                self.results["dataset"].append(dataset)
                self.results["method"].append(method_pretty)
                self.results["seed"].append(seed)
                for metric_name, metric in self.metrics.items():
                    if not metric_name in self.results:
                        self.results[metric_name] = []
                    self.results[metric_name].append(self._get_metric_results(metric, results_path))
        self.results = pd.DataFrame(self.results)
        self.results = self.results.round(self.rounding_decimals)
        self.results_by_method_dataset = self.results.groupby(["dataset", "method"]).mean(numeric_only=True) # numeric only to avoid pandas warning and ensure that the mean is computed only on the numeric columns
        self.results_by_method_dataset = self.results_by_method_dataset.drop(columns = ["seed"])
        self.results_by_method_dataset = self.results_by_method_dataset.reset_index()
        self.results_by_method_dataset = self.results_by_method_dataset.round(self.rounding_decimals)

        if trying_to_load_or_save_evaluated_results_binaries:
            dump(self, os.path.join(evaluations_binaries_folder, "results_" + str(last_id + 1) + ".joblib"))

        return self


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
        self.results_by_method = self.results_by_method_dataset.groupby(["method"]).mean(numeric_only=True)
        self.results_by_method = self.results_by_method.sort_values(by = list(self.metrics.keys()), ascending = False)
        self.results_by_method = self.results_by_method.reset_index()
        self.results_by_method.set_index("method", inplace = True)
        self.results_by_method = self.results_by_method.round(self.rounding_decimals)
        return self.results_by_method
    

    def get_results_by_dataset_metric(self, metric, results_by_method_dataset = None):
        if results_by_method_dataset is None:
            results_by_method_dataset = self.results_by_method_dataset

        df = results_by_method_dataset.sort_values(["dataset", "method"])
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


    def adrian_test_format_results(self, metric_file_dict):
        for metric, output_file in metric_file_dict.items():
            metric_results_by_dataset = self.get_results_by_dataset_metric(metric)
            new_cols = []
            for c in metric_results_by_dataset.columns:
                if c == "dataset":
                    continue
                new_cols.append(c)
                rank_col = "Rank. " + c
                metric_results_by_dataset[rank_col] = ""
                new_cols.append(rank_col)

            metric_results_by_dataset = metric_results_by_dataset[new_cols]
            metric_results_by_dataset['Datasets'] = metric_results_by_dataset.index.values
            metric_results_by_dataset = metric_results_by_dataset.set_index("Datasets")
            metric_results_by_dataset.to_excel(output_file)
        return metric_results_by_dataset