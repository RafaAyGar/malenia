import os
import pandas as pd

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
        for dataset in os.listdir(os.path.join(self.results_path, global_method_name, specif_method_name)):
            for seed in range(self.seeds):
                if dataset == "config.json" or (not dataset in self.datasets):
                    continue
                path = os.path.join(
                    self.results_path,
                    global_method_name,
                    specif_method_name,
                    dataset,
                    "seed_" + str(seed) + "_" + train_or_test + ".csv"
                )
                yield (
                    dataset,
                    seed,
                    path
                )

    def _get_metric_results(self, metric, results_path):
        df = pd.read_csv(results_path)
        y_true = df["y_true"]
        y_pred = df["y_pred"]
        return metric(y_true, y_pred)


    def evaluate(self):
        for metric_name, metric in self.metrics.items():
            self.results[metric_name] = []
            for method_pretty, method_real in self.methods.items():
                for dataset, seed, results_path in self._method_results_info_generator(method_real, "test"):
                    if os.path.exists(results_path) == False:
                        print("* WARNING: results file not found: ", results_path)
                        continue
                    self.results["dataset"].append(dataset)
                    self.results["method"].append(method_pretty)
                    self.results["seed"].append(seed)
                    self.results[metric_name].append(self._get_metric_results(metric, results_path))
        self.results = pd.DataFrame(self.results)


    def get_results_by_method(self):
        self.results_by_method = self.results.groupby(["method"]).mean()
        self.results_by_method = self.results_by_method.drop(columns = ["seed"])
        self.results_by_method = self.results_by_method.sort_values(by = list(self.metrics.keys()), ascending = False)
        self.results_by_method = self.results_by_method.reset_index()
        return self.results_by_method
    
    def get_results_by_dataset_metric(self, metric):
        self.results_by_dataset = self.results.groupby(["dataset", "method"]).mean()
        self.results_by_dataset = self.results_by_dataset.drop(columns = ["seed"])
        self.results_by_dataset = self.results_by_dataset.reset_index()

        df = self.results_by_dataset.sort_values(["dataset", "method"])
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






