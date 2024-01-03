import os
import numpy as np
import pandas as pd



def ensemble_columns(results):
    columns_separated_datasets = []
    for dataset in results.datasets:
        if "_dim_" in dataset:
            columns_separated_datasets.append(dataset)

    methods_real_names = []
    manager = dict()
    for method_pretty, method_real in results.methods.items():
        methods_real_names.append(method_real)
        for dataset, seed, results_path in results._method_results_info_generator(method_real, "test"):
            if not dataset in columns_separated_datasets:
                continue
            index = method_real + "__" + dataset.split("_dim_")[0] + "__" + str(seed)
            df = pd.read_csv(results_path)
            probas_list = [np.fromstring(string[1:-1], sep=' ').astype(np.float64) for string in df["y_proba"].values]
            if not index in manager:
                manager[index] = [probas_list]
            else:
                manager[index].append(probas_list)
            
    manager_ensembled = manager.copy()
    for index, predictions in manager.items():
        ensembled_preds = np.mean(predictions, axis=0)
        ensembled_preds = np.round(ensembled_preds, decimals=5)
        manager_ensembled[index] = ensembled_preds

    for index, predictions in manager_ensembled.items():
        Method, Dataset, Seed = index.split("__")
        print("Method", Method)

        if "_" in Method:
            method_general = Method.split("_")[0]
            method_specific = ""
            for part in Method.split("_")[1:]:
                method_specific += part + "_"
            method_specific = method_specific[:-1]
            method_dataset_results_path = os.path.join(results.results_path, method_general, method_specific, Dataset)
        else:
            method_dataset_results_path = os.path.join(results.results_path, Method, Dataset)

        existing_preds = pd.read_csv(os.path.join(method_dataset_results_path + "_dim_0", "seed_" + Seed +  "_test.csv"))
        existing_preds['y_proba'] = pd.Series(predictions.tolist())

        classes = np.unique(existing_preds['y_true'])
        classes = np.sort(classes)
        existing_preds['y_pred'] = existing_preds['y_proba'].apply(lambda x: classes[np.argmax(x)])

        if not os.path.exists(method_dataset_results_path):
            os.makedirs(method_dataset_results_path)
        csv_path = method_dataset_results_path + "/seed_" + Seed +  "_test.csv"
        if os.path.isdir(csv_path):
            os.rmdir(csv_path)
        
        existing_preds.to_csv(csv_path, index=False) # <-- write ensembled predictions to csv