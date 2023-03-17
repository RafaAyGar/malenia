
import malenia
from malenia.results import Results
from malenia.results_utils import Metric
from sklearn.metrics import accuracy_score

correct_results_by_dataset = {
    "Father-A-Son-A/data_A" : ( (0.9 + 0.4 + 0.0 + 1.0 + 0.3 + 0.1) / 6), # converted from univariate predictions
    "Father-A-Son-A/data_B" : ( (0.8 + 0.3 + 0.2) / 3),
    "Father-A-Son-A/data_C" : ( (1.0 + 0.4 + 0.2) / 3),
    "Father-A-Son-B/data_A" : ( (0.9 + 0.6 + 0.2) / 3),
    "Father-A-Son-B/data_B" : ( (0.8 + 0.9 + 0.2) / 3),
    "Father-A-Son-B/data_C" : ( (1.0 + 1.0 + 0.2) / 3),
    "Father-B-Son-A/data_A" : ( (0.9 + 0.8 + 0.3) / 3),
    "Father-B-Son-A/data_B" : ( (0.9 + 0.9 + 0.2) / 3),
    "Father-B-Son-A/data_C" : ( (1.0 + 1.0 + 0.3) / 3),
}

correct_results_by_method = {
    "Father-A-Son-A" : ( ( (0.8 + 0.3 + 0.2)/3 + (0.9 + 0.4 + 0.0 + 1.0 + 0.3 + 0.1)/6 + (1.0 + 0.4 + 0.2)/3 ) / 3),
    "Father-A-Son-B" : ( ( (0.8 + 0.9 + 0.2) + (0.9 + 0.6 + 0.2) + (1.0 + 1.0 + 0.2) ) / 9),
    "Father-B-Son-A" : ( ( (0.9 + 0.9 + 0.2) + (0.9 + 0.8 + 0.3) + (1.0 + 1.0 + 0.3) ) / 9),
}

##
RESULTS_PATH = malenia.__path__[0] + "/test_results"
DATA = [
    "data_A_dim_0",
    "data_A_dim_1",
    "data_A",
    "data_B",
    "data_C",
]
METHODS = {
    "Father-A-Son-A" : ["FatherA_Son_A", False],
    "Father-A-Son-B" : ["FatherA_Son_B", True],
    "Father-B-Son-A" : ["FatherB_Son_A", True],
}
METRICS = {
    "ccr": Metric(accuracy_score),
}
SEEDS = 3
ROUNDING_DECIMALS = 6
##

r = Results(datasets=DATA, methods=METHODS, metrics=METRICS, results_path=RESULTS_PATH, seeds=SEEDS, rounding_decimals=ROUNDING_DECIMALS)
r.evaluate()

def test_results_by_dataset():
    results_by_dataset = r.get_results_by_dataset_metric("ccr")
    data_without_separated = [d for d in DATA if "dim" not in d]
    assert(len(data_without_separated) == len(results_by_dataset.index.values))
    for dataset in data_without_separated:
        for method in METHODS.keys():
            obtained = round(results_by_dataset.loc[dataset, method], r.rounding_decimals)
            correct = round(correct_results_by_dataset[method + "/" + dataset], r.rounding_decimals)
            assert(obtained == correct)

def test_results_by_method():
    results_by_method = r.get_results_by_method()
    for method in METHODS.keys():
        obtained = round(results_by_method.loc[method, "ccr"], r.rounding_decimals)
        correct = round(correct_results_by_method[method], r.rounding_decimals)
        assert(obtained == correct)


