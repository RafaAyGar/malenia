import itertools
import pandas as pd
from scipy import stats

# aeon method
def wilcoxon_test(results_per_method_dataset):
    wilcoxon_df = pd.DataFrame()
    prod = itertools.combinations(results_per_method_dataset.keys(), 2)
    for p in prod:
        estim_1 = p[0]
        estim_2 = p[1]
        w, p_val = stats.wilcoxon(
            results_per_method_dataset[estim_1], results_per_method_dataset[estim_2],
            zero_method="wilcox"
        )

        w_test = {
            "estimator_1": estim_1,
            "estimator_2": estim_2,
            "statistic": w,
            "p_val": p_val,
        }

        wilcoxon_df = pd.concat(
            [wilcoxon_df, pd.DataFrame(w_test, index=[0])], ignore_index=True
        )

    return wilcoxon_df