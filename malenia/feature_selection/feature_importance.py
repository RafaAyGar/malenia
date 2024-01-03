import numpy as np
import pandas as pd
import scipy.stats as stats

from itertools import combinations
from malenia.feature_selection._utils import get_pdfs_from_distribution



def get_feature_importance(X, y, target_column = 'target'):

    data = X.copy()
    if type(data) is np.ndarray:
        data = pd.DataFrame(data)
        
    features = data.columns.values
    data[target_column] = y

    # pdfs = dict()
    # for feature in X.columns:
    #     if feature != target_column:
    #         pdfs[feature] = get_pdfs_from_distribution(data, feature, target_column, distribution = stats.norm, plot=False)
    
    data_by_class = dict()
    target_uniques = np.unique(data[target_column])
    for target in target_uniques:
        data_by_class[target] = data[data[target_column] == target]

    feature_importances_by_classes = dict()

    feature_importances_global = dict()
    class_combinations = combinations(target_uniques, 2)
    n_combinations = len(list(class_combinations))
    for feature in features:
        feature_importance = 0
        feature_pdfs_along_classes = get_pdfs_from_distribution(data, feature, target_column, distribution = stats.norm, plot=False)
        test_ks_pvalues = []
        test_mw_pvalues = []
        for class_combination in combinations(target_uniques, 2):
            f1 = feature_pdfs_along_classes[class_combination[0]]
            f2 = feature_pdfs_along_classes[class_combination[1]]
            test_ks = stats.ks_2samp(f1, f2)
            test_ks_pvalues.append(test_ks.pvalue)
            f1_values = data_by_class[class_combination[0]][feature].values
            f2_values = data_by_class[class_combination[1]][feature].values
            test_mw = stats.mannwhitneyu(f1_values, f2_values)
            test_mw_pvalues.append(test_mw.pvalue)
            feature_importance += ((test_ks.pvalue) + (test_mw.pvalue)) / 2
            # feature_importance += (test_mw.pvalue)

        feature_importances_by_classes[feature] = feature_importance / n_combinations

        # print(test_mw_pvalues)
        values_by_class = []
        for label in target_uniques:
            values_by_class.append(data[data[target_column] == label][feature].values)
        _, p_value = stats.f_oneway(*values_by_class)
        feature_importances_global[feature] = p_value

    return feature_importances_global, feature_importances_by_classes
    # return feature_importances_by_classes