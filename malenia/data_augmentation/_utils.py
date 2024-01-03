import random
import numpy as np
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy, from_3d_numpy_to_nested



def augmentate(X, y, label, N_aug, augmentation_techniques, is_ts_data):

    if is_ts_data:
        X_array = from_nested_to_3d_numpy(X)
    else:
        X_array = X.to_numpy() # not tested

    new_X = X_array.copy()
    new_y = y
    
    y_uniques = np.unique(y)
    if not label in y_uniques:
        raise ValueError("Label not found in the dataset")

    done_augs = 0
    while done_augs < N_aug:

        for i in range(len(X_array)):

            if y[i] != label:
                continue

            ## Augmentate a pattern and add to the dataset
            #
            new_pattern = []
            aug_technique = random.choice(augmentation_techniques)
            
            for series in X_array[i]:
                new_pattern.append(list(aug_technique(series)))
            new_pattern = np.array(new_pattern)
            
            new_X = np.append(new_X, [new_pattern], axis=0)
            new_y = np.append(new_y, [y[i]], axis=0)
            done_augs += 1

            if done_augs >= N_aug:
                break

    new_X = from_3d_numpy_to_nested(new_X, column_names=X.columns)

    return new_X, new_y



def compute_needed_counts_by_class(y, ideal_ir):
    train_y_uniques, train_y_counts = np.unique(y, return_counts=True)
    needed_counts = dict()
    for i in range(len(train_y_uniques)):
        count_i = train_y_counts[i]
        count_rest = np.sum(train_y_counts) - count_i
        ir = count_rest / ( len(train_y_uniques) * count_i )
        if ir < ideal_ir:
            needed_counts[train_y_uniques[i]] = -1
        else:
            needed_counts[train_y_uniques[i]] = int(count_rest / (len(train_y_uniques) * ideal_ir)) - count_i

    return needed_counts