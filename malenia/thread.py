import os
import sys
from logging import StreamHandler, getLogger

import numpy as np
from joblib import dump, load
from malenia.internal_cv_extractor import extract_internal_cv_results
from malenia.save_and_load import save_cv_results, save_method, save_predictions
from pandas import Timestamp

## Uncomment when using sktime-dl methods
# sys.path.append("/home/rayllon/GitHub/sktime-dl")
# import sktime_dl


console = StreamHandler()
log = getLogger()
log.addHandler(console)


## Read console arguments
#
dataset_path = str(sys.argv[1]).strip()
method_path = str(sys.argv[2]).strip()
cv_path = str(sys.argv[3]).strip()
fold = int(sys.argv[4].strip())
overwrite_strategies = eval(sys.argv[5])
overwrite_predictions = eval(sys.argv[6])
predict_on_train = eval(sys.argv[7])
save_fitted_strategies = eval(sys.argv[8])
job_info = str(sys.argv[9])
results_path = str(sys.argv[10])
dataset_name = str(sys.argv[11]).strip()
data_aug_path = str(sys.argv[12]).strip()
save_transformed_data_to_disk = str(sys.argv[13]).strip()
transformed_data_path = str(sys.argv[14]).strip()
do_save_cv_results = eval(sys.argv[15])


### Load Dataset
##
#
with open(dataset_path, "rb") as dataset_binary:
    dataset = load(dataset_binary)
#
# Check if dataset has a load_crude() method
#
if not hasattr(dataset, "load_crude"):
    raise ValueError(f"Dataset {dataset.name} must implement a load_crude() method")
#
###


### Load Method
##
#
with open(method_path, "rb") as method_binary:
    method = load(method_binary)
#
# Check that, in the case we want to load or save transformed data, the method has a predicting_on_train__ attribute.
#
requiring_transformed_data = (transformed_data_path != "None") or (
    save_transformed_data_to_disk != "None"
)
if requiring_transformed_data:
    if not hasattr(method, "predicting_on_train__"):
        raise ValueError(f"Method {method} must have a predicting_on_train__ attribute")
#
# Check that method random state is equal to fold. This must be always true
#
assert (
    method.random_state == fold
), f"Method random state (= {method.random_state}) must be equal to fold (= {fold})"
#
###


### Load CV method
##
#
with open(cv_path, "rb") as cv_binary:
    cv = load(cv_binary)
#
###


### Load Data Augmentation method
##
#
if data_aug_path != "None":
    with open(data_aug_path, "rb") as data_aug_binary:
        data_aug = load(data_aug_binary)
else:
    data_aug = None
#
###


### Load transformed data from disk if required. Meant for future time saving with methods with built-in transformer
##
#
if transformed_data_path != "None":
    transformed_data_path_train = os.path.join(
        transformed_data_path, dataset.name, f"train_fold_{fold}.pkl"
    )
    transformed_data_path_test = os.path.join(
        transformed_data_path, dataset.name, f"test_fold_{fold}.pkl"
    )
    with open(transformed_data_path_train, "rb") as transformed_data_binary:
        transformed_data_train = load(transformed_data_binary)
    with open(transformed_data_path_test, "rb") as transformed_data_binary:
        transformed_data_test = load(transformed_data_binary)
else:
    transformed_data_train = None
    transformed_data_test = None
#
# Assign transformed data to method if desired
#
if (not transformed_data_train is None) and (not transformed_data_test is None):
    method.train_X_t__ = transformed_data_train
    method.test_X_t__ = transformed_data_test
    method.use_cached_transform = True
#
###


### Data Augmentation stuff (currently not used)
# if not data_aug is None:
#     y_train_unique, y_train_counts = np.unique(y_train, return_counts=True)
#     X_train, y_train = data_aug.balance_dataset(
#         X = X_train,
#         y = y_train,
#         is_ts_data=True
#     )
#     y_train_unique, y_train_counts = np.unique(y_train, return_counts=True)
# else:
#     print("No data augmentation was performed!")


### Load data, apply CV and reset indexes
##
#
if hasattr(cv, "apply"):
    X_train, y_train, X_test, y_test = dataset.load_crude()
    X_train, y_train, X_test, y_test = cv.apply(X_train, y_train, X_test, y_test, fold)
elif hasattr(cv, "get_fold_from_disk"):
    X_train, y_train, X_test, y_test = cv.get_fold_from_disk(dataset, fold)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    # The next thing is being printed for debugging purposes
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
#
del cv
#
if hasattr(X_train, "reset_index"):
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
#
###


### If the method is an ensemble, set up members paths
##
#
IS_ENSEMBLE = hasattr(method, "set_up_members_paths")
if IS_ENSEMBLE:
    method.set_up_members_paths(dataset.name, fold)
else:
    print("Member path not set up!")
#
###


### Fit estimator measuring time
##
#
fit_estimator_start_time = Timestamp.now()
method.fit(X_train, y_train)
fit_estimator_end_time = Timestamp.now()
#
###


### Save fitted method to disk if required
##
#
if save_fitted_strategies:
    try:
        save_method(method, job_info, results_path)
    except Exception as e:
        log.warning("ERROR SAVING method!")
        log.warning(f"SAVING method ERROR - " f"Fit - {job_info} - " f"* EXCEPTION: \n{e}")
#
###


### Save CV results to disk if required
##
#
if do_save_cv_results:
    cv_results = extract_internal_cv_results(method)
    save_cv_results(cv_results, job_info, results_path)
#
###


### Predict on train if required. If method has a _get_train_probs() method, save oob_probas
##
#
if predict_on_train:
    # If we are dealing with a method that requires transformed data, set predicting_on_train__ to True.
    # This tells the methods that the prediction is being performed on train.
    if requiring_transformed_data:
        method.predicting_on_train__ = True
    if IS_ENSEMBLE:
        method.test_or_train = "train"
    predict_estimator_start_time = Timestamp.now()
    y_pred = method.predict(X_train)
    predict_estimator_end_time = Timestamp.now()

    try:
        if IS_ENSEMBLE:
            method.test_or_train = "train"
        y_proba = method.predict_proba(X_train)
    except:
        n_classes = len(np.unique(y_train))
        y_proba = np.zeros((X_train.shape[0], n_classes))

    oob_train_probs = None
    if hasattr(method, "_get_train_probs"):
        oob_train_probs = method._get_train_probs(X_train, y_train)

    save_predictions(
        y_true=y_train,
        y_pred=y_pred,
        y_proba=y_proba,
        oob_probas=oob_train_probs,
        fit_estimator_start_time=fit_estimator_start_time,
        fit_estimator_end_time=fit_estimator_end_time,
        predict_estimator_start_time=predict_estimator_start_time,
        predict_estimator_end_time=predict_estimator_end_time,
        train_or_test="train",
        job_info=job_info,
        results_path=results_path,
    )
#
del X_train, y_train
#
###


### Predict on test
##
#
# If we are dealing with a method that requires transformed data, set predicting_on_train__ to False.
# This tells the methods that the prediction is being performed on test.
if requiring_transformed_data:
    method.predicting_on_train__ = False
#
if IS_ENSEMBLE:
    method.test_or_train = "test"
predict_estimator_start_time = Timestamp.now()
y_pred = method.predict(X_test)
predict_estimator_end_time = Timestamp.now()
#
try:
    if IS_ENSEMBLE:
        method.test_or_train = "test"
    y_proba = method.predict_proba(X_test)
except:
    n_classes = len(np.unique(y_test))
    y_proba = np.zeros((X_test.shape[0], n_classes))
#
del X_test
#
try:
    save_predictions(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        oob_probas=None,
        fit_estimator_start_time=fit_estimator_start_time,
        fit_estimator_end_time=fit_estimator_end_time,
        predict_estimator_start_time=predict_estimator_start_time,
        predict_estimator_end_time=predict_estimator_end_time,
        train_or_test="test",
        job_info=job_info,
        results_path=results_path,
    )
    log.warning(f"Done! - Fit {job_info} saved!")
except Exception as e:
    log.warning(f"SAVING PREDICTIONS ERROR - " f"Fit - {job_info} - " f"* EXCEPTION: \n{e}")
#
###


### Save transformed data to disk if required
##
#
if save_transformed_data_to_disk != "None":
    save_transformed_data_to_disk = os.path.join(save_transformed_data_to_disk, dataset.name)
    if not os.path.exists(save_transformed_data_to_disk):
        os.makedirs(save_transformed_data_to_disk)
    # Save transformed data to sikd using pickle dump() function
    with open(os.path.join(save_transformed_data_to_disk, f"train_fold_{fold}.pkl"), "wb") as f:
        dump(method.train_X_t__, f)
    with open(os.path.join(save_transformed_data_to_disk, f"test_fold_{fold}.pkl"), "wb") as f:
        dump(method.test_X_t__, f)
#


del dataset, method, y_test, y_pred, y_proba
