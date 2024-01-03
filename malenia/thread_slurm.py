import sys
from logging import StreamHandler, getLogger

import numpy as np
from pandas import Timestamp
from sktime.benchmarking.data import UEADataset
from sktime.classification.convolution_based import Arsenal
from sktime.classification.interval_based._drcif import DrCIF
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification_ordinal.kernel_based._arsenal_ordinal import \
    ArsenalOrdinal
from sktime.classification_ordinal.shapelet_based import \
    ShapeletTransformClassifierOrdinal
from sktime.series_as_features.model_selection import StratifiedCV

from malenia.save_and_load import save_method, save_predictions

console = StreamHandler()
log = getLogger()
log.addHandler(console)

# Read console arguments
dataset_path = str(sys.argv[1]).strip()
method = str(sys.argv[2]).strip()
# cv_path = str(sys.argv[3]).strip()
fold = int(sys.argv[4].strip())
overwrite_strategies = eval(sys.argv[5])
overwrite_predictions = eval(sys.argv[6])
predict_on_train = eval(sys.argv[7])
save_fitted_strategies = eval(sys.argv[8])
job_info = str(sys.argv[9])
results_path = str(sys.argv[10])

## El código de abajo espera que el path del dataset sea del tipo: "ruta_base/tsoc/dataset_name", ejemplo: "ruta_base/tsoc/AAPL"
dataset_name = dataset_path.split("/")[-1]
dataset = UEADataset(path=dataset_path, name=dataset_name)

SEEDS = 30
cv = StratifiedCV(
    total_folds=SEEDS
)  # Como no cambia, se declara aquí a pelo y lo hacemos más fácil

## Esto es bastante chustero, pero por ahora nos vale.
#
if method == "DrCIF_Default":
    method = DrCIF(random_state=fold)
elif method == "Arsenal_Default":
    method = Arsenal(random_state=fold)
elif method == "ArsenalOrdinal_Default":
    method = ArsenalOrdinal(random_state=fold)
elif method == "STC_Default":
    method = ShapeletTransformClassifier(random_state=fold)
elif method == "STCOrdinal_Default":
    method = ShapeletTransformClassifierOrdinal(random_state=fold)

# Do the work that standard condor.fit_predict() does in every for iteration
if not hasattr(dataset, "load_crude"):
    raise ValueError(f"Dataset {dataset.name} must implement a load_crude() method")
X_train, y_train, X_test, y_test = dataset.load_crude()

# Apply stratified resample
X_train, y_train, X_test, y_test = cv.apply(X_train, y_train, X_test, y_test, fold)

del cv

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)


fit_estimator_start_time = Timestamp.now()
method.fit(X_train, y_train)
fit_estimator_end_time = Timestamp.now()

if save_fitted_strategies:
    try:
        save_method(method, job_info, results_path)
    except Exception as e:
        log.warning("ERROR SAVING method!")
        log.warning(f"SAVING method ERROR - " f"Fit - {job_info} - " f"* EXCEPTION: \n{e}")

if predict_on_train:
    predict_estimator_start_time = Timestamp.now()
    y_pred = method.predict(X_train)
    predict_estimator_end_time = Timestamp.now()

    try:
        y_proba = method.predict_proba(X_train)
    except:
        n_classes = len(np.unique(y_train))
        y_proba = np.zeros((X_train.shape[0], n_classes))

    save_predictions(
        y_true=y_train,
        y_pred=y_pred,
        y_proba=y_proba,
        fit_estimator_start_time=fit_estimator_start_time,
        fit_estimator_end_time=fit_estimator_end_time,
        predict_estimator_start_time=predict_estimator_start_time,
        predict_estimator_end_time=predict_estimator_end_time,
        train_or_test="train",
        job_info=job_info,
        results_path=results_path,
    )

del X_train, y_train

predict_estimator_start_time = Timestamp.now()
y_pred = method.predict(X_test)
predict_estimator_end_time = Timestamp.now()

try:
    y_proba = method.predict_proba(X_test)
except:
    n_classes = len(np.unique(y_test))
    y_proba = np.zeros((X_test.shape[0], n_classes))
# else:
#     n_classes = len(np.unique(y_test))
#     y_proba = np.zeros((X_test.shape[0], n_classes))
#     for i in range(len(y_pred)):
#         y_proba[i, y_pred[i]] = 1

del X_test

try:
    save_predictions(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
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

del dataset, method, y_test, y_pred, y_proba
