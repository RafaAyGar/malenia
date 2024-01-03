# -*- coding: utf-8 -*-
from copy import copy

import numpy as np
from aeon.base._base import _clone_estimator
from aeon.classification.base import BaseClassifier
from aeon.classification.sklearn import RotationForestClassifier
from aeon.transformations.collection.shapelet_transform import \
    RandomShapeletTransform
from aeon.utils.validation.panel import check_X_y
from sklearn.model_selection import cross_val_predict

from malenia.methods._base_saver_transformer import BaseSaveLoadTransformation
from malenia.methods.my_stc.my_random_st_aeon import RandomShapeletTransform
from malenia.methods.my_stc.my_st import MyShapeletTransform

# from malenia.methods.my_stc.my_random_st_debugging import MyRandomShapeletTransformDebug


class MyShapeletTransformClassifier(BaseClassifier, BaseSaveLoadTransformation):
    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "algorithm_type": "shapelet",
    }

    def __init__(
        self,
        transformer="my_rst",
        n_shapelet_samples=10000,
        max_shapelets=None,
        max_shapelet_length=None,
        estimator=None,
        transformer_quality_measure="ig",
        transform_limit_in_minutes=0,
        time_limit_in_minutes=0,
        contract_max_n_shapelet_samples=np.inf,
        save_transformed_data=False,
        ignore_saved_transform=False,
        predicting_on_train=False,
        train_X_t__=None,
        test_X_t__=None,
        predicting_on_train__=False,
        n_jobs=1,
        batch_size=100,
        random_state=None,
    ):
        self.n_shapelet_samples = n_shapelet_samples
        self.max_shapelets = max_shapelets
        self.max_shapelet_length = max_shapelet_length
        self.estimator = estimator
        self.transformer_quality_measure = transformer_quality_measure

        self.transform_limit_in_minutes = transform_limit_in_minutes
        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_shapelet_samples = contract_max_n_shapelet_samples

        self.save_transformed_data = save_transformed_data
        self.ignore_saved_transform = ignore_saved_transform
        self.predicting_on_train = predicting_on_train
        self.train_X_t__ = train_X_t__
        self.test_X_t__ = test_X_t__
        self.predicting_on_train__ = predicting_on_train__

        self.random_state = random_state
        self.batch_size = batch_size
        self.n_jobs = n_jobs

        self.n_instances_ = 0
        self.n_dims_ = 0
        self.series_length_ = 0
        self.transformed_data_ = []

        self.transformer = transformer
        self._transformer = None
        self._estimator = estimator
        self._transform_limit_in_minutes = 0
        self._classifier_limit_in_minutes = 0

        super(MyShapeletTransformClassifier, self).__init__()

    def _fit(self, X, y):
        self.n_instances_, self.n_dims_, self.series_length_ = X.shape

        y = y.astype(int)

        if self.time_limit_in_minutes > 0:
            # contracting 2/3 transform (with 1/5 of that taken away for final
            # transform), 1/3 classifier
            third = self.time_limit_in_minutes / 3
            self._classifier_limit_in_minutes = third
            self._transform_limit_in_minutes = (third * 2) / 5 * 4
        elif self.transform_limit_in_minutes > 0:
            self._transform_limit_in_minutes = self.transform_limit_in_minutes

        ### Set up transformer
        ##
        #
        if self.transformer == "rst":
            self._transformer = RandomShapeletTransform(
                n_shapelet_samples=self.n_shapelet_samples,
                max_shapelets=self.max_shapelets,
                max_shapelet_length=self.max_shapelet_length,
                quality_measure=self.transformer_quality_measure,
                time_limit_in_minutes=self._transform_limit_in_minutes,
                contract_max_n_shapelet_samples=self.contract_max_n_shapelet_samples,
                n_jobs=self.n_jobs,
                batch_size=self.batch_size,
                random_state=self.random_state,
            )
        elif self.transformer == "my_rst_debug":
            self._transformer = MyRandomShapeletTransformDebug(
                n_shapelet_samples=self.n_shapelet_samples,
                max_shapelets=self.max_shapelets,
                max_shapelet_length=self.max_shapelet_length,
                # quality_measure=self.transformer_quality_measure,
                time_limit_in_minutes=self._transform_limit_in_minutes,
                contract_max_n_shapelet_samples=self.contract_max_n_shapelet_samples,
                n_jobs=self.n_jobs,
                batch_size=self.batch_size,
                random_state=self.random_state,
            )
        else:
            raise ValueError("Unknown transformer type: {}".format(self.transformer))
        ###

        ### Set up estimator
        ##
        #
        self._estimator = _clone_estimator(
            RotationForestClassifier() if self.estimator is None else self.estimator,
            self.random_state,
        )
        #
        if isinstance(self._estimator, RotationForestClassifier):
            self._estimator.save_transformed_data = self.save_transformed_data
        #
        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs
        #
        m = getattr(self._estimator, "time_limit_in_minutes", None)
        if m is not None and self.time_limit_in_minutes > 0:
            self._estimator.time_limit_in_minutes = self._classifier_limit_in_minutes
        ###

        if self.train_X_t__ is None:
            print("Fitting transformer and estimator.")
            self.train_X_t__ = self._transformer.fit_transform(X, y)
        else:
            print("Using saved transformed data from disk.")

        if self.save_transformed_data:
            self.transformed_data_ = self.train_X_t__

        self._estimator.fit(self.train_X_t__, y)

        return self

    def _predict(self, X) -> np.ndarray:
        X_t = self._get_transformation(X)

        return self._estimator.predict(X_t)

    def _predict_proba(self, X) -> np.ndarray:
        X_t = self._get_transformation(X)

        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(X_t)
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(X_t)
            for i in range(0, X.shape[0]):
                dists[i, np.where(self.classes_ == preds[i])] = 1
            return dists

    def _get_train_probs(self, X, y) -> np.ndarray:
        y = y.astype(int)

        self.check_is_fitted()
        X, y = check_X_y(X, y, coerce_to_pandas=True)

        n_instances, n_dims = X.shape

        if n_instances != self.n_instances_ or n_dims != self.n_dims_:
            raise ValueError(
                "n_instances, n_dims mismatch. X should be "
                "the same as the training data used in fit for generating train "
                "probabilities."
            )

        if not self.save_transformed_data:
            raise ValueError("Currently only works with saved transform data from fit.")

        if (isinstance(self.estimator, RotationForestClassifier)) or (self.estimator is None):
            return self._estimator._get_train_probs(self.transformed_data_, y)
        else:
            m = getattr(self._estimator, "predict_proba", None)
            if not callable(m):
                raise ValueError("Estimator must have a predict_proba method.")

            cv_size = 10
            _, counts = np.unique(y, return_counts=True)
            min_class = np.min(counts)
            if min_class < cv_size:
                cv_size = min_class

            estimator = _clone_estimator(self.estimator, self.random_state)
            # estimator = copy(self._estimator)

            return cross_val_predict(
                estimator,
                X=self.transformed_data_,
                y=y,
                cv=cv_size,
                method="predict_proba",
                n_jobs=self._n_jobs,
            )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from sklearn.ensemble import RandomForestClassifier

        if parameter_set == "results_comparison":
            return {
                "estimator": RandomForestClassifier(n_estimators=5),
                "n_shapelet_samples": 50,
                "max_shapelets": 10,
                "batch_size": 10,
            }
        elif parameter_set == "contracting":
            return {
                "time_limit_in_minutes": 5,
                "estimator": RotationForestClassifier(contract_max_n_estimators=2),
                "contract_max_n_shapelet_samples": 10,
                "max_shapelets": 3,
                "batch_size": 5,
            }
        elif parameter_set == "train_estimate":
            return {
                "estimator": RotationForestClassifier(n_estimators=2),
                "n_shapelet_samples": 10,
                "max_shapelets": 3,
                "batch_size": 5,
                "save_transformed_data": True,
            }
        else:
            return {
                "estimator": RotationForestClassifier(n_estimators=2),
                "n_shapelet_samples": 10,
                "max_shapelets": 3,
                "batch_size": 5,
            }
