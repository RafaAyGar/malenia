import numpy as np
from aeon.base._base import _clone_estimator
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.classification.sklearn import RotationForestClassifier
from aeon.utils.validation.panel import check_X_y
from malenia.methods._base_saver_transformer import BaseSaveLoadTransformation
from malenia.methods.my_stc.custom_rst import CustomRandomShapeletTransform
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline


class CustomShapeletTransformClassifier(ShapeletTransformClassifier, BaseSaveLoadTransformation):
    def __init__(
        self,
        shapelet_quality_measure,
        n_shapelet_samples=10000,
        max_shapelets=None,
        max_shapelet_length=None,
        estimator=None,
        transform_limit_in_minutes=0,
        time_limit_in_minutes=0,
        contract_max_n_shapelet_samples=np.inf,
        save_transformed_data=False,
        n_jobs=1,
        batch_size=100,
        random_state=None,
        ignore_saved_transform=False,
        train_X_t__=None,
        test_X_t__=None,
        predicting_on_train__=False,
    ):
        self.shapelet_quality_measure = shapelet_quality_measure

        ShapeletTransformClassifier.__init__(
            self,
            n_shapelet_samples=n_shapelet_samples,
            max_shapelets=max_shapelets,
            max_shapelet_length=max_shapelet_length,
            estimator=estimator,
            transform_limit_in_minutes=transform_limit_in_minutes,
            time_limit_in_minutes=time_limit_in_minutes,
            contract_max_n_shapelet_samples=contract_max_n_shapelet_samples,
            save_transformed_data=save_transformed_data,
            n_jobs=n_jobs,
            batch_size=batch_size,
            random_state=random_state,
        )

        BaseSaveLoadTransformation.__init__(
            self,
            ignore_saved_transform=ignore_saved_transform,
            train_X_t__=train_X_t__,
            test_X_t__=test_X_t__,
            predicting_on_train__=predicting_on_train__,
        )

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

        self._transformer = CustomRandomShapeletTransform(
            shapelet_quality_measure=self.shapelet_quality_measure,
            n_shapelet_samples=self.n_shapelet_samples,
            max_shapelets=self.max_shapelets,
            max_shapelet_length=self.max_shapelet_length,
            time_limit_in_minutes=self._transform_limit_in_minutes,
            contract_max_n_shapelet_samples=self.contract_max_n_shapelet_samples,
            n_jobs=self.n_jobs,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )

        self._estimator = _clone_estimator(
            RotationForestClassifier() if self.estimator is None else self.estimator,
            self.random_state,
        )

        if isinstance(self._estimator, RotationForestClassifier):
            self._estimator.save_transformed_data = self.save_transformed_data

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        m = getattr(self._estimator, "time_limit_in_minutes", None)
        if m is not None and self.time_limit_in_minutes > 0:
            self._estimator.time_limit_in_minutes = self._classifier_limit_in_minutes

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

            if isinstance(self.estimator, Pipeline):
                if "gridsearchcv" in self.estimator.named_steps.keys():
                    # Use best estimator from trained gridsearchcv
                    estimator = _clone_estimator(
                        self._estimator["gridsearchcv"].best_estimator_,
                        self.random_state,
                    )
                    print("Using best estimator from gridsearchcv in oob probas.")
                else:
                    estimator = _clone_estimator(self.estimator[-1], self.random_state)
            else:
                estimator = _clone_estimator(self.estimator, self.random_state)

            # We shuffle the data in this custom version because some of our datasets
            # have a lot of instances of the same class in a row.
            data = np.column_stack((self.transformed_data_, y))
            np.random.shuffle(data)
            X_shuffled = data[:, :-1]
            y_shuffled = data[:, -1]
            y_shuffled = y_shuffled.astype(int)

            return cross_val_predict(
                estimator,
                X=X_shuffled,
                y=y_shuffled,
                cv=cv_size,
                method="predict_proba",
                n_jobs=self._n_jobs,
            )
