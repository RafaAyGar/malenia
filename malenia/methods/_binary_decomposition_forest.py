import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier


class OBDForest(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, random_state=None, **base_estimator_params):
        self.base_estimator = base_estimator
        self.random_state = random_state
        self.base_estimator_params = base_estimator_params

        if self.base_estimator is None:
            self.base_estimator = RandomForestClassifier(
                random_state=self.random_state, **self.base_estimator_params
            )

    def fit(self, X, y=None):
        self.classes_ = np.unique(y)
        self.n_instances_ = X.shape[0]
        self.n_classes_ = len(self.classes_)

        print("Base estimator:", self.base_estimator)

        y = y.astype(int)

        y_binary_decompositions = []
        self.trained_binary_estimators = []
        for r in range(1, self.n_classes_):
            y_bd = np.where(y >= self.classes_[r], 1, 0)
            y_binary_decompositions.append(y_bd)

            rf = clone(self.base_estimator)
            rf.fit(X, y_bd)
            self.trained_binary_estimators.append(rf)

        return self

    def predict(self, X, y=None):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X, y=None):
        pred_cumprobas_bd = np.ones((X.shape[0], self.n_classes_))
        for r in range(1, self.n_classes_):
            pred_cumprobas_bd[:, r] = self.trained_binary_estimators[r - 1].predict_proba(X)[:, 1]

        final_probas = np.zeros((X.shape[0], self.n_classes_))
        for r in range(0, self.n_classes_ - 1):
            final_probas[:, r] = pred_cumprobas_bd[:, r] - pred_cumprobas_bd[:, r + 1]
        final_probas[:, -1] = pred_cumprobas_bd[:, -1]

        return final_probas

    def get_params(self, deep=True):
        params = {"base_estimator": self.base_estimator}
        if deep:
            params.update(self.base_estimator.get_params(deep=True))
        params.update(self.base_estimator_params)
        return params

    def set_params(self, **params):
        super().set_params(**params)
        base_estimator_params_to_set = {}
        base_estimator_params = self.base_estimator.get_params(deep=True)
        for k, v in params.items():
            if k in base_estimator_params:
                base_estimator_params_to_set[k] = v
        self.base_estimator.set_params(**base_estimator_params_to_set)
        return self
