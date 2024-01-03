import numpy as np
from scipy.stats import logistic
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from malenia.metrics import mmae, amae


class OrdinalRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_search_forests=1000, random_state=None, best_rcp=None):
        # self.base_estimator = base_estimator
        self.n_search_forests = n_search_forests
        self.random_state = random_state

        self.best_rcp = best_rcp

    def _compute_preds_from_latent_preds(self, y_latent_preds, thresholds):
        bottom_bounds = thresholds[:-1]
        upper_bounds = thresholds[1:]
        y_test_preds = np.argmax(
            (y_latent_preds[:, None] >= bottom_bounds)
            & (y_latent_preds[:, None] < upper_bounds),
            axis=1,
        )

        # close = np.allclose(y_test_preds, y_test_preds_eff)
        return y_test_preds

    def fit(self, X, y=None):
        self.classes_ = np.unique(y)
        self.n_instances_ = X.shape[0]
        self.n_classes_ = len(self.classes_)

        # if self.base_estimator is None:
        #     self.base_estimator = RandomForestClassifier(random_state=self.random_state)

        y = y.astype(int)

        if self.best_rcp is not None:
            best_latent_y = np.zeros(self.n_instances_)
            for i in range(self.n_instances_):
                best_latent_y[i] = logistic.ppf(
                    (self.best_rcp[y[i]] + self.best_rcp[y[i] + 1]) / 2
                )

            self.best_thresholds = logistic.ppf(self.best_rcp)

            self.final_forest = RandomForestRegressor()
            self.final_forest.fit(X, best_latent_y)

            return self

        random_cum_probas = np.random.rand(self.n_search_forests, self.n_classes_ - 1)
        random_cum_probas = np.sort(random_cum_probas, axis=1)
        random_cum_probas = np.hstack(
            [
                np.zeros((self.n_search_forests, 1)),
                random_cum_probas,
                np.ones((self.n_search_forests, 1)),
            ]
        )

        y_latent_variables = logistic.ppf(
            (random_cum_probas[:, y] + random_cum_probas[:, y + 1]) / 2
        )

        regression_forests = []
        oob_scores = np.zeros(self.n_search_forests)
        curr_iterations = 1
        # y_oob_preds = np.zeros((self.n_search_forests, self.n_instances_))
        for b in range(self.n_search_forests):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_latent_variables[b], test_size=0.33
            )

            rf = DecisionTreeRegressor()
            rf.fit(X_train, y_train)

            y_test_latent_preds = rf.predict(X_test)

            oob_score = r2_score(y_test, y_test_latent_preds)
            oob_scores[b] = oob_score
            regression_forests.append(rf)

            ###
            print(f"Progress: {curr_iterations}/{self.n_search_forests}", end="\r")
            curr_iterations += 1
            ###

        regression_forests = np.array(regression_forests)

        ### V: Keep the best n_best oob_scores and their corresponding RFs.
        ##
        #
        n_best = int(self.n_search_forests * 0.025)
        # best_oob_scores_indices = np.argsort(oob_scores)[:n_best]
        best_oob_scores_indices = np.argsort(oob_scores)[-n_best:]

        # Extract best random cumulative probabilities (rcp)
        self.best_rcp = np.mean(random_cum_probas[best_oob_scores_indices], axis=0)

        best_latent_y = np.zeros(self.n_instances_)
        for i in range(self.n_instances_):
            best_latent_y[i] = logistic.ppf(
                (self.best_rcp[y[i]] + self.best_rcp[y[i] + 1]) / 2
            )

        self.best_thresholds = logistic.ppf(self.best_rcp)

        self.final_forest = RandomForestRegressor()
        self.final_forest.fit(X, best_latent_y)

        self.is_fitted = True

        return self

    def predict(self, X):
        return self._compute_preds_from_latent_preds(
            self.final_forest.predict(X), self.best_thresholds
        )

    def predict_proba(self, X):
        y_preds = self.predict(X)
        n_entries_test = X.shape[0]
        probas = np.zeros((n_entries_test, self.n_classes_))
        for i in range(n_entries_test):
            probas[i][y_preds[i]] = 1
        return probas
