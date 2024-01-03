from malenia.feature_selection import get_feature_importance
from sklearn.base import BaseEstimator, TransformerMixin



class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        # y,
        features_pct = 0.1,
        by = "global_and_class"
    ):
        # self.y = y
        self.features_pct = features_pct
        self.by = by

    def fit(self, X, y=None):
        
        if y is None:
            raise ValueError("y must be provided for feature selection")

        # print("The feature selector receives a dataset with shape: {}".format(X.shape))
        fi_global, fi_byClass = get_feature_importance(X, y)

        self.final_fis = dict()
        if self.by == "global_and_class":
            for feature in fi_global.keys():
                fi = ( fi_global[feature] + fi_byClass[feature] ) / 2
                self.final_fis[feature] = fi

        self.final_fis = sorted(self.final_fis.items(), key=lambda x: x[1], reverse=False)

        return self

    def transform(self, X, y=None):
        
        best_features = []

        n_features = int(len(self.final_fis) * self.features_pct)
        for i in range(n_features):
            best_features.append(self.final_fis[i][0])

        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        # print(X)
        # print(best_features)
        # print("The feature selector returns a dataset with shape: {}".format(X[best_features].shape))
        X_best_features = X[best_features]
        X_best_features = X_best_features.reindex(sorted(X_best_features.columns), axis=1)
        return X_best_features