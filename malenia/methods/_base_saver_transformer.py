class BaseSaveLoadTransformation:
    def __init__(
        self,
        ignore_saved_transform=False,
        train_X_t__=None,
        test_X_t__=None,
        predicting_on_train__=False,
    ):
        self.ignore_saved_transform = ignore_saved_transform
        self.train_X_t__ = train_X_t__
        self.test_X_t__ = test_X_t__
        self.predicting_on_train__ = predicting_on_train__

    def _get_transformation(self, X):
        if self.ignore_saved_transform:
            return self._transformer.transform(X)
        else:
            if self.predicting_on_train__:
                if self.train_X_t__ is None:
                    print("Fitting transformer and estimator for train.")
                    self.train_X_t__ = self._transformer.transform(X)
                return self.train_X_t__
            else:
                if self.test_X_t__ is None:
                    print("Fitting transformer and estimator for test.")
                    self.test_X_t__ = self._transformer.transform(X)
                return self.test_X_t__
