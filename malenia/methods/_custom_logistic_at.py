import numpy as np
from mord import LogisticAT, threshold_fit
from sklearn.utils.validation import check_X_y


class CustomLogisticAT(LogisticAT):

    """

    This is a custom implementation of the LogisticAT class from the mord package.

    The only difference is that this class allows for missing classes in y when the
        parameter 'fix_missing_classes' is set to True.

    Its purpose is to be used in oob predictions, where having every class in each oob y
        is not allways possible.

    The coded method for fixing missing classes is to add a new instance to the training
        data, with the same features as a nearest instance of the missing class.

    """

    def __init__(self, fix_missing_classes=False, alpha=1.0, verbose=0, max_iter=1000):
        self.fix_missing_classes = fix_missing_classes
        super().__init__(alpha=alpha, verbose=verbose, max_iter=max_iter)

    def fit(self, X, y, sample_weight=None):
        _y = np.array(y).astype(int)
        if np.abs(_y - y).sum() > 0.1:
            raise ValueError("y must only contain integer values")
        self.classes_ = np.unique(y)
        self.n_class_ = self.classes_.max() - self.classes_.min() + 1
        y_tmp = y - y.min()  # we need classes that start at zero

        if self.fix_missing_classes:
            X, y_tmp = _fix_missing_classes(X, y_tmp)
            self.classes_ = np.unique(y_tmp)
            self.n_class_ = self.classes_.max() - self.classes_.min() + 1
        self.coef_, self.theta_ = threshold_fit(
            X,
            y_tmp,
            self.alpha,
            self.n_class_,
            mode="AE",
            verbose=self.verbose,
            max_iter=self.max_iter,
            sample_weight=sample_weight,
        )
        return self


def _fix_missing_classes(X, y):
    unique_y = np.sort(np.unique(y))
    if np.all(unique_y == np.arange(unique_y.size)):
        # No need to fix anything
        return X, y

    missing_classes = np.setdiff1d(np.arange(unique_y.size), unique_y)

    for missing_class in missing_classes:
        if missing_class > 0:
            neighbour_class_idx = np.where(y == missing_class - 1)[0][0]
        else:
            neighbour_class_idx = np.where(y == missing_class + 1)[0][0]
        X = np.vstack((X, X[neighbour_class_idx]))
        y = np.append(y, missing_class)

    return X, y
