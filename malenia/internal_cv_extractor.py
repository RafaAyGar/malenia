from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from aeon.classification.base import BaseClassifier as AEONBaseClassifier


def extract_internal_cv_results(method):
    if isinstance(method, GridSearchCV):
        return method.cv_results_
    elif isinstance(method, Pipeline):
        return extract_internal_cv_results(method.steps[-1][1])

    elif isinstance(method, AEONBaseClassifier):
        if hasattr(method, "estimator"):
            return extract_internal_cv_results(method._estimator)
        elif hasattr(method, "base_estimator"):
            return extract_internal_cv_results(method.base_estimator)
        else:
            raise ValueError("Cannot extract internal CV results from AEON classifier")
    else:
        raise ValueError("Cannot extract internal CV results from {}".format(type(method)))
