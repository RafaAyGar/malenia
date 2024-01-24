from aeon.classification.base import BaseClassifier as AEONBaseClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def extract_internal_cv_results(method):
    cv_results = None
    if isinstance(method, GridSearchCV):
        cv_results = method.cv_results_
    elif isinstance(method, Pipeline):
        cv_results = extract_internal_cv_results(method.steps[-1][1])

    elif isinstance(method, AEONBaseClassifier):
        if hasattr(method, "estimator"):
            cv_results = extract_internal_cv_results(method._estimator)
        elif hasattr(method, "base_estimator"):
            cv_results = extract_internal_cv_results(method.base_estimator)

    return cv_results
