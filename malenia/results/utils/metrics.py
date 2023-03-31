import numpy as np
from sklearn.metrics import confusion_matrix

## Class to declare metrics to be used in the evaluation of the results
#
class Metric:
    def __init__(self, metric, **kwargs):
        self.metric = metric
        self.kwargs = kwargs

    def compute(self, y_true, y_pred):
        return self.metric(y_true, y_pred, **self.kwargs)


## Metrics
#
def accuracy_off1(y_true, y_pred):
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    conf_mat = confusion_matrix(y_true, y_pred)
    n = conf_mat.shape[0]
    mask = np.eye(n, n) + np.eye(n, n, k=1), + np.eye(n, n, k=-1)
    correct = mask * conf_mat

    return 1.0 * np.sum(correct) / np.sum(conf_mat)