


class Metric:
    def __init__(self, metric, **kwargs):
        self.metric = metric
        self.kwargs = kwargs

    def compute(self, y_true, y_pred):
        return self.metric(y_true, y_pred, **self.kwargs)
