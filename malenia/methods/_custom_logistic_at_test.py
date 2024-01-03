import numpy as np
from _custom_logistic_at import CustomLogisticAT

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6]])
y = np.array([1, 2, 1, 1, 2])

clf = CustomLogisticAT(fix_missing_classes=True)
clf.fit(X, y)
