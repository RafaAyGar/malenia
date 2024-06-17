import numpy as np
from sklearn.metrics import confusion_matrix


def amae(y, ypred):
    cm = confusion_matrix(y, ypred)
    n_class = cm.shape[0]
    costes = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
    costes = np.abs(costes - np.transpose(costes))
    errores = costes * cm
    non_zero_cm_rows = ~np.all(cm == 0, axis=1)
    cm_ = cm[non_zero_cm_rows]
    errores_ = errores[non_zero_cm_rows]
    amaes = np.sum(errores_, axis=1) / np.sum(cm_, axis=1).astype("double")
    # amaes_no_nans = amaes[~np.isnan(amaes)]
    return np.mean(amaes)


def mmae(y, ypred):
    cm = confusion_matrix(y, ypred)
    n_class = cm.shape[0]
    costes = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
    costes = np.abs(costes - np.transpose(costes))
    errores = costes * cm
    amaes = np.sum(errores, axis=1) / np.sum(cm, axis=1).astype("double")
    amaes = amaes[~np.isnan(amaes)]
    return amaes.max()
