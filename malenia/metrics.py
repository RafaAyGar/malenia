import scipy
import numpy as np
from sklearn.metrics import confusion_matrix


def rank_probability_score(y, yproba):
    y = np.array(y)
    yproba = np.array([list(map(float, item.strip('[]').split())) for item in yproba])

    y = np.clip(y, 0, yproba.shape[1] - 1)

    yoh = np.zeros(yproba.shape)
    yoh[np.arange(len(y)), y] = 1

    yoh = yoh.cumsum(axis=1)
    yproba = yproba.cumsum(axis=1)

    rps = 0
    for i in range(len(y)):
        if y[i] in np.arange(yproba.shape[1]):
            rps += np.power(yproba[i] - yoh[i], 2).sum()
        else:
            rps += 1
    return rps / len(y)


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

def tkendall(y, ypred):
    """
    Value of +1 indicate strong agreement and value of -1 indicate strong DISagreement.
    """
    corr, _ = scipy.stats.kendalltau(y, ypred)
    return corr

def spearman(y, ypred):
    """
    Values of +1 and -1 indicate strong realtionship and value of 0 indicate NO relationship.
    """
    # n = len(y)
    # num = ((y - np.repeat(np.mean(y), n)) * (ypred - np.repeat(np.mean(ypred), n))).sum()
    # div = np.sqrt((pow(y - np.repeat(np.mean(y), n), 2)).sum() * (pow(ypred - np.repeat(np.mean(ypred), n), 2)).sum())

    # if num == 0:
    #     return 0
    # else:
    #     return num / div
    corr, _ = scipy.stats.spearmanr(y, ypred)
    return corr

# def gm(y, ypred):
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         cm = np.transpose(confusion_matrix(y, ypred))
#         sum_byclass = np.sum(cm,axis=1)
#         sensitivities = np.diag(cm)/sum_byclass.astype('double')
#         sensitivities[sum_byclass==0] = 1
#         gm_result = pow(np.prod(sensitivities),1.0/cm.shape[0])
#         return gm_result

# def ms(y, ypred):
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         cm = np.transpose(confusion_matrix(y, ypred))
#         sum_byclass = np.sum(cm,axis=1)
#         sensitivities = np.diag(cm)/sum_byclass.astype('double')
#         sensitivities[sum_byclass==0] = 1
#         ms = np.min(sensitivities)

#         return ms

# def mze(y, ypred):
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")

#         confusion = confusion_matrix(y, ypred)
#         return 1 - np.diagonal(confusion).sum() / confusion.sum()

# def wkappa(y, ypred):
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")

#         confusion = confusion_matrix(y, ypred)
#         m = len(confusion)
#         J = np.mgrid[0:m, 0:m][1]
#         I = np.flipud(np.rot90(J))
#         f = 1 - np.abs(I - J) / 4.0
#         x = np.copy(confusion)

#         n = x.sum()
#         x = x/n

#         r = x.sum(axis=1) # Row sum
#         s = x.sum(axis=0) # Col sum
#         Ex = r.reshape(-1, 1) * s
#         po = (x * f).sum()
#         pe = (Ex * f).sum()
#         return (po - pe) / (1 - pe)