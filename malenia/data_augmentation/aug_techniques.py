import numpy as np
from scipy.interpolate import CubicSpline


def assign_aug_techniques(aug_techniques):
    aug_techniques_list = []
    for aug_technique in aug_techniques:
        if aug_technique == "noise":
            aug_techniques_list.append(noise)
        elif aug_technique == "invert":
            aug_techniques_list.append(invert)
        elif aug_technique == "jittering":
            aug_techniques_list.append(jittering)
        elif aug_technique == "homogeneous_scaling":
            aug_techniques_list.append(homogeneous_scaling)
        elif aug_technique == "magnitude_warping":
            aug_techniques_list.append(magnitude_warping)
        elif aug_technique == "time_warping":
            aug_techniques_list.append(time_warping)
        elif aug_technique == "permutation":
            aug_techniques_list.append(permutation)
        else:
            raise ValueError("Augmentation technique not found")
    return aug_techniques_list


# Tested OK
def noise(time_series):
    fluc_extension = np.random.choice([0.25, 0.5, 0.75, 1])  # in percentage
    fluc_variation = np.random.choice([0.025, 0.05])
    if fluc_extension != 1:
        fluc_place = np.random.randint(
            0, time_series.shape[0] - (time_series.shape[0] * fluc_extension)
        )
    else:
        fluc_place = 0

    time_series_z = (time_series - np.mean(time_series)) / np.std(time_series)

    for i in range(int(fluc_extension * time_series.shape[0])):
        time_series_z[fluc_place + i] = time_series_z[fluc_place + i] + np.random.normal(
            0, fluc_variation
        )

    return (time_series_z * np.std(time_series)) + np.mean(time_series)


# Tested OK
def invert(time_series):
    return time_series * -1


# Tested OK
def jittering(time_series, sigma=0.05):
    time_series_z = (time_series - np.mean(time_series)) / np.std(time_series)

    for i in range(time_series.shape[0]):
        time_series_z[i] = time_series_z[i] + np.random.normal(0, sigma)

    return (time_series_z * np.std(time_series)) + np.mean(time_series)


# Tested OK
def homogeneous_scaling(time_series, sigma=0.1):
    time_series_z = (time_series - np.mean(time_series)) / np.std(time_series)

    time_series_z = time_series_z * np.random.normal(1, sigma)

    return (time_series_z * np.std(time_series)) + np.mean(time_series)


# Util function for Magnitude and Time Warping
from scipy.interpolate import CubicSpline


def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((1, 1)) * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, 1))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:, 0], yy[:, 0])
    return np.array(cs_x(x_range)).transpose()


# Tested OK
def magnitude_warping(X, sigma=0.2):
    X_z = (X - np.mean(X)) / np.std(X)
    result = X_z * GenerateRandomCurves(X, sigma)
    return (result * np.std(X)) + np.mean(X)


def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma)
    tt_cum = np.cumsum(tt, axis=0)
    print(tt_cum.shape)
    t_scale = [(X.shape[0] - 1) / tt_cum[-1]]
    tt_cum = tt_cum * t_scale[0]
    return tt_cum


# Tested OK
def time_warping(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new = np.interp(x_range, tt_new, X)
    return X_new


def permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm + 1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0] - minSegLength, nPerm - 1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:] - segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]] : segs[idx[ii] + 1]]
        X_new[pp : pp + len(x_temp)] = x_temp
        pp += len(x_temp)
    return X_new


from transforms3d.axangles import axangle2mat


def rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X, axangle2mat(axis, angle))
