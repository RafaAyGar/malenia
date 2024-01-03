import heapq
import math
import time

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from numba.typed.typedlist import List
from sklearn import preprocessing
from sklearn.utils._random import check_random_state
from sklearn.metrics import r2_score

from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.utils.numba.general import z_normalise_series
from aeon.utils.validation import check_n_jobs
from scipy.stats import linregress


class MyRandomShapeletTransform(BaseCollectionTransformer):
    # _tags = {
    #     "output_data_type": "Tabular",
    #     "capability:multivariate": True,
    #     "capability:unequal_length": True,
    #     "X_inner_type": ["np-list", "numpy3D"],
    #     "y_inner_type": "numpy1D",
    #     "requires_y": True,
    #     "algorithm_type": "shapelet",
    #     "fit_is_empty": False,
    # }
    _tags = {
        "scitype:transform-output": "Primitives",
        "fit_is_empty": False,
        "univariate-only": False,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "numpy1D",
        "requires_y": True,
    }

    def __init__(
        self,
        n_shapelet_samples=10000,
        max_shapelets=None,
        min_shapelet_length=3,
        max_shapelet_length=None,
        remove_self_similar=True,
        time_limit_in_minutes=0.0,
        contract_max_n_shapelet_samples=np.inf,
        n_jobs=1,
        parallel_backend=None,
        batch_size=100,
        random_state=None,
    ):
        self.n_shapelet_samples = n_shapelet_samples
        self.max_shapelets = max_shapelets
        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.remove_self_similar = remove_self_similar

        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_shapelet_samples = contract_max_n_shapelet_samples

        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.batch_size = batch_size
        self.random_state = random_state

        # The following set in method fit
        self.n_classes_ = 0
        self.n_instances_ = 0
        self.n_channels_ = 0
        self.min_series_length_ = 0
        self.classes_ = []
        self.shapelets = []

        # Protected attributes
        self._max_shapelets = max_shapelets
        self._max_shapelet_length = max_shapelet_length
        self._n_jobs = n_jobs
        self._batch_size = batch_size
        self._class_counts = []
        self._class_dictionary = {}
        self._sorted_indicies = []

        super(MyRandomShapeletTransform, self).__init__()

    def _fit(self, X, y=None):
        self._n_jobs = check_n_jobs(self.n_jobs)

        self.classes_, self._class_counts = np.unique(y, return_counts=True)
        self.n_classes_ = self.classes_.shape[0]
        for index, classVal in enumerate(self.classes_):
            self._class_dictionary[classVal] = index

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        self.n_instances_ = len(X)
        self.n_channels_ = X[0].shape[0]
        # Set series length to the minimum
        self.min_series_length_ = X[0].shape[1]
        for i in range(1, self.n_instances_):
            if X[i].shape[1] < self.min_series_length_:
                self.min_series_length_ = X[i].shape[1]

        if self.max_shapelets is None:
            self._max_shapelets = min(10 * self.n_instances_, 1000)
        if self._max_shapelets < self.n_classes_:
            self._max_shapelets = self.n_classes_
        if self.max_shapelet_length is None:
            self._max_shapelet_length = self.min_series_length_

        max_shapelets_per_class = int(self._max_shapelets / self.n_classes_)
        if max_shapelets_per_class < 1:
            max_shapelets_per_class = 1
        # shapelet list content: quality, length, position, channel, inst_idx, cls_idx
        shapelets = List(
            [List([(-1.0, -1, -1, -1, -1, -1)]) for _ in range(self.n_classes_)]
        )
        n_shapelets_extracted = 0

        rng = check_random_state(self.random_state)

        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        fit_time = 0
        if time_limit > 0:
            while (
                fit_time < time_limit
                and n_shapelets_extracted < self.contract_max_n_shapelet_samples
            ):
                candidate_shapelets = Parallel(
                    n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
                )(
                    delayed(self._extract_random_shapelet)(
                        X,
                        y,
                        n_shapelets_extracted + i,
                        shapelets,
                        max_shapelets_per_class,
                        check_random_state(rng.randint(np.iinfo(np.int32).max)),
                    )
                    for i in range(self._batch_size)
                )

                for i, heap in enumerate(shapelets):
                    self._merge_shapelets(
                        heap,
                        List(candidate_shapelets),
                        max_shapelets_per_class,
                        i,
                    )

                if self.remove_self_similar:
                    for i, heap in enumerate(shapelets):
                        to_keep = self._remove_self_similar_shapelets(heap)
                        shapelets[i] = List([n for (n, b) in zip(heap, to_keep) if b])

                n_shapelets_extracted += self._batch_size
                fit_time = time.time() - start_time
        else:
            while n_shapelets_extracted < self.n_shapelet_samples:
                n_shapelets_to_extract = (
                    self._batch_size
                    if n_shapelets_extracted + self._batch_size
                    <= self.n_shapelet_samples
                    else self.n_shapelet_samples - n_shapelets_extracted
                )

                ### Genera un conjunto de shapelets candidatos de forma aleatoria.
                ##
                #
                candidate_shapelets = Parallel(
                    n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
                )(
                    delayed(self._extract_random_shapelet)(
                        X,
                        y,
                        n_shapelets_extracted + i,
                        shapelets,
                        max_shapelets_per_class,
                        check_random_state(rng.randint(np.iinfo(np.int32).max)),
                    )
                    for i in range(n_shapelets_to_extract)
                )
                ## Non parallel implementation
                # candidate_shapelets = []
                # for i in range(n_shapelets_to_extract):
                #     candidate_shapelet = self._extract_random_shapelet(
                #         X,
                #         y,
                #         n_shapelets_extracted + i,
                #         shapelets,
                #         max_shapelets_per_class,
                #         check_random_state(rng.randint(np.iinfo(np.int32).max)),
                #     )
                #     candidate_shapelets.append(candidate_shapelet)

                ### Añade a la pila (heap) de shapelets de cada clase los candidatos que tengan una calidad mayor que cero, también cuidando que no se exceda el límite de shapelets por clase.
                ##
                #
                for i, heap in enumerate(shapelets):
                    self._merge_shapelets(
                        heap,
                        List(candidate_shapelets),
                        max_shapelets_per_class,
                        i,
                    )

                ### Elimina los shapelets que sean muy similares entre sí.
                ##
                #
                if self.remove_self_similar:
                    for i, heap in enumerate(shapelets):
                        to_keep = self._remove_self_similar_shapelets(heap)
                        shapelets[i] = List([n for (n, b) in zip(heap, to_keep) if b])

                n_shapelets_extracted += n_shapelets_to_extract

        ### Forma un nuevo conjunto de shapelets donde cada shapelet es una n-tupla tal que: (calidad, longitud, posición, canal, instancia, clase, shapelet).
        ##
        #
        self.shapelets = [
            (
                s[0],
                s[1],
                s[2],
                s[3],
                s[4],
                self.classes_[s[5]],
                z_normalise_series(X[s[4]][s[3]][s[2] : s[2] + s[1]]),
            )
            for class_shapelets in shapelets
            for s in class_shapelets
            if s[0] > 0
        ]

        ### Ordena los shapelets por: 1º->calidad, 2º->menos la longitud, 3º->posición, 4º->canal, 5º->instancia.
        ##
        #
        self.shapelets.sort(reverse=True, key=lambda s: (s[0], -s[1], s[2], s[3], s[4]))

        to_keep = self._remove_identical_shapelets(List(self.shapelets))
        self.shapelets = [n for (n, b) in zip(self.shapelets, to_keep) if b]

        self._sorted_indicies = []
        for s in self.shapelets:
            sabs = np.abs(s[6])
            self._sorted_indicies.append(
                np.array(
                    sorted(range(s[1]), reverse=True, key=lambda j, sabs=sabs: sabs[j])
                )
            )
        # find max shapelet length
        self.max_shapelet_length_ = max(self.shapelets, key=lambda x: x[1])[1]

        return self

    def _transform(self, X, y=None):
        """Transform X according to the extracted shapelets.

        Parameters
        ----------
        X : np.ndarray shape (n_time_series, n_channels, series_length)
            The input data to transform.

        Returns
        -------
        output : 2D np.array of shape = (n_instances, n_shapelets)
            The transformed data.
        """
        output = np.zeros((len(X), len(self.shapelets)))

        for i in range(0, len(X)):
            if X[i].shape[1] < self.max_shapelet_length_:
                raise ValueError(
                    "The shortest series in transform is smaller than "
                    "the min shapelet length, pad to min length prior to "
                    "calling transform."
                )

        for i, series in enumerate(X):
            dists = Parallel(
                n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
            )(
                delayed(_online_shapelet_distance)(
                    series[shapelet[3]],
                    shapelet[6],
                    self._sorted_indicies[n],
                    shapelet[2],
                    shapelet[1],
                )
                for n, shapelet in enumerate(self.shapelets)
            )

            output[i] = dists

        return output

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        if parameter_set == "results_comparison":
            return {"max_shapelets": 10, "n_shapelet_samples": 500}
        else:
            return {"max_shapelets": 5, "n_shapelet_samples": 50, "batch_size": 20}

    def _extract_random_shapelet(
        self, X, y, i, shapelets, max_shapelets_per_class, rng
    ):
        inst_idx = i % self.n_instances_
        cls_idx = int(y[inst_idx])
        worst_quality = (
            shapelets[cls_idx][0][0]
            if len(shapelets[cls_idx]) == max_shapelets_per_class
            else -1
        )

        length = (
            rng.randint(0, self._max_shapelet_length - self.min_shapelet_length)
            + self.min_shapelet_length
        )
        position = rng.randint(0, self.min_series_length_ - length)
        channel = rng.randint(0, self.n_channels_)

        shapelet = z_normalise_series(
            X[inst_idx][channel][position : position + length]
        )
        sabs = np.abs(shapelet)
        sorted_indicies = np.array(
            sorted(range(length), reverse=True, key=lambda j: sabs[j])
        )

        quality = self._find_shapelet_quality(
            X,
            y,
            shapelet,
            sorted_indicies,
            position,
            length,
            channel,
            inst_idx,
            self._class_counts[cls_idx],
            self.n_instances_ - self._class_counts[cls_idx],
            worst_quality,
        )

        return np.round(quality, 8), length, position, channel, inst_idx, cls_idx

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _find_shapelet_quality(
        X,
        y,
        shapelet,
        sorted_indicies,
        position,
        length,
        dim,
        inst_idx,
        this_cls_count,
        other_cls_count,
        worst_quality,
    ):
        # This is slow and could be optimised, we spend 99% of time here
        orderline = []
        this_cls_traversed = 0
        other_cls_traversed = 0

        for i, series in enumerate(X):
            if i != inst_idx:
                distance = _online_shapelet_distance(
                    series[dim], shapelet, sorted_indicies, position, length
                )
            else:
                distance = 0

            if y[i] == y[inst_idx]:
                cls = 1
                this_cls_traversed += 1
            else:
                cls = -1
                other_cls_traversed += 1

            orderline.append((distance, cls))
            orderline.sort()

            # worst_quality = 0.000001
            # if worst_quality > 0:
            # print(
            #     f"Early correlation in {i} -> {_calc_correlation(orderline, y[inst_idx], l_norm=1)}"
            # )

            # print(
            #     f"_calc_early_correlation output -> {_calc_early_correlation(orderline, y[inst_idx], y)}"
            # )

            # quality = _calc_early_correlation(orderline, y[inst_idx], y)

            # if quality <= worst_quality:
            #     return -1

        quality = _calc_correlation(orderline, y[inst_idx], l_norm=1)

        return round(quality, 12)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _merge_shapelets(
        shapelet_heap, candidate_shapelets, max_shapelets_per_class, cls_idx
    ):
        for shapelet in candidate_shapelets:
            if shapelet[5] == cls_idx and shapelet[0] > 0:
                if (
                    len(shapelet_heap) == max_shapelets_per_class
                    and shapelet[0] < shapelet_heap[0][0]
                ):
                    continue

                heapq.heappush(shapelet_heap, shapelet)

                if len(shapelet_heap) > max_shapelets_per_class:
                    heapq.heappop(shapelet_heap)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _remove_self_similar_shapelets(shapelet_heap):
        to_keep = [True] * len(shapelet_heap)

        for i in range(len(shapelet_heap)):
            if to_keep[i] is False:
                continue

            for n in range(i + 1, len(shapelet_heap)):
                if to_keep[n] and _is_self_similar(shapelet_heap[i], shapelet_heap[n]):
                    if (shapelet_heap[i][0], -shapelet_heap[i][1]) >= (
                        shapelet_heap[n][0],
                        -shapelet_heap[n][1],
                    ):
                        to_keep[n] = False
                    else:
                        to_keep[i] = False
                        break

        return to_keep

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _remove_identical_shapelets(shapelets):
        to_keep = [True] * len(shapelets)

        for i in range(len(shapelets)):
            if to_keep[i] is False:
                continue

            for n in range(i + 1, len(shapelets)):
                if (
                    to_keep[n]
                    and shapelets[i][1] == shapelets[n][1]
                    and np.array_equal(shapelets[i][6], shapelets[n][6])
                ):
                    to_keep[n] = False

        return to_keep


@njit(fastmath=True, cache=True)
def _online_shapelet_distance(series, shapelet, sorted_indicies, position, length):
    subseq = series[position : position + length]

    sum = 0.0
    sum2 = 0.0
    for i in subseq:
        sum += i
        sum2 += i * i

    mean = sum / length
    std = math.sqrt(max(sum2 - mean * mean * length, 0.0) / length)
    if std > 0:
        subseq = (subseq - mean) / std
    else:
        subseq = np.zeros(length)

    best_dist = 0
    for i, n in zip(shapelet, subseq):
        temp = i - n
        best_dist += temp * temp

    i = 1
    traverse = [True, True]
    sums = [sum, sum]
    sums2 = [sum2, sum2]

    while traverse[0] or traverse[1]:
        for n in range(2):
            mod = -1 if n == 0 else 1
            pos = position + mod * i
            traverse[n] = pos >= 0 if n == 0 else pos <= len(series) - length

            if not traverse[n]:
                continue

            start = series[pos - n]
            end = series[pos - n + length]

            sums[n] += mod * end - mod * start
            sums2[n] += mod * end * end - mod * start * start

            mean = sums[n] / length
            std = math.sqrt(max(sums2[n] - mean * mean * length, 0.0) / length)

            dist = 0
            use_std = std != 0
            for j in range(length):
                val = (series[pos + sorted_indicies[j]] - mean) / std if use_std else 0
                temp = shapelet[sorted_indicies[j]] - val
                dist += temp * temp

                if dist > best_dist:
                    break

            if dist < best_dist:
                best_dist = dist

        i += 1

    return best_dist if best_dist == 0 else 1 / length * best_dist


@njit(fastmath=True, cache=True)
def _calc_correlation(orderline, shp_class, l_norm):
    orderline = np.array(orderline)

    # check if all values within orderline[:, 0] are the same
    if np.all(orderline[:, 0] == orderline[0, 0]):
        r_value = 0
    else:
        r_value = np.corrcoef(
            orderline[:, 0], np.power(np.abs(orderline[:, 1] - shp_class), l_norm)
        )
        r_value = 0.0 if np.isnan(r_value[0, 1]) else r_value[0, 1]
        # _, _, r_value, _, _ = linregress(
        #     orderline[:, 0], np.power(np.abs(orderline[:, 1] - shp_class), l_norm)
        # )

    return r_value**2


# def _calc_early_correlation(orderline, shp_class, y):
#     orderline = np.array(orderline)

#     # y = list(dict(y).values())  # extract classes in the correct order of round robin

#     if len(orderline) < len(y):
#         orderline_aux = np.vstack((orderline, np.zeros((len(y) - len(orderline), 2))))

#         for i in range(len(orderline), len(y)):
#             orderline_aux[i, :] = [np.abs(y[i] - shp_class), y[i]]

#     else:
#         orderline_aux = orderline

#     r2_value = r2_score(orderline_aux[:, 0], np.abs(orderline_aux[:, 1] - shp_class))

#     return r2_value


# # @njit(fastmath=True, cache=True)
# def _calc_early_binary_ig(
#     orderline,
#     c1_traversed,
#     c2_traversed,
#     c1_to_add,
#     c2_to_add,
#     worst_quality,
# ):
#     initial_ent = _binary_entropy(
#         c1_traversed + c1_to_add,
#         c2_traversed + c2_to_add,
#     )

#     total_all = c1_traversed + c2_traversed + c1_to_add + c2_to_add

#     bsf_ig = 0
#     # actual observations in orderline
#     c1_count = 0
#     c2_count = 0

#     # evaluate each split point
#     for split in range(len(orderline)):
#         next_class = orderline[split][1]  # +1 if this class, -1 if other
#         if next_class > 0:
#             c1_count += 1
#         else:
#             c2_count += 1

#         # optimistically add this class to left side first and other to right
#         left_prop = (split + 1 + c1_to_add) / total_all
#         ent_left = _binary_entropy(c1_count + c1_to_add, c2_count)

#         # because right side must optimistically contain everything else
#         right_prop = 1 - left_prop

#         ent_right = _binary_entropy(
#             c1_traversed - c1_count,
#             c2_traversed - c2_count + c2_to_add,
#         )

#         ig = initial_ent - left_prop * ent_left - right_prop * ent_right
#         bsf_ig = max(ig, bsf_ig)

#         # now optimistically add this class to right, other to left
#         left_prop = (split + 1 + c2_to_add) / total_all
#         ent_left = _binary_entropy(c1_count, c2_count + c2_to_add)

#         # because right side must optimistically contain everything else
#         right_prop = 1 - left_prop

#         ent_right = _binary_entropy(
#             c1_traversed - c1_count + c1_to_add,
#             c2_traversed - c2_count,
#         )

#         ig = initial_ent - left_prop * ent_left - right_prop * ent_right
#         bsf_ig = max(ig, bsf_ig)

#         if bsf_ig > worst_quality:
#             return bsf_ig

#     return bsf_ig


# # @njit(fastmath=True, cache=True)
# def _calc_binary_ig(orderline, c1, c2):
#     initial_ent = _binary_entropy(c1, c2)

#     total_all = c1 + c2

#     bsf_ig = 0
#     c1_count = 0
#     c2_count = 0

#     # evaluate each split point
#     for split in range(len(orderline)):
#         next_class = orderline[split][1]  # +1 if this class, -1 if other
#         if next_class > 0:
#             c1_count += 1
#         else:
#             c2_count += 1

#         left_prop = (split + 1) / total_all
#         ent_left = _binary_entropy(c1_count, c2_count)

#         right_prop = 1 - left_prop
#         ent_right = _binary_entropy(
#             c1 - c1_count,
#             c2 - c2_count,
#         )

#         ig = initial_ent - left_prop * ent_left - right_prop * ent_right
#         bsf_ig = max(ig, bsf_ig)

#     return bsf_ig


# # @njit(fastmath=True, cache=True)
# def _binary_entropy(c1, c2):
#     ent = 0
#     if c1 != 0:
#         ent -= c1 / (c1 + c2) * np.log2(c1 / (c1 + c2))
#     if c2 != 0:
#         ent -= c2 / (c1 + c2) * np.log2(c2 / (c1 + c2))
#     return ent


@njit(fastmath=True, cache=True)
def _is_self_similar(s1, s2):
    # not self similar if from different series or dimension
    if s1[4] == s2[4] and s1[3] == s2[3]:
        if s2[2] <= s1[2] <= s2[2] + s2[1]:
            return True
        if s1[2] <= s2[2] <= s1[2] + s1[1]:
            return True

    return False
