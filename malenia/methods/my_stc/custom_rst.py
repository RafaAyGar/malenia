import numpy as np
from aeon.transformations.collection.shapelet_based._shapelet_transform import (
    RandomShapeletTransform,
    _calc_binary_ig,
    _calc_early_binary_ig,
    _online_shapelet_distance,
)
from aeon.utils.numba.general import z_normalise_series
from numba import njit


class CustomRandomShapeletTransform(RandomShapeletTransform):
    def __init__(
        self,
        shapelet_quality_measure,
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
        self.shapelet_quality_measure = shapelet_quality_measure

        super().__init__(
            n_shapelet_samples=n_shapelet_samples,
            max_shapelets=max_shapelets,
            min_shapelet_length=min_shapelet_length,
            max_shapelet_length=max_shapelet_length,
            remove_self_similar=remove_self_similar,
            time_limit_in_minutes=time_limit_in_minutes,
            contract_max_n_shapelet_samples=contract_max_n_shapelet_samples,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            batch_size=batch_size,
            random_state=random_state,
        )

    def _extract_random_shapelet(self, X, y, i, shapelets, max_shapelets_per_class, rng):
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

        shapelet = z_normalise_series(X[inst_idx][channel][position : position + length])
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
            self.shapelet_quality_measure,
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
        shapelet_quality_measure,
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

            # if worst_quality > 0:
            #     quality = _calc_early_binary_ig(
            #         orderline,
            #         this_cls_traversed,
            #         other_cls_traversed,
            #         this_cls_count - this_cls_traversed,
            #         other_cls_count - other_cls_traversed,
            #         worst_quality,
            #     )

            #     if quality <= worst_quality:
            #         return -1

        if shapelet_quality_measure == "ig":
            quality = _calc_binary_ig(orderline, this_cls_count, other_cls_count)
        elif shapelet_quality_measure == "r2":
            quality = _calc_correlation(orderline, y[inst_idx], l_norm=1)
        else:
            quality = None  # can't rise an error here in njit

        return round(quality, 12)


@njit(fastmath=True, cache=True)
def _calc_correlation(orderline, shp_class, l_norm):
    orderline = np.array(orderline)

    if np.all(orderline[:, 0] == orderline[0, 0]):
        r_value = 0
    else:
        # Numpy calculation
        r_value = np.corrcoef(
            orderline[:, 0], np.power(np.abs(orderline[:, 1] - shp_class), l_norm)
        )
        # Scipy calculation
        # _, _, r_value, _, _ = linregress(
        #     orderline[:, 0], np.power(np.abs(orderline[:, 1] - shp_class), l_norm)
        # )
        r_value = 0.0 if np.isnan(r_value[0, 1]) else r_value[0, 1]

    return r_value**2
