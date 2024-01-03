import random
from itertools import zip_longest
from time import time

import numpy as np
from aeon.transformations.collection.base import BaseCollectionTransformer
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution


class MyShapeletTransform(BaseCollectionTransformer):
    _tags = {
        "scitype:transform-output": "Primitives",
        "fit_is_empty": False,
        "univariate-only": True,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "numpy1D",
        "requires_y": True,
    }

    def __init__(
        self,
        min_shapelet_length=3,
        max_shapelet_length=np.inf,
        max_shapelets_to_store_per_class=200,
        random_state=None,
        verbose=0,
        remove_self_similar=True,
    ):
        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.max_shapelets_to_store_per_class = max_shapelets_to_store_per_class
        self.random_state = random_state
        self.verbose = verbose
        self.remove_self_similar = remove_self_similar
        self.predefined_ig_rejection_level = 0.05
        self.shapelets = None

        super(MyShapeletTransform, self).__init__()

    def _fit(self, X, y=None):
        X_lens = np.repeat(X.shape[-1], X.shape[0])

        self.n_instances = len(y)
        self.series_length = X.shape[-1]
        self.classes = np.unique(y)
        self._random_state = check_random_state(self.random_state)

        # Check class attributes
        if self.max_shapelet_length > X.shape[-1]:
            self.max_shapelet_length = X.shape[-1]
        #

        candidate_starts_and_lens = [
            [start, length]
            for start in range(0, self.series_length - self.min_shapelet_length + 1)
            for length in range(self.min_shapelet_length, self.max_shapelet_length + 1)
            if start + length <= self.series_length
        ]

        shapelet_heaps_by_class = {i: [] for i in self.classes}
        ids_by_class = {i: np.where(y == i)[0] for i in self.classes}
        num_instances_per_class = {i: len(ids_by_class[i]) for i in ids_by_class}

        round_robin_case_order = _round_robin(*[list(v) for k, v in ids_by_class.items()])
        cases_to_visit = [(i, y[i]) for i in round_robin_case_order]

        case_idx = 0
        while case_idx < len(cases_to_visit):
            case_series_id = cases_to_visit[case_idx][0]
            case_series_len = len(X[case_series_id][0])
            case_class = cases_to_visit[case_idx][1]

            # minus 1 to remove this candidate from sums
            case_class_n_instances = num_instances_per_class[case_class] - 1
            other_classes_n_instances = self.n_instances - case_class_n_instances - 1

            num_visited_this_class = 0
            num_visited_other_class = 0
            candidate_rejected = False

            evaluation_times = []
            candidate_i = 1
            case_random_candidate_starts_and_lens = random.sample(
                candidate_starts_and_lens, int(len(candidate_starts_and_lens) * 0.2)
            )
            for (
                canditate_start_pos,
                candidate_len,
            ) in case_random_candidate_starts_and_lens:
                start_time = time()

                candidate = standardize(
                    X[case_series_id][:, canditate_start_pos : canditate_start_pos + candidate_len]
                )

                cases_to_compare = [case for case in cases_to_visit if case[0] != case_series_id]
                cases_to_compare = random.sample(cases_to_compare, int(len(cases_to_compare) * 0.2))
                for case_to_compare in cases_to_compare:
                    comparison_series_id = case_to_compare[0]
                    comparison_series_len = len(X[comparison_series_id][0])
                    comparison_class = case_to_compare[1]

                    if case_class == comparison_class:
                        num_visited_this_class += 1
                        binary_class_identifier = 1  # positive for this class
                    else:
                        num_visited_other_class += 1
                        binary_class_identifier = -1  # negative for any
                        # other class

                    bsf_dist = np.inf

                    start_left = canditate_start_pos
                    start_right = canditate_start_pos + 1

                    if comparison_series_len == candidate_len:
                        start_left = 0
                        start_right = 0

                    orderline = []

                    for _ in range(
                        max(1, int(np.ceil((comparison_series_len - candidate_len) / 2)))
                    ):
                        # left
                        if start_left < 0:
                            start_left = comparison_series_len - 1 - candidate_len

                        comparison = standardize(
                            X[comparison_series_id][:, start_left : start_left + candidate_len]
                        )
                        dist_left = np.linalg.norm(candidate - comparison)
                        bsf_dist = min(dist_left * dist_left, bsf_dist)

                        # for odd lengths
                        if start_left == start_right:
                            continue

                        # right
                        if start_right == comparison_series_len - candidate_len + 1:
                            start_right = 0
                        comparison = standardize(
                            X[comparison_series_id][:, start_right : start_right + candidate_len]
                        )
                        dist_right = np.linalg.norm(candidate - comparison)
                        bsf_dist = min(dist_right * dist_right, bsf_dist)

                        start_left -= 1
                        start_right += 1

                    orderline.append((bsf_dist, binary_class_identifier))
                    # sorting required after each add for early IG abandon.
                    orderline.sort()

                    if len(orderline) > 2:
                        ig_upper_bound = calc_early_binary_ig(
                            orderline,
                            num_visited_this_class,
                            num_visited_other_class,
                            case_class_n_instances - num_visited_this_class,
                            other_classes_n_instances - num_visited_other_class,
                        )
                        # print("upper: "+str(ig_upper_bound))
                        if ig_upper_bound <= self.predefined_ig_rejection_level:
                            candidate_rejected = True
                            break
                # end for loop

                if candidate_rejected is False:
                    final_ig = calc_binary_ig(
                        orderline,
                        case_class_n_instances,
                        other_classes_n_instances,
                    )
                    accepted_candidate = _Shapelet(
                        case_series_id,
                        canditate_start_pos,
                        candidate_len,
                        final_ig,
                        candidate,
                    )

                    # add to min heap to store shapelets for this class
                    shapelet_heaps_by_class[case_class].append(accepted_candidate)

                    # informal, but extra 10% allowance for self similar later
                    # if (
                    #     len(shapelet_heaps_by_class[case_class])
                    #     > self.max_shapelets_to_store_per_class * 3
                    # ):
                    #     shapelet_heaps_by_class[case_class].pop()

                candidate_i += 1
                evaluation_times.append(time() - start_time)
                print(
                    f"Progress: Case_idx {case_idx} / {len(cases_to_visit)} | Candidate {candidate_i} / {len(case_random_candidate_starts_and_lens)} | Mean candidate evaluation time: {np.mean(evaluation_times)}",
                    end="\r",
                )

                # # Check max time is not exceeded
                # if (
                #     hasattr(self, "time_contract_in_mins")
                #     and self.time_contract_in_mins > 0
                # ):
                #     time_now = time_taken()
                #     time_this_shapelet = time_now - time_last_shapelet
                #     if time_this_shapelet > max_time_calc_shapelet:
                #         max_time_calc_shapelet = time_this_shapelet
                #         if self.verbose > 0:
                #             print(f"Max time{max_time_calc_shapelet}")  # noqa
                #     time_last_shapelet = time_now

                #     # add a little 1% leeway to the timing incase one run was
                #     # slightly faster than another based on the CPU.
                #     time_in_seconds = self.time_contract_in_mins * 60
                #     max_shapelet_time_percent = (max_time_calc_shapelet / 100.0) * 0.75
                #     if (time_now + max_shapelet_time_percent) > time_in_seconds:
                #         time_finished = True
                #         break

                # if case_idx >= num_series_to_visit:
                #     if (
                #         hasattr(self, "time_contract_in_mins")
                #         and time_finished is not True
                #     ):
                #         case_idx = 0
                # elif case_idx >= num_series_to_visit or time_finished:
                #     if self.verbose > 0:
                #         print("Stopping search")  # noqa
                #     break

            case_idx += 1

        # # Remove self similar shapelets
        # self.shapelets = []
        # for class_val in distinct_class_vals:
        #     by_class_descending_ig = sorted(
        #         shapelet_heaps_by_class[class_val].get_array(),
        #         key=itemgetter(0),
        #         reverse=True,
        #     )

        #     if self.remove_self_similar and len(by_class_descending_ig) > 0:
        #         by_class_descending_ig = (
        #             ShapeletTransform.remove_self_similar_shapelets(
        #                 by_class_descending_ig
        #             )
        #         )
        #     else:
        #         # need to extract shapelets from tuples
        #         by_class_descending_ig = [x[2] for x in by_class_descending_ig]

        #     # if we have more than max_shapelet_per_class, trim to that
        #     # amount here
        #     if len(by_class_descending_ig) > self.max_shapelets_to_store_per_class:
        #         max_n = self.max_shapelets_to_store_per_class
        #         by_class_descending_ig = by_class_descending_ig[:max_n]

        #     self.shapelets.extend(by_class_descending_ig)

        # final sort so that all shapelets from all classes are in
        # descending order of information gain
        self.shapelets.sort(key=lambda x: x.info_gain, reverse=True)

        # warn the user if fit did not produce any valid shapelets
        if len(self.shapelets) == 0:
            raise ValueError("No valid shapelets were extracted.")

        return self

    def _transform(self, X, y=None):
        raise NotImplementedError


class _Shapelet:
    """A simple class to model a Shapelet with associated information.

    Parameters
    ----------
    series_id: int
        The index of the series within the data (X) that was passed to fit.
    start_pos: int
        The starting position of the shapelet within the original series
    length: int
        The length of the shapelet
    info_gain: flaot
        The calculated information gain of this shapelet
    data: array-like
        The (z-normalised) data of this shapelet.
    """

    def __init__(self, series_id, start_pos, length, info_gain, data):
        self.series_id = series_id
        self.start_pos = start_pos
        self.length = length
        self.info_gain = info_gain
        self.data = data

    def __str__(self):
        """Print."""
        return "Series ID: {0}, start_pos: {1}, length: {2}, info_gain: {3}," " ".format(
            self.series_id, self.start_pos, self.length, self.info_gain
        )


def _round_robin(*iterables):
    sentinel = object()
    return (a for x in zip_longest(*iterables, fillvalue=sentinel) for a in x if a != sentinel)


def standardize(a, axis=0, ddof=0):
    """Return the normalised version of series.

    This mirrors the scipy implementation
    with a small difference - rather than allowing /0, the function
    returns output = np.zeroes(len(input)).
    This is to allow for sensible processing of candidate
    shapelets/comparison subseries that are a straight
    line. Original version:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats
    .zscore.html

    Parameters
    ----------
    a : array_like
        An array like object containing the sample data.

    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over
        the whole array a.

    ddof : int, optional
        Degrees of freedom correction in the calculation of the standard
        deviation. Default is 0.

    Returns
    -------
    zscore : array_like
        The z-scores, standardized by mean and standard deviation of
        input array a.
    """
    zscored = np.empty(a.shape)
    for i, j in enumerate(a):
        sstd = j.std(axis=axis, ddof=ddof)

        # special case - if shapelet is a straight line (i.e. no
        # variance), zscore ver should be np.zeros(len(a))
        if sstd == 0:
            zscored[i] = np.zeros(len(j))
        else:
            mns = j.mean(axis=axis)
            if axis and mns.ndim < j.ndim:
                zscored[i] = (j - np.expand_dims(mns, axis=axis)) / np.expand_dims(sstd, axis=axis)
            else:
                zscored[i] = (j - mns) / sstd
    return zscored


def binary_entropy(num_this_class, num_other_class):
    """Binary entropy."""
    ent = 0
    if num_this_class != 0:
        ent -= (
            num_this_class
            / (num_this_class + num_other_class)
            * np.log2(num_this_class / (num_this_class + num_other_class))
        )
    if num_other_class != 0:
        ent -= (
            num_other_class
            / (num_this_class + num_other_class)
            * np.log2(num_other_class / (num_this_class + num_other_class))
        )
    return ent


def calc_binary_ig(orderline, total_num_this_class, total_num_other_class):
    """Binary information gain."""
    # def entropy(ent_class_counts, all_class_count):

    initial_ent = binary_entropy(total_num_this_class, total_num_other_class)
    bsf_ig = 0

    count_this_class = 0
    count_other_class = 0

    total_all = total_num_this_class + total_num_other_class

    # evaluate each split point
    for split in range(0, len(orderline) - 1):
        next_class = orderline[split][1]  # +1 if this class, -1 if other
        if next_class > 0:
            count_this_class += 1
        else:
            count_other_class += 1

        # optimistically add this class to left side first and other to
        # right
        left_prop = (split + 1) / total_all
        ent_left = binary_entropy(count_this_class, count_other_class)

        right_prop = 1 - left_prop  # because right side must
        # optimistically contain everything else
        ent_right = binary_entropy(
            total_num_this_class - count_this_class,
            total_num_other_class - count_other_class,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

    return bsf_ig


def calc_early_binary_ig(
    orderline,
    num_this_class_in_orderline,
    num_other_class_in_orderline,
    num_to_add_this_class,
    num_to_add_other_class,
):
    """Early binary IG."""
    # def entropy(ent_class_counts, all_class_count):

    initial_ent = binary_entropy(
        num_this_class_in_orderline + num_to_add_this_class,
        num_other_class_in_orderline + num_to_add_other_class,
    )
    bsf_ig = 0

    # actual observations in orderline
    count_this_class = 0
    count_other_class = 0

    total_all = (
        num_this_class_in_orderline
        + num_other_class_in_orderline
        + num_to_add_this_class
        + num_to_add_other_class
    )

    # evaluate each split point
    for split in range(0, len(orderline) - 1):
        next_class = orderline[split][1]  # +1 if this class, -1 if other
        if next_class > 0:
            count_this_class += 1
        else:
            count_other_class += 1

        # optimistically add this class to left side first and other to
        # right
        left_prop = (split + 1 + num_to_add_this_class) / total_all
        ent_left = binary_entropy(count_this_class + num_to_add_this_class, count_other_class)

        right_prop = 1 - left_prop  # because right side must
        # optimistically contain everything else
        ent_right = binary_entropy(
            num_this_class_in_orderline - count_this_class,
            num_other_class_in_orderline - count_other_class + num_to_add_other_class,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

        # now optimistically add this class to right, other to left
        left_prop = (split + 1 + num_to_add_other_class) / total_all
        ent_left = binary_entropy(count_this_class, count_other_class + num_to_add_other_class)

        right_prop = 1 - left_prop  # because right side must
        # optimistically contain everything else
        ent_right = binary_entropy(
            num_this_class_in_orderline - count_this_class + num_to_add_this_class,
            num_other_class_in_orderline - count_other_class,
        )
        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

    return bsf_ig
