from malenia.data_augmentation._utils import (augmentate,
                                              compute_needed_counts_by_class)
from malenia.data_augmentation.aug_techniques import assign_aug_techniques


class DataAugmentation:
    def __init__(self, aug_techniques, ideal_ir):
        self.aug_techniques = aug_techniques
        self.ideal_ir = ideal_ir

    def balance_dataset(self, X, y, is_ts_data=True):
        aug_techniques_list = assign_aug_techniques(self.aug_techniques)

        needed_counts = compute_needed_counts_by_class(y, self.ideal_ir)
        for label in needed_counts:
            if needed_counts[label] > 0:
                X, y = augmentate(
                    X,
                    y,
                    label,
                    needed_counts[label],
                    augmentation_techniques=aug_techniques_list,
                    is_ts_data=is_ts_data,
                )

        return X, y
