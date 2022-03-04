from typing import List, Tuple

import numpy as np


def add_and_remove_instances(X_train: np.ndarray, y_train: np.ndarray, X_pool: np.ndarray,
                             top_k_indices: List) -> Tuple:
    X_pool = np.delete(X_pool, top_k_indices, 0)

    return X_train, y_train, X_pool
