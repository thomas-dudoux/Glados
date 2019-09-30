#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
The helpers module
It contains some utilities functions
"""


from typing import Tuple

import numpy as np


def shuffle_two_array_unison(arr1: np.ndarray, arr2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffle two array in unison, so array the shuffled the same way
    :param arr1: The first array to be shuffled
    :param arr2: The second array to be shuffled
    :return: A tuple with the two new shuffled array
    """
    indices = np.arange(arr1.shape[0])
    np.random.shuffle(indices)
    return arr1[indices], arr2[indices]
