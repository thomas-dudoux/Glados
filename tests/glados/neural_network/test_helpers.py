#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
The module for testing the glados.neural_network.helpers module
"""


from random import randint

import numpy as np
import pytest

from glados.neural_network import helpers


@pytest.fixture
def unison_arr():
    """
    Generate the two array that have to be merged in unison
    :return: A tuple of two numpy array
    """
    arr_size = randint(10, 100)
    arr1 = np.asarray([randint(0, 10) for _ in range(arr_size)])
    arr2 = np.asarray([randint(0, 10) for _ in range(arr_size)])
    return arr1, arr2


def test_shuffle_two_array_unison(unison_arr):
    """
    Test the glados.neural_network.helpers.shuffle_two_array_unison function
    :param unison_arr: The two array that have to be merged in unison
    """
    zipped_array = list(zip(unison_arr[0], unison_arr[1]))
    shuffled_unison_array = helpers.shuffle_two_array_unison(unison_arr[0], unison_arr[1])
    zipped_shuffled_unison_array = list(zip(shuffled_unison_array[0], shuffled_unison_array[1]))
    assert sorted(zipped_array) == sorted(zipped_shuffled_unison_array)
