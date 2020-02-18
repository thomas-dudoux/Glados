#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
The Metrics module
Define all the metrics that can be used for a neural network
"""


from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from glados.utils.types import NPVector


def confusion_matrix(validation: NPVector, predicted: NPVector) -> NPVector:
    """
    Computes a confusion matrix using numpy for two np.arrays true and pred.
    :param validation: The array of true values
    :param predicted: The array of prediction values
    """
    num_classes = len(np.unique(validation))
    cm = np.empty((num_classes, num_classes), dtype=np.float32)
    for v, p in zip(validation, predicted):
        vi, pi = int(round(v)), int(round(p))
        cm[vi][pi] += 1.0
    return cm
