#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
This activation module
It contains the activations function usable by a neuron
"""


from __future__ import annotations

from typing import Callable

import numpy as np


ActivationFunction = Callable[[float], float]


def sigmoid(x: float, derivative=False) -> float:
    """
    Calculate the y value of x on a sigmoid
    :param x: The value to calculate on the sigmoid
    :param derivative: If True calculate the derivative on x instead
    :return: The y value of x for a sigmoid
    """
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def tanh(x: float, derivative=False) -> float:
    """
    Return the hyperbolic tangent of x
    :param x: The value to calculate the hyperbolic tangent
    :param derivative: If True calculate the derivative on x instead
    :return: The hyperbolic tangent of x
    """
    th = np.tanh(x)
    if derivative:
        return 1.0 - th**2
    return th


def relu(x: float, derivative=False) -> float:
    """
    Return the Rectified linear units of x
    :param x: The value to calculate the hyperbolic tangent
    :param derivative: If True calculate the derivative on x instead
    :return: The hyperbolic tangent of x
    """
    if derivative:
        return 0 if x <= 0 else 1
    return np.maximum(0, x)


def leaky_relu(x: float, derivative=False, alpha=0.01) -> float:
    """
    Return the leaky rectified linear units of x
    :param x: The value to calculate the hyperbolic tangent
    :param derivative: If True calculate the derivative on x instead
    :param alpha: The coefficient for the negative part of the leaky relu
    :return: The hyperbolic tangent of x
    """
    if derivative:
        return alpha if x <= 0 else 1
    return alpha*x if x <= 0 else x
