#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
The loss module
It contains the loss function definitions usable by a neural network
"""


from typing import List

import numpy as np


class Loss:
    """
    Base class for all Loss
    """

    @staticmethod
    def compute(x: List[float], y: List[float]) -> float:
        raise NotImplementedError

    @staticmethod
    def derivative(x: float, y: float) -> float:
        raise NotImplementedError


class MSE(Loss):
    """
    Class which define the mean squared error function
    """

    @staticmethod
    def compute(x: List[float], y: List[float]) -> float:
        """
        Compute the mean squared error between two vectors
        :param x: The first vector (aka y_true)
        :param y: The second vectors (aka y_pred)
        :return: The error between y_true and y_pred
        """
        return np.square(np.subtract(y, x)).mean()

    @staticmethod
    def derivative(x: float, y: float) -> float:
        """
        Compute the derivative the mse for x and y
        :param x: The predicted value
        :param y: The expected value
        :return: The mse derivative according to x and y
        """
        return 2 * (x - y)


class LogCosH(Loss):
    """
    Class which define the logcosh function
    """

    @staticmethod
    def compute(x: List[float], y: List[float]) -> float:
        """
        The logarithm of the hyperbolic cosine of the prediction error
        :param x: The first vector (aka y_true)
        :param y: The second vectors (aka y_pred)
        :return: The error between y_true and y_pred
        """
        return np.sum(np.log(np.cosh(y - x)))

    @staticmethod
    def derivative(x: float, y: float) -> float:
        """
        Compute the derivative the logcosh for x and y
        :param x: The predicted value
        :param y: The expected value
        :return: The logcosh derivative according to x and y
        """
        raise NotImplementedError
