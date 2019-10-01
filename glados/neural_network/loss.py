#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
The loss module
It contains the loss function definitions usable by a neural network
"""


from __future__ import annotations

import numpy as np


class Loss:
    """
    Base class for all Loss
    """

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError

    @staticmethod
    def derivative(y_true: float, y_pred: float) -> float:
        raise NotImplementedError


class MSE(Loss):
    """
    Class which define the mean squared error function
    """

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean squared error between two vectors
        :param y_true: The second vectors contains the true values
        :param y_pred: The first vector, contains the predicted values
        :return: The error between y_true and y_pred
        """
        return np.square(np.subtract(y_true, y_pred)).mean()

    @staticmethod
    def derivative(y_true: float, y_pred: float) -> float:
        """
        Compute the derivative of the mse for y_pred and y_true
        :param y_true: The expected value
        :param y_pred: The predicted value
        :return: The mse derivative according to y_pred and y_true
        """
        return 2 * (y_true - y_pred)


class LogCosH(Loss):
    """
    Class which define the logcosh function
    """

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        The logarithm of the hyperbolic cosine of the prediction error
        :param y_true: The second vectors contains the true values
        :param y_pred: The first vector, contains the predicted values
        :return: The error between y_true and y_pred
        """
        return np.sum(np.log(np.cosh(np.subtract(y_true, y_pred))))

    @staticmethod
    def derivative(y_true: float, y_pred: float) -> float:
        """
        Compute the derivative of the logcosh for y_pred and y_true
        :param y_true: The expected value
        :param y_pred: The predicted value
        :return: The mse derivative according to y_pred and y_true
        """
        return np.tanh(y_true - y_pred)
