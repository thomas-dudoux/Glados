#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
The optimizers module
It contains the optimizers function usable by a neural network
"""


from __future__ import annotations

from random import sample
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from sklearn.metrics import confusion_matrix

from glados.neural_network.helpers import shuffle_two_array_unison

if TYPE_CHECKING:
    from glados.neural_network.neuron import Neuron


class Optimizer:
    """
    Base class for all other Optimizers
    """

    def __init__(self):
        """
        Initialize all optimizer class, setup loss and precision history
        """
        self.train_loss_history = list()
        self.train_precision_history = list()
        self.val_loss_history = list()
        self.val_precision_history = list()

    def compute(self, neuron: Neuron, x_train: np.ndarray, y_train: np.ndarray,
                x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                iteration=100, batch_size=32, verbose=True) -> None:
        raise NotImplementedError

    @staticmethod
    def _calculate_precision(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Print the information about the progression of the optimizer
        :param y_pred: The vector of predictions
        :param y_true: The vector of true values
        """
        if len(y_pred.shape) > 1:
            rounded_pred = np.array([round(p) for yp in y_pred for p in yp])
        else:
            rounded_pred = np.array([round(p) for p in y_pred])
        cm = confusion_matrix(y_true, rounded_pred)
        precision = np.nan_to_num(np.diag(cm) / np.sum(cm, axis=0))
        return np.mean(precision)

    def _compute_pred_error_precision(self, neuron: Neuron, xdata: np.ndarray, ytrue: np.ndarray,
                                      history: str) -> Tuple[np.ndarray, float, float]:
        """
        Compute the vector of prediction, the loss and the precision for a given ML structure and append
        the loss and precision into the given history
        :param neuron: The neuron to use to compute the prediction, loss and precision
        :param xdata: The data corresponding either to y_pred or x_val
        :param ytrue: The vector containing the true value of the xdata (aka y_true)
        :param history: The history to which the error and precision should be added (train or validation)
        :raise ValueError: If the history argument is not either 'train' or 'validation'
        """
        pred = np.asarray([neuron.forward(xt) for xt in xdata])
        error = neuron.loss.compute(pred, ytrue)
        precision = self._calculate_precision(pred, ytrue)
        if history == 'train':
            self.train_loss_history.append(error)
            self.train_precision_history.append(precision)
        elif history == 'validation':
            self.val_loss_history.append(error)
            self.val_precision_history.append(precision)
        else:
            raise ValueError('The history argument can only be either "train" or "validation"')
        return pred, error, precision


class SGD(Optimizer):
    """
    Class to compute the SGD algorithm
    """

    def __init__(self):
        """
        Initialize The SGD like his parent
        """
        super().__init__()

    def compute(self, neuron: Neuron, x_train: np.ndarray, y_train: np.ndarray,
                x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                iteration=100, batch_size=32, verbose=True) -> None:
        """
        Execute the mini batch Stochastic Gradient Descent algorithm on a ML structure
        :param neuron: The neuron to apply the SGD on
        :param x_train: The learning data
        :param y_train: The learning prediction
        :param x_val: The validation data
        :param y_val: The validation prediction
        :param iteration: The number of iteration for the neuron to learn
        :param batch_size: The number of randomly picked element to take at each iteration
        :param verbose: If the algorithm should print information about his status
        """
        for i in range(iteration):
            random_indices = sample(range(len(x_train)), batch_size)
            it_x_train, it_y_train = shuffle_two_array_unison(x_train[random_indices], y_train[random_indices])
            train_pred, train_error, train_precision = self._compute_pred_error_precision(neuron, it_x_train,
                                                                                          it_y_train, 'train')
            if verbose:
                print(f'Iteration : {i}  |  Train loss : {train_error}  |  Train precision : {train_precision}')
            for xi, xt in enumerate(it_x_train):
                loss_derivative = neuron.loss.derivative(it_y_train[xi], train_pred[xi])
                activation_derivative = neuron.activation(train_pred[xi], derivative=True)
                base_gradient = loss_derivative * activation_derivative
                for w in range(len(neuron.weights)):
                    neuron.weights[w] += neuron.learning_rate * (base_gradient * xt[w])
                neuron.bias += neuron.learning_rate * base_gradient
            if x_val and y_val:
                it_x_val, it_y_val = shuffle_two_array_unison(x_val[random_indices], y_val[random_indices])
                _, val_error, val_precision = self._compute_pred_error_precision(neuron, it_x_val,
                                                                                 it_y_val, 'validation')
                if verbose:
                    print(f'Val loss : {val_error}  |  Val Precision : {val_precision}')
