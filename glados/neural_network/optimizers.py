#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
The optimizers module
It contains the optimizers function usable by a neural network
"""


from __future__ import annotations

from dataclasses import dataclass
from random import sample
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from glados.utils.metrics import confusion_matrix
from glados.utils.helpers import shuffle_two_array_unison

if TYPE_CHECKING:
    from glados.neural_network.neuron import NeuralNetwork
    from glados.utils.types import NPVector


class Optimizer:
    """
    Base class for all other Optimizers
    """

    def __init__(self):
        """
        Initialize all optimizer class, setup loss and precision history
        """
        self.train_precision_history: List[Precision] = list()
        self.val_precision_history: List[Precision] = list()

    def compute(self, nn: NeuralNetwork, x_train: NPVector, y_train: NPVector,
                x_val: Optional[NPVector] = None, y_val: Optional[NPVector] = None,
                iteration=100, batch_size=32, batch=False, verbose=True) -> None:
        raise NotImplementedError

    @staticmethod
    def _calculate_precision(y_pred: NPVector, y_true: NPVector) -> float:
        """
        Print the information about the progression of the optimizer
        :param y_pred: The vector of predictions
        :param y_true: The vector of true values
        """
        if len(y_pred.shape) > 1:
            rounded_pred = np.array([round(p) for yp in y_pred for p in yp])
        else:
            rounded_pred = np.array([round(p) for p in y_pred])
        cm = confusion_matrix(y_true.flatten(), rounded_pred.flatten())
        precision = np.nan_to_num(np.diag(cm) / np.sum(cm, axis=0))
        return np.mean(precision)

    def _compute_pred_error_precision(self, nn: NeuralNetwork, xdata: NPVector, ytrue: NPVector,
                                      history: str) -> Tuple[NPVector, float, float]:
        """
        Compute the vector of prediction, the loss and the precision for a given ML structure by doing a forward pass
        for each element in the given dataset and append the loss and precision into the given history
        :param nn: The Neural Network to use to compute the prediction, loss and precision
        :param xdata: The data corresponding either to y_pred or x_val
        :param ytrue: The vector containing the true value of the xdata (aka y_true)
        :param history: The history to which the error and precision should be added (train or validation)
        :raise ValueError: If the history argument is not either 'train' or 'validation'
        """
        preds = np.asarray([nn.forward(xt) for xt in xdata], np.float32)
        error = nn.loss.compute(preds, ytrue)
        accuracy = self._calculate_precision(preds, ytrue)
        if history == 'train':
            self.train_precision_history.append(Precision(error, accuracy))
        elif history == 'validation':
            self.val_precision_history.append(Precision(error, accuracy))
        else:
            raise ValueError('The history argument can only be either "train" or "validation"')
        return preds, error, accuracy

    def _logs(self, iteration: int) -> None:
        """
        Logs the status of the Neural network
        :param iteration: The number of iteration the NN is at
        """
        train_loss = self.train_precision_history[-1].error
        train_accuracy = self.train_precision_history[-1].accuracy * 100
        print(f'Iteration : {iteration} =>')
        print(f'    Train loss : {train_loss}  |  Train precision : {train_accuracy}')
        if self.val_precision_history:
            val_loss = self.val_precision_history[-1].error
            val_accuracy = self.val_precision_history[-1].accuracy * 100
            print(f'    Val loss : {val_loss}  |  Val Precision : {val_accuracy}')
        print('*' * 20)


class SGD(Optimizer):
    """
    Class to compute the SGD algorithm
    """

    def __init__(self):
        """
        Initialize The SGD like his parent
        """
        super().__init__()

    def compute(self, neural_network: NeuralNetwork, x_train: NPVector, y_train: NPVector,
                x_val: Optional[NPVector] = None, y_val: Optional[NPVector] = None,
                iteration=100, batch_size=32, verbose=True, verbose_frequency=10) -> None:
        """
        Execute the mini batch Stochastic Gradient Descent algorithm on a ML structure
        :param neural_network: The NeuralNetwork to apply the SGD on
        :param x_train: The learning data
        :param y_train: The learning prediction
        :param x_val: The validation data
        :param y_val: The validation prediction
        :param iteration: The number of iteration for the neuron to learn
        :param batch_size: The number of randomly picked element to take at each iteration
        :param verbose: If the algorithm should print information about his status
        :param verbose_frequency: At how much frequency should we output some info
        """
        for i in range(iteration):
            random_indices = sample(range(len(x_train)), batch_size)
            it_x_train, it_y_train = shuffle_two_array_unison(x_train[random_indices], y_train[random_indices])
            train_pred, _, _ = self._compute_pred_error_precision(neural_network, it_x_train, it_y_train, 'train')
            neural_network.backpropagate_error(train_pred, it_y_train)
            neural_network.batch_weights_update(it_x_train)
            # neural_network.update_weights(it_x_train)
            if x_val is not None and y_val is not None:
                random_indices = sample(range(len(x_val)), batch_size)
                it_x_val, it_y_val = shuffle_two_array_unison(x_val[random_indices], y_val[random_indices])
                self._compute_pred_error_precision(neural_network, it_x_val, it_y_val, 'validation')
            if verbose and i % verbose_frequency == 0:
                self._logs(i)
            neural_network.clear_layers_output()


@dataclass
class Precision:
    error: float
    accuracy: float
