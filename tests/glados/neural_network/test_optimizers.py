#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
The module for testing the glados.neural_network.optimizers module
"""


import numpy as np
import pytest
from random import random

from glados.neural_network.activations import sigmoid
from glados.neural_network.loss import MSE
from glados.neural_network.neuron import Neuron
from glados.neural_network.optimizers import SGD


@pytest.fixture
def simple_neuron():
    """
    Instantiate a simple neuron with SGD, MSE, a single weight and bias
    :return: A simple neuron with one weight and a bias
    """
    return Neuron(1, sigmoid, MSE(), SGD())


@pytest.fixture
def neuron2():
    """
    Instantiate a simple neuron with SGD, MSE, a single weight and bias
    :return: A simple neuron with one weight and a bias
    """
    return Neuron(2, sigmoid, MSE(), SGD())


@pytest.fixture
def dataset1():
    """
    Create a simple Dataset for logistic regression purposes
    :return: A tuple representing the dataset with x_train and y_train
    """
    dataset = list()
    for _ in range(100):
        x = random()
        y = 1. if x >= 0.5 else 0.
        dataset.append((x, y))
    x_train = np.asarray([[d[0]] for d in dataset], dtype=np.float32)
    y_train = np.asarray([d[1] for d in dataset], dtype=np.float32)
    return x_train, y_train


@pytest.fixture
def dataset2():
    """
    Create a simple Dataset for logistic regression purposes (AND Gate dataset)
    :return: A tuple representing the dataset with x_train and y_train
    """
    dataset = list()
    for _ in range(100):
        x = (round(random()), round(random()))
        y = 1. if x[0] == 1. and x[1] == 1. else 0.
        dataset.append((x, y))
    x_train = np.asarray([d[0] for d in dataset], dtype=np.float32)
    y_train = np.asarray([d[1] for d in dataset], dtype=np.float32)
    return x_train, y_train


def test_sgd_compute_n1d1(simple_neuron, dataset1):
    """
    Test the glados.neural_network.optimizer.SGD.compute method
    It test that the neuron learn well from the data with this optimizer and simple separable data
    :param simple_neuron: The neuron fixture to test on
    :param dataset1: The dataset to test the neuron learning with SGD
    """
    x_train, y_train = dataset1
    simple_neuron.learn(x_train, y_train, iteration=100, verbose=False)
    assert round(simple_neuron.predict([0.75])) == 1.
    assert round(simple_neuron.predict([0.25])) == 0.


def test_sgd_compute_n2d2(neuron2, dataset2):
    """
    Test the glados.neural_network.optimizer.SGD.compute method
    It test that the neuron learn well from the data with this optimizer and simple separable data (AND Gate)
    :param neuron2: The neuron fixture to test on
    :param dataset2: The dataset to test the neuron learning with SGD
    """
    x_train, y_train = dataset2
    neuron2.learn(x_train, y_train, iteration=100, verbose=False)
    assert round(neuron2.predict([1., 1.])) == 1.
    assert round(neuron2.predict([0., 1.])) == 0.
    assert round(neuron2.predict([1., 0.])) == 0.
    assert round(neuron2.predict([0., 0.])) == 0.
