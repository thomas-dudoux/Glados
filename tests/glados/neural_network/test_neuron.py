#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
The module for testing the glados.neural_network.neuron module
"""


import pytest

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


def test_neuron_forward(simple_neuron):
    """
    Test the glados.neural_network.neuron.Neuron.forward method
    THis test that the neuron do well the dot product of the inputs by the weights and bias
    :param simple_neuron: The neuron fixture to test on
    """
    alpha = simple_neuron.weights[0]
    beta = simple_neuron.bias
    assert simple_neuron.predict([0.8]) == sigmoid(alpha*0.8 + beta)
