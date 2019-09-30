#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
The module for testing the glados.neural_network.activations module
"""


from math import isclose

from glados.neural_network import activations


def test_sigmoid():
    """
    Test the glados.neural_network.activations.sigmoid function
    """
    assert activations.sigmoid(0) == 0.5
    assert isclose(activations.sigmoid(10 ** 10), 1.0) is True
    assert isclose(activations.sigmoid(-10 ** 10), 0.0) is True
    assert activations.sigmoid(0, derivative=True) == 0.25
    assert isclose(activations.sigmoid(10 ** 10, derivative=True), 0.0) is True
    assert isclose(activations.sigmoid(-10 ** 10, derivative=True), 0.0) is True


def test_tanh():
    """
    Test the glados.neural_network.activations.tanh function
    """
    assert activations.tanh(0) == 0
    assert isclose(activations.tanh(10 ** 10), 1) is True
    assert isclose(activations.tanh(-10 ** 10), -1) is True
    assert activations.tanh(0, derivative=True) == 1
    assert isclose(activations.tanh(10 ** 10, derivative=True), 0) is True
    assert isclose(activations.tanh(-10 ** 10, derivative=True), 0) is True


def test_relu():
    """
    Test the glados.neural_network.activations.relu function
    """
    assert activations.relu(1) == 1
    assert activations.relu(0, derivative=True) == 0
    assert activations.relu(1, derivative=True) == 1


def test_leaky_relu():
    """
    Test the glados.neural_network.activations.leaky_relu function
    with alpha = 0.01
    """
    assert activations.leaky_relu(1) == 1
    assert activations.leaky_relu(-1) == -0.01
    assert activations.leaky_relu(1, derivative=True) == 1
    assert activations.leaky_relu(-1, derivative=True) == 0.01
