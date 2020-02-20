#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
The Neuron module
It contains all the necessary to define an artificial neuron
"""


from __future__ import annotations

from typing import List, Optional, Sequence, Type, TYPE_CHECKING, Union
from random import random

import numpy as np

if TYPE_CHECKING:
    from glados.neural_network.activations import ActivationFunction
    from glados.neural_network.loss import Loss
    from glados.neural_network.optimizers import Optimizer
    from glados.utils.types import NPTensor, NPVector


class NeuralNetwork:
    """
    Class that act as an entire neural network
    """

    def __init__(self, loss: Loss, optimizer: Optimizer, learning_rate=0.01):
        """
        Initialize the neural network class
        :param loss: The loss function to use to calculate the error
        :param optimizer: The optimizer function to use to reduce the error
        :param learning_rate: The learning rate to use
        """
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.layers: List[Layer] = list()

    def backpropagate_error(self, preds: NPTensor, y_true: NPTensor) -> None:
        """
        Apply the error backpropagation algortihm on a Neural Network
        :param preds: The list of predictions made by the neural network
        :param y_true: The list of true values from the dataset
        """
        for il in reversed(range(len(self.layers))):
            layer = self.layers[il]
            if layer == self.layers[-1]:  # if it's the output layer
                layer.error = self.loss.compute(y_true, preds)
                # layer.error = y_train - preds
                layer.delta = layer.error * layer.activation(preds, derivative=True)
            else:  # if it's the other layers
                previous_layer = self.layers[il+1]
                layer.error = np.dot(previous_layer.neurons['weights'], previous_layer.delta)
                layer.delta = layer.error * layer.activation(layer.output, derivative=True)

    def update_weights(self, x_train: NPTensor) -> None:
        """
        Update the all the weight of a neural network given the error of each neuron
        :param x_train: The input data that have caused the error and that we should use to update the weights
        """
        for il, layer in enumerate(self.layers):
            input_data = np.atleast_2d(x_train if il == 0 else self.layers[il-1].output)
            layer.neurons['weights'] += layer.delta * input_data.T * self.learning_rate
            layer.neurons['bias'] += layer.delta * self.learning_rate

    def add(self, layer: Type[Layer], num_neurons: int, activation: ActivationFunction,
            num_inputs: Optional[int] = None, bias=True, dropout: Optional[float] = None) -> None:
        """
        Add a layer to the neural network, be it either a Input or a Layer
        :param layer: The class of the layer type to add in the NN
        :param num_neurons: The number of neurons that the layer should have
        :param activation: The activation function to use with this layer
        :param num_inputs: The number of input for each neuron to have,
            equals to the number of neuron in the previous layer if None, defaults to None
        :param bias: If the layer should have a bias neuron, defaults to True
        :param dropout: The dropout proportion to apply on the forward and backpropagation pass, defaults to None
        """
        if layer != Input and not self.layers:
            raise ValueError("The first layer should be an Input layer")
        # if layer == Input and num_inputs is None:
        #     num_inputs = 1
        if layer != Input and num_inputs is None:
            num_inputs = self.layers[-1].neurons['weights'].shape[-1]
            # num_inputs = len(self.layers[-1].neurons['weights'])
        new_layer = layer(num_neurons, activation, num_inputs, bias, dropout)
        self.layers.append(new_layer)

    def learn(self, x_train: NPVector, y_train: NPVector, x_val: Optional[NPVector] = None,
              y_val: Optional[NPVector] = None, iteration=100, batch_size=32, verbose=True) -> None:
        """
        Make the neural network learn from the data (Basically just a wrapper around the optimizer)
        :param x_train: The learning data
        :param y_train: The learning prediction
        :param x_val: The validation data
        :param y_val: The validation prediction
        :param iteration: The number of iteration for the neuron to learn
        :param batch_size: The number of randomly picked element to take at each iteration
        :param verbose: If the learning process should print information about his status
        """
        self.optimizer.compute(self, x_train, y_train, x_val, y_val, iteration, batch_size, verbose)

    def forward(self, data: NPVector, clear_output=False) -> NPVector:
        """
        Get a prediction from the neural network
        :param data: The data to predict from
        :param clear_output: If we should clear the output of our layers after the forward pass
        """
        layer_output = self.layers[0].forward(data)
        for li in range(1, len(self.layers)):
            layer_output = self.layers[li].forward(layer_output)
        nn_out = self.layers[-1].output
        if clear_output:
            self.clear_layers_output()
        return nn_out

    def predict(self, data: NPVector,) -> NPVector:
        """
        Normalize the predicted values
        :param data: The data to predict from
        :return: The vector of predicted values
        """
        res = self.forward(data)
        axis = None if res.ndim == 1 else 1
        return np.argmax(res, axis=axis)

    def clear_layers_output(self) -> None:
        """
        Reset the output of each layer in the NN
        """
        for layer in self.layers:
            layer.clear()


class Layer:
    """
    Class that act as a layer containing multiple neurons
    """

    def __init__(self, num_neurons: int, activation: ActivationFunction,
                 num_inputs: int, bias=True, dropout: Optional[float] = None):
        """

        :param num_neurons: [description]
        :param num_inputs: [description]
        :param activation: [description]
        :param bias: [description], defaults to True
        :param dropout: [description], defaults to None
        """
        self.neurons = {
            'weights': np.random.random((num_inputs, num_neurons)),
            'bias': np.random.random(num_neurons)
        }
        self.activation = activation
        self.output = np.zeros(num_neurons, dtype=np.float32)
        self.dropout = dropout
        self.delta = 0.0
        self.error = 0.0

    def forward(self, input_data: NPVector) -> NPVector:
        """
        Make a forward pass onto the layer
        """
        out = np.dot(input_data, self.neurons['weights']) + self.neurons['bias']
        out = self.activation(out)
        self.output.append(out)
        return out

    def clear(self) -> None:
        """
        Clear the inputs and outputs buffer lists
        """
        self.output = np.zeros(self.neurons, dtype=np.float32)


class Input(Layer):
    """
    Class that act as a layer of input for a NN
    """

    def __init__(self, num_neurons: int, activation: ActivationFunction,
                 num_inputs: Optional[int] = 1, bias=True, dropout: Optional[float] = None):
        """
        Initialize an Input layer with one input per neuron
        All the parameters are the same than the parent "Layer" class
        """
        super().__init__(num_neurons, activation, num_inputs, bias, dropout)


class Dense(Layer):
    """
    Class that a layer of dense neuron
    """

    def __init__(self, num_neurons: int, activation: ActivationFunction,
                 num_inputs: int, bias=True, dropout: Optional[float] = None):
        """
        Initialize a Dense layer for a NN
        All the parameters are the same than the parent "Layer" class
        """
        super().__init__(num_neurons, activation, num_inputs, bias, dropout)
