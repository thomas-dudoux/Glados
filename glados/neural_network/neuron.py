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
from toolz import curry

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

    def backpropagate_error(self, preds: NPVector, y_train: NPVector) -> None:
        """
        Apply the error backpropagation algortihm on a Neural Network
        :param preds: The list of predictions made by the neural network
        :param y_train: The list of true values from the dataset
        """
        for onn, output_neuron in enumerate(self.layers[-1].neurons):  # Calculate the error for the output layer
            # loss_derivative = self.loss.derivative(y_train[onn], preds[onn])
            neuron_loss = self.loss.compute(y_train[onn], preds[onn])
            # neuron_loss = y_train[onn] - preds[onn]
            activation_derivative = output_neuron.activation(preds[onn], derivative=True)
            output_neuron.error = neuron_loss * activation_derivative
        for nl, layer in enumerate(self.layers[::-1][:-1]):  # Calculate the error for all the other layers
            for neuron in layer.neurons:
                for wl, weight in enumerate(neuron.weights):
                    backpropagated_neuron = self.layers[::-1][nl+1].neurons[wl]
                    backpropagated_neuron.error += (weight * neuron.error) * neuron.activation(neuron.output, derivative=True)

    def update_weights(self, x_train: NPTensor) -> None:
        """
        Update the all the weight of a neural network given the error of each neuron
        :param x_train: The input data that have caused the error and that we should use to update the weights
        """
        for inn, input_neuron in enumerate(self.layers[0].neurons):
            input_neuron.update_weights(self.learning_rate, x_train[inn])
        for nl, layer in enumerate(self.layers[1:], start=1):
            inputs_data = np.asarray([neuron.output for neuron in self.layers[nl-1].neurons], dtype=np.float32)
            for neuron in layer.neurons:
                for weight in neuron.weights:
                    neuron.update_weights(self.learning_rate, inputs_data)

    @curry
    def add(self, layer: Type[Layer], num_neurons: int, neuron: Type[Neuron],
            num_inputs: int, activation: ActivationFunction, dropout: float = None) -> None:
        """
        Add a layer to the neural network, be it either a Input or a Layer
        :param layer: The layer that you want to add to the NN
        """
        new_layer = layer([neuron(num_inputs, activation) for _ in range(num_neurons)], dropout)
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

    def forward(self, data: NPVector) -> NPVector:
        """
        Get a prediction from the neural network
        :param data: The data to predict from
        """
        self.layers[0].layer_inputs = data
        self.layers[0].forward()
        for l in range(1, len(self.layers)):
            self.layers[l].layer_inputs = self.layers[l-1].layer_outputs
            self.layers[l].forward()
        res = self.layers[-1].layer_outputs
        for layer in self.layers:
            layer.clear()
        return res

    predict = forward  # Alias for the forward function


class Layer:
    """
    Class that act as a layer containing multiple neurons
    """

    def __init__(self, neurons: Sequence[Neuron], dropout: float = None):
        """
        Initialize layer containing multiple neurons
        :param neurons: The list of neurons to be in the layer
        :param dropout: The dropout probability to apply on the layer during forward pass
        """
        self.neurons = neurons
        self.layer_outputs = np.zeros(len(neurons), dtype=np.float32)
        self.layer_inputs = np.zeros(len(neurons), dtype=np.float32)
        self.dropout = dropout

    def forward(self) -> None:
        """
        Make a forward pass onto the layer
        """
        for n in range(len(self.neurons)):
            if self.dropout and random() <= self.dropout:
                self.layer_outputs[n] = 0.0
            else:
                self.layer_outputs[n] = self.neurons[n].forward(self.layer_inputs)

    def clear(self) -> None:
        """
        Clear the inputs and outputs buffer lists
        """
        self.layer_outputs = np.zeros(len(self.neurons), dtype=np.float32)
        self.layer_inputs = np.zeros(len(self.neurons), dtype=np.float32)


class Input(Layer):
    """
    Class that act as a layer containing multiple neurons
    """

    def __init__(self, neurons: Sequence[Neuron]):
        """
        Initialize layer containing multiple neurons
        :param neurons: The list of neurons to be in the layer
        """
        super().__init__(neurons)

    def forward(self) -> None:
        """
        Make a forward pass onto the layer
        """
        for n in range(len(self.neurons)):
            self.layer_outputs[n] = self.neurons[n].forward(self.layer_inputs[n])


class Neuron:
    """
    Class that act as an artificial neuron
    """

    def __init__(self, num_inputs: int, activation: ActivationFunction, bias=True):
        """
        Initialize the neuron class with random weight and given loss and activation function
        :param num_inputs: The number of inputs that the neuron will have
        :param activation: The activation function that the neuron will use
        :param bias: If the neuron should have a bias or not
        """
        self.weights = np.array([random() for _ in range(num_inputs)])
        self.bias = random() if bias else None
        self.activation = activation
        self.error = 0.0
        self.output = 0.0

    def forward(self, inputs: NPVector) -> float:
        """
        Do a forward pass on the neuron (aka sum of the inputs weighted plus the bias)
        :param inputs: The list of inputs data to predict from
        :return: The value returned by the activation function
        """
        if self.bias is not None:
            out = self.bias + np.dot(inputs, self.weights)
        else:
            out = np.dot(inputs, self.weights)
        self.output = self.activation(out)
        return self.output

    def update_weights(self, learning_rate: float, input_data: Union[NPVector, float]) -> None:
        """
        Update the weights of the neuron for a given learning rate and input
        :param learning_rate: The learning to update the weights with
        :param input_data: The input that cause the weight change
        """
        lr_err = learning_rate * self.error
        for w in range(len(self.weights)):
            try:  # If it's a vector aka a neuron in the hidden or output layer
                self.weights[w] += lr_err * input_data[w]
            except (TypeError, IndexError):  # If it's a float aka a neuron in the input layer
                self.weights[w] += lr_err * input_data
        if self.bias is not None:
            self.bias += lr_err
