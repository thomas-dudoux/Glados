#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
The Neuron module
It contains all the necessary to define an artificial neuron
"""


from __future__ import annotations

from typing import List, Optional, Type, TYPE_CHECKING

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

    @property
    def output_layer(self) -> Layer:
        """
        Return the last layer (output layer) of the neural network
        """
        return self.layers[-1]

    @property
    def input_layer(self) -> Layer:
        """
        Return the first layer (input layer) of the neural network
        """
        return self.layers[0]

    def backpropagate_error(self, preds: NPTensor, y_true: NPTensor) -> None:
        """
        Apply the error backpropagation algortihm on a Neural Network
        :param preds: The list of predictions made by the neural network
        :param y_true: The list of true values from the dataset
        """
        self.output_layer.errors = y_true - preds
        self.output_layer.deltas = self.output_layer.errors * self.output_layer.activation(preds, derivative=True)
        for li in reversed(range(len(self.layers[:-1]))):
            layer = self.layers[li]
            previous_layer = self.layers[li+1]
            layer.errors = previous_layer.deltas.dot(previous_layer.neurons['weights'].T)
            layer.deltas = layer.errors * layer.activation(layer.outputs, derivative=True)

    def batch_weights_update(self, x_train: NPTensor) -> None:
        """
        Update all the weight of a neural network given the error of each neuron
        Using the batch method averaging
        :param x_train: The input data that have caused the error and that we should use to update the weights
        """
        for il, layer in enumerate(self.layers):
            input_data = x_train if il == 0 else np.asarray(self.layers[il-1].outputs, np.float32)
            deltas = np.asarray(layer.deltas, dtype=np.float32)
            if il == 0 and layer.num_inputs == 1:
                layer.neurons['weights'] += np.multiply(input_data, deltas).T.mean(axis=1) * self.learning_rate
            else:
                layer.neurons['weights'] += input_data.T.dot(deltas) * self.learning_rate
            layer.neurons['bias'] += deltas.mean(axis=0) * self.learning_rate

    def update_weights(self, x_train: NPTensor) -> None:
        """
        Update all the weight of a neural network given the error of each neuron
        :param x_train: The input data that have caused the error and that we should use to update the weights
        """
        for xti in range(len(x_train)):
            for il, layer in enumerate(self.layers):
                input_data = x_train[xti] if il == 0 else np.asarray(self.layers[il-1].outputs[xti], np.float32)
                if il == 0 and layer.num_inputs == 1:
                    layer.neurons['weights'] += (layer.deltas[xti] * input_data) * self.learning_rate
                else:
                    layer.neurons['weights'] += (layer.deltas[xti] * np.expand_dims(input_data, axis=1)) * self.learning_rate
                layer.neurons['bias'] += layer.deltas[xti] * self.learning_rate

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
        if layer == Input and num_inputs is None:
            num_inputs = 1
        if layer != Input and num_inputs is None:
            num_inputs = self.output_layer.neurons['weights'].shape[-1]
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
        layer_output = self.input_layer.forward(data)
        for li in range(1, len(self.layers)):
            layer_output = self.layers[li].forward(layer_output)
        if clear_output:
            self.clear_layers_output()
        return layer_output

    def predict(self, data: NPVector) -> NPVector:
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
        Initialize a layer with random weights and bias
        :param num_neurons: The number of neurons in the layer
        :param num_inputs: The number of input of each neurons in the layer
        :param activation: The activation function used by each neurons in the layer
        :param bias: If the layer should have a bias neuron or not, defaults to True
        :param dropout: The percentage of dropout in the layer, defaults to None
        """
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.neurons = {
            'weights': np.random.random((num_inputs, num_neurons)),
            'bias': np.random.random(num_neurons) if bias else 0.
        }
        self.activation = activation
        # self.output = np.zeros(num_neurons, dtype=np.float32)
        self.dropout = dropout
        self._outputs: List[np.ndarray] = list()
        self.deltas: np.ndarray = None
        self.errors: np.ndarray = None

    def forward(self, input_data: NPVector) -> NPVector:
        """
        Make a forward pass onto the layer
        """
        out = np.dot(input_data, self.neurons['weights']) + self.neurons['bias']
        out = self.activation(out)
        self._outputs.append(out)
        return out

    def clear(self) -> None:
        """
        Clear the inputs and outputs buffer lists
        """
        # self.output = np.zeros(self.neurons, dtype=np.float32)
        self._outputs = list()
        self.errors = None
        self.deltas = None

    @property
    def outputs(self) -> NPTensor:
        """
        Return the outputs as a vector of numpy array
        """
        return np.asarray(self._outputs, dtype=np.float32)


class Input(Layer):
    """
    Class that act as a layer of input for a NN
    """

    def __init__(self, num_neurons: int, activation: ActivationFunction,
                 num_inputs: Optional[int] = 1, bias=False, dropout: Optional[float] = None):
        """
        Initialize an Input layer with one input per neuron
        All the parameters are the same than the parent "Layer" class
        """
        super().__init__(num_neurons, activation, num_inputs, bias, dropout)

    def forward(self, input_data: NPVector) -> NPVector:
        """
        Make a forward pass onto the layer
        """
        if self.neurons['weights'].shape[0] == 1:
            out = input_data * self.neurons['weights'][0]
        else:
            out = super().forward(input_data)
        self._outputs.append(out)
        return out


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
