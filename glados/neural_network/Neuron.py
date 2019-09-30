#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
The Neuron module
It contains all the necessary to define an artificial neuron
"""


from typing import List
from random import random, sample

import numpy as np

from activations import ActivationFunction, sigmoid
from loss import Loss, MSE
from helpers import shuffle_two_array_unison


class Neuron:
    """
    Class that act as an artificial neuron
    """

    def __init__(self, num_inputs: int, activation: ActivationFunction, loss: Loss, learning_rate=0.01):
        """
        Initialize the neuron class with random weight and given loss and activation function
        :param num_inputs: The number of inputs that the neuron will have
        :param activation: The activation function that the neuron will use
        :param loss: The loss function to use to calculate the error
        :param learning_rate: The learning rate to use
        """
        self.weights = [random() for _ in range(num_inputs)]
        self.bias = random()
        self.activation = activation
        self.loss = loss
        self.learning_rate = learning_rate

    def learn(self, x_train, y_train, x_val=None, y_val=None, iteration=100, batch_size=32) -> None:
        """
        Make the neuron learn from the data (Basically Mini-batch SGD ATM)
        :param x_train: The learning data
        :param y_train: The learning prediction
        :param x_val: The validation data
        :param y_val: The validation prediction
        :param iteration: The number of iteration for the neuron to learn
        :param batch_size: The number of randomly picked element to take at each iteration
        """
        for i in range(iteration):
            print(f'Iteration : {i}')
            random_indices = sample(range(len(x_train)), batch_size)
            it_x_train, it_y_train = shuffle_two_array_unison(x_train[random_indices], y_train[random_indices])
            pred = [self.forward(xt) for xt in it_x_train]
            error = self.loss.compute(pred, it_y_train)
            print(f'Loss : {error}')
            for xi, xt in enumerate(it_x_train):
                base_gradient = self.loss.derivative(it_y_train[xi], pred[xi]) * self.activation(pred[xi], derivative=True)
                for w in range(len(self.weights)):
                    self.weights[w] += self.learning_rate * (base_gradient * xt[w])
                self.bias += self.learning_rate * base_gradient
            if x_val and y_val:
                pass

    def forward(self, inputs: List[float]) -> float:
        """
        Do a forward pass on the neuron (aka sum of the inputs weighted plus the bias)
        :param inputs: The list of inputs data to predict from
        :return: The value returned by the activation function
        """
        act = self.bias + np.dot(inputs, self.weights)
        return self.activation(act)

    predict = forward  # Alias for the forward function


if __name__ == '__main__':
    dataset = list()
    for _ in range(1000):
        x = random()
        y = 1.0 if x >= 0.5 else 0.0
        dataset.append((x, y))
    x_train = np.asarray([[d[0]] for d in dataset], dtype=np.float32)
    y_train = np.asarray([d[1] for d in dataset], dtype=np.float32)
    neuron = Neuron(1, sigmoid, MSE())
    neuron.learn(x_train, y_train, iteration=1000)
    predicted = neuron.predict([0.8])
    print(f'Predicted value for 0.8 -> 1: {predicted}')
    predicted = neuron.predict([0.2])
    print(f'Predicted value for 0.2 -> 0: {predicted}')
