#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from random import random

import numpy as np

from glados.neural_network.neuron import Dense, Input, NeuralNetwork
from glados.neural_network.loss import MSE
from glados.neural_network.optimizers import SGD
from glados.neural_network.activations import sigmoid


dsize = 384
dataset = np.empty((dsize, 3), dtype=np.float32)
for i in range(dsize):
    r1 = random() * 10.
    r2 = random() * 10.
    res = 1.0 if (r1 + r2) >= 10. else 0.0
    dataset[i] = np.asarray([r1, r2, res], dtype=np.float32)

x_train = dataset[:320][:, :2] / 10.
y_train = dataset[:320][:, 2:]
x_val = dataset[320:][:, :2] / 10.
y_val = dataset[320:][:, 2:]

dorothy = NeuralNetwork(MSE(), SGD())
dorothy.add(Input, 2, sigmoid, 2)
dorothy.add(Dense, 3, sigmoid)
dorothy.add(Dense, 1, sigmoid)
dorothy.learn(x_train, y_train, x_val, y_val)
