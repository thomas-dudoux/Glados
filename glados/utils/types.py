#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
The Types module
Define all the custom types for the Glados package
"""


from nptyping import NDArray, Float


NPVector = NDArray[Float]
NPTensor = NDArray[NPVector]  # AKA a matrix
