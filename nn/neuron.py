#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy


def decision(X, W, dev=False):
    """
    Decision value is the inner product of data and weight.
    Definition: X * W
    Derivative of decision function directly return input data.
    Definition: X

    Args:
        X (numpy array): M x N
        W (numpy array): (N + 1) x K

    Returns:
        if dev:
            Z' (numpy array): M x N, gradient of decision
        else:
            Z (numpy array): M x K, decision value
    """
    if dev:
        return X
    else:
        X = add_bias(X)
        return numpy.dot(X, W)


def activate(Z, func='sigmoid', dev=False):
    """

    Args:
        Z (numpy array): M x K, decision value
        func (str): determine activation funciton
                sigmoid -> sigmoid
                tanh -> tanh
                relu -> relu
                other -> no activate
        dev (boolean): True for getting derivative of activation

    Returns:
        if dev:
            A' (numpy array): M x K, gradient of activation
        else:
            A (numpy array): M x K, activate value
    """
    if func == 'sigmoid':
        if dev:
            return dev_sigmoid(Z)
        else:
            return sigmoid(Z)
    elif func == 'tanh':
        if dev:
            return dev_tanh(Z)
        else:
            return tanh(Z)
    elif func == 'relu':
        if dev:
            return dev_relu(Z)
        else:
            return relu(Z)
    else:
        if dev:
            return 1
        else:
            return Z


def output(Z, func='softmax', dev=False):
    """

    Args:
        Z (numpy array): M x K, decision value
        func (str): determine output funciton
                softmax -> softmax
                sigmoid -> sigmoid
                tanh -> tanh
                relu -> relu
                other -> no output
        dev (boolean): True for getting derivative of output

    Returns:
        if dev:
            O' (numpy array): N x K, gradient of output
        else:
            O (numpy array): N x K, output value
    """
    if func == 'softmax':
        if dev:
            return dev_softmax(Z)
        else:
            return softmax(Z)
    elif func == 'sigmoid':
        if dev:
            return dev_sigmoid(Z)
        else:
            return sigmoid(Z)
    elif func == 'tanh':
        if dev:
            return dev_tanh(Z)
        else:
            return tanh(Z)
    elif func == 'relu':
        if dev:
            return dev_relu(Z)
        else:
            return relu(Z)
    else:
        if dev:
            return 1
        else:
            return Z


def add_bias(X):
    return numpy.hstack([X, numpy.ones((len(X), 1))])


def sigmoid(Z):
    """
    This function converts input value to 0 ~ 1.
    Definition: 1 / (1 + exp(-Z))
    Reference: https://en.wikipedia.org/wiki/Sigmoid_function

    Args:
        Z (numpy array): M x K, decision value

    Returns:
        value (numpy array): M x K, sigmoid value
    """
    temp = 1 + numpy.exp(-Z)
    return numpy.divide(1.0, temp)


def dev_sigmoid(Z):
    """
    Derivative of sigmoid function
    Definition: sigmoid(z) * (1- sigmoid(z))
    Reference: https://en.wikipedia.org/wiki/Sigmoid_function

    Args:
        Z (numpy array): M x K, decision value

    Returns:
        result (numpy array): M x K, derivative of sigmoid
    """
    return numpy.multiply(sigmoid(Z), (1 - sigmoid(Z)))


def tanh(Z):
    """
    Hyperbolic tangent function
    Definition: (exp(Z) - exp(-Z))/ (exp(Z) + exp(-Z))
    Reference: https://en.wikipedia.org/wiki/Hyperbolic_function

    Args:
        Z (numpy array): M x K, decision value

    Returns:
        value (numpy array): M x K, tanh value
    """
    temp1 = numpy.exp(Z) - numpy.exp(-Z)
    temp2 = numpy.exp(Z) + numpy.exp(-Z)
    return numpy.divide(temp1, temp2)


def dev_tanh(Z):
    """
    Derivative of tanh function
    Definition: 1 - tanh(Z)^2
    Reference: https://en.wikipedia.org/wiki/Hyperbolic_function

    Args:
        Z (numpy array): M x K, decision value

    Returns:
        result (numpy array): M x K, derivative of tanh
    """
    return 1 - numpy.power(tanh(Z), 2)


def relu(Z):
    """
    Rectified Linear Units is an activation funciton for neural network.
    Definition: max(0, Z)
    Reference: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

    Args:
        Z (numpy array): M x K, decision value

    Returns:
        value (numpy array): M x K, tanh value
    """
    return numpy.maximum(0, Z)


def dev_relu(Z):
    """
    Derivative of relu function
    Definition:
        Z <= 0  -> 0
        Z  > 0  -> 1
    Reference: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

    Args:
        Z (numpy array): M x K, decision value

    Returns:
        result (numpy array): M x K, derivative of tanh
    """
    value = 0.0000001 * numpy.ones(Z.shape)
    value[Z > 0] = 1
    return value


def softmax(Z):
    """
    Definition: exp(Z) / (sum(exp(Z)))
    Reference: https://en.wikipedia.org/wiki/Softmax_function

    Args:
        Z (numpy array): M x K, decision value

    Returns:
        value (numpy array): M x K, softmax value
    """
    temp = numpy.exp(Z) + 0.0000001 * numpy.ones(Z.shape)
    return numpy.divide(temp, numpy.sum(temp))


def dev_softmax(Z):
    """
    Definition: softmax(Z) * (1 - softmax(Z))
    Reference: https://en.wikipedia.org/wiki/Softmax_function

    Args:
        Z (numpy array): M x K, decision value

    Returns:
        result (numpy array): M x K, derivative of softmax
    """
    temp = softmax(Z)
    return numpy.multiply(temp, 1 - temp)
