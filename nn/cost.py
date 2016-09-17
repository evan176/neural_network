#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy


def loss(O, Y, func='square', dev=False):
    """Select loss function

    Args:
        O (numpy array): M x K, output value
        Y (numpy array): M x K
        func (str): determine loss funciton
                square -> square_loss
                cross_entropy -> cross_entropy
        dev (boolean): True for getting derivative of loss

    Returns:
        if dev:
            L' (numpy array): M x K, gradient of loss
        else:
            L (numpy array): M x 1, loss value
    """
    if func == 'square':
        if dev:
            return dev_square_loss(O, Y)
        else:
            return square_loss(O, Y)
    elif func == 'cross_entropy':
        if dev:
            return dev_cross_entropy(O, Y)
        else:
            return cross_entropy(O, Y)


def square_loss(O, Y):
    """
    Sum of square error
    Definition: (O - Y)**2 / 2.0

    Args:
        O (numpy array): N x K, output value
        Y (numpy array): N x K

    Returns:
        result (float): error value
    """
    temp = O - Y
    temp = numpy.dot(temp, temp.T)
    return numpy.divide(temp, 2.0)


def dev_square_loss(O, Y):
    """
    Derivative of square loss
    Definition: O - Y

    Args:
        O (numpy array): N x K, output value
        Y (numpy array): N x K

    Returns:
        result (float): error value
    """
    return O - Y


def cross_entropy(O, Y):
    """
    Sum of square error
    Definition: (O - Y)**2 / 2.0

    Args:
        O (numpy array): N x K, output value
        Y (numpy array): N x K

    Returns:
        result (float): error value
    """
    return -numpy.dot(numpy.log(O), Y.T)


def dev_cross_entropy(O, Y):
    """
    Derivative of square loss
    Definition: O - Y

    Args:
        O (numpy array): N x K, output value
        Y (numpy array): N x K

    Returns:
        result (float): error value
    """
    #return numpy.multiply(O - Y, Y)
    return O - Y
