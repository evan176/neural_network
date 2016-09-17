#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
from .neuron import add_bias, activation, output
from .cost import loss


def calculate_gradient(neurons_state, cursis):
    """Select loss function

    Args:
        neurons_state (list): state of each layer
        cursis (list): 

    Returns:
        gradients (list):
    """
    gradients = list()
    for i in range(len(cursis)):
        a_ = add_bias(neurons_state[i]['out'])
        w_ = numpy.dot(a_.T, cursis[i])
        gradients.append(w_)
    return gradients


def calculate_cursis(Y, neurons_state, weights, activate_func,
                     output_func, loss_func):
    """

    Args:
        Y (numpy array): actual target value of output
        neurons_state (list): state of each layer
        weights (list): list of each layer's weight
        activate_func (str): activation function
                sigmoid -> sigmoid
                tanh -> tanh
                relu -> relu
                other -> no activate
        output_func (str): output function
                softmax -> softmax,
                sigmoid -> sigmoid,
                tanh -> tanh,
                relu -> relu,
                other -> no output
        loss_func (str): loss function
                square -> square_loss
                cross_entropy -> cross_entropy

    Returns:
        cursis (list):
    """
    cursis = [None] * len(weights)
    cursis[-1] = compute_top_cursi(
        neurons_state[-1]['in'], Y, output_func, loss_func
    )
    for i in range(len(neurons_state) - 2, 0, -1):
        cursi_up = cursis[i]
        w_up = weights[i]
        temp = compute_next_cursi(
            neurons_state[i]['in'], cursi_up, w_up, activate_func
        )
        cursis[i - 1] = temp
    return cursis


def compute_top_cursi(Z, Y, output_func, loss_func):
    """

    Args:
        Z (numpy array): input value of last layer
        Y (numpy array): actual target value of output
        output_func (str): output function
                softmax -> softmax,
                sigmoid -> sigmoid,
                tanh -> tanh,
                relu -> relu,
                other -> no output
        loss_func (str): loss function
                square -> square_loss
                cross_entropy -> cross_entropy

    Returns:
        cursi (numpy array):
    """
    dev_o = output(Z, func=output_func, dev=True)
    O = output(Z, func=output_func)
    dev_l = loss(O, Y, func=loss_func, dev=True)
    return numpy.multiply(dev_l, dev_o)


def compute_next_cursi(Z, cursi_up, w_up, activate_func):
    """

    Args:
        Z (numpy array): input value of given layer
        cursi_up (numpy array): cursi value of upper layer
        w_up (numpy array): weight of upper layer
        activate_func (str): activation function
                sigmoid -> sigmoid
                tanh -> tanh
                relu -> relu
                other -> no activate

    Returns:
        cursi (numpy array):
    """
    dev_a = activation(Z, func=activate_func, dev=True)
    cursi_sum = numpy.dot(cursi_up, w_up.T)[:, :-1]
    return numpy.multiply(cursi_sum, dev_a)
