#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy


e = 0.00000001


def sgd(g, learning_rate):
    """
    Stochastic gradient descent

    Reference:
        https://en.wikipedia.org/wiki/Stochastic_gradient_descent

    Args:
        g (numpy array): M x K, gradient of weight
        learning_rate (float): learning rate for descent algorithm

    Returns:
        w_update (numpy array): M x K, weight updated amount
    """
    w_update = -learning_rate * g
    return w_update


def momentum(g, v, learning_rate, alpha):
    """
    Stochastic gradient descent with momentum

    Definition: v(k-1) = learning_rate * w_ + alpha * v(k-2)
                w(k) = w(k-1) - v(k-1)
    Reference: 
        http://sebastianruder.com/optimizing-gradient-descent/index.html#adagrad
    Args:
        g (numpy array): M x K, gradient of weight
        v (numpy array): M x K, velocity of previous movement
        learning_rate (float): learning rate for descent algorithm

    Returns:
        w_update (numpy array): M x K, weight updated amount
        v (numpy array): M x K, current velocity
    """
    v = learning_rate * g + alpha * v
    w_update = -v
    return w_update, v


def adagrad(g, g_2, learning_rate):
    """
    Reference: 
        http://sebastianruder.com/optimizing-gradient-descent/index.html#adagrad
    Args:
        g (numpy array): M x K, gradient of weight
        g_2 (numpy array): M x K, gradient square sum
        learning_rate (float): learning rate for descent algorithm

    Returns:
        w_update (numpy array): M x K, weight updated amount
        g_2 (numpy array): M x K, gradient square sum
    """
    g_2 = g_2 + numpy.multiply(g, g)
    temp = numpy.sqrt(g_2 + e)
    w_ = numpy.multiply(numpy.divide(learning_rate, temp), g)
    w_update = -w_
    return w_update, g_2


def rmsprop(g, g_2, learning_rate, alpha):
    """
    Reference: 
        http://sebastianruder.com/optimizing-gradient-descent/index.html#adagrad
    Args:
        g (numpy array): M x K, gradient of weight
        g_2 (numpy array): M x K, gradient square sum
        learning_rate (float): learning rate for descent algorithm
        alpha (float): decaying average argument

    Returns:
        w_update (numpy array): M x K, weight updated amount
        g_2 (numpy array): M x K, gradient square sum
    """
    g_2 = alpha * g_2 + (1 - alpha) * numpy.multiply(g, g)
    temp = numpy.sqrt(g_2 + e)
    w_ = numpy.multiply(numpy.divide(learning_rate, temp), g)
    w_update = -w_
    return w_update, g_2


def adadelta(g, g_2, w_2, learning_rate, alpha):
    """
    Reference: 
        http://sebastianruder.com/optimizing-gradient-descent/index.html#adagrad
    Args:
        g (numpy array): M x K, gradient of weight
        g_2 (numpy array): M x K, gradient square sum
        w_2 (numpy array): M x K, gradient square sum unit
        learning_rate (float): learning rate for descent algorithm
        alpha (float): decaying average argument

    Returns:
        w_update (numpy array): M x K, weight updated amount
        g_2 (numpy array): M x K, gradient square sum
        w_2 (numpy array): M x K, gradient square sum unit
    """
    g_2 = alpha * g_2 + (1 - alpha) * numpy.multiply(g, g)
    temp1 = numpy.sqrt(w_2 + e)
    temp2 = numpy.sqrt(g_2 + e)
    w_ = numpy.multiply(numpy.divide(temp1, temp2), g)
    w_2 = alpha * w_2 + (1 - alpha) * numpy.multiply(w_, w_)
    w_update = -learning_rate * w_
    return w_update, g_2, w_2


def adam(g, m, v, learning_rate, alpha, beta):
    """
    Reference: 
        http://sebastianruder.com/optimizing-gradient-descent/index.html#adagrad
    Args:
        g (numpy array): M x K, gradient of weight
        m (numpy array): M x K, gradient mean
        v (numpy array): M x K, gradient variance
        learning_rate (float): learning rate for descent algorithm
        alpha (float): decaying average argument of mean
        beta (float): decaying average argument of variance

    Returns:
        w_update (numpy array): M x K, weight updated amount
        m (numpy array): M x K, gradient mean
        v (numpy array): M x K, gradient variance
    """
    m = alpha * m + (1 - alpha) * g
    v = beta * v + (1 - beta) * numpy.multiply(g, g)
    m_ = numpy.divide(m, 1 - alpha)
    v_ = numpy.divide(v, 1 - beta)
    w_ = numpy.divide(m_, numpy.sqrt(v_) + e)
    w_update = -learning_rate * w_
    return w_update, m, v
