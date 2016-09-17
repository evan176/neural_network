#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy


e = 0.00000001


def sgd(w, g, learning_rate):
    """
    Stochastic gradient descent

    Reference:
        https://en.wikipedia.org/wiki/Stochastic_gradient_descent

    Args:
        w (numpy array): M x K, weight
        g (numpy array): M x K, gradient of weight
        learning_rate (float): learning rate for descent algorithm

    Returns:
        w (numpy array): M x K, updated weights
    """
    return w - learning_rate * g


def momentum(w, g, v, learning_rate, alpha):
    """
    Stochastic gradient descent with momentum

    Definition: v(k-1) = learning_rate * w_ + alpha * v(k-2)
                w(k) = w(k-1) - v(k-1)
    Reference: 
        http://sebastianruder.com/optimizing-gradient-descent/index.html#adagrad
    Args:
        w (numpy array): M x K, weight
        g (numpy array): M x K, gradient of weight
        v (numpy array): M x K, velocity of previous movement
        learning_rate (float): learning rate for descent algorithm

    Returns:
        w (numpy array): M x K, updated weights
        v (numpy array): M x K, current velocity
    """
    v = learning_rate * g + alpha * v
    return w - v, v


def adagrad(w, g, g_2, learning_rate):
    """
    Reference: 
        http://sebastianruder.com/optimizing-gradient-descent/index.html#adagrad
    Args:
        w (numpy array): M x K, weight
        g (numpy array): M x K, gradient of weight
        g_2 (numpy array): M x K, gradient square sum
        learning_rate (float): learning rate for descent algorithm

    Returns:
        w (numpy array): M x K, updated weights
        g_2 (numpy array): M x K, gradient square sum
    """
    g_2 = g_2 + numpy.multiply(g, g)
    temp = numpy.sqrt(g_2 + e)
    w_ = numpy.multiply(numpy.divide(learning_rate, temp), g)
    return w - w_, g_2


def rmsprop(w, g, g_2, learning_rate, alpha):
    """
    Reference: 
        http://sebastianruder.com/optimizing-gradient-descent/index.html#adagrad
    Args:
        w (numpy array): M x K, weight
        g (numpy array): M x K, gradient of weight
        g_2 (numpy array): M x K, gradient square sum
        learning_rate (float): learning rate for descent algorithm
        alpha (float): decaying average argument

    Returns:
        w (numpy array): M x K, updated weights
        g_2 (numpy array): M x K, gradient square sum
    """
    g_2 = alpha * g_2 + (1 - alpha) * numpy.multiply(g, g)
    temp = numpy.sqrt(g_2 + e)
    w_ = numpy.multiply(numpy.divide(learning_rate, temp), g)
    return w - w_, g_2


def adadelta(w, g, g_2, w_2, learning_rate, alpha):
    """
    Reference: 
        http://sebastianruder.com/optimizing-gradient-descent/index.html#adagrad
    Args:
        w (numpy array): M x K, weight
        g (numpy array): M x K, gradient of weight
        g_2 (numpy array): M x K, gradient square sum
        w_2 (numpy array): M x K, gradient square sum unit
        learning_rate (float): learning rate for descent algorithm
        alpha (float): decaying average argument

    Returns:
        w (numpy array): M x K, updated weights
        g_2 (numpy array): M x K, gradient square sum
        w_2 (numpy array): M x K, gradient square sum unit
    """
    g_2 = alpha * g_2 + (1 - alpha) * numpy.multiply(g, g)
    temp1 = numpy.sqrt(w_2 + e)
    temp2 = numpy.sqrt(g_2 + e)
    w_ = numpy.multiply(numpy.divide(temp1, temp2), g)
    w_2 = alpha * w_2 + (1 - alpha) * numpy.multiply(w_, w_)
    return w - learning_rate * w_, g_2, w_2


def adam(w, g, m, v, learning_rate, alpha, beta):
    """
    Reference: 
        http://sebastianruder.com/optimizing-gradient-descent/index.html#adagrad
    Args:
        w (numpy array): M x K, weight
        g (numpy array): M x K, gradient of weight
        m (numpy array): M x K, gradient mean
        v (numpy array): M x K, gradient variance
        learning_rate (float): learning rate for descent algorithm
        alpha (float): decaying average argument of mean
        beta (float): decaying average argument of variance

    Returns:
        w (numpy array): M x K, updated weights
        m (numpy array): M x K, gradient mean
        v (numpy array): M x K, gradient variance
    """
    m = alpha * m + (1 - alpha) * g
    v = beta * v + (1 - beta) * numpy.multiply(g, g)
    m_ = numpy.divide(m, 1 - alpha)
    v_ = numpy.divide(v, 1 - beta)
    w_ = numpy.divide(m_, numpy.sqrt(v_) + e)
    return w - learning_rate * w_, m, v
