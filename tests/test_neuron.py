#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import unittest
import numpy
from nn.neuron import decision, sigmoid, dev_sigmoid, tanh, dev_tanh
from nn.neuron import relu, dev_relu, softmax, dev_softmax


logger = logging.getLogger(__name__)


class DecisionTest(unittest.TestCase):
    """
    """
    def setUp(self):
        self.W = numpy.ones((3, 1))
        self.X = numpy.array(
                     [[4.9, 4.1]]
                 )
        self.epsilon = 0.0001

    def test_decision(self):
        error = numpy.fabs(decision(self.X, self.W) - 10)
        self.assertTrue(error < self.epsilon)

    def test_dev_decision(self):
        """
        formula: (J(w+e) - J(w-e)) / (2 * e) ~= J'(w)
        """
        for i in range(self.W.shape[0] - 1):
            e = numpy.zeros(self.W.shape)
            e[i][0] = self.epsilon
            temp1 = decision(self.X, self.W + e)
            temp2 = decision(self.X, self.W - e)
            approximate_derivative = (temp1 - temp2) / self.epsilon / 2.0
            derivative = decision(self.X, self.W, dev=True)[0][i]
            error = numpy.fabs(derivative - approximate_derivative)
            self.assertTrue(error < self.epsilon)


class SigmoidTest(unittest.TestCase):
    """
    """
    def setUp(self):
        self.Z = numpy.array(
                     [[0, 1.5, -1.5]]
                 )
        self.epsilon = 0.0001

    def test_sigmoid(self):
        error = numpy.fabs(sigmoid(self.Z) - [0.5, 0.817574, 0.182425])
        self.assertTrue((error < self.epsilon).all())

    def test_dev_sigmoid(self):
        """
        (S(z+e) - S(z-e)) / (2 * e) ~= S'(z)
        """
        for i in range(self.Z.shape[1]):
            e = numpy.zeros(self.Z.shape)
            e[0][i] = self.epsilon
            temp1 = sigmoid(self.Z + e)
            temp2 = sigmoid(self.Z - e)
            approximate_derivative = (temp1 - temp2) / self.epsilon / 2.0
            derivative = dev_sigmoid(self.Z)
            error = numpy.fabs(derivative - approximate_derivative)[0][i]
            self.assertTrue(error < self.epsilon)


class TanhTest(unittest.TestCase):
    """
    """
    def setUp(self):
        self.Z = numpy.array(
                     [[0, 1.5, -1.5]]
                 )
        self.epsilon = 0.0001

    def test_tanh(self):
        error = numpy.fabs(tanh(self.Z) - [0, 0.905148, -0.905148])
        self.assertTrue((error < self.epsilon).all())

    def test_dev_tanh(self):
        """
        (T(z+e) - T(z-e)) / (2 * e) ~= T'(z)
        """
        for i in range(self.Z.shape[1]):
            e = numpy.zeros(self.Z.shape)
            e[0][i] = self.epsilon
            temp1 = tanh(self.Z + e)
            temp2 = tanh(self.Z - e)
            approximate_derivative = (temp1 - temp2) / self.epsilon / 2.0
            derivative = dev_tanh(self.Z)
            error = numpy.fabs(derivative - approximate_derivative)[0][i]
            self.assertTrue(error < self.epsilon)


class ReluTest(unittest.TestCase):
    """
    """
    def setUp(self):
        self.Z = numpy.array(
                     [[0, 1.5, -1.5]]
                 )
        self.epsilon = 0.0001

    def test_relu(self):
        error = numpy.fabs(relu(self.Z) - [0, 1.5, 0])
        self.assertTrue((error < self.epsilon).all())

    def test_dev_relu(self):
        """
        """
        error = numpy.fabs(dev_relu(self.Z) - [[0, 1, 0]])
        self.assertTrue((error < self.epsilon).all())


class SoftmaxTest(unittest.TestCase):
    """
    """
    def setUp(self):
        self.Z = numpy.array(
                     [[0, 1.5, -1.5]]
                 )
        self.epsilon = 0.0001

    def test_softmax(self):
        error = numpy.fabs(softmax(self.Z) - [0.175290, 0.785597, 0.039112])
        self.assertTrue((error < self.epsilon).all())

    def test_dev_softmax(self):
        """
        (T(z+e) - T(z-e)) / (2 * e) ~= T'(z)
        """
        for i in range(self.Z.shape[1]):
            e = numpy.zeros(self.Z.shape)
            e[0][i] = self.epsilon
            temp1 = softmax(self.Z + e)
            temp2 = softmax(self.Z - e)
            approximate_derivative = (temp1 - temp2) / self.epsilon / 2.0
            derivative = dev_softmax(self.Z)
            error = numpy.fabs(derivative - approximate_derivative)[0][i]
            self.assertTrue(error < self.epsilon)
