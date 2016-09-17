#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import unittest
import numpy
from nn.cost import square_loss, dev_square_loss, cross_entropy, dev_cross_entropy


logger = logging.getLogger(__name__)


class SquareLossTest(unittest.TestCase):
    """
    """
    def setUp(self):
        self.O = numpy.array(
                     [[0.1, 1.5, -1.0]]
                 )
        self.Y = numpy.array(
                     [[0, 1.5, -1.5]]
                 )
        self.epsilon = 0.0001

    def test_square_loss(self):
        error = numpy.fabs(square_loss(self.O, self.Y) - 0.13)
        self.assertTrue(error < self.epsilon)

    def test_dev_square_loss(self):
        """
        (T(z+e) - T(z-e)) / (2 * e) ~= T'(z)
        """
        for i in range(self.O.shape[1]):
            e = numpy.zeros(self.O.shape)
            e[0][i] = self.epsilon
            temp1 = square_loss(self.O + e, self.Y)
            temp2 = square_loss(self.O - e, self.Y)
            approximate_derivative = (temp1 - temp2) / self.epsilon / 2.0
            derivative = dev_square_loss(self.O, self.Y)
            error = numpy.fabs(derivative - approximate_derivative)[0][i]
            self.assertTrue((error < self.epsilon).all())
