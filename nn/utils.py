#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy


eps = 1e-8


def safe_divide(A, B):
    """
    with numpy.errstate(divide='ignore'):
        C = numpy.divide(A, B)
        C[~ numpy.isfinite(C)] = 0.0
    return C

    Args:
        A (numpy array): dividend
        B (numpy array): divisor

    Returns:
        result (numpy array): divide result
    """
    return numpy.divide(A, B + eps)
