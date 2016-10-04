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
    """
    return numpy.divide(A, B + eps)
