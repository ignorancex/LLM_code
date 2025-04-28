#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import sqrt
EPS = 1e-8
LESS_EPS = 1e-8

def distance(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def same(a, b):
    return abs(a - b) < EPS


def same_points(a, b):
    return same(a[0], b[0]) and same(a[1], b[1]) and same(a[2], b[2])


def somewhat_same(a, b):
    return abs(a - b) < LESS_EPS


def somewhat_same_points(a, b):
    return somewhat_same(a[0], b[0]) and somewhat_same(a[1], b[1]) and somewhat_same(a[2], b[2])
