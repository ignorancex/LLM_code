# _*_ coding: utf-8 _*_
# This file is created by C. Zhang for personal use.
# @Time         : 25/03/2022 19:56
# @Author       : tl22089
# @File         : __init__.py.py
# @Affiliation  : University of Bristol
from abc import ABC, abstractmethod


class Compressor(ABC):
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self, average=True, tensors_size_are_same=True):
        self.average = average
        self.tensors_size_are_same = tensors_size_are_same

    @abstractmethod
    def compress(self, tensor, name):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        raise NotImplemented("compress was not implemented.")

    @abstractmethod
    def decompress(self, tensors, ctx):
        """Decompress the tensor with the given context."""
        raise NotImplemented("decompress was not implemented.")

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        return sum(tensors)