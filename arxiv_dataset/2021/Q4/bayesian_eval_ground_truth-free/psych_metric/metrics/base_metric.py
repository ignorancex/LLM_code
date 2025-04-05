"""
Base Metric class to be inherited by other dataset classes
Overwrite these methods
"""

class BaseMetric(object):

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def calculate(self):
        raise NotImplementedError
