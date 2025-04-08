import random

from .Filter import Filter


class Transform(Filter):
    def __init__(self, p=None, run_on_test = False, **args):
        super().__init__(run_on_test = run_on_test, **args)
        self._p = p

    def __call__(self, element, test=False):
        if self._p is None or random.random() < self._p:
            return super().__call__(element, test=test)
        return element
