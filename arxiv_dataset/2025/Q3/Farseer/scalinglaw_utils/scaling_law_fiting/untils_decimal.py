
import bisect
import unittest
from abc import ABC, abstractmethod
import numpy as np
from decimal import InvalidOperation

class MathUtil(ABC):

    @staticmethod
    @abstractmethod
    def exp(x):
        pass

    @staticmethod
    @abstractmethod
    def log10(x):
        pass

    @staticmethod
    @abstractmethod
    def log(x):
        pass

    @staticmethod
    def power(base, exp):
        return (base ** exp)

class NumpyMathUtil(MathUtil):

    @staticmethod
    def exp(x):
        return np.exp(x)

    @staticmethod
    def log10(x):
        return np.log10(x)

    @staticmethod
    def log(x):
        return np.log(x)

class DecimalMathUtil(MathUtil):

    @staticmethod
    def exp(x):
        return x.exp()

    @staticmethod
    def log10(x):
        if (x <= 0):
            raise ValueError(f'log10参数必须为正数，当前输入: {x}')
        try:
            return x.log10()
        except InvalidOperation as e:
            raise InvalidOperation(f'log10计算错误: {e} (x={x})')

    @staticmethod
    def log(x):
        return x.ln()

def find_closest_indices(arr1, arr2):
    reversed_arr2 = [(- x) for x in arr2]
    result = []
    for x in arr1:
        pos = bisect.bisect_right(reversed_arr2, (- x))
        if (pos == 0):
            candidate = 0
        elif (pos == len(reversed_arr2)):
            candidate = (len(arr2) - 1)
        else:
            left_val = arr2[(pos - 1)]
            right_val = arr2[pos]
            left_diff = abs((left_val - x))
            right_diff = abs((right_val - x))
            if (left_diff < right_diff):
                candidate = (pos - 1)
            elif (right_diff < left_diff):
                candidate = pos
            else:
                candidate = pos
        result.append(candidate)
    return result

class TestClosestIndices(unittest.TestCase):

    def test_exact_match(self):
        arr1 = [3]
        arr2 = [5, 4, 3, 2, 1]
        self.assertEqual(find_closest_indices(arr1, arr2), [2])

    def test_insert_at_beginning(self):
        arr1 = [6]
        arr2 = [5, 4, 3, 2, 1]
        self.assertEqual(find_closest_indices(arr1, arr2), [0])

    def test_insert_at_end(self):
        arr1 = [0]
        arr2 = [5, 4, 3, 2, 1]
        self.assertEqual(find_closest_indices(arr1, arr2), [4])

    def test_equal_distance(self):
        arr1 = [3]
        arr2 = [5, 4, 2, 1]
        self.assertEqual(find_closest_indices(arr1, arr2), [2])

    def test_multiple_elements(self):
        arr1 = [5, 2, 1]
        arr2 = [6, 4, 3, 2, 0]
        self.assertEqual(find_closest_indices(arr1, arr2), [1, 3, 4])
