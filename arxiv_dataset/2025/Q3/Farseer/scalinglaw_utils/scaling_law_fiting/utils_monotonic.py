
from decimal import Decimal
import numpy as np
import unittest

def is_monotonic(arr, order='ascending', strict=False):
    "\n    检查数组是否为单调递增或递减。\n\n    参数:\n        arr (list[Decimal] 或 np.ndarray): 输入数组，一维。\n        order (str): 可选值为 'ascending'（升序）或 'descending'（降序），默认为 'ascending'。\n        strict (bool): 是否严格单调，默认为False。\n\n    返回:\n        bool: 数组是否符合指定的单调性条件。\n\n    异常:\n        ValueError: 如果order不是'ascending'或'descending'，或数组不是一维。\n    "
    arr = np.asarray(arr)
    if (arr.ndim != 1):
        raise ValueError('输入数组必须是一维的。')
    if (len(arr) <= 1):
        return True
    diffs = np.diff(arr)
    if (order == 'ascending'):
        if strict:
            return bool(np.all((diffs > 0)))
        else:
            return bool(np.all((diffs >= 0)))
    elif (order == 'descending'):
        if strict:
            return bool(np.all((diffs < 0)))
        else:
            return bool(np.all((diffs <= 0)))
    else:
        raise ValueError("参数order必须是'ascending'或'descending'。")

def check_monotonic_indices(arr, order='ascending', strict=False):
    "\n    检查数组的单调性并返回破坏条件的元素索引\n\n    Args:\n        arr: list[Decimal] 或 np.ndarray\n        order: 排序方向 'ascending' 或 'descending'\n        strict: 是否严格单调\n\n    Returns:\n        list: 破坏单调性的元素索引列表\n        (空列表表示符合条件，单元素数组也返回空列表)\n\n    Raises:\n        ValueError: 无效参数或多维数组\n    "
    arr = np.asarray(arr)
    if (arr.ndim != 1):
        raise ValueError('输入必须是一维数组')
    if (len(arr) <= 1):
        return []
    diffs = np.diff(arr)
    if (order == 'ascending'):
        if strict:
            invalid_pos = np.where((diffs <= 0))[0]
        else:
            invalid_pos = np.where((diffs < 0))[0]
    elif (order == 'descending'):
        if strict:
            invalid_pos = np.where((diffs >= 0))[0]
        else:
            invalid_pos = np.where((diffs > 0))[0]
    else:
        raise ValueError("排序方向必须是 'ascending' 或 'descending'")
    return [int((pos + 1)) for pos in invalid_pos]

class TestMonotonicCheck(unittest.TestCase):

    def test_basic_cases(self):
        self.assertEqual(check_monotonic_indices([1, 2, 3, 4]), [])
        self.assertEqual(check_monotonic_indices([1, 3, 2], 'ascending'), [2])
        self.assertEqual(check_monotonic_indices([1, 1, 2], 'ascending', True), [1])
        self.assertEqual(check_monotonic_indices([4, 3, 2], 'descending'), [])
        self.assertEqual(check_monotonic_indices([5, 3, 4], 'descending'), [2])
        dec_arr = [Decimal('1'), Decimal('2'), Decimal('1.5')]
        self.assertEqual(check_monotonic_indices(dec_arr), [2])

    def test_edge_cases(self):
        self.assertEqual(check_monotonic_indices([]), [])
        self.assertEqual(check_monotonic_indices([5]), [])
        self.assertEqual(check_monotonic_indices([2, 2, 2], 'ascending'), [])
        self.assertEqual(check_monotonic_indices([2, 2, 2], 'ascending', True), [1, 2])
        inf_arr = [float('-inf'), 0, float('inf')]
        self.assertEqual(check_monotonic_indices(inf_arr, strict=True), [])

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            check_monotonic_indices([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            check_monotonic_indices([1, 2, 3], 'invalid')

class TestIsMonotonic(unittest.TestCase):

    def test_empty_array(self):
        self.assertTrue(is_monotonic([]))

    def test_single_element(self):
        self.assertTrue(is_monotonic([5]))
        self.assertTrue(is_monotonic(np.array([3.14])))

    def test_strict_ascending(self):
        self.assertTrue(is_monotonic([1, 2, 3], 'ascending', True))
        self.assertTrue(is_monotonic(np.array([0.1, 0.2, 0.3]), 'ascending', True))

    def test_non_strict_ascending(self):
        self.assertTrue(is_monotonic([1, 2, 2, 3], 'ascending'))
        self.assertTrue(is_monotonic([Decimal('1'), Decimal('2'), Decimal('3')], 'ascending'))

    def test_strict_descending(self):
        self.assertTrue(is_monotonic([3, 2, 1], 'descending', True))
        self.assertTrue(is_monotonic(np.array([5.0, 4.0, 3.0]), 'descending', True))

    def test_non_strict_descending(self):
        self.assertTrue(is_monotonic([3, 3, 2, 1], 'descending'))
        self.assertTrue(is_monotonic([Decimal('3'), Decimal('3'), Decimal('2')], 'descending'))

    def test_ascending_failure(self):
        self.assertFalse(is_monotonic([1, 3, 2], 'ascending'))
        self.assertFalse(is_monotonic(np.array([1, 3, 2]), 'ascending', True))

    def test_descending_failure(self):
        self.assertFalse(is_monotonic([3, 1, 2], 'descending'))
        self.assertFalse(is_monotonic([Decimal('3'), Decimal('1'), Decimal('2')], 'descending'))

    def test_strict_vs_non_strict(self):
        self.assertFalse(is_monotonic([1, 21, 21], strict=True))
        self.assertTrue(is_monotonic([1, 5.1, 5.1], strict=False))

    def test_all_equal_elements(self):
        self.assertTrue(is_monotonic([5, 5, 5], 'ascending'))
        self.assertTrue(is_monotonic([10, 10, 10], 'descending'))

    def test_infinity_values(self):
        arr = [float('-inf'), (- 1), 0, float('inf')]
        self.assertTrue(is_monotonic(arr, 'ascending', True))

    def test_multi_dimension_array(self):
        with self.assertRaises(ValueError):
            is_monotonic([[1, 2], [3, 4]])

    def test_invalid_order(self):
        with self.assertRaises(ValueError):
            is_monotonic([1, 2, 3], 'invalid')
if (__name__ == '__main__'):
    unittest.main()
