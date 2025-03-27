import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/bv")
from executing import execute_quantum_program

class bv_M2Scenario1K3Test(unittest.TestCase):

	def test_bv_M2_90(self):
		input = '1011010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_101(self):
		input = '1100101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_43(self):
		input = '0101011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_54(self):
		input = '0110110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_31(self):
		input = '0011111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_17(self):
		input = '0010001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_77(self):
		input = '1001101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_67(self):
		input = '1000011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_96(self):
		input = '1100000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_110(self):
		input = '1101110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_4(self):
		input = '0000100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_121(self):
		input = '1111001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_56(self):
		input = '0111000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_116(self):
		input = '1110100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

if __name__ == '__main__':
	unittest.main()