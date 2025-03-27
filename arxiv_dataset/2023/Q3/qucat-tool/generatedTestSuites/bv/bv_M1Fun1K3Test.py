import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/bv")
from executing import execute_quantum_program

class bv_M1Scenario1K3Test(unittest.TestCase):

	def test_bv_M1_25(self):
		input = '0011001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_103(self):
		input = '1100111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_104(self):
		input = '1101000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_94(self):
		input = '1011110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_34(self):
		input = '0100010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_123(self):
		input = '1111011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_52(self):
		input = '0110100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_80(self):
		input = '1010000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_45(self):
		input = '0101101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_1(self):
		input = '0000001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_31(self):
		input = '0011111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_2(self):
		input = '0000010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_87(self):
		input = '1010111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_14(self):
		input = '0001110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_69(self):
		input = '1000101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

if __name__ == '__main__':
	unittest.main()