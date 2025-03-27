import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/bv")
from executing import execute_quantum_program

class bv_M1Scenario1K2Test(unittest.TestCase):

	def test_bv_M1_127(self):
		input = '1111111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_32(self):
		input = '0100000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_21(self):
		input = '0010101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_66(self):
		input = '1000010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_14(self):
		input = '0001110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_88(self):
		input = '1011000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_65(self):
		input = '1000001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

if __name__ == '__main__':
	unittest.main()