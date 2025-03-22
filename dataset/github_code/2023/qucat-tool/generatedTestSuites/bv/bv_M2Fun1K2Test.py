import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/bv")
from executing import execute_quantum_program

class bv_M2Scenario1K2Test(unittest.TestCase):

	def test_bv_M2_122(self):
		input = '1111010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_69(self):
		input = '1000101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_11(self):
		input = '0001011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_61(self):
		input = '0111101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_16(self):
		input = '0010000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_102(self):
		input = '1100110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

if __name__ == '__main__':
	unittest.main()