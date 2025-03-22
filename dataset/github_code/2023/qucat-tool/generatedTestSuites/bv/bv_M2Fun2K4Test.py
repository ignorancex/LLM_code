import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/bv")
from executing import execute_quantum_program

class bv_M2Scenario2K4Test(unittest.TestCase):

	def test_bv_M2_90(self):
		input = '1011010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_39(self):
		input = '0100111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_105(self):
		input = '1101001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_21(self):
		input = '0010101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_12(self):
		input = '0001100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

	def test_bv_M2_118(self):
		input = '1110110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M2", 100))

if __name__ == '__main__':
	unittest.main()