import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/bv")
from executing import execute_quantum_program

class bv_M1Scenario2K4Test(unittest.TestCase):

	def test_bv_M1_72(self):
		input = '1001000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_53(self):
		input = '0110101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_99(self):
		input = '1100011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

	def test_bv_M1_62(self):
		input = '0111110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M1", 100))

if __name__ == '__main__':
	unittest.main()