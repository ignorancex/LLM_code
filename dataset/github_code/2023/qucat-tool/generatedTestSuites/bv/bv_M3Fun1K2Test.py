import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/bv")
from executing import execute_quantum_program

class bv_M3Scenario1K2Test(unittest.TestCase):

	def test_bv_M3_58(self):
		input = '0111010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_85(self):
		input = '1010101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_9(self):
		input = '0001001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_103(self):
		input = '1100111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_96(self):
		input = '1100000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_22(self):
		input = '0010110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_95(self):
		input = '1011111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

if __name__ == '__main__':
	unittest.main()