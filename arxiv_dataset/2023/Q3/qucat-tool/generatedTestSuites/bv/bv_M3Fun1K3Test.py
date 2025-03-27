import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/bv")
from executing import execute_quantum_program

class bv_M3Scenario1K3Test(unittest.TestCase):

	def test_bv_M3_7(self):
		input = '0000111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_122(self):
		input = '1111010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_44(self):
		input = '0101100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_16(self):
		input = '0010000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_118(self):
		input = '1110110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_72(self):
		input = '1001000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_30(self):
		input = '0011110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_97(self):
		input = '1100001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_43(self):
		input = '0101011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_95(self):
		input = '1011111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_53(self):
		input = '0110101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_83(self):
		input = '1010011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_70(self):
		input = '1000110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

	def test_bv_M3_93(self):
		input = '1011101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], 14, input, "bv_M3", 100))

if __name__ == '__main__':
	unittest.main()