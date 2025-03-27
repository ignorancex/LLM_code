import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/ce")
from executing import execute_quantum_program

class ce_M2Scenario1K2Test(unittest.TestCase):

	def test_ce_M2_733(self):
		input = '01011011101'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M2", 400))

	def test_ce_M2_1442(self):
		input = '10110100010'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M2", 400))

	def test_ce_M2_820(self):
		input = '01100110100'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M2", 400))

	def test_ce_M2_1097(self):
		input = '10001001001'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M2", 400))

	def test_ce_M2_23(self):
		input = '00000010111'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M2", 400))

	def test_ce_M2_1658(self):
		input = '11001111010'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M2", 400))

	def test_ce_M2_813(self):
		input = '01100101101'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M2", 400))

	def test_ce_M2_1990(self):
		input = '11111000110'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M2", 400))

	def test_ce_M2_1003(self):
		input = '01111101011'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M2", 400))

if __name__ == '__main__':
	unittest.main()