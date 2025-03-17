import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/ce")
from executing import execute_quantum_program

class ce_M2Scenario2K4Test(unittest.TestCase):

	def test_ce_M2_402(self):
		input = '00110010010'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M2", 400))

	def test_ce_M2_1901(self):
		input = '11101101101'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M2", 400))

	def test_ce_M2_712(self):
		input = '01011001000'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M2", 400))

	def test_ce_M2_1077(self):
		input = '10000110101'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M2", 400))

	def test_ce_M2_583(self):
		input = '01001000111'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M2", 400))

	def test_ce_M2_510(self):
		input = '00111111110'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M2", 400))

if __name__ == '__main__':
	unittest.main()