import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/ce")
from executing import execute_quantum_program

class ce_M1Scenario1K2Test(unittest.TestCase):

	def test_ce_M1_1450(self):
		input = '10110101010'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M1", 400))

	def test_ce_M1_629(self):
		input = '01001110101'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M1", 400))

	def test_ce_M1_1871(self):
		input = '11101001111'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M1", 400))

	def test_ce_M1_145(self):
		input = '00010010001'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M1", 400))

	def test_ce_M1_964(self):
		input = '01111000100'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M1", 400))

	def test_ce_M1_122(self):
		input = '00001111010'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M1", 400))

	def test_ce_M1_1308(self):
		input = '10100011100'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M1", 400))

	def test_ce_M1_1699(self):
		input = '11010100011'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M1", 400))

if __name__ == '__main__':
	unittest.main()