import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/ce")
from executing import execute_quantum_program

class ce_M3Scenario1K2Test(unittest.TestCase):

	def test_ce_M3_703(self):
		input = '01010111111'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M3", 400))

	def test_ce_M3_320(self):
		input = '00101000000'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M3", 400))

	def test_ce_M3_2010(self):
		input = '11111011010'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M3", 400))

	def test_ce_M3_1317(self):
		input = '10100100101'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M3", 400))

	def test_ce_M3_1172(self):
		input = '10010010100'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M3", 400))

	def test_ce_M3_107(self):
		input = '00001101011'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M3", 400))

	def test_ce_M3_1741(self):
		input = '11011001101'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M3", 400))

	def test_ce_M3_562(self):
		input = '01000110010'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M3", 400))

if __name__ == '__main__':
	unittest.main()