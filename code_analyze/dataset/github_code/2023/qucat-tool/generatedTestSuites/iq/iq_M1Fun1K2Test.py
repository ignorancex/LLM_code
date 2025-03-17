import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/iq")
from executing import execute_quantum_program

class iq_M1Scenario1K2Test(unittest.TestCase):

	def test_iq_M1_189(self):
		input = '0010111101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_962(self):
		input = '1111000010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_616(self):
		input = '1001101000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_343(self):
		input = '0101010111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_302(self):
		input = '0100101110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_913(self):
		input = '1110010001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_35(self):
		input = '0000100011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_604(self):
		input = '1001011100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

if __name__ == '__main__':
	unittest.main()