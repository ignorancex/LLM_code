import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/iq")
from executing import execute_quantum_program

class iq_M2Scenario1K2Test(unittest.TestCase):

	def test_iq_M2_766(self):
		input = '1011111110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_321(self):
		input = '0101000001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_532(self):
		input = '1000010100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_299(self):
		input = '0100101011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_408(self):
		input = '0110011000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_677(self):
		input = '1010100101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_66(self):
		input = '0001000010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_827(self):
		input = '1100111011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_308(self):
		input = '0100110100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

if __name__ == '__main__':
	unittest.main()