import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/iq")
from executing import execute_quantum_program

class iq_M3Scenario1K2Test(unittest.TestCase):

	def test_iq_M3_918(self):
		input = '1110010110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M3", 102400))

	def test_iq_M3_104(self):
		input = '0001101000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M3", 102400))

	def test_iq_M3_507(self):
		input = '0111111011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M3", 102400))

	def test_iq_M3_773(self):
		input = '1100000101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M3", 102400))

	def test_iq_M3_719(self):
		input = '1011001111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M3", 102400))

	def test_iq_M3_563(self):
		input = '1000110011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M3", 102400))

	def test_iq_M3_188(self):
		input = '0010111100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M3", 102400))

	def test_iq_M3_194(self):
		input = '0011000010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M3", 102400))

if __name__ == '__main__':
	unittest.main()