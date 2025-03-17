import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/as")
from executing import execute_quantum_program

class as_M3Scenario1K2Test(unittest.TestCase):

	def test_as_M3_60(self):
		input = '111100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_11(self):
		input = '001011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_5(self):
		input = '000101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_51(self):
		input = '110011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_38(self):
		input = '100110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_24(self):
		input = '011000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

if __name__ == '__main__':
	unittest.main()