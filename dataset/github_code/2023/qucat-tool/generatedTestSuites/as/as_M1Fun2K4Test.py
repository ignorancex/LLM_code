import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/as")
from executing import execute_quantum_program

class as_M1Scenario2K4Test(unittest.TestCase):

	def test_as_M1_9(self):
		input = '001001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_54(self):
		input = '110110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_28(self):
		input = '011100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_39(self):
		input = '100111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_42(self):
		input = '101010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

if __name__ == '__main__':
	unittest.main()