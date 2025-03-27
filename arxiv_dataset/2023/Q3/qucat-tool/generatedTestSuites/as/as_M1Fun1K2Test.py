import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/as")
from executing import execute_quantum_program

class as_M1Scenario1K2Test(unittest.TestCase):

	def test_as_M1_54(self):
		input = '110110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_25(self):
		input = '011001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_10(self):
		input = '001010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_45(self):
		input = '101101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_35(self):
		input = '100011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_20(self):
		input = '010100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

if __name__ == '__main__':
	unittest.main()