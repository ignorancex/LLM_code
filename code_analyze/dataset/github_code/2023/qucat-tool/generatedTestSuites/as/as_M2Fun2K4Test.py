import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/as")
from executing import execute_quantum_program

class as_M2Scenario2K4Test(unittest.TestCase):

	def test_as_M2_38(self):
		input = '100110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_9(self):
		input = '001001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_51(self):
		input = '110011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_56(self):
		input = '111000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_21(self):
		input = '010101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_14(self):
		input = '001110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_49(self):
		input = '110001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_13(self):
		input = '001101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

if __name__ == '__main__':
	unittest.main()