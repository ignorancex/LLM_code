import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/as")
from executing import execute_quantum_program

class as_M2Scenario1K2Test(unittest.TestCase):

	def test_as_M2_53(self):
		input = '110101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_11(self):
		input = '001011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_34(self):
		input = '100010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_12(self):
		input = '001100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_62(self):
		input = '111110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_17(self):
		input = '010001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

if __name__ == '__main__':
	unittest.main()