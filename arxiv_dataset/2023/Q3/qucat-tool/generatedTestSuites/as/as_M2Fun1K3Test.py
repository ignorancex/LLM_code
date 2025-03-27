import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/as")
from executing import execute_quantum_program

class as_M2Scenario1K3Test(unittest.TestCase):

	def test_as_M2_63(self):
		input = '111111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_6(self):
		input = '000110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_17(self):
		input = '010001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_40(self):
		input = '101000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_11(self):
		input = '001011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_13(self):
		input = '001101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_20(self):
		input = '010100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_50(self):
		input = '110010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_26(self):
		input = '011010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_53(self):
		input = '110101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_35(self):
		input = '100011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_60(self):
		input = '111100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

	def test_as_M2_37(self):
		input = '100101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M2", 400))

if __name__ == '__main__':
	unittest.main()