import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/as")
from executing import execute_quantum_program

class as_M1Scenario1K3Test(unittest.TestCase):

	def test_as_M1_19(self):
		input = '010011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_28(self):
		input = '011100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_48(self):
		input = '110000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_41(self):
		input = '101001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_58(self):
		input = '111010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_14(self):
		input = '001110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_9(self):
		input = '001001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_39(self):
		input = '100111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_63(self):
		input = '111111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_0(self):
		input = '000000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_5(self):
		input = '000101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_53(self):
		input = '110101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_38(self):
		input = '100110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

	def test_as_M1_10(self):
		input = '001010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M1", 400))

if __name__ == '__main__':
	unittest.main()