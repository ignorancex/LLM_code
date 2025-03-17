import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/as")
from executing import execute_quantum_program

class as_M3Scenario1K3Test(unittest.TestCase):

	def test_as_M3_47(self):
		input = '101111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_28(self):
		input = '011100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_2(self):
		input = '000010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_49(self):
		input = '110001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_27(self):
		input = '011011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_36(self):
		input = '100100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_54(self):
		input = '110110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_9(self):
		input = '001001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_5(self):
		input = '000101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_18(self):
		input = '010010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_42(self):
		input = '101010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_23(self):
		input = '010111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_56(self):
		input = '111000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

if __name__ == '__main__':
	unittest.main()