import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/as")
from executing import execute_quantum_program

class as_M3Scenario1K4Test(unittest.TestCase):

	def test_as_M3_26(self):
		input = '011010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_15(self):
		input = '001111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_34(self):
		input = '100010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_44(self):
		input = '101100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_5(self):
		input = '000101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_9(self):
		input = '001001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_33(self):
		input = '100001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_51(self):
		input = '110011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_39(self):
		input = '100111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_6(self):
		input = '000110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_61(self):
		input = '111101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_42(self):
		input = '101010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_56(self):
		input = '111000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_52(self):
		input = '110100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_62(self):
		input = '111110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_43(self):
		input = '101011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_0(self):
		input = '000000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_12(self):
		input = '001100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_23(self):
		input = '010111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_27(self):
		input = '011011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_17(self):
		input = '010001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_2(self):
		input = '000010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_28(self):
		input = '011100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

	def test_as_M3_18(self):
		input = '010010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 8, input, "as_M3", 400))

if __name__ == '__main__':
	unittest.main()