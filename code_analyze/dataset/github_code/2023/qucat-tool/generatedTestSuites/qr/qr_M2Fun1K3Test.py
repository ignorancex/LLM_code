import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/qram")
from executing import execute_quantum_program

class qram_M2Scenario1K3Test(unittest.TestCase):

	def test_qram_M2_509(self):
		input = '111111101'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_130(self):
		input = '010000010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_122(self):
		input = '001111010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_321(self):
		input = '101000001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_279(self):
		input = '100010111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_169(self):
		input = '010101001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_280(self):
		input = '100011000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_228(self):
		input = '011100100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_303(self):
		input = '100101111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_462(self):
		input = '111001110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_116(self):
		input = '001110100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_435(self):
		input = '110110011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_12(self):
		input = '000001100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_211(self):
		input = '011010011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_404(self):
		input = '110010100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_31(self):
		input = '000011111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_294(self):
		input = '100100110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

if __name__ == '__main__':
	unittest.main()