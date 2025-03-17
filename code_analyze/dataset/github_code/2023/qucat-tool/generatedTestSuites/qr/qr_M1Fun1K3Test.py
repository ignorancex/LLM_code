import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/qram")
from executing import execute_quantum_program

class qram_M1Scenario1K3Test(unittest.TestCase):

	def test_qram_M1_395(self):
		input = '110001011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_380(self):
		input = '101111100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_132(self):
		input = '010000100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_87(self):
		input = '001010111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_34(self):
		input = '000100010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 100))

	def test_qram_M1_189(self):
		input = '010111101'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_485(self):
		input = '111100101'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_334(self):
		input = '101001110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_273(self):
		input = '100010001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 100))

	def test_qram_M1_498(self):
		input = '111110010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_216(self):
		input = '011011000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_318(self):
		input = '100111110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_105(self):
		input = '001101001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_255(self):
		input = '011111111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

if __name__ == '__main__':
	unittest.main()