import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/qram")
from executing import execute_quantum_program

class qram_M2Scenario1K2Test(unittest.TestCase):

	def test_qram_M2_250(self):
		input = '011111010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_263(self):
		input = '100000111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_433(self):
		input = '110110001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_68(self):
		input = '001000100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 100))

	def test_qram_M2_361(self):
		input = '101101001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_414(self):
		input = '110011110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_55(self):
		input = '000110111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_457(self):
		input = '111001001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

if __name__ == '__main__':
	unittest.main()