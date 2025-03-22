import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/qram")
from executing import execute_quantum_program

class qram_M1Scenario1K2Test(unittest.TestCase):

	def test_qram_M1_253(self):
		input = '011111101'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_386(self):
		input = '110000010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_371(self):
		input = '101110011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_4(self):
		input = '000000100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_286(self):
		input = '100011110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_104(self):
		input = '001101000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_457(self):
		input = '111001001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_55(self):
		input = '000110111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

if __name__ == '__main__':
	unittest.main()