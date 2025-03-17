import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/qram")
from executing import execute_quantum_program

class qram_M2Scenario2K4Test(unittest.TestCase):

	def test_qram_M2_273(self):
		input = '100010001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 100))

	def test_qram_M2_174(self):
		input = '010101110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_476(self):
		input = '111011100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 100))

	def test_qram_M2_103(self):
		input = '001100111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 100))

	def test_qram_M2_104(self):
		input = '001101000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

	def test_qram_M2_219(self):
		input = '011011011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M2", 200))

if __name__ == '__main__':
	unittest.main()