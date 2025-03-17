import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/qram")
from executing import execute_quantum_program

class qram_M3Scenario1K2Test(unittest.TestCase):

	def test_qram_M3_474(self):
		input = '111011010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_293(self):
		input = '100100101'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_179(self):
		input = '010110011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_28(self):
		input = '000011100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_64(self):
		input = '001000000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_463(self):
		input = '111001111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_232(self):
		input = '011101000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_67(self):
		input = '001000011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

if __name__ == '__main__':
	unittest.main()