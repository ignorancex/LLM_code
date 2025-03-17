import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/iq")
from executing import execute_quantum_program

class iq_M2Scenario1K3Test(unittest.TestCase):

	def test_iq_M2_799(self):
		input = '1100011111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_672(self):
		input = '1010100000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_468(self):
		input = '0111010100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_250(self):
		input = '0011111010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_73(self):
		input = '0001001001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_615(self):
		input = '1001100111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_300(self):
		input = '0100101100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_274(self):
		input = '0100010010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_953(self):
		input = '1110111001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_963(self):
		input = '1111000011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_117(self):
		input = '0001110101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_135(self):
		input = '0010000111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_1006(self):
		input = '1111101110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_540(self):
		input = '1000011100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_819(self):
		input = '1100110011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_463(self):
		input = '0111001111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_832(self):
		input = '1101000000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

	def test_iq_M2_979(self):
		input = '1111010011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M2", 102400))

if __name__ == '__main__':
	unittest.main()