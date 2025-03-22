import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/qram")
from executing import execute_quantum_program

class qram_M3Scenario2K4Test(unittest.TestCase):

	def test_qram_M3_389(self):
		input = '110000101'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_123(self):
		input = '001111011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_342(self):
		input = '101010110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_172(self):
		input = '010101100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_16(self):
		input = '000010000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_330(self):
		input = '101001010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_435(self):
		input = '110110011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_472(self):
		input = '111011000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_249(self):
		input = '011111001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_454(self):
		input = '111000110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_44(self):
		input = '000101100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_3(self):
		input = '000000011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_432(self):
		input = '110110000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_285(self):
		input = '100011101'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_114(self):
		input = '001110010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_485(self):
		input = '111100101'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_158(self):
		input = '010011110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_362(self):
		input = '101101010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_95(self):
		input = '001011111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_393(self):
		input = '110001001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_80(self):
		input = '001010000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_311(self):
		input = '100110111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_231(self):
		input = '011100111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_498(self):
		input = '111110010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_400(self):
		input = '110010000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_511(self):
		input = '111111111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 100))

if __name__ == '__main__':
	unittest.main()