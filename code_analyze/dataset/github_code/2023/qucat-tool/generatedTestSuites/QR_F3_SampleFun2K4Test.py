import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_sample/qram")
from executing import execute_quantum_program

class QR_F3_SampleFun2K4Test(unittest.TestCase):

	def test_QR_F3_Sample_326(self):
		input = '101000110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F3_Sample", 200))

	def test_QR_F3_Sample_185(self):
		input = '010111001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F3_Sample", 200))

	def test_QR_F3_Sample_467(self):
		input = '111010011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F3_Sample", 200))

	def test_QR_F3_Sample_108(self):
		input = '001101100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F3_Sample", 200))

	def test_QR_F3_Sample_143(self):
		input = '010001111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F3_Sample", 200))

	def test_QR_F3_Sample_272(self):
		input = '100010000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F3_Sample", 100))

	def test_QR_F3_Sample_246(self):
		input = '011110110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F3_Sample", 200))

	def test_QR_F3_Sample_361(self):
		input = '101101001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F3_Sample", 200))

	def test_QR_F3_Sample_348(self):
		input = '101011100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F3_Sample", 200))

	def test_QR_F3_Sample_130(self):
		input = '010000010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F3_Sample", 200))

	def test_QR_F3_Sample_429(self):
		input = '110101101'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F3_Sample", 200))

	def test_QR_F3_Sample_113(self):
		input = '001110001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F3_Sample", 200))

	def test_QR_F3_Sample_31(self):
		input = '000011111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F3_Sample", 200))

	def test_QR_F3_Sample_440(self):
		input = '110111000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F3_Sample", 200))

	def test_QR_F3_Sample_459(self):
		input = '111001011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F3_Sample", 200))

	def test_QR_F3_Sample_354(self):
		input = '101100010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F3_Sample", 200))

	def test_QR_F3_Sample_4(self):
		input = '000000100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F3_Sample", 200))

	def test_QR_F3_Sample_238(self):
		input = '011101110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F3_Sample", 100))

if __name__ == '__main__':
	unittest.main()