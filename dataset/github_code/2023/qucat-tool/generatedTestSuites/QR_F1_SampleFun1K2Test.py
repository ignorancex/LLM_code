import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_sample/qram")
from executing import execute_quantum_program

class QR_F1_SampleFun1K2Test(unittest.TestCase):

	def test_QR_F1_Sample_422(self):
		input = '110100110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F1_Sample", 200))

	def test_QR_F1_Sample_91(self):
		input = '001011011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F1_Sample", 200))

	def test_QR_F1_Sample_192(self):
		input = '011000000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F1_Sample", 200))

	def test_QR_F1_Sample_381(self):
		input = '101111101'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F1_Sample", 200))

	def test_QR_F1_Sample_40(self):
		input = '000101000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F1_Sample", 200))

	def test_QR_F1_Sample_7(self):
		input = '000000111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F1_Sample", 200))

	def test_QR_F1_Sample_403(self):
		input = '110010011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F1_Sample", 200))

	def test_QR_F1_Sample_216(self):
		input = '011011000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "QR_F1_Sample", 200))

if __name__ == '__main__':
	unittest.main()