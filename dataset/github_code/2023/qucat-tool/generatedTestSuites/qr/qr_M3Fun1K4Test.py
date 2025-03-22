import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/qram")
from executing import execute_quantum_program

class qram_M3Scenario1K4Test(unittest.TestCase):

	def test_qram_M3_44(self):
		input = '000101100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_147(self):
		input = '010010011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_73(self):
		input = '001001001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_224(self):
		input = '011100000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_421(self):
		input = '110100101'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_414(self):
		input = '110011110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_326(self):
		input = '101000110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_207(self):
		input = '011001111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_84(self):
		input = '001010100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_266(self):
		input = '100001010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_370(self):
		input = '101110010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_383(self):
		input = '101111111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_313(self):
		input = '100111001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_451(self):
		input = '111000011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_400(self):
		input = '110010000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_469(self):
		input = '111010101'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_182(self):
		input = '010110110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_460(self):
		input = '111001100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_39(self):
		input = '000100111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_29(self):
		input = '000011101'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_245(self):
		input = '011110101'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_90(self):
		input = '001011010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_99(self):
		input = '001100011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_494(self):
		input = '111101110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_277(self):
		input = '100010101'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_249(self):
		input = '011111001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_171(self):
		input = '010101011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 100))

	def test_qram_M3_506(self):
		input = '111111010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_48(self):
		input = '000110000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_136(self):
		input = '010001000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 100))

	def test_qram_M3_352(self):
		input = '101100000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_275(self):
		input = '100010011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_418(self):
		input = '110100010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_1(self):
		input = '000000001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 100))

	def test_qram_M3_271(self):
		input = '100001111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_70(self):
		input = '001000110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_457(self):
		input = '111001001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_284(self):
		input = '100011100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

if __name__ == '__main__':
	unittest.main()