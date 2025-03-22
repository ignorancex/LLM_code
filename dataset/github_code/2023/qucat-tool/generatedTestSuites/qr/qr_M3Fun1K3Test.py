import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/qram")
from executing import execute_quantum_program

class qram_M3Scenario1K3Test(unittest.TestCase):

	def test_qram_M3_410(self):
		input = '110011010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_108(self):
		input = '001101100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_321(self):
		input = '101000001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_51(self):
		input = '000110011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_207(self):
		input = '011001111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_416(self):
		input = '110100000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_445(self):
		input = '110111101'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_6(self):
		input = '000000110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_351(self):
		input = '101011111'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_212(self):
		input = '011010100'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_137(self):
		input = '010001001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 100))

	def test_qram_M3_502(self):
		input = '111110110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_120(self):
		input = '001111000'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_298(self):
		input = '100101010'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_227(self):
		input = '011100011'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

	def test_qram_M3_261(self):
		input = '100000101'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M3", 200))

if __name__ == '__main__':
	unittest.main()