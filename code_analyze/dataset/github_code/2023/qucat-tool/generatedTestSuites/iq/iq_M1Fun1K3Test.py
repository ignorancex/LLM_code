import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/iq")
from executing import execute_quantum_program

class iq_M1Scenario1K3Test(unittest.TestCase):

	def test_iq_M1_404(self):
		input = '0110010100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_879(self):
		input = '1101101111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_712(self):
		input = '1011001000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_121(self):
		input = '0001111001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_162(self):
		input = '0010100010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_598(self):
		input = '1001010110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_264(self):
		input = '0100001000'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_787(self):
		input = '1100010011'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_996(self):
		input = '1111100100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_159(self):
		input = '0010011111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_954(self):
		input = '1110111010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_549(self):
		input = '1000100101'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_508(self):
		input = '0111111100'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_370(self):
		input = '0101110010'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_961(self):
		input = '1111000001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_327(self):
		input = '0101000111'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_945(self):
		input = '1110110001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

	def test_iq_M1_222(self):
		input = '0011011110'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

if __name__ == '__main__':
	unittest.main()