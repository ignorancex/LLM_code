import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/qram")
from executing import execute_quantum_program

class qram_M1Scenario2K4Test(unittest.TestCase):

	def test_qram_M1_481(self):
		input = '111100001'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

	def test_qram_M1_62(self):
		input = '000111110'
		print(execute_quantum_program([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3], 13, input, "qram_M1", 200))

if __name__ == '__main__':
	unittest.main()