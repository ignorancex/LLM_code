import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/ce")
from executing import execute_quantum_program

class ce_M1Scenario2K4Test(unittest.TestCase):

	def test_ce_M1_777(self):
		input = '01100001001'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M1", 400))

	def test_ce_M1_1278(self):
		input = '10011111110'
		print(execute_quantum_program([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15, input, "ce_M1", 400))

if __name__ == '__main__':
	unittest.main()