import unittest
import sys
sys.path.append("/Users/xinyi/Documents/qucat_experiment/iq")
from executing import execute_quantum_program

class iq_M1Scenario2K4Test(unittest.TestCase):

	def test_iq_M1_497(self):
		input = '0111110001'
		print(execute_quantum_program([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, input, "iq_M1", 102400))

if __name__ == '__main__':
	unittest.main()