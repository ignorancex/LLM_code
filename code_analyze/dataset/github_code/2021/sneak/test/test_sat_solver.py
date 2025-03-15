import os
import sys
cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cur_dir)
from unittest import TestCase
from src.sneak.sneak_helper.sat_solver import SatSolver
from src.sneak.sneak_helper.oracle import Oracle

class TestRanker(TestCase):
    def test_no_file_error(self):
        self.assertRaises(FileNotFoundError, SatSolver.get_solutions, "Unknown_file", "EVAL_FILE")

    def test_getsolution(self):
        self.assertRaises(FileNotFoundError, SatSolver.get_solutions, "Unknown_file", "EVAL_FILE")
        Items = SatSolver.get_solutions(cur_dir + '/test/test_resources/method_bin.csv',
                        cur_dir + '/test/test_resources/method_eval.csv')
        t = TestCase()
        t.assertEqual(len(Items), 200)