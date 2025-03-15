import os
import sys
cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cur_dir)
from unittest import TestCase
from src.sneak.sneak_helper.method import Method
from src.sneak.sneak_helper.oracle import Oracle

class TestMethod(TestCase):
    def test_no_file_error(self):
        self.assertRaises(FileNotFoundError, Method, "abc", "EVAL_FILE")

    def test_method_init(self):
        method = Method(cur_dir+'/test/test_resources/method_bin.csv', cur_dir+'/test/test_resources/method_eval.csv')
        t = TestCase()
        t.assertEqual(len(method.rank), 124)
        t.assertEqual(len(method.weights), 200)
        t.assertEqual(len(method.items), 200)


    def test_find_node_empty(self):
        method = Method(cur_dir+'/test/test_resources/method_bin.csv', cur_dir+'/test/test_resources/method_eval.csv')
        method.tree = None
        t = TestCase()
        path_id, node = method.find_node()
        t.assertEqual(node, None)
        t.assertEqual(path_id, None)

    def test_find_node(self):
        method = Method(cur_dir+'/test/test_resources/method_bin.csv', cur_dir+'/test/test_resources/method_eval.csv')
        t = TestCase()
        path, node = method.find_node()
        t.assertIsNotNone(path)
        t.assertIsNotNone(node)

    def test_pick_questions(self):
        method = Method(cur_dir+'/test/test_resources/method_bin.csv', cur_dir+'/test/test_resources/method_eval.csv')
        t = TestCase()
        _, node = method.find_node()
        questions = method.pick_questions(node)
        t.assertIsNotNone(questions)

    def test_adjust_weights(self):
        method = Method(cur_dir+'/test/test_resources/method_bin.csv', cur_dir+'/test/test_resources/method_eval.csv')
        t = TestCase()
        o = Oracle(len(method.rank))
        _, node = method.find_node()
        q_idx = method.pick_questions(node)
        picked = o.pick(q_idx, node)
        asked = node.asked
        method.adjust_weights(node, picked, q_idx)
        t.assertEqual(asked+1, node.asked)

    def test_get_item(self):
        method = Method(cur_dir+'/test/test_resources/method_bin.csv', cur_dir+'/test/test_resources/method_eval.csv')
        t = TestCase()
        path_id, _ = method.find_node()
        item = method.get_item(path_id)
        t.assertIsNotNone(item)