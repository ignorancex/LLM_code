import os
import sys
cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cur_dir)
from unittest import TestCase
from src.sneak.sneak_helper.method import Method
from src.sneak.sneak_helper.search import Search


def test_bfs():
    m = Method(cur_dir + "/src/whun/XOMO/Scrum10k.csv", cur_dir + "/src/whun/XOMO/flight_eval.csv", "")
    t = TestCase()
    path, node = Search.bfs(m.tree, m.cur_best_node)
    t.assertIsNotNone(path)
    t.assertIsNotNone(node)


def test_bfs_final():
    m = Method(cur_dir + "/src/whun/XOMO/Scrum10k.csv", cur_dir + "/src/whun/XOMO/flight_eval.csv", "")
    t = TestCase()
    result = Search.bfs_final(m.tree, m.cur_best_node)
    t.assertIsNone(result)


def test_get_all_items():
    m = Method(cur_dir + "/src/whun/XOMO/Scrum10k.csv", cur_dir + "/src/whun/XOMO/flight_eval.csv", "")
    t = TestCase()
    result = Search.get_all_items(m.tree)
    print(result)
    t.assertIsNotNone(result)

