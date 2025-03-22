import os
import sys
import pytest
from unittest import TestCase

cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cur_dir)
from src.sneak.sneak_helper.ranker import Ranker
from src.sneak.sneak_helper.method import Method
from src.sneak.sneak_helper.item import Item
from src.sneak.utils.utils import sway


def test_none_root_none_data():
    result = Ranker.level_rank_features(None, None)
    assert result is None


def test_none_root_empty_data():
    result = Ranker.level_rank_features({}, {})
    assert result is None


def test_non_empty_items_rank():
    m = Method(cur_dir + "/src/whun/XOMO/Scrum10k.csv", cur_dir + "/src/whun/XOMO/flight_eval.csv", "")
    rank_result = Ranker.level_rank_features(m.tree, m.weights)
    assert (rank_result == m.rank).all


def test_none_root_node_none_data():
    result = Ranker.rank_nodes(None, None)
    assert result is None


def test_none_root_node_empty_data():
    result = Ranker.rank_nodes({}, {})
    assert result is None


def test_non_empty_root_node():
    m = Method(cur_dir + "/src/whun/XOMO/Scrum10k.csv", cur_dir + "/src/whun/XOMO/flight_eval.csv", "")
    result = Ranker.level_rank_features(m.tree, m.rank)
    assert (result == m.cur_best_node).all


def test_none_pr_level_none_data():
    result = Ranker.pr_level(None)
    assert result is None


def test_none_pr_level_empty_data():
    result = Ranker.pr_level({})
    assert result is None


def test_non_empty_pr_level():
    m = Method(cur_dir + "/src/whun/XOMO/Scrum10k.csv", cur_dir + "/src/whun/XOMO/flight_eval.csv", "")
    result = Ranker.pr_level(m.tree)
    t = TestCase()
    t.assertIsNotNone(result)


def test_none_check_solution_none_data():
    result = Ranker.check_solution(None)
    assert result is None


def test_none_check_solution_empty_data():
    result = Ranker.check_solution({})
    assert result is None


def test_non_empty_check_solution():
    m = Method(cur_dir + "/src/whun/XOMO/Scrum10k.csv", cur_dir + "/src/whun/XOMO/flight_eval.csv", "")
    result = Ranker.check_solution(m.tree)
    t = TestCase()
    t.assertIn(result, [None, -1, 1])


def test_leaf_check_solution():
    item_val = Item(
            [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0,
             0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1,
             1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0,
             1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1], [0, 1, 0, 0, 0, 0])
    tree_node_root = sway([item_val], 100)
    result = Ranker.check_solution(tree_node_root)
    t = TestCase()
    t.assertIn(result, [None, -1, 1])
