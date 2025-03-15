import os
import sys
import numpy as np
cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cur_dir)
from unittest import TestCase
from src.sneak.sneak_helper.oracle import Oracle
from src.sneak.sneak_helper.method import Method
from src.sneak.sneak_helper.item import Item
from src.sneak.sneak_helper.tree_node import TreeNode


def test_non_empty_init():
    m = Method(cur_dir + "/src/whun/XOMO/Scrum10k.csv", cur_dir + "/src/whun/XOMO/flight_eval.csv", "")
    o = Oracle(len(m.rank))
    assert len(o.picked) == len(m.rank)


def test_pick():
    for i in range(5):
        m = Method(cur_dir + "/src/whun/XOMO/Scrum10k.csv", cur_dir + "/src/whun/XOMO/flight_eval.csv", "")
        o = Oracle(len(m.rank))
        asked = 0
        first_qidx = set()
        t = TestCase()
        while True:
            _, node = m.find_node()
            q_idx = m.pick_questions(node)
            for q in q_idx:
                first_qidx.add(q)
            asked += 1
            picked = o.pick(q_idx, node)
            t.assertIn(picked, [0, 1])
            m.adjust_weights(node, picked, q_idx)
            m.re_rank()
            solutions = m.check_solution()
            if solutions is not None:
                if solutions == -1:
                    break
                for item in solutions:
                    item.selectedpoints = np.sum(np.multiply(
                        item.item, o.picked)) / np.sum(o.picked) * 100
                break

def test_update_picked_array():
    m = Method(cur_dir + "/src/whun/XOMO/Scrum10k.csv", cur_dir + "/src/whun/XOMO/flight_eval.csv", "")
    o = Oracle(len(m.rank))
    item1 = Item(
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0])
    item2 = Item(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0])
    east_item = [item1, [1, 2], [3, 4]]
    west_item = [item2, [1, 2], [3, 4]]
    node = TreeNode(east_item, west_item, None, None, None)
    q_idx = [1, 2, 3, 4]
    t = TestCase()
    result = o.update_picked_array(1, q_idx, node)
    t.assertIsNotNone(result)