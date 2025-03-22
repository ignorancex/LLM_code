import os
import sys

cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cur_dir)
from src.sneak.sneak_helper.input_output import InputOutput

def test_get_question_text():
    list_expected = ['a', 'b', 'c', 'd']
    with open(cur_dir+'/test/test_resources/question.csv', 'r') as file:
        list_actual = InputOutput.get_question_text(file, 'question')
        assert list_actual==list_expected

def test_read_dimacs():
    a_exp = ['sampl', '', '0', '-']
    a,b = InputOutput.read_dimacs(cur_dir+'/test/test_resources/dimacs_file.txt')
    assert a==a_exp
    assert b==[]
