#coding=utf8
import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import tempfile
import subprocess
import re

from .domain_base import Domain


def set_evaluator_path(path):
    global _evaluator_path
    os.environ['EVALUATOR_PATH'] = _evaluator_path = path
    # e.g. _evaluator_path == ./data/overnight/evaluator


SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'overnight.sh')


class OvernightDomain(Domain):

    def __init__(self, dataset):
        self.dataset = dataset
        self.denotation = True

    def normalize(self, lf_list):
        
        def format_overnight(lf):
            replacements = [
                ('(', ' ( '), # make sure ( and ) must have blank space around
                (')', ' ) '),
                ('! ', ' !'),
                ('SW', 'edu.stanford.nlp.sempre.overnight.SimpleWorld'),
            ]
            for a, b in replacements:
                lf = lf.replace(a, b)
            # remove redundant blank spaces
            lf = re.sub(' +', ' ', lf)
            return lf.strip()

        return [format_overnight(lf) for lf in lf_list]

    def obtain_denotations(self, lf_list):
        tf = tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.examples')
        for line in lf_list:
            tf.write(line + '\n')
        tf.flush()
        msg = subprocess.check_output([SCRIPT_PATH, self.dataset, tf.name])
        msg = msg.decode('utf8')
        tf.close()
        denotations = [
            line.split('\t')[1] for line in msg.split('\n') if line.startswith('targetValue\t')
        ]
        return denotations

    def is_valid(self, ans):
        return not ('BADJAVA' in ans or 'ERROR' in ans or 'Exception' in ans or 'name what' in ans or not bool(ans))
