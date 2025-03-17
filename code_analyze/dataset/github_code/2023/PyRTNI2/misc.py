import itertools
from sympy.combinatorics import Permutation
from sympy import symbols, sympify #symbols, simplify, sympify
import os
import json


class _IteratorPairings:
    def __init__(self, n):
        self.n = n
        self.it = self.iterate_pairs(list(range(n)))
    def __iter__(self):
        return self
    def __next__(self):
        x = next(self.it)
        return x
    
    def iterate_pairs(self, ns):
        if ns==[]:
            yield []
            return
        ns_shorter = ns[1:]
        for i, n in enumerate(ns_shorter):
            for pairs in self.iterate_pairs(ns_shorter[:i] + ns_shorter[i+1:]):
                yield [(ns[0], n)] + pairs

                
def _generate_iterator4reconnections(size, is_group, is_complex):
    if is_complex:
        if is_group:
            it = (([(i, j) for i,j in enumerate(c)], Permutation(c, size=size)) for c in itertools.permutations(range(size)))
        else:
            it = (([(i, j) for i,j in enumerate(c)], None) for c in itertools.permutations(range(size)))
    else:
        if is_group:
            it =((c, Permutation(c, size=size)) for c in _IteratorPairings(size*2))
        else:
            it =((c, None) for c in _IteratorPairings(size*2))
    # returning the iterator. 
    return it    
        
        
class _WeingartenFormulas:

    def functions(is_complex, size):

        n = symbols('n')

        if is_complex:
            with open(os.path.join('wfu_json', f'{size:02d}.json'), 'r') as file:
                wfs= json.load(file)
        else:
            with open(os.path.join('wfo_json', f'{size:02d}.json'), 'r') as file:
                wfs= json.load(file)

        return {sympify(young_diagram_str): sympify(wf_str) for young_diagram_str, wf_str in wfs.items()}

    def nums(is_complex, size, dim):

        n = symbols('n')

        if is_complex:
            with open(os.path.join('wfu_json', f'{size:02d}.json'), 'r') as file:
                wfs= json.load(file)
        else:
            with open(os.path.join('wfo_json', f'{size:02d}.json'), 'r') as file:
                wfs= json.load(file)

        return {sympify(young_diagram_str): sympify(wf_str).subs(n, dim) for young_diagram_str, wf_str in wfs.items()}

    def nums_low_dims(is_complex, size, dim):

        n = symbols('n')

        if is_complex:
            with open(os.path.join('wfu_json_nums', f'{size:02d}.json'), 'r') as file:
                wfs= json.load(file)
        else:
            with open(os.path.join('wfo_json_nums', f'{size:02d}.json'), 'r') as file:
                wfs= json.load(file)

        return {sympify(young_diagram_str): sympify(wf_str) for young_diagram_str, wf_str in wfs[str(dim)].items()}
    
class _WeingartenNote:
    def __init__(self,):
        self._note_dict = {'formula':{}, 'nums':{}}
        self.n = symbols('n')
    def get(self, is_symbolic, is_complex, size, dim, yd):
        if is_symbolic:
            if (is_complex, size) not in self._note_dict['formula']:
                self._note_dict['formula'][(is_complex, size)] = _WeingartenFormulas.functions(is_complex, size)
#                 print(self._note_dict['formula'][(is_complex, size)])
            return self._note_dict['formula'][(is_complex, size)][yd].subs(self.n, dim)
        else:
            if (is_complex, size, dim) not in self._note_dict['nums']:
                if size <= dim:
                    self._note_dict['nums'][(is_complex, size, dim)] = _WeingartenFormulas.nums(is_complex, size, dim)
                else:
                    self._note_dict['nums'][(is_complex, size, dim)] = _WeingartenFormulas.nums_low_dims(is_complex, size, dim)
            return self._note_dict['nums'][(is_complex, size, dim)][yd]

