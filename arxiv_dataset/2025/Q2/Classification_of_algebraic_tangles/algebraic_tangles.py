import re
import partition
import pandas as pd
from rational_tangles import RationalTangle
from other_classes import MyFraction, OneOverZero
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

def tree_from_string(string):
    tree_top = Node(parent=None, is_left=False)
    tree_top.load_from_string(string)
    return tree_top

def ratmul2frac(seq):
    frac = MyFraction(seq[0])
    for s in seq[1:]:
        frac = frac.inverse() + s
    return frac
   
def is_any_inside(set1, set2):
    for t in set1:
        if t in set2:
            return True
    return False 

def get_crossnum(string):
    return sum([abs(int(x)) for x in re.split('[ )(]+', string) if x])

def sorting_features(string):
    numbers = [int(x) for x in re.split('[ )(]+', string) if x]
    abs_numbers = [abs(x) for x in numbers]
    count_neg = sum([1 if x<0 else 0 for x in numbers])
    return sum(abs_numbers), count_neg, len(string), numbers[0]

def get_nestedness(string):
    max_nestedness = 0
    count = 0
    for elem in string:
        if elem == '(':
            count += 1
            max_nestedness = max(max_nestedness, count)
        elif elem == ')':
            count -= 1
    return max_nestedness

def ljust(s):
    s = s.astype(str).str.strip()
    return s.str.ljust(s.str.len().max())

def re_canonical(a): 
    a = a.group(1) 
    numbers = [int(x) for x in a.split(' ')] 
    last = numbers[-1] 
    all_but_last = [-x for x in numbers[:-1]] 
    last = last-1 
    if all_but_last[-1] == 1: 
        all_but_last[-2] += 1 
        all_but_last.pop() 
    else: 
        all_but_last[-1] -= 1 
        all_but_last.append(1) 
    if len(all_but_last) >= 2 and 1==all_but_last[0]==all_but_last[1]: 
        all_but_last[1] += 1 
        all_but_last.pop(0) 
    canonical = [str(x) for x in all_but_last] + [str(last)] 
    return '('+' '.join(canonical)+')' 

def re_canonical2(a): 
    numbers = [int(x) for x in a.split(' ')] 
    if len(numbers) == 1 or numbers[0]>0:
        return a
    last = numbers[-1] 
    all_but_last = [-x for x in numbers[:-1]] 
    last = last-1 
    if all_but_last[-1] == 1: 
        all_but_last[-2] += 1 
        all_but_last.pop() 
    else: 
        all_but_last[-1] -= 1 
        all_but_last.append(1) 
    if len(all_but_last) >= 2 and 1==all_but_last[0]==all_but_last[1]: 
        all_but_last[1] += 1 
        all_but_last.pop(0) 
    canonical = [str(x) for x in all_but_last] + [str(last)] 
    return ' '.join(canonical)

def get_representative(a):
    eq = set([])
    # true canonical
    for e in set(a.equivalent.values()):
        new_e = re.sub(r'\((-[^)(]+)\)', re_canonical, e)
        eq.add(new_e)
        assert tree_from_string(new_e).calc_frac() == tree_from_string(e).calc_frac()
    # choosing representative one: number of crossings, negative terms, nestedness, sort by numbers, shortest string
    eq_sorting = []
    for e in eq:
        crossnum, count_neg, length, numbers = sorting_features(e)
        nestedness = get_nestedness(e)
        length = len(e)
        if ')' in e:
            closing = e.index(')')
        else:
            closing = 0
        eq_sorting.append((crossnum,count_neg,nestedness,numbers,length,e))
    eq_sorting.sort()
    representative = eq_sorting[0][-1]
    return representative

def classify(n):
    tangles = set()
    tangles_eq = set()
    for part in tqdm(list(partition.get_unique(n))):
        a = AlgebraicTangle(part)
        if a.crossnum > n:
            continue
        representative = get_representative(a)
        a = AlgebraicTangle(representative)
        sym = a.symmetry
        vhx = a.vhx
        frac = a.frac
        crossnum = a.crossnum
        if crossnum == get_crossnum(representative):
            minimal = '-'
        else:
            minimal = a.minimal_str.replace(' ',',')
        representative = representative.replace(' ',',')
        eq = set([])
        canonical2mini = {}
        canonical2relation = defaultdict(list)
        crossnums = defaultdict(list)
        # true canonical
        for k,v in a.equivalent.items():
            if '(' in v:
                new_v = re.sub(r'\((-[^)(]+)\)', re_canonical, v).replace(' ',',')
            else:
                new_v = re_canonical2(v).replace(' ',',')
            canonical2relation[new_v].append(k)
            if not new_v in canonical2mini:
                t = tree_from_string(v)
                crossnum, _, _, minimal_str, _ = t.get_minimal()
                crossnums[crossnum].append(minimal_str)
                minimal_str = minimal_str.replace(' ',',')
                if minimal_str == new_v:
                    minimal_str = '-'
                canonical2mini[new_v] = minimal_str
                eq.add(new_v)
        assert len(crossnums) == 1, ('\n   '+ '\n  '.join(['{:2d}: {}'.format(k,str(v)) for k,v in crossnums.items()]))
        group = []
        for canonical in canonical2mini:
            mini = canonical2mini[canonical]
            relations = str(tuple(canonical2relation[canonical])).replace(' ','')
            group.append(tuple([canonical, mini, relations]))
        assert len(group) in [16,8,4,2]
        record = (representative, vhx[0], vhx[1], sym[0], sym[1], frac, minimal, crossnum, str(group).replace(' ',''))
        tangles.add(record)
        tangles_eq |= eq
    tangles = pd.DataFrame(sorted(tangles), columns = ['tangle', 'vhx', 'compnum', 'symnum', 'symmetry', 'fraction', 'minimal', 'crossnum', 'groupinfo'])
    tangles.sort_values(by=['crossnum','symnum','tangle'], inplace=True)
#    output_group = tangles[['tangle','groupinfo']].to_string(index=False, justify='left')
    with open('up_to_{:d}_group.txt'.format(n), 'w') as f:
        f.write('groupinfo\n')
    with open('up_to_{:d}_group.txt'.format(n), 'a+') as f:
        for val in tangles['groupinfo']:
            f.write(val+'\n')
    tangles = tangles.apply(ljust)
    output_repr = tangles.drop('groupinfo', axis=1).to_string(index=False, justify='left')
    print(output_repr)
    with open('up_to_{:d}.txt'.format(n), 'w') as f:
        f.write(output_repr)


class AlgebraicTangle():
    def __init__(self, string):
        self.string = string
        self.tangle = self.get_tangle_from_string()
        self.string = str(self.tangle)
        self.frac = self.tangle.calc_frac()
        self.crossnum, _, _, self.minimal_str, self.minimal = self.get_minimal()
        self.vhx = '{}{:d}'.format(*self.tangle.get_vhx())
        self.equivalent = self.get_symmetry_group_tangles()
        self.symmetry = self.get_symmetry_type()

    def get_minimal(self):
        tree_top = Node(parent=None, is_left=False)
        tree_top.load_from_string(self.string)
        minimal = tree_top.get_minimal()
        assert minimal[-1].calc_frac() == self.frac, (self.string, str(minimal))
        return minimal

    def get_symmetry_group_tangles(self):
        dic = defaultdict(str)
        dic['e'] = self.string
        frac = self.frac
        frac0 = self.frac.inverse()
        # e -(x)-> rho_x -(eta)-> munu -(mu)-> nu
        t = self.get_tangle_from_string()
        t.rotate_x()
        t.to_canonical()
        dic['x'] = str(t)
        assert t.calc_frac() == frac
        t.invert()
        t.to_canonical()
        dic['ηx'] = str(t)
        assert t.calc_frac() == frac0
        t.negate()
        t.to_canonical()
        dic['μηx'] = str(t)
        assert t.calc_frac() == -frac0
        # e -(y)-> rho_y -(eta)-> munununu -(mu) -> nununu
        t = self.get_tangle_from_string()
        t.rotate_y()
        t.to_canonical()
        dic['y'] = str(t)
        assert t.calc_frac() == frac
        t.invert()
        t.to_canonical()
        dic['ηy'] = str(t)
        assert t.calc_frac() == frac0
        t.negate()
        t.to_canonical()
        dic['μηy'] = str(t)
        assert t.calc_frac() == -frac0
        # e -(x)-> rho_x -(y)-> z -(eta)-> eps -(mu)-> mu eps
        t = self.get_tangle_from_string()
        t.rotate_z()
        t.to_canonical()
        dic['z'] = str(t)
        assert t.calc_frac() == frac
        t.invert()
        t.to_canonical()
        dic['ηz'] = str(t)
        assert t.calc_frac() == frac0
        t.negate()
        t.to_canonical()
        dic['μηz'] = str(t)
        assert t.calc_frac() == -frac0
        # e -(mu)-> mu -(x)-> mux -(y)-> muz -(x)-> muy
        t = self.get_tangle_from_string()
        t.negate()
        t.to_canonical()
        dic['μ'] = str(t)
        assert t.calc_frac() == -frac
        t.rotate_x()
        t.to_canonical()
        dic['μx'] = str(t)
        assert t.calc_frac() == -frac
        t.rotate_y()
        t.to_canonical()
        dic['μz'] = str(t)
        assert t.calc_frac() == -frac
        t.rotate_x()
        t.to_canonical()
        dic['μy'] = str(t)
        assert t.calc_frac() == -frac
        # e -(eta)-> eta -(mu)-> etamu
        t = self.get_tangle_from_string()
        t.invert()
        t.to_canonical()
        dic['η'] = str(t)
        assert t.calc_frac() == frac0
        t.negate()
        t.to_canonical()
        dic['μη'] = str(t)
        assert t.calc_frac() == -frac0
        # test if z-rotation = xy-rotations
        t = self.get_tangle_from_string()
        t.rotate_y()
        t.to_canonical()
        t.rotate_x()
        t.to_canonical()
        assert dic['z'] == str(t)
        return dic

    def get_symmetry_type(self):
        e = self.equivalent['e']
        s = {sym: tangle==e for sym, tangle in self.equivalent.items()}
        rot = '{}{}{}{}{}{}{}'.format(
                    'z' if s['z'] else '.',
                    'x' if s['x'] else '.',
                    'y' if s['y'] else '.',
                    'η' if s['μη'] else '.',
                    'ν' if s['μηx'] else '.',
                    '3' if s['μηy'] else '.',
                    'ε' if s['μηz'] else '.')
        mirr = '{}{}{}{}{}{}{}{}'.format(
                    'μ' if s['μ'] else '.',
                    'z' if s['μz'] else '.',
                    'x' if s['μx'] else '.',
                    'y' if s['μy'] else '.',
                    'η' if s['η'] else '.',
                    'ε' if s['ηz'] else '.',
                    'ν' if s['ηx'] else '.',
                    '3' if s['ηy'] else '.')
        fmt = '{}|{}'.format(rot,mirr)
        num = sum([2**i for i, x in enumerate(rot+mirr) if x != '.'])
        unique_tangles = defaultdict(int)
        for t in self.equivalent.values():
            unique_tangles[t] += 1
        assert len(unique_tangles) in [2,4,8,16], '\n' + '\n'.join(['{:3}: {}'.format(k,v) for k,v in self.equivalent.items()])
        assert len(set(unique_tangles.values())) == 1
        return num, fmt

    def __repr__(self):
        t = '"{}"'.format(self.string)
        mini = '"{}"'.format(self.minimal_str)
        symnum, symtype = self.symmetry
        frac = str(self.frac)
        eq_mini = defaultdict(str)
        for k,v in self.equivalent.items():
            top = Node(None, None)
            top.load_from_string(v)
            minimal = top.get_minimal()
            eq_mini[k] = str(minimal)
        eq_mini_str = '\n  '.join(['{:3}: {:20} --> {:20}'.format(k,self.equivalent[k], v) for k,v in eq_mini.items()])
        return '{:20} {:20} {:d} {} {}\n  {}'.format(t, mini, self.crossnum, self.vhx,
                  symtype, eq_mini_str)

    def get_tangle_from_string(self):
        tree_top = Node(parent=None, is_left=False)
        tree_top.load_from_string(self.string)
        tree_top.to_canonical()
        return tree_top


class Node():
    vhx_mul  = {'HV':'V', 'HH':'V', 'HX':'V',                                   
                'VV':'V', 'VH':'H', 'VX':'X',                                   
                'XV':'V', 'XH':'X', 'XX':'H'}# HV gives +1 component 
    def __init__(self, parent=None, is_left=None):
        self.parent = parent
        if self.parent == None:
            self.is_top = True
            self.is_left = False
        else:
            self.is_top = False
            self.is_left = is_left
        self.left = None
        self.right = None
        self.tangle = None

    def move_down_while(self):
        assert self.is_top

    def to_canonical(self, debug=False):
        assert self.is_top
        self.flype_ones()
        if debug:
            print(f'1 | {str(self)}')
        self.collect_rationals()
        if debug:
            print(f'2 | {str(self)}')
        self.flype_integrals()
        if debug:
            print(f'3 | {str(self)}')
        self_str, self_str_old = str(self), ''
        while self_str != self_str_old:
            self.move_down()
            self_str, self_str_old = str(self), self_str
            if debug:
                print(f'4 | {str(self)}')
        frac = self.calc_frac()
        self_str, self_str_old = str(self), ''
        while self_str != self_str_old:
            self.flype_integrals()
            if debug:
                print(f'A | {str(self)}')
            assert frac == self.calc_frac()
            self.remove_zeroes()
            if debug:
                print(f'B | {str(self)}')
            self.collect_rationals()
            if debug:
                print(f'C | {str(self)}')
            self_str, self_str_old = str(self), self_str
        self.push_rings()

    def get_minimal(self, debug=False):
        assert self.is_top
        frac = self.calc_frac()
        copy = self.copy()
        copy.unfold_twists()
        copy.remove_zeroes()
        copy.collect_rationals()
        # copy goes to stack and str(copy) goes to potentials
        # if copy can be flyped then its
        stack = [copy] # stack of tangle trees to try to flype
        potentials = set([str(copy)]) # only strings, for checking if such tangle already occurred
        minimals = []
        while stack:
            current = stack.pop()
            places_to_flype = current.find_potential_flypes() # (depth, sum(left), position))
#            print(str(current))
            #print(f' potential | {str(places_to_flype)}')
            copies = []
            for place_to_flype in places_to_flype:
                tangle = current.copy()
                position = place_to_flype[2] # position at which to perform flype
                node = tangle.get_node(position)
#                print('    ' +str(node))
                node.flype_over()
                tangle.remove_zeroes()
                tangle.collect_rationals()
                tangle.test_tree()
                assert frac == tangle.calc_frac()
                tangle_str = str(tangle)
                if not tangle_str in potentials:
                    stack.append(tangle)
                    potentials.add(tangle_str)
            crossnum = current.get_crossnum()
            str_current = str(current)
            length = len(str_current)
            number_of_int = len([int(x) for x in re.split('[ )(]+', str_current) if x])
            result = (crossnum, number_of_int, length, str_current, current)
            minimals.append(result)
        minimals.sort()
#        for minimal in minimals:
#            print(minimal)
        minimal = minimals[0]
        #if len(minimals)>1:
        #    print('-----------------------------------------------------')
        #    print(minimals)
        for m in minimals[1:]:
            del m
        return minimal
        
    #=================== FORMAT FIXING METHODS etc ========================================

    def collect_rationals(self):
        leafs = [f for f in self.get_leafs_left_to_right() if f.is_left and f.parent.right.is_integral()]
        for f in leafs:
            pr = f.parent.right
            if pr.is_integral:
                f.invert()
                num = f.tangle.fraction.numerator
                den = f.tangle.fraction.denominator
                N = f.parent.right.tangle.fraction.numerator
                f.parent.tangle = RationalTangle.new(num+N*den, den)
                f.parent.right = None
                f.parent.left = None
                del f, pr

    def remove_zeroes(self):
        old_tangle, tangle = str(self), ''
        while old_tangle != tangle:
            leafs = [f for f in self.get_leafs_left_to_right() if f.is_zero() and f.parent.is_left]
            for f in leafs:
                A = f.parent.left
                B = f.parent.parent.right
                Ar = A.rightmost()
                if Ar.divide_rational():
                    Ar = Ar.right
                N = Ar.tangle.fraction.numerator
                if N%2:
                    B.rotate_x()
                B.rightmost().twist(N)
                if Ar.is_left: # Ar == A
                    f.parent.left = B
                    B.parent = f.parent
                    B.is_left = True
                else:
                    B.parent = Ar.parent
                    Ar.parent.right = B
                new_left = f.parent.left.left
                new_right = f.parent.left.right
                new_left.parent = f.parent.parent
                new_right.parent = f.parent.parent
                f.parent.parent.left = new_left
                f.parent.parent.right = new_right
            old_tangle, tangle = tangle, str(self)

    def test_tree(self):
        if self.is_top:
            assert self.parent == None
            assert self.is_left == False
        else:
            assert self.parent
        if self.tangle:
            assert self.right == self.left == None
        else:
            assert self.left.parent == self, self
            assert self.right.parent == self
            assert self.left.is_left == True
            assert self.right.is_left == False
            self.left.test_tree()
            self.right.test_tree()


    #=================== FLYPE METHODS ========================================

    def find_potential_flypes(self):
        places_to_flype = []
        leafs = self.get_leafs_left_to_right()
        for t in leafs:
            lca = t # last common ancestor
            while not (lca.is_left or lca.is_top):
                lca = lca.parent
            if lca.is_top: continue
            lca = lca.parent # lca found
            cousin = lca.rightmost()
            frac1 = t.tangle.fraction
            frac2 = cousin.tangle.fraction
            #print('leaf', t, frac1, 'right', cousin, frac2)
            if frac1*frac2 < 0: # before >=1
                pos = lca.left.get_position()
                places_to_flype.append((len(pos), sum([int(x) for x in pos]), pos))
            elif frac2 == 0:
                lca_lca = lca # last common ancestor
                while not (lca_lca.is_left or lca_lca.is_top):
                    lca_lca = lca_lca.parent
                if lca_lca.is_top: continue
                lca_lca = lca_lca.parent # lca found
                cousin_farther = lca_lca.rightmost()
                frac3 = cousin_farther.tangle.fraction
                if frac1*frac3 < 0: # before >=1
                    pos = lca.left.get_position()
                    places_to_flype.append((len(pos), sum([int(x) for x in pos]), pos))
                
#                if str(cousin.parent) != str(lca):
#                    other_leafs = cousin.parent.get_leafs_left_to_right()[:-1]
#                    if any([frac1*leaf.tangle.fraction<0 for leaf in other_leafs]):
#                        pos = lca.left.get_position()
#                        places_to_flype.append((len(pos), sum([int(x) for x in pos]), pos))
        places_to_flype.sort()
        return places_to_flype
        
    def flype_rings(self):
        leafs = [f for f in self.get_leafs_left_to_right() if f.is_left and f.tangle.fraction==-2]
        for f in leafs:
            if f.parent.is_ring():
                f.parent.negate()

    def flype_integrals(self):
        if self.tangle:
            return
        if not self.is_top:
            assert not (self.right.is_integral() and self.left.is_integral())
            if self.right.is_integral(): 
                if self.is_left:
                    if self.right.tangle.fraction<0:
                        self.flype_over()
                        return self.flype_integrals()
                    elif abs(self.right.tangle) == 1 and not self.left.tangle:
                        self.flype_one()
                self.left.flype_integrals()
                return
            elif self.left.tangle and self.left.tangle.fraction<0:
                self.left.flype_rational()
                self.right.flype_integrals()
                return
        self.right.flype_integrals()
        self.left.flype_integrals()

    # flype over self (insert node between self and parent and add child 1/-1 to the node)
    def flype_over(self):
        assert self.is_left
        child = self.rightmost()
        cousin = self.parent.rightmost()
        frac1 = child.tangle.fraction
        frac2 = cousin.tangle.fraction
        #assert frac1*frac2 < 0 and abs(frac1) >= 1 and abs(frac2) >= 1
        lca = self.parent
        right_branch = lca.right
        num1 = frac1.numerator
        den1 = frac1.denominator
        num2 = frac2.numerator
        den2 = frac2.denominator
        if num1 != 0:
            sign = num1//abs(num1)
        else:
            sign = -num2//abs(num2)
        # altering twistable
        child.tangle = RationalTangle.new(num1-sign*den1, den1)
        cousin.tangle = RationalTangle.new(num2+sign*den2, den2)
        # adding 1 on left branch
        lca.left = Node(parent=lca, is_left=True)
        lca.left.left = self
        self.parent = lca.left
        lca.left.right = Node(parent=lca.left, is_left=False)
        lca.left.right.tangle = RationalTangle.new(-sign,1)
        # finalyzing flype
        self.negate()
        lca.right.rotate_x()
        
    def flype_one(self):
        r = self.right
        l = self.left
        one = r.tangle.fraction.numerator # 1 or -1
        assert one in [1,-1]
        if self.is_left and not self.is_top: # if it is "A1B"
            pr = self.parent.right # B
            if pr.is_zero() and self.parent.is_left:
                # it is A10B
                pass # it will be twisted
            elif self.left and self.left.is_integral():
                pass
            elif self.left.right and self.left.right.is_integral():
                pass
            else: # it is A1B (and not An1B)
                # modifying "A"
                l.negate()
                l.move_down(-one)
                # modifying "B"
                pr.rotate_x()
                pr.move_down(one)
                # updating tree structure
                self.parent.left = self.left
                self.left.parent = self.parent
                r = self.parent.right
                del self.right, self
        else: # then it is "A1)B" or "A1"
            pass # I guess it is ok

    def flype_ones(self):
        #looking for A1 to flype. A is not integral.
        # A1B -> -(A01)(B01)_x
        # A10B raise
        # A10) -> -(A01)1)
        if not self.tangle:
            r = self.right
            l = self.left
            if r.is_one() and not l.tangle:
                self.flype_one()
            r.flype_ones()
            l.flype_ones()

    def flype_ones_back(self):
        #looking for A1 to flype. A is not integral.
        # A1B -> -(A01)(B01)_x
        # A10B raise
        # A10) -> -(A01)1)
        if not self.tangle:
            r = self.right
            l = self.left
            if r.is_one() and not l.tangle:
                one = r.tangle.fraction.numerator # 1 or -1
                if self.is_left and not self.is_top: # if it is "A1B"
                    pr = self.parent.right # B
                    if pr.is_zero() and self.parent.is_left:
                        # it is A10B
                        pass # it will be twisted
                    elif self.left and self.left.is_integral():
                        pass
                    elif self.left.right and self.left.right.is_integral():
                        pass
                    else: # it is A1B (and not An1B)
                        # modifying "A"
                        l.negate()
                        l.move_down(-one)
                        # modifying "B"
                        pr.rotate_x()
                        pr.move_down(one)
                        # updating tree structure
                        self.parent.left = self.left
                        self.left.parent = self.parent
                        r = self.parent.right
                        del self.right, self
                else: # then it is "A1)B" or "A1"
                    pass # I guess it is ok
            r.flype_ones()
            l.flype_ones()

    # n...k (A 0) -> -n...-(k-1) -1 (A 1)_x
    def flype_rational(self):
        assert self.tangle
        assert self.is_left
        sign = self.tangle.sign
        assert sign in (-1,1)
        self.tangle = self.tangle.plus(-sign)
        if self.tangle.fraction.denominator:
            self.parent.right.rotate_x()
            self.parent.right.move_down(sign)
        else: # when after tangle.plus(-sign) we got infty tangle
            p = self.parent
            pr = self.parent.right
            prr = self.parent.right.right
            prl = self.parent.right.left
            p.right = prr
            p.left = prl
            prl.parent = p
            prr.parent = p
            pr = self.parent.right
            self.parent.rotate_x()
            self.parent.move_down(sign)
            del pr, self

    #=================== TWIST UP METHODS ========================================

    def unfold_twists(self):
        # assume its canonical
        # unfolding collected twists up the tree
        leafs = [f for f in self.get_leafs_left_to_right() if not (f.is_top or f.is_left)]
#        print('leafs', leafs)
        for f in leafs:
#            print('leaf', f)
            possible_twists = int(f.tangle.fraction)
            if possible_twists != 0:
                starting_node = f.parent
                unused_twists = starting_node.twist_up(possible_twists)
                used_twists = possible_twists - unused_twists
                # update original tangle
                num = f.tangle.fraction.numerator
                den = f.tangle.fraction.denominator
                f.tangle = RationalTangle.new(num-den*used_twists, den)
#                print(num, den)
#                print('possible', possible_twists, 'used', used_twists, 'tangle', self)
        self.test_tree()

    def twist_up(self, twists=0):
        assert twists != 0
        twist_sign = twists//abs(twists)
        l = self.left
        # adding one twist and using it to flype
        if l.tangle and l.tangle.fraction.numerator * twist_sign < 0:
#            print('move_up 1', self)
            num = l.tangle.fraction.numerator
            den = l.tangle.fraction.denominator
            l.tangle = RationalTangle.new(num, den+num*twist_sign)
            self.right.rotate_x()
            twists -= twist_sign
        # twisting the next parent if possible
        if twists and not (self.is_left or self.is_top):
            twists = self.parent.twist_up(twists)
        # otherwise return number of unused twists
        return twists

    #=================== MOVE DOWN METHODS ========================================

    def twist(self, twists):
        assert not self.is_top
        num = self.tangle.fraction.numerator
        den = self.tangle.fraction.denominator
        self.tangle = self.tangle.new(num + twists*den, den)

    def add_ring(self): # add ring to the left
        self.left = Node(parent=self, is_left=True)
        self.left.left = Node(parent=self.left, is_left = True)
        self.left.left.tangle = RationalTangle.new(2,1)
        self.left.right = Node(parent=self.left, is_left = False)
        self.left.right.tangle = RationalTangle.new(-1,2)

    def add_rings(self, rings):
        self.right = Node(parent=self, is_left=False)
        self.right.tangle = self.tangle
        self.tangle = None
        self.add_ring()
        for i in range(rings-1):
            r = self.right
            l = self.left
            self.right = Node(parent=self, is_left=False)
            self.right.right = r
            self.right.left = l
            r.parent = self.right
            l.parent = self.right
            self.add_ring()

    def divide_rational(self):
        new_tangle = self.tangle.__invert__()
        #if not self.is_integral() and new_tangle.fraction <= 1:
        if not self.is_integral():
            new_tangle, new_twists = new_tangle.untwist()
            self.right = Node(parent=self, is_left = False)
            self.left = Node(parent=self, is_left = True)
            self.left.tangle = new_tangle
            self.right.tangle = RationalTangle.new(new_twists,1)
            self.tangle = None
            return True
        return False

    def untwist_rational(self):
        aaa = str(self.left.tangle)
        new_tangle, new_twists = self.left.tangle.untwist()
        if new_tangle.fraction < 0:
            new_tangle = new_tangle.plus()
            new_twists -= 1
        if new_twists%2:
            self.right.rotate_x()
        if new_tangle.fraction.denominator == 0:
            l = self.left
            r = self.right
            rr = self.right.right
            rl = self.right.left
            self.right = rr
            self.left = rl
            self.right.parent = self
            self.left.parent = self
            del r,l
        else:
            self.left.tangle = new_tangle
        return new_twists

    def collect_a_ring(self):
        l = self.left
        r = self.right
        changed = False
        if r.tangle:
            extra_rings = 0
            # if tangle is integral do not collect a ring (it is deep enough)
            if r.tangle.fraction.denominator != 1: # is non integral
                new_tangle, new_twists = r.tangle.__invert__().untwist()
                self.left = Node(parent=self, is_left=True) 
                self.left.tangle = new_tangle
                self.right = Node(parent=self, is_left=False)
                self.right.left = l
                self.right.left.parent = self.right
                self.right.right = Node(parent=self.right, is_left=False)
                self.right.right.tangle = RationalTangle.new(new_twists, 1)
                del r
                changed = True
        else:
            extra_rings = 1
            self.right = r.right
            self.right.parent = self
            self.left = r.left
            self.left.parent = self
            del r, l.left, l.right, l
            changed = True
        return extra_rings, changed

    def move_right_with_zero(self):
        lr = self.left.right # == 0
        llr = self.left.left.right
        lll = self.left.left.left
        if self.right.tangle and self.right.tangle.fraction.numerator == 0:
            r = self.right
            ll = self.left.left
            self.left = lll
            self.right = llr
            self.left.parent = self
            self.right.parent = self
            del r, lr, ll
            new_twists = 0
        elif llr.is_integral(): # is integral tangle
            new_twists = llr.tangle.fraction.numerator
            if new_twists%2:
                self.right.rotate_x()
            # removing "n 0" 
            self.left = lll
            self.left.parent = self
            del llr, lr
        else:
            new_twists = 0
            r = self.right
            l = self.left
            ll = self.left.left
            self.left = lll
            self.left.parent = self
            self.right = Node(parent=self, is_left=False)
            self.right.right = r
            self.right.right.parent = self.right
            self.right.left = Node(parent=self.right, is_left=True)
            if llr.tangle: # is rational
                self.right.left.tangle = llr.tangle.__invert__()
                del llr
            else:
                self.right.left.right = lr
                lr.parent = self.right.left
                self.right.left.left = llr
                llr.is_left = True
                llr.parent = self.right.left
            del l, ll
        changed = True
        return new_twists, changed

    def move_down(self, twists = 0, rings = 0):
        changed = True
#        print('===========================')
        while changed:
#            print(self)
#            self.print_tree()
#            self.test_tree()
            changed = False
            if self.tangle: # self is rational
                if twists:
                    self.twist(twists)
#                    print('========= twist ', twists)
                if rings:
                    is_divided = self.divide_rational()
                    if is_divided:
#                        print('divided')
                        self.right.add_rings(rings)
                    else:
#                        print('raw')
                        self.add_rings(rings)
#                    print('========= rings ', rings)
                return
            if self.left.tangle:
#                print('left rational')
                # joining rational tangles into one rational tangle if possible
                if self.right.is_integral():
                    new_tangle = self.left.tangle.fraction.inverse() + self.right.tangle.fraction
                    self.tangle = RationalTangle.new(new_tangle.numerator, new_tangle.denominator)
                    l = self.left
                    r = self.right
                    del r,l
                    self.left = None
                    self.right = None
                    changed = True
                # reducing left tangle to have fraction > 1 and collecting twists
                elif self.left.tangle.fraction <= 1:
                    new_twists = self.untwist_rational()
                    twists += new_twists
            elif self.left.right.tangle:
                if self.left.is_ring():
#                    print('left ring')
                    extra_rings, changed = self.collect_a_ring()
                    rings += extra_rings
                elif self.left.right.is_zero():
#                    print('left right zero')
                    new_twists, changed = self.move_right_with_zero()
                    twists += new_twists
                    # checking if there is "0 n" on the right
                    lr = self.left.right
                    if lr and lr.is_zero():
                        r = self.right
                        if r.is_integral():
                            twists += r.tangle.fraction.numerator
                            #print('DDDDDDDDDDD')
                            # removing "0 n"
                            self.right = self.left.left.right
                            self.right.parent = self
                            self.left = self.left.left.left
                            self.left.parent = self
                            changed = True
                            del lr, r
        self.right.move_down(twists, rings)
        self.left.move_down()

    def push_rings(self, rings=0):
        changed = True
        while changed:
            changed = False
            if self.tangle: # self is rational
                if rings:
                    is_divided = self.divide_rational()
                    if is_divided:
                        self.right.add_rings(rings)
                    else:
                        self.add_rings(rings)
                return
            if self.left.is_ring():
                extra_rings, changed = self.collect_a_ring()
                rings += extra_rings
        self.right.push_rings(rings)
        self.left.push_rings()

        

    #=================== TREE and MINOR METHODS ========================================

    def get_crossnum(self):
        if self.tangle:
            return self.tangle.get_crossnum()
        return self.left.get_crossnum() + self.right.get_crossnum()

    def is_ring(self):
        if self.tangle: return False
        if self.left.tangle and self.right.tangle:
            frac1 = self.left.tangle.fraction
            frac2 = self.right.tangle.fraction
            if (frac1 == 2 and frac2 == -1/2) or (frac1 == -2 and frac2 == 1/2):
                return True
        return False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self.tangle:
            return self.tangle.representative
        str_left = repr(self.left)
        str_right = repr(self.right)
        if ' ' in str_right:
            str_right = '({})'.format(str_right)
        return '{} {}'.format(str_left, str_right)

    def get_vhx(self):
        if self.tangle:
            res,c = self.tangle.vhx
            components = int(c)
        else:
            left, c1 = self.left.get_vhx()
            right, c2 = self.right.get_vhx()
            key = left+right
            res = Node.vhx_mul[key]
            components = c1 + c2 + int(key=='HV') # HV gives )o(
        return res, components

    def calc_frac(self):
        if self.tangle:
            return self.tangle.fraction
        return self.left.calc_frac().inverse() + self.right.calc_frac()

    def is_integral(self):
        return bool(self.tangle) and self.tangle.fraction.denominator == 1

    def is_one(self):
        return bool(self.tangle) and abs(self.tangle.fraction) == 1

    def is_zero(self):
        return bool(self.tangle) and self.tangle.fraction.numerator == 0

    # left to right order
    def get_leafs_left_to_right(self):
        if self.tangle:
            return [self]
        return [*self.left.get_leafs_left_to_right()] + [*self.right.get_leafs_left_to_right()]

    # breadth-first order
    def get_leafs_bfs(self):
        leafs = []
        queue = [self]
        while queue:
            node = queue.pop(0)
            if node.tangle:
                leafs.append(node)
            else:
                queue += [node.left, node.right]
        return leafs

    def invert(self):
        if self.tangle:
            self.tangle = self.tangle.mirror2
        else:
            old_left = self.left
            old_right = self.right
            self.left = Node(parent=self, is_left=True)
            self.right = Node(parent=self, is_left=False)
            self.right.tangle = RationalTangle.new(0,1)
            self.left.left = old_left
            self.left.right = old_right
            self.left.left.parent = self.left
            self.left.right.parent = self.left

    def negate(self):
        if self.tangle:
            self.tangle = self.tangle.mirror
        else:
            self.left.negate()
            self.right.negate()

    def rotate_x(self):
#        aaa = str(self)
        if not self.tangle:
            self.left.rotate_y()
            self.right.rotate_x()
#            print('      rotx | {:30} -> {}'.format(aaa, str(self)))

    def rotate_y(self):
#        aaa = str(self)
        if self.tangle: return
        # analytically worked out cases which let us join rotation with
        # afterward twist move or ring move
        if self.right.tangle:
            case = abs(self.right.tangle.fraction)
            if case == 0: # [A0]y = [A]x 0
                self.left.rotate_x()
#                print('      rotyA| {:30} -> {}'.format(aaa, str(self)))
                return
            elif case == 1: # [A1]y = 1[A0]y = [A0]z 01 = [A]z 1
                self.left.rotate_z()
#                print('      rotyB| {:30} -> {}'.format(aaa, str(self)))
                return
        elif self.right.right.tangle and self.right.right.is_integral(): 
            # [A(2(-20)N)]y = 2(-20) ([A]xy^N N) = [A]xy^N (2(-20)N)
            if self.right.left.is_ring():
                N = self.right.right.tangle.fraction.numerator
                if N%2:
                    self.left.rotate_z()
#                    print('      rotyC| {:30} -> {}'.format(aaa, str(self)))
                    return
                else:
                    self.left.rotate_x()
#                    print('      rotyD| {:30} -> {}'.format(aaa, str(self)))
                    return
            # [A(BN)]y = [BN0]x [A0]y = [B]xy^N ([A]xy^N N)
            else:
                N = self.right.right.tangle.fraction.numerator
                self.left, self.right.left = self.right.left, self.left
                self.left.parent = self
                self.right.left.parent = self.right
                if N%2:
                    self.left.rotate_z()
                    self.right.left.rotate_z()
                else:
                    self.left.rotate_x()
                    self.right.left.rotate_x()
#                print('      rotyE| {:30} -> {}'.format(aaa, str(self)))
                return
        # regular rotation
        self.left, self.right = self.right, self.left
        self.left.is_left = True
        self.right.is_left = False
        self.left.rotate_y()
        self.right.rotate_x()
        self.left.invert()
        self.right.invert()
        if self.left.tangle and abs(self.left.tangle.fraction) < 1:
            twists = self.untwist_rational()
            rightmost = self.right.rightmost()
            rightmost.twist(twists)
#        print('      rotyF| {:30} -> {}'.format(aaa, str(self)))

    def rotate_z(self):
#        aaa = str(self)
        self.rotate_x()
        self.rotate_y()
#        print('      rotz | {:30} -> {}'.format(aaa, str(self)))

    def rightmost(self):
        if self.tangle:
            return self
        return self.right.rightmost()

    def get_sign(self):
        if self.tangle:
            if self.tangle.fraction > 0:
                return 1
            elif self.tangle.fraction == 0:
                return 0
            else:
                return -1
        signs = [l.get_sign() for l in self.get_leafs_left_to_right()]
        if (1 in signs): 
            if (-1 in signs):
                return 0
            return 1
        elif (-1 in signs):
            return -1
        return 0

    def get_depth(self):
        return len(self.get_position())

    def get_position(self, pos=''):
        if self.is_top: return tuple([int(s) for s in pos[::-1]])
        return self.parent.get_position('{}{:d}'.format(pos,int(self.is_left)))

    def get_node(self, pos):
        node = self
        for s in pos:
            if s:
                node = node.left
            else:
                node = node.right
        return node

    def load_from_string(self, string):
        string = self.clean_string(string)
        if ')' in string:
            if string[-1] == ')':
            # extracting the bracket
                closed = 1
                for i,s in reversed(list(enumerate(string[:-1]))):
                    if s == ')':
                        closed += 1
                    elif s == '(':
                        closed -= 1
                        if closed == 0:
                            left_string = string[:i]
                            right_string = string[i+1:-1]
                            break
            else:
            #extracting last number
                right_string = ''
                for i,s in reversed(list(enumerate(string))):
                    if s in ' )': 
                        break
                    right_string += s
                right_string = right_string[::-1]
                left_string = string[:i+1]
            self.left = Node(parent=self, is_left=True)
            self.left.load_from_string(left_string)
            self.right = Node(parent=self, is_left=False)
            self.right.load_from_string(right_string)
        else:
            frac = ratmul2frac([int(x) for x in string.split(' ')])
            self.tangle = RationalTangle.new(frac.numerator, frac.denominator)

    def print_tree(self, last=True, header=''):
        elbow = "└───"
        pipe = "│   "
        tee = "├───"
        blank = "    "
        print(header + pipe)
        print(header + (elbow if last else tee) + ' ' + repr(self))
        if not self.tangle:
            self.right.print_tree(header=header + (blank if last else pipe), last=False)
            self.left.print_tree(header=header + (blank if last else pipe), last=True)
            
    @staticmethod
    def clean_string(string):
        string = string.strip()
        while '  ' in string:
            string.replace('  ',' ')
        string.replace('(1 0)', '1')
        string.replace('(1 1', '(2')
        while 1:
            old_string = string
            #remove unnecessary bracket
            if string[0] == '(':
                opened = 1
                for i,s in enumerate(string[1:], start=1):
                    if s == '(':
                        opened += 1
                    elif s == ')':
                        opened -= 1
                    if opened == 0:
                        string = string[1:i] + string[i+1:]
                        string = string.strip()
                        while '  ' in string:
                            string.replace('  ',' ')
                        break
            if string == old_string: break
        return string

    def copy(self):
        new_top = Node()
        new_top.load_from_string(str(self))
        return new_top

def test_one():
    s = '2 (-2 -1) 2 (2 (2 (-2 0) (-2 0)) -1)'
    s = '2 (-2 0) 2 (2 (2 (2 (-2 0) -1)) 0)'
    s = '2 (-2 0) 2 (2 (2 (2 (-2 0) -1)) 0)'
    s = '2 (-2 -1) (2 0) (-2 0) 1'
    s = '-2 (2 (2 (-2 (-2 (2 0) -1) 0 (-2 0) 1)) 0) 0'
    a = tree_from_string(s)
    frac = a.calc_frac()
    a.rotate_x()
    print('   x |', a)
    assert frac == a.calc_frac()
    a.to_canonical()
    print('   x |', a)
    assert frac == a.calc_frac()
    a.rotate_y()
    print('   z |', a)
    assert frac == a.calc_frac()
    a.to_canonical()
    print('   z |', a)
    assert frac == a.calc_frac()
    a.rotate_x()
    print('   y |', a)
    assert frac == a.calc_frac()
    a.to_canonical()
    print('   y |', a)
    assert frac == a.calc_frac()
#    a.unfold_twists()
#    print(a)
#    assert frac == a.calc_frac()
#    minimal = a.flype_reduce()
#    print(minimal)

def test_few():
    ss = ['-2 (-2 -1) (2 0) 1 (-2 (-2 -1) 0)', '-2 (2 (2 1) 0) -1 (2 (2 1) 0) 0', '-2 (2 (2 1) 0) -1 (2 (2 1) 0)', '2 (2 1) (-2 0) -1 (2 (2 1) 0) 0', '2 (-2 (-2 -1) 0) 1 (-2 (-2 -1) 0)', '2 (2 1) (-2 0) -1 (2 (2 1) 0)', '-2 (-2 -1) (2 0) 1 (-2 (-2 -1) 0) 0', '2 (-2 (-2 -1) 0) 1 (-2 (-2 -1) 0) 0']
    ss = ['2 (2 (2 (-2 0) -1) 1 1 -1) 0', '2 (2 (-2 0) -1) 1 1 (-2 0) 0', '2 (2 (2 (-2 0) -1) 1 1 -1) ']
    for s in ss:
        a = tree_from_string(s)
        frac = a.calc_frac()
        a.to_canonical()
        print(a)
        assert frac == a.calc_frac()
        minimal = a.get_minimal()
        print(minimal)
        assert frac == minimal[-1].calc_frac()
        print('==============')

def check_symmetry():
    tangles = '2 (3 (-3 0) 0) -1', '2 (4 (-4 0) 0) -1', '2 (5 (-5 0) 0) -1'
    tangles = '3(2(-3 0))-1', '4(2(-4 0))-1', '5(2(-5 0))-1'
    tangles = '3(-3 0)(2 1(-2 -1 0)0)', '3(-3 0)(-3(3 0)0)', '4(-4 0)(-4(4 0)0)','5(-5 0)(-5(5 0)0)','25(-25 0)(-25(25 0)0)'
    for s in tangles:
        a = AlgebraicTangle(s)
        print(a)

def to_representative():
    tangles = ['2 (2 (2 0) 2 -2)', '2 (2 (2 2) -2)', '2 (2 1) -2 (-2 0)', '3 (2 (2 1) -2)']
    for t in tangles:
        a = AlgebraicTangle(t)
        r = get_representative(a)
        print('{:20} {:20}'.format(t,r))

def test_algebraic():
    tangles = ['2 (2 0) 1 (2 0)',
               '2 (2 0) -1 (-2 0)',
               '2 (-2 0) 1 (2 0)']
    tangles = ['2 (2 0) (-2 0) (2 0) -1']
    tangles = ['2 (-2 0) 2 (2 (2 (2 (-2 0) -1)) 0)']
    tangles = ['3 (-2 -1)']
    tangles = ['2 (2 (2 0) 1 0) (-2 (-2 0) -1 0)']
    for tangle in tangles:
        a = AlgebraicTangle(tangle)
        print(a)
        print('--------------')

def test_minimal():
    s = '-2 (2 (2 (-2 (-2 (2 0) -1) 0 (-2 0) 1)) 0)'
    s = '-2 (2 (2 (-2 (-2 (2 0) -1) 0 (-2 0) 1)) 0) 0'
    s = '2 (2 (2 0) 1 0) (-2 (-2 0) -1 0)'
    s = '2 (-2 (-2 -1) 0) 1 (-2 (-2 -1) 0) 0'
    s = '2 (-2 (-2 -1) 0) 1 (-2 (-2 -1) 0)'
    s = '-2 (-2 -1) (2 0) 1 (-2 (-2 -1) 0)'
    s = '2 (2 (2 (-2 0) 0) (2 0) 0)'
    a = tree_from_string(s)
    #frac = a.calc_frac()
    #a.to_canonical()
    #print('   canon', a)
    #assert frac == a.calc_frac()
    #new_a = re.sub(r'\((-[^)(]+)\)', re_canonical, v).replace(' ',',')
    minimal = a.get_minimal(debug=True)
    print('   minim', minimal)
    

if __name__ == '__main__':
    #check_symmetry()
    #test_algebraic()
    #test_rotations()
    #test_one()
    #test_few()
    classify(8)
    #to_representative()
    #test_minimal()






