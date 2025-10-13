import re
from itertools import product

def nested_partitions(n):
    # Nested partitions of integers, with non-leading 0's
    yield (n,) 
    yield (n,0)
    for i in range(2, n):  # split "n" to "(i) + (n-i)"
        for q in nested_partitions(i):  # partitions of i
# new condition, we do not need "q 0 p" since it is equal q(p 0)
            if q[-1] != 0: 
                for p in nested_partitions(n-i):  # partitions of (n-i)
                    yield q + p  # flatten both
# new condition, we do not need "(1 0)" since they are equal 1
                    if len(p) > 1 and p != (1,0):
                        yield q + (p,)  # flatten only right side
                        yield q + (p, 0)
# not needed since "(q) p" == "q p"
#                    if len(q) > 1:
#                        yield (q,) + p  # flatten only left side
#                        yield (q, 0) + p
# not needed since "(q) (p)" == "q (p)"
#                    if len(p) > 1 and len(q) > 1:
#                        yield (q,) + (p,)  # flatten neither
#                        yield (q, 0) + (p, 0)

def replace_0_with_negative(string, n):
    if '0' in string:
        #crossnum = sum([abs(int(num)) for num in string.replace(')',' ').replace('(',' ').split()])
        splitted = string.split(' 0')
        nest_levels = []
        for part in splitted[1:]:
            nest_level = 0
            for x in part:
                if x == ')':
                    nest_level += 1
                elif x == ' ':
                    continue
                else:
                    break
            nest_levels.append(nest_level)
        to_insert = []
        for k in nest_levels:
            to_insert.append([-x for x in range(k+2)])
        new_tangles = []
        for prod in product(*to_insert):
            new_tangle = ''
            for s,p in zip(splitted[:-1],prod):
                new_tangle += '{} {:d}'.format(s, p)
            new_tangle += splitted[-1]
            new_tangles.append(new_tangle)
        return new_tangles
    return [string]

def get_unique(n):
    unique = set()
    for k in range(1,n+1):
        for tangle in nested_partitions(k):
            if tangle[-1] == 0: continue
            tangle = str(tangle)[1:-1].replace(',', '').replace(')0', ') 0')
            for t in replace_0_with_negative(tangle, n):
                unique.add(t)
    return sorted(unique)
