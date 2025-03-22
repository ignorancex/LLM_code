'''
Python interpreter
'''

# Python imports
import importlib
import math
import string

from copy import deepcopy

# Feedback lib imports
from .py_parser import PyParser

from .interpreter import Interpreter, addlanginter, RuntimeErr, UndefValue
from .model import Var, Op, VAR_IN, VAR_OUT, VAR_RET, prime


def eargs(fun):
    '''
    Decorator to evaluate args
    '''

    # Wrapper function that calls original 'fun'
    def wrap(self, f, mem):
        args = [self.execute(x, mem) for x in f.args]
        for a in args:
            if isinstance(a, UndefValue):
                raise RuntimeErr('undefined value')
        
        return fun(self, *args)
    return wrap

DEFAULT = object()

class PyInterpreter(Interpreter):

    BINARY_OPS = set()
    UNARY_OPS = set()
    DEFAULT_RETURN = None

    def execute_Const(self, c, mem):

        c = c.value
        
        # Undef
        if c == '?':
            return UndefValue()

        # Integer
        try:
            if len(c) > 0 and c[-1] == 'l':
                c = c[:-1]
            return int(c)
        except ValueError:
            pass

        # Float
        try:
            return float(c)
        except ValueError:
            pass

        # Complex
        if len(c) > 0 and c[-1] == 'j':
            try:
                return complex(c)
            except ValueError:
                pass
        
        # String
        if len(c) >= 2 and c[0] == c[-1] == '"':
            return c[1:-1]

        # Bool
        if c in ('True', 'False'):
            return c == 'True'

        if c == 'list':
            return list
        if c == 'tuple':
            return tuple
        elif c == 'int':
            return int
        elif c == 'dict':
            return dict
        elif c == 'float':
            return float
        elif c == 'bool':
            return bool

        # None
        if c == 'None':
            return None

        if c == 'break_outside_loop':
            return UndefValue()

        assert False, 'unknown constant: %s' % (c,)

    def execute_builtin_fnc(self, fnc, args, mem):
        args = [self.execute(x, mem) for x in args]
        for a in args:
            if isinstance(a, UndefValue):
                raise RuntimeErr('undefined value')
        ans = fnc(*args)
        if (fnc.__name__ == 'map'):
            ans = list(ans)
        return ans
    
    def execute_lib_fnc(self, lib, fnc, args, mem):
        lib = importlib.import_module(lib)
        args = [self.execute(x, mem) for x in args]
        for a in args:
            if isinstance(a, UndefValue):
                raise RuntimeErr('undefined value')
        f = getattr(lib, fnc)
        return f(*args)

    @eargs
    def execute_ListInit(self, *a):
        return list(a)

    @eargs
    def execute_DictInit(self, *d):
        return {k: v for (k, v) in zip(d[0::2], d[1::2])}


    @eargs
    def execute_SetInit(self, *s):
        return set(s)

    @eargs
    def execute_TupleInit(self, *t):
        return tuple(t)

    @eargs
    def execute_Not(self, x):
        return not x

    def execute_And(self, f, mem):
        x = self.execute(f.args[0], mem)
        if not x:
            return False
        return self.execute(f.args[1], mem)

    def execute_Or(self, f, mem):
        x = self.execute(f.args[0], mem)
        if x:
            return x
        return self.execute(f.args[1], mem)

    def execute_GetAttr(self, g, mem):
        name = g.args[1].value

        if isinstance(g.args[0], Var) \
           and g.args[0].name in PyParser.MODULE_NAMES:
            mname = g.args[0].name
            if mname == 'string':
                return getattr(string, name)
            elif name == 'math':
                return getattr(math, name)
            
        return getattr(self.execute(g.args[0], mem), name)

    @eargs
    def execute_Pow(self, x, y):
        return x ** y

    @eargs
    def execute_math_pow(self, *a):
        return math.pow(*a)

    @eargs
    def execute_math_ceil(self, *a):
        return math.ceil(*a)
    
    @eargs
    def execute_math_log(self, x, y):
        return math.log(x,y)

    @eargs
    def execute_Invert(self, x):
        return ~x

    @eargs
    def execute_UAdd(self, x):
        return +x

    @eargs
    def execute_USub(self, x):
        return -x

    @eargs
    def execute_BitAnd(self, x, y):
        return x & y

    @eargs
    def execute_BitOr(self, x, y):
        return x | y

    @eargs
    def execute_BitXor(self, x, y):
        return x ^ y

    @eargs
    def execute_RShift(self, x, y):
        return x >> y

    @eargs
    def execute_LShift(self, x, y):
        return x << y

    @eargs
    def execute_Mod(self, x, y):
        return x % y

    @eargs
    def execute_Add(self, x, y):
        return x + y

    @eargs
    def execute___add__(self, x, y):
        return x + y

    # See Python parser for explanation of this hack
    @eargs
    def execute_AssAdd(self, x, y):
        if isinstance(x, list) and isinstance(y, tuple):
            y = list(y)
        return x + y

    @eargs
    def execute_Sub(self, x, y):
        return x - y

    @eargs
    def execute_Div(self, x, y):
        return x / y

    @eargs
    def execute_FloorDiv(self, x, y):
        return x // y

    @eargs
    def execute_Mult(self, x, y):
        return x * y

    @eargs
    def execute_Lt(self, x, y):
        return x < y

    @eargs
    def execute_LtE(self, x, y):
        return x <= y

    @eargs
    def execute_Gt(self, x, y):
        return x > y

    @eargs
    def execute_GtE(self, x, y):
        return x >= y

    @eargs
    def execute_Eq(self, x, y):
        return x == y

    @eargs
    def execute_NotEq(self, x, y):
        return x != y

    @eargs
    def execute_In(self, x, y):
        return x in y

    @eargs
    def execute_NotIn(self, x, y):
        return x not in y

    @eargs
    def execute_Is(self, x, y):
        return x is y

    @eargs
    def execute_IsNot(self, x, y):
        return x is not y

    @eargs
    def execute_GetElement(self, x, y):
        return x[y]

    @eargs
    def execute_Slice(self, a, b, c):
        return slice(a, b, c)

    @eargs
    def execute_AssignElement(self, l, i, v):
        l = deepcopy(l)
        l[i] = v
        return l

    @eargs
    def execute_append(self, l, e):
        l = deepcopy(l)
        l.append(e)
        return l

    @eargs
    def execute_sort(self, l, *a):
        l = deepcopy(l)
        l.sort(*a)
        return l
    
    @eargs
    def execute_values(self, *a):
        return a.values()

    def execute_xrange(self, e, mem):
        return self.execute_range(e, mem)

    @eargs
    def execute_extend(self, l, i):
        l = deepcopy(l)
        l.extend(i)
        return l

    @eargs
    def execute_remove(self, l, i):
        l = deepcopy(l)
        l.remove(i)
        return l

    @eargs
    def execute_insert(self, l, i, v):
        l = deepcopy(l)
        l.insert(i, v)
        return l

    @eargs
    def execute_items(self, d):
        return list(d.items())

    @eargs
    def execute_keys(self, d):
        return list(d.keys())

    @eargs
    def execute_index(self, l, v, *a):
        return l.index(v, *a)

    @eargs
    def execute_count(self, l, *a):
        return l.count(*a)

    @eargs
    def execute_strip(self, s):
        return s.strip()

    @eargs
    def execute_pop(self, l, *a):
        nl = deepcopy(l)
        res = nl.pop(*a)
        return (nl, res)

    @eargs
    def execute_join(self, s, l):
        return s.join(l)

    @eargs
    def execute_Delete(self, l, i):
        nl = deepcopy(l)
        del nl[i]
        return nl

    @eargs
    def execute_reverse(self, l):
        return list(reversed(l))

    @eargs
    def execute_split(self, l, m=" "):
        return l.split(m)
    
    @eargs
    def execute_replace(self, s, o, n):
        return s.replace(o, n)

    @eargs
    def execute_lower(self, l):
        return l.lower()

    @eargs
    def execute_upper(self, l):
        return l.upper()

    @eargs
    def execute_startswith(self, s, elem):
        return s.startswith(elem)
    
    @eargs
    def execute_find(self, s, elem):
        return s.find(elem)
    
    @eargs
    def execute_capitalize(self, s):
        return s.capitalize()
    
    @eargs
    def execute_swapcase(self, s):
        return s.swapcase()

    @eargs
    def execute_title(self, s):
        return s.title()

    @eargs
    def execute_ignore_none(self, s):
        return

    def execute_map(self, m, mem):
        if isinstance(m.args[0], Var) and m.args[0].name == 'mul':
            import operator
            f = operator.mul
        else:
            f = self.execute(m.args[0], mem)
        ls = [self.execute(x, mem) for x in m.args[1:]]
        return list(map(f, *ls))

    def execute_reversed(self, o, mem):
        return self.execute_reverse(o, mem)

    def execute_BoundVar(self, c, mem):
        var = int(c.args[0].value)
        return mem['#__bound'][var]

    def execute_ListComp(self, lc, mem):

        # Arg #1 is an elements expresion
        eexpr = lc.args[1]

        # Arg #2 is a list expression
        l = self.execute(lc.args[2], mem)

        # Arg #3 is a filter
        filt = lc.args[3]

        # Arg #0 is a number of bound variables
        boundlen = int(lc.args[0].value)

        mem = deepcopy(mem)
        bound = mem['#__bound'] = ([None for _ in range(boundlen)] \
                                   + mem.get('#__bound', []))

        # Construct a new list
        nl = []
        for el in l:

            # Construct a new memory from a list element
            if boundlen == 1:
                bound[0] = el
            else:
                el = tuple(el)
                if len(el) != boundlen:
                    raise RuntimeErr('Cannot unpack')
                for var, val in zip(range(boundlen), el):
                    bound[var] = val

            # Apply filter
            ok = self.execute(filt, mem)
            if not ok:
                continue

            # Get a value of an element by executing it
            el = self.execute(eexpr, mem)
            nl.append(el)

        return nl

    def execute_DictComp(self, lc, mem):

        # Arg #1,#2 are key val exprs
        kexpr = lc.args[1]
        vexpr = lc.args[2]

        # Arg #3 is a list expression
        l = self.execute(lc.args[3], mem)

        # Arg #4 is a filter
        filt = lc.args[4]

        # Arg #0 is a number of bound variables
        boundlen = int(lc.args[0].value)

        mem = deepcopy(mem)
        bound = mem['#__bound'] = ([None for _ in range(boundlen)] \
                                   + mem.get('#__bound', []))

        # Construct a dict
        nd = {}
        for el in l:

            # Construct a new memory from a list element
            if boundlen == 1:
                bound[0] = el
            else:
                el = tuple(el)
                if len(el) != boundlen:
                    raise RuntimeErr('Cannot unpack')
                for var, val in zip(range(boundlen), el):
                    bound[var] = val

            # Apply filter
            ok = self.execute(filt, mem)
            if not ok:
                continue

            # Get a value of an element by executing it
            key = self.execute(kexpr, mem)
            val = self.execute(vexpr, mem)
            nd[key] = val

        return nd

    def execute_SetComp(self, sc, mem):
        return set(self.execute_ListComp(sc, mem))

    def execute_GeneratorExp(self, sc, mem):
        return self.execute_ListComp(sc, mem)

    # def execute_DictComp(self, dc, mem):

    #     # Comprehension expression
    #     vars, l = self.execute(dc.args[2], mem)

    #     # Key, value exprs
    #     ke, ve = dc.args[:2]

    #     # Construct a dict
    #     nd = {}
    #     for el in l:

    #         # New memory from a list element
    #         newmem = deepcopy(mem)
    #         for var, val in zip(vars, el):
    #             newmem[var] = val

    #         # Get key and value by executing expressions
    #         key = self.execute(ke, newmem)
    #         val = self.execute(ve, newmem)
    #         nd[key] = val

    #     return nd

    def extract_names(self, node):
        # Single variable
        if isinstance(node, Var):
            return [node.name]

        # Tuple of variables
        if isinstance(node, Op) and node.name == 'TupleInit':
            vars = []
            for arg in node.args:
                assert isinstance(arg, Var), 'unsupported list of names'
                vars.append(arg.name)
            return vars

        assert False, 'unsupported list of names'
        
    def convert(self, v, t):
        return v


addlanginter('py', PyInterpreter)
