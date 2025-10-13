from inspect import stack
from numbers import Real
from fractions import Fraction
from functools import total_ordering
from copy import deepcopy

def fs():
    fs = []
    sta = stack()
    for _, s in enumerate(sta):
        ee = str(s[3])
        if ee not in ['__init__', 'new']:
            fs.append(ee)
    return '/'.join(fs[-4:0:-1])

@total_ordering
class RationalStr(str):
    alphabet="-0123456789/() _.,:|+♭♯∞"
    order = {elem:i for i,elem in enumerate(alphabet)}
    def __gt__(self, other):
        for s,o in zip(self, other):
            s = self.order[s]
            o = self.order[o]
            if s == o: continue
            else: return s>o
        return len(self) > len(other)

    def __eq__(self, other):
        for s,o in zip(self, other):
            s = self.order[s]
            o = self.order[o]
            if s == o: continue
            else: return False
        return len(self) == len(other)

@total_ordering
class MyFraction(Fraction):
    def __repr__(self):
        if self.numerator == 0:
            return '0'
        elif self.numerator < 0:
            return '-{:d}/{:d}'.format(abs(self.numerator), self.denominator)
        else:
            return '+{:d}/{:d}'.format(abs(self.numerator), self.denominator)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        if type(other) == OneOverZero:
            return False
        else:
            return super().__eq__(other)

    def __gt__(self, other):
        if type(other) == OneOverZero:
            return False
        else:
            return super().__gt__(other)

    def __add__(self,other):
        if type(other) == OneOverZero:
            return other
        frac = super().__add__(other)
        return MyFraction(frac.numerator, frac.denominator)

    def inverse(self):
        if self.numerator == 0:
            return OneOverZero()
        return MyFraction(self.denominator, self.numerator)

    def __hash__(self):
        return hash((self.numerator,self.denominator))


@total_ordering
class OneOverZero(float):
    created = None
    def __new__(cls):
        if cls.created:
            return cls.created
        else:
            instance = super().__new__(cls)
            cls.created = instance
            return instance

    def __init__(self):
        self.numerator = 1
        self.denominator = 0

    def inverse(self):
        return MyFraction(0,1)

    def __add__(self,other):
        return self

    def __float__(self):
        return float('inf')

    def __int__(self):
        raise ValueError

    def __eq__(self, other):
        return type(other) == OneOverZero

    def __gt__(self, other):
        other_type = type(other)
        if other_type == OneOverZero:
            return False
        elif issubclass(other.__class__, Real):
            return True
        else:
            raise

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '∞'

    def __neg__(self):
        return self

    def __abs__(self):
        return float(self)

    def __hash__(self):
        return hash((1,0))
