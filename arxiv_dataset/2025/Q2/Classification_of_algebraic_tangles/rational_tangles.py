from copy import deepcopy, copy
from functools import total_ordering
from other_classes import OneOverZero, MyFraction, RationalStr

@total_ordering
class Tangle():
    vhx_mul0 = {'H':'V', 'V':'H', 'X':'X'}
    vhx_sum  = {'HV':'V', 'HH':'H', 'HX':'X',                                
                'VV':'V', 'VH':'V', 'VX':'V',                                
                'XV':'V', 'XH':'X', 'XX':'H'}  

    def __new__(cls, key, *args):
        #if (num,den) in cls.cached:
        #    raise ValueError('Use RationalTangle.new(num,den)')
        instance = super().__new__(cls)
        cls.cached[key] = instance
        return instance

    def __init__(self):
        self.sign = self.get_sign()
        self.mul_repr = self.get_mul_repr()
        self.fractional_repr = self.get_fractional_repr()
        self.fractional_repr_str = self.fmt_fractional()
        self.representative = self.get_representative()
        #self.jones = split(bracket(self.mul_repr))
        self.signature = self.get_signature()
        self.crossnum = self.get_crossnum()
        self.vhx = self.get_vhx()
        self.mirror = -self
        self.mirror2 = ~self
        self.rotated_x = self.rotate_x() # rotated are always (at least) equivalent
        self.rotated_y = self.rotate_y() 
        self.rotated_z = self.rotate_z()
#        self.twisted = self.get_twisted()
#        self.twisted_x_reflected = -self.twisted.rotate_x()
#        self.twisted_y_reflected = -self.twisted.rotate_y()
#        self.twisted_z_reflected = -self.twisted.rotate_z()
        self.symmetry = self.get_symmetry_type()
        self.equivalent = self.get_equivalent()
#        self.alternative = self.get_alternative()

    def __repr__(self):
        return self.representative

    def save(self):
        out = ';'.join([self.representative, str(self.crossnum), self.vhx, self.symmetry_fmt(), self.signature, self.fractional_repr_str, str(self.mul_repr).replace(' ','')])
        return out

    def __str__(self):
        representative = repr(self)
        vhx = self.vhx
        fractional_repr = self.fractional_repr_str
        fraction = str(self.fraction)
        symmetry = self.symmetry_fmt()
        std_out = ' {:2d} {}  {:34} {} {:8} {:50}'.format(self.crossnum, vhx,
                        representative, symmetry, fraction, fractional_repr)
        #std_out = ' {:2d} {}  {:34} {:34} {} {:8} {:50}'.format(self.crossnum, vhx,
        #                representative, repr(self.mirror), symmetry, fraction, fractional_repr)
        return std_out

    def __eq__(self, other):
        if issubclass(other.__class__, Tangle):
            return self.representative == other.representative
            #return self.signature == other.signature and self.fraction == other.fraction
        elif other == None:
            return False
        else:
            raise TypeError
            #return self.fraction == other

    def __gt__(self, other):
        if issubclass(other.__class__, Tangle):
            if self.signature == other.signature:
                if self.fractional_repr == other.fractional_repr:
                    return self.representative > other.representative
                else:
                    return self.fractional_repr > other.fractional_repr
            else:
                return self.signature > other.signature
#        return self.representative > other.representative
        else: 
            raise TypeError
            #return self.fraction > other

    def __hash__(self):
        return hash(self.key)

    def __abs__(self):
        return abs(self.fraction)

    def symmetry_fmt(self):
        s = self.symmetry
        fmt = '{}{}{}{}'.format(
                    'μ' if s['mirror'] else 'η' if s['mirror2'] else '.',
                    'x' if s['x_rot'] else '.',
                    'y' if s['y_rot'] else '.',
                    'z' if s['z_rot'] else '.'
                    )
        return fmt

    def get_symmetry_type(self):
        symmetry = {
            'x_rot': self == self.rotated_x,
            'y_rot': self == self.rotated_y,
            'z_rot': self == self.rotated_z,
            'mirror': self == self.mirror,
            'mirror2': self == self.mirror2
            }
        return symmetry

    def get_equivalent(self):
        equivalent = set([self])
        if not self.symmetry['x_rot']:
            equivalent.add(self.rotated_x)
        if not self.symmetry['y_rot']:
            equivalent.add(self.rotated_y)
        if not self.symmetry['z_rot']:
            equivalent.add(self.rotated_z)
        return equivalent

    def equivalent_fmt(self):
        eqv = copy(self.equivalent)
        text = []
        if self in eqv: eqv.remove(self)
        if eqv:
            text.append('EQ:{}'.format(';'.join([e.representative for e in eqv])))
        return ' '.join(text)

    @staticmethod
    def get_vhx(integral_tangles):
        dict_vhx = {'':'V0', '0':'H0', '1':'X0', '00':'V0', '01':'V0', '10':'X0', '11':'H0'}
        reduced = [num%2 for num in integral_tangles]
        length = len(reduced)
        if length:
            i = 0
            mem = reduced[0] # 0 or 1
            while i < length-2:
                if mem == 0:
                    i += 2
                    mem = reduced[i]
                elif mem == 1:
                    i += 1
                    mem = 1-reduced[i]
            key = str(mem)
            if i == length-2:
                key += str(reduced[i+1])
        else:
            key = ''
        return dict_vhx[key]

    @staticmethod
    def euclid_fmt(sequence, plus_notation=False):
        to_add = ''
        if len(sequence):
            if plus_notation:
                if sequence[-1] == 0 and len(sequence)>1:
                    num = sequence[-2]
                    sequence = sequence[:-2]
                    if num > 0:
                        to_add += '+'*num
                    elif num < 0:
                        to_add += '-'*abs(num)
        return ' '.join([str(x) for x in sequence]) + to_add

    def minus(self, times=1):
        return self.plus(-times)

    def fmt_fractional(self):
        pass

class RationalTangle(Tangle):
    cached = {}
    def __init__(self, key, num, den):
        self.fraction = self.get_fraction(num, den)
        self.key = key
        super().__init__()

    @staticmethod
    def new(num,den):
        if den < 0:
            num *= -1
            den *= -1
        if den == 0:
            num = 1
        key = '{}/{}'.format(str(num), str(den))
        if key in RationalTangle.cached:
            return RationalTangle.cached[key]
        else:
            return RationalTangle(key, num, den)

    def __int__(self):
        assert abs(self.fraction.numerator) == 1
        return self.fraction.numerator * self.fraction.denominator

    def get_sign(self):
        if 0 in [self.fraction.numerator, self.fraction.denominator]:
            return 0
        if self.fraction > 0: return 1
        elif self.fraction < 0 : return -1
        else: raise

#    def get_rank(self):
#        return self.fraction.denominator//abs(self.fraction.numerator)

    def __len__(self):
        return 1

    def __str__(self):
        std_out = super().__str__()
        return '{} {}'.format(std_out, self.equivalent_fmt())

    def get_representative(self):
        res = self.euclid_fmt(self.mul_repr, plus_notation=False)
        return res

    def get_crossnum(self):
        return sum([abs(x) for x in self.mul_repr])

    def get_signature(self):
        return str(abs(self.fraction.numerator))

    def get_fraction(self, num, den):
        if den == 0:
            return OneOverZero()
        else:
            return MyFraction(num, den)

    def rotate_90(self):
        new_num = self.fraction.denominator
        new_den = -self.fraction.numerator
        return self.new(new_num, new_den)

    def rotate_x(self):
        return self

    def rotate_y(self):
        return self

    def rotate_z(self):
        return self

    def __neg__(self):
        return self.new(-self.fraction.numerator, self.fraction.denominator)

    def __invert__(self):
        return self.new(self.fraction.denominator, self.fraction.numerator)

    def get_twisted(self, orig_sign=None):
        num = abs(self.fraction.numerator)
        den = abs(self.fraction.denominator)
        sign = self.sign
        if 0 in [num,den]:
            raise TypeError('0 and 00 tangle cannot be twisted')
        new_den = den-num
        return self.new(num*sign, new_den)

#    def get_twisted(self, orig_sign=None):
#        num = self.fraction.numerator
#        den = self.fraction.denominator
#        sign = num//abs(num)
#        if abs(num) <= 1:
#            new_den = -den 
#        else:
#            times = abs(den)//abs(num)
#            new_den = den - num*(1+2*times)*sign
#        if new_den < 0:
#            new_den, num = -new_den, -num
##        print('twist: {:d}/{:d} --> {:d}/{:d}'.format(num, den, num, new_den))
#        return self.new(num, new_den)

    def get_vhx(self):
        return super().get_vhx(self.mul_repr)

    def untwist(self):
        num = abs(self.fraction.numerator)
        den = abs(self.fraction.denominator)
        if num > den or num == 0:
            return self, 0
        else:
            new_den = den%num
            twists = den//num
            return RationalTangle.new(num*self.sign, new_den), twists*self.sign
        
    def plus(self, times=1):
        new_den = times*self.fraction.numerator + self.fraction.denominator
        return self.new(self.fraction.numerator, new_den)

    def get_fractional_repr(self):
        return self.fraction

    def fmt_fractional(self):
        if self.fraction.numerator == 0:
            return '0'
        if self.fraction.denominator == 0:
            return '∞'
        else:
            if self.sign == 1:
                sign = '+'
            elif self.sign == -1:
                sign = '-'
            return '{}{:d}/{:d}'.format(sign, abs(self.fraction.numerator), abs(self.fraction.denominator))

#    def get_fractional_repr(self):
#        if self.fraction:
#            num = self.fraction.numerator
#            den = self.fraction.denominator
#            assert den > 0, den
#            if num == -0:
#                num = 0
#            rank = abs(den)//abs(num)
#            if abs(num) == 1:
#                rank -= 1
#            minimal_num, minimal_den = self.get_minimal()
#            if minimal_num < 0:
#                rank = rank + 1
#                plus = '-'*rank
#                num, den = abs(minimal_num), abs(minimal_den+minimal_num)
#            else:
#                plus = '+'*rank
#                num, den = minimal_num, minimal_den
#            res = '{:d}/{:d}{}'.format(num,den,plus)
#        elif self.fraction == 0:
#            res = '0'
#        else:
#            res = ''
#        return RationalStr(res)

    # euclid algorithm, get representative name and number of crossings
    def get_mul_repr(self):
        num = abs(self.fraction.numerator)
        den = abs(self.fraction.denominator)
        if num in [0,-0]:
            return (0,)
        elif den in [0,-0]:
            return tuple()
        sign = self.fraction.numerator // abs(self.fraction.numerator)
        tangle = [0]
        while num != den:
            if den > num:
                num, den = den, num
                tangle.append(0)
            num -= den
            tangle[-1] += 1
        tangle[-1] += 1
        if sign == -1:
            tangle = [-x for x in tangle]
        tangle = tangle[::-1]
        return tuple(tangle)


