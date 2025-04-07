from sage.all import *


class NumericGross:
    r"""
    Given a supersingular primes p>=37 and D1, then
    numerically computes the reduced Gram matrix of the Gross lattice. 
    """

    def __init__(self, p):
        r"""
        Initialize the NumericGross class with prime p.
        """
        assert is_pseudoprime(p), "p must be a pseudo prime"
        self.p = p

    def gross(self, D1):
        r"""
        All the reduced Gram matrix for given p and D1.
        """
        assert isinstance(D1, (int, Integer)) and D1 > 0, "D1 must be a positive integer"
        sols = []
        for x, D2 in self.beta12(D1):
            bound37 = self.beta13(D1)
            if isinstance(bound37, list):
                for y, D3 in bound37:
                    if D2 <= D3:
                        for zsol in self.beta123(D1, D2, D3, x, y):
                            z = zsol.rhs()
                            if self.beta23(D2, D3, z):
                                sols.append(Matrix(ZZ, [[D1, x, y], [x, D2, z], [y, z, D3]]))
        return sols # multiple possible

    def beta12(self, D1):
        r"""
        Step 1: D1*D2-x^2 = 4p such that 0 <= x <= D1/2 and D2 is 0,3 mod 4.
        """
        c = floor(D1/2)
        x = 0
        sols = []
        while x < c+1:
            f = 4*(self.p) + x**2
            if f % D1 == 0:
                d2 = f // D1  # use integer division to avoid floating point errors
                if d2 % 4 == 0 or d2 % 4 == 3:
                    sols.append((x, d2))  # multiple possible
            x += 1
        return sols

    def beta13(self, D1):
        r"""
        Step 2: D1*D3-y^2 = 4np such that, n>=1, 0 <= y <= D1/2 and D3 is 0,3 mod 4 for p >= max(37,N_E) and j != 0 
        p <= D3 <= 8p/7 + 7/4 
        pD1 - D1^2/4 <= 4np <= 8pD1/7 + 7D1/4 for p >= 37
        D1/4 - D1^2/16*p <= n <= 2*D1/7 + 7*D1/16*p for p >= 37
        """
        assert self.p%3 != 2 or D1 != 3, "j-invariant can't be 0"        
        p_bound = max(37, self.p)
        lower = ceil(D1/4 - D1**2/(16*p_bound))
        upper = floor((2*D1)/7 + (7*D1)/(16*p_bound))
        n_values = [i for i in range(lower, upper+1)]  # multiple possible
        
        sols = {n: [] for n in n_values}  # multiple possible
        c = floor(D1/2)
        y = 0
        for n in n_values:
            while y < c+1:
                f = 4*n*(self.p) + y**2
                if f % D1 == 0:
                    d3 = f // D1  # use integer division to avoid floating point errors
                    if d3 % 4 == 0 or d3 % 4 == 3:
                        sols[n].append((y, d3))  # multiple possible
                y += 1
        # remove n_values which didn't produce any solution
        sols = {key: value for key, value in sols.items() if value}
        if len(sols) == 1:
            return next(iter(sols.values()))
        else:
            return len(sols) #False

    def beta123(self, D1, D2, D3, x, y):
        r"""
        Step 3: D_1*D_2*D_3 + 2xyz - D_1*z^2 - D_2*y^2 - D_3*x^2 = 4p^2, solve for z.
        """
        ## Az^2 + Bz + C
        A = D1
        B = -2*x*y
        C = 4*self.p**2 + D3*x**2 + D2*y**2 - D1*D2*D3
        z = var('z')
        sols = (A*z**2 + B*z + C).solve(z)  # multiple possible
        return sols

    def beta23(self, D2, D3, z):
        r"""
        Step 4: check that D2 < D3 and D2*D3 - z^2 = 4*m*p for some m>=1.
        """
        val = (D2*D3 - z**2) / (4*self.p)
        if val in IntegerRing():
            return True
        else:
            return False



