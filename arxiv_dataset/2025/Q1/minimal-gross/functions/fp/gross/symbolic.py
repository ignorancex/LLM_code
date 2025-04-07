from sage.all import *

class SymbolicGross:
    r"""
    Given a family of supersingular primes p = a mod b > N, and D1, then
    symbolically computes the reduced Gram matrix of the Gross lattice. 
    """
    def __init__(self, a, b):
        r"""
        Initialize the SymbolicGross class with parameters a and b.
        p = a mod b
        """
        assert isinstance(a, (int, Integer)) and isinstance(b, (int, Integer)) and 0 <= a <= b, "a and b must be integers with 0 <= a <= b"
        self.a = a
        self.b = b
        self.p = var('p')

    def gross(self, D1, N):
        r"""
        The unique reduced Gram matrix for given p = a mod b, D1 and p>N.
        """
        assert isinstance(D1, (int, Integer)) and D1 > 0, "D1 must be a positive integer"
        assert isinstance(N, (int, Integer)) and N > 0, "N must be a positive integer"
        sols = []
        for x, D2 in self.beta12(D1):
            for y, D3 in self.beta13(D1, N):
                d2 = D2.subs(self.p==N)
                d3 = D3.subs(self.p==N)
                if d2 <= d3:
                    for zsol in self.beta123(D1, D2, D3, x, y):
                        z = zsol.rhs()
                        if self.beta23(D2, D3, z):
                            sols.append(Matrix([[D1, x, y], [x, D2, z], [y, z, D3]]))
        if len(sols) == 1:
            return sols[0]
        else:
            return False

    def beta12(self, D1):
        r"""
        Step 1: D1*D2-x^2 = 4p such that 0 <= x <= D1/2 and D2 is 0,3 mod 4.
        """
        R = PolynomialRing(ZZ, 'k')
        k = R.gen()
        c = floor(D1/2)
        x = 0
        sols = []
        while x < c+1:
            f = 4*(self.a + self.b*k) + x**2
            if f % D1 == 0:
                d2 = f // D1  # use integer division to avoid floating point errors
                if d2 % 4 == 0 or d2 % 4 == 3:
                    D2 = (4*self.p + x**2) / D1
                    sols.append((x, D2))  # multiple possible
            x += 1
        return sols

    def beta13(self, D1, N):
        r"""
        Step 2: D1*D3-y^2 = 4np such that, n>=1, 0 <= y <= D1/2 and D3 is 0,3 mod 4 for p >= max(37,N_E) and j != 0 
        p <= D3 <= 8p/7 + 7/4 
        pD1 - D1^2/4 <= 4np <= 8pD1/7 + 7D1/4 for p >= 37
        D1/4 - D1^2/16*p <= n <= 2*D1/7 + 7*D1/16*p for p >= 37
        """
        assert self.a != 2 or self.b != 3 or D1 != 3, "j-invariant can't be 0"        
        p_bound = max(37, N)
        lower = ceil(D1/4 - D1**2/(16*p_bound))
        upper = floor((2*D1)/7 + (7*D1)/(16*p_bound))
        n_values = [i for i in range(lower, upper+1)]  # multiple possible
        
        sols = {n: [] for n in n_values}  # multiple possible
        R = PolynomialRing(ZZ, 'k')
        k = R.gen()
        c = floor(D1/2)
        y = 0
        for n in n_values:
            while y < c+1:
                f = 4*n*(self.a + self.b*k) + y**2
                if f % D1 == 0:
                    d3 = f // D1  # use integer division to avoid floating point errors
                    if d3 % 4 == 0 or d3 % 4 == 3:
                        D3 = (4*n*self.p + y**2) / D1
                        sols[n].append((y, D3))  # multiple possible
                y += 1
        # remove n_values which didn't produce any solution
        sols = {key: value for key, value in sols.items() if value}
        if len(sols) == 1:
            return next(iter(sols.values()))
        else:
            return False

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
        R = PolynomialRing(ZZ, 'k')
        k = R.gen()
        val = val.subs(self.p==self.a + self.b*k)
        val = val.simplify_rational()
        if val in R:
            return True
        else:
            return False



