from sage.all import *

class IbukiyamaMaximalOrder:
    @classmethod
    def get_matrix(cls, jj, p):
        r"""
        wrapper function.
        """
        # ibukiyama order
        B, order_base, q, r, order_type = cls.maximal_order(jj, p)
        OT_base = cls.gross_lattice(B, order_base)
        G = cls.gram_matrix(OT_base)
        U = G.LLL_gram()
        Gred = U.transpose() * G * U
        Mred = cls.minimal_basis_matrix(Gred)
        return [order_type, (q, r), order_base, OT_base, Gred, Mred]

    @classmethod
    def maximal_order(cls, jj, p):
        r"""
        Computing the Ibukiyama maximal order O(q,r) and O'(q,r) of supersingular
        elliptic curves over Fp.

        This code is based on Algorithms 1 and 2 in the Appendix of:
        S. Li, Y. Ouyang, and Z. Xu, "Endomorphism rings of supersingular elliptic curves over Fp," Finite Fields and Their Applications, vol. 62. Elsevier BV, p. 101619, Feb. 2020. 
        https://doi.org/10.1016/j.ffa.2019.101619 
        https://zbmath.org/1451.11054
        https://mathscinet.ams.org/mathscinet-getitem?mr=4038249
        """
        assert p>3 and is_pseudoprime(p) == True, "must enter a prime number greater than 3."

        F = GF(p)
        
        assert EllipticCurve(j=F(jj)).is_supersingular() == True, "must enter a supersingular prime for the given j-invariant."

        for q in primes(ceil(p*(log(p,2))**2)):
            q = ZZ(q)
            if kronecker(p, q) == -1 and q % 8 == 3:

                # End(E(jj)) = O(q,r) -> order_type = True
                H_1 = hilbert_class_polynomial(-q)
                root1 = []
                Root1 = H_1.change_ring(F).roots()
                for m1 in Root1:
                    root1.append(m1[0])

                H_2 = hilbert_class_polynomial(-4 * p)
                root2 = []
                Root2 = H_2.change_ring(F).roots()
                for m2 in Root2:
                    root2.append(m2[0])

                K = GF(q)
                r = ZZ(K(-p).sqrt())
                H_3 = hilbert_class_polynomial(-4 * (r**2 + p) / q)
                root3 = []
                Root3 = H_3.change_ring(F).roots()
                for m3 in Root3:
                    root3.append(m3[0])

                A = (Set(root1).intersection(Set(root2))).intersection(Set(root3))
                if A.cardinality() == 1 and A == Set([F(jj)]):
                    B = QuaternionAlgebra(QQ, -q, -p)
                    i, j, k = B.gens()
                    return [B, [1, (1+i)/2, (j+k)/2, (r*i-k)/q], q, r, True]

                # End(E(jj)) = O'(q,r) -> order_type = False
                if p % 4 == 3:
                    H_4 = hilbert_class_polynomial(-4 * q)
                    root4 = []
                    Root4 = H_4.change_ring(F).roots()
                    for m4 in Root4:
                        root4.append(m4[0])

                    H_5 = hilbert_class_polynomial(-p)
                    root5 = []
                    Root5 = H_5.change_ring(F).roots()
                    for m5 in Root5:
                        root5.append(m5[0])

                    rprime_list = K(-p).sqrt(extend=False, all=True)
                    for rr in rprime_list:
                        integerr = ZZ(rr) ** 2 + ZZ(p)
                        if integerr % 4 == 0:
                            rprime = ZZ(rr)
                            H_6 = hilbert_class_polynomial(-(integerr) / q)
                            break
                    root6 = []
                    Root6 = H_6.change_ring(F).roots()
                    for m6 in Root6:
                        root6.append(m6[0])

                    Aprime = (Set(root4).intersection(Set(root5))).intersection(Set(root6))
                    if Aprime.cardinality() == 1 and Aprime == Set([F(jj)]):
                        B = QuaternionAlgebra(QQ, -q, -p)
                        i, j, k = B.gens()
                        return [B, [1, (1+j)/2, i, (rprime*i-k)/(2*q)], q, rprime, False]
        # if no suitable q found
        return False
    
    @classmethod
    def gross_lattice(cls, B, order_base):
        r"""
        Comptutes the Gross lattice inside the maximal order.

        This code is based on SageMath's implementation of ternary_quadratic_form()
        function for quaternion orders.

        https://doc.sagemath.org/html/en/reference/quat_algebras/sage/algebras/quatalg/quaternion_algebra.html#sage.algebras.quatalg.quaternion_algebra.QuaternionOrder.ternary_quadratic_form
        """
        O = B.quaternion_order(order_base)
        twoO = O.free_module().scale(2)
        Z = twoO.span([B(1).coefficient_tuple()], ZZ)
        S = twoO + Z
        v = [b.reduced_trace() for b in B.basis()]
        M = matrix(QQ, 4, 1, v)
        tr0 = M.kernel()
        S0 = tr0.intersection(S)
        OT_base = [B(a) for a in S0.basis()]
        return OT_base # [2*aa - aa.reduced_trace() for aa in order_base[1:]]


    @classmethod
    def gram_matrix(cls, base):
        r"""
        The entries of Gram matrix are the inner-product values of lattice basis.

        Here the inner-product is 
        <x,y> =  1/2 * x.pair(y) = 1/2*(x.conjugate()*y).reduced_trace()
        """
        G = matrix(ZZ, [[(x.pair(y))/2 for x in base] for y in base])
        return G
    
    @classmethod
    def minimal_basis_matrix(cls, M):
        r"""
        Obtaining Gram matrix of the minimal Gross lattice from LLL-reduced matrix.

        Rearrange LLL-reduced Gram matrix such that D1 <= D2 <= D3 and x,y>0 (flip angles accordingly).
        """
        # Create a deep copy of the input matrix M
        Gred = deepcopy(M)
        
        # if D2 > D3 then swap D2 and D3, and x and y 
        if Gred[1, 1] > Gred[2, 2]:
            # Swap D2 and D3
            Gred[1, 1], Gred[2, 2] = Gred[2, 2], Gred[1, 1]
            # Swap x and y, 
            Gred[0, 1], Gred[0, 2] = Gred[0, 2], Gred[0, 1]
            Gred[1, 0], Gred[2, 0] = Gred[2, 0], Gred[1, 0]
        
        # make sure x and y are positive.
        if Gred[0, 1] < 0 and Gred[0, 2] >= 0:
            Gred[0, 1] = -Gred[0, 1]
            Gred[1, 0] = -Gred[1, 0]
            Gred[1, 2] = -Gred[1, 2]
            Gred[2, 1] = -Gred[2, 1]
        elif Gred[0, 1] >= 0 and Gred[0, 2] < 0:
            Gred[0, 2] = -Gred[0, 2]
            Gred[2, 0] = -Gred[2, 0]
            Gred[1, 2] = -Gred[1, 2]
            Gred[2, 1] = -Gred[2, 1]
        elif Gred[0, 1] < 0 and Gred[0, 2] < 0:
            Gred[0, 1] = -Gred[0, 1]
            Gred[1, 0] = -Gred[1, 0]
            Gred[0, 2] = -Gred[0, 2]
            Gred[2, 0] = -Gred[2, 0]
        
        return Gred