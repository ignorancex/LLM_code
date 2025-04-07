from sage.all import magma

def get_maximal_order(a, b, d):
    r"""
    This is essentially a SageMath wrapper for Magma's Quaternion Algebra
    functions like ConjugacyClasses.
    
    It constructs a quaternion algebra with i^2=-a and j^2=-b, and then gives
    the maximal order that contains a Gross lattice with the first successive 
    minima, D1, equal to d.

    If d = 0 then this function outputs matrices for all orders.
    """
    results = []
    m = magma
    B = m.QuaternionAlgebra(m.RationalField(), a, b)
    results.append((a, b)) # quaternion algebra
    MaxO = B.MaximalOrder()
    results.append(MaxO.Basis()) # an endomorphism ring whose j-invariant is known
    
    for O in MaxO.ConjugacyClasses():
        # maximal order
        basis = [B(bb) for bb in O.ReducedBasis()]
        if basis[0] == 1:
            # Gross lattice
            OTbasis = [2*aa - aa.Trace() for aa in basis[1:]]
            assert len(OTbasis) == 3, "check a and b"
            n = 3
            # Gram matrix
            G = m.Matrix([[m.RationalField()((OTbasis[i] * OTbasis[j].Conjugate()).Trace()) for j in range(n)] for i in range(n)])
            G = 1/2 * G 
            Gred = G.LLLGram(Delta=0.99, Eta=0.51)
            Gred = m.Matrix([[m.Integers()(Gred[i][j]) for j in range(1, n+1)] for i in range(1, n+1)])  # Convert entries to Magma integers
            if Gred[1][1] == d:
                results.append((basis, Gred)) # the maximal order and Gram matrix of Gross lattice.
            elif d == 0:
                results.append((basis, Gred)) # all the maximal orders and Gram matrix of Gross lattice.
    
    return results
