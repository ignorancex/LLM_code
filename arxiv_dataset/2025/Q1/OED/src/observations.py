import numpy as np

from src.multiplier import FourierMultiplier

class Observation(FourierMultiplier):
    @property
    def OstarO(self):
        O = self.matrix
        return np.einsum('ki, kj->ij', O.conjugate(), O)

    
class PointObservation(Observation):
    """A point obeservation, at arbitrary location not necessarily on the
    computational grid. Calculations carried out in the frequency
    domain.
    """

    def __init__(self,
                 measurements=[],
                 **kwargs):
        super().__init__(**kwargs, size=len(measurements))
        self.measurements = np.array(measurements)
        self.multiplier = self.normalized_block(self.measurements).T

    def __str__(self):
        return 'Point observations at ' + ', '.join([f'{me:.4f}' for me in self.measurements])

    def _matvec(self, v):
        v_hat = self.to_freq_domain(v)
        return np.einsum('ij, j -> i', self.multiplier, v_hat)

    def _matmat(self, M):
        M_hat = self.to_freq_domain(M, axis=0)
        return np.einsum('ij, jk -> ik', self.multiplier, M_hat)

    # @property
    # def matrix(self):
    #     return self.to_freq_domain_from_right(self.multiplier)

    def eigenvalues(self):
        O = self.multiplier
        return np.linalg.eigh(O @ O.conjugate().T)[0] * self.h

