import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix


class WMATRIX:
    def __init__(self,
                 geometry,
                 pops,
                 sigma=10000
                 ):
        self.centroids = [poly.centroid for poly in geometry]
        csx = np.array([c.x for c in self.centroids])
        csy = np.array([c.y for c in self.centroids])
        self.network = np.array([csx, csy]).T
        self.length = self.network.shape[0]
        self.pops = pops
        self.dmat = distance_matrix(self.network, self.network)
        self.sigma = sigma

    def make_exposure_mat(self):
        """[Summary]
        :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
        :type [ParamName]: [ParamType](, optional)
        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: [ReturnDescription]
        :rtype: [ReturnType]
        """
        pops_form = np.array(self.pops)[np.newaxis, :]
        w_matrix = pops_form * np.exp(- self.dmat**2 / (2 * self.sigma**2))
        row_sums = w_matrix.sum(axis=1)
        w_matrix = w_matrix/row_sums[:, np.newaxis]
        w_matrix = csr_matrix(w_matrix)
        return w_matrix
