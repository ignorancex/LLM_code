from __future__ import print_function, division

import numpy as np
from nbodykit.source.mesh.field import FieldMesh

class Smoother(object):
    """
    Class to apply smoothing to field.
    """
    def __init__(self):
        raise NotImplementedError

    def get_smoothing_kernel_of_Nth_iteration(self, N):
        raise NotImplementedError


class GaussianSmoother(Smoother):
    """
    Apply Gaussian smoothing to field.
    """
    def __init__(
        self,
        R=20.0, # smoothing scale R in Mpc/h
        name='Gaussian'
        ):
        self.R = R
        self.name = name

    def get_smoothing_kernel(self):
        R = self.R
        if R is None or R==0.:
            def kernel_fcn(k3vec, val):
                return val
        else:
            def kernel_fcn(k3vec, val):
                k2 = sum(ki**2 for ki in k3vec)  # |\vk|^2 on the mesh
                return np.exp(- 0.5 * R**2 * k2) * val
        return kernel_fcn

    def apply_smoothing(self, meshsource):
        # make a copy
        out = FieldMesh(meshsource.compute(mode='complex'))
        kernel_fcn = self.get_smoothing_kernel()
        out = out.apply(kernel_fcn, kind='wavenumber', mode='complex')
        return out

    def to_dict(self):
        return dict(
            R=self.R, name=self.name)

    def __str__(self):
        return json.dumps(self.to_dict())

    def __repr__(self):
        return self.__str__()
