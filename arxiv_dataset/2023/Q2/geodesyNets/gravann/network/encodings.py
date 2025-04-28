from abc import ABC, abstractmethod

import numpy as np
import torch


class Encoding(ABC):
    """Abstract encoding
    """

    @abstractmethod
    def __call__(self, sp):
        pass


class DirectionalEncoding(Encoding):
    """ Directional encoding
    x = [x,y,z] is encoded as [ix, iy, iz, r]
    """

    def __init__(self):
        self.dim = 4
        self.name = "directional_encoding"

    # sp: sampled points

    def __call__(self, sp):
        unit = sp / torch.norm(sp, dim=1).view(-1, 1)
        return torch.cat((unit, torch.norm(sp, dim=1).view(-1, 1)), dim=1)


class PositionalEncoding(Encoding):
    """ Positional encoding
    x = [x,y,z] is encoded as [sin(pi x), sin(pi y), sin(pi z), cos(pi x), cos(pi y), cos(pi z), sin(2 pi x), ....]
    """

    def __init__(self, N):
        self.dim = 6 * N
        self.name = "positional_encoding_" + str(N)

    def __call__(self, sp):
        retval = torch.cat((torch.sin(np.pi * sp).view(-1, 3), torch.cos(np.pi * sp).view(-1, 3)), dim=1)
        for i in range(1, self.dim // 6):
            retval = torch.cat(
                (retval, torch.sin(2 ** i * np.pi * sp).view(-1, 3), torch.cos(2 ** i * np.pi * sp).view(-1, 3)), dim=1)
        return retval


class DirectEncoding(Encoding):
    """Direct encoding:
    x = [x,y,z] is encoded as [x,y,z]
    """

    def __init__(self):
        self.dim = 3
        self.name = "direct_encoding"

    def __call__(self, sp):
        return sp


class SphericalCoordinates(Encoding):
    """Spherical encoding:
    x = [x,y,z] is encoded as [r,phi,theta] (i.e. spherical coordinates)
    """

    def __init__(self):
        self.dim = 3
        self.name = "spherical_coordinates"

    def __call__(self, sp):
        phi = torch.atan2(sp[:, 1], sp[:, 0]) / np.pi
        r = torch.norm(sp, dim=1)
        theta = torch.div(sp[:, 2], r)
        return torch.cat((r.view(-1, 1), phi.view(-1, 1), theta.view(-1, 1)), dim=1)


def get_encoding(encoding_name: str) -> Encoding:
    """Returns an encoding for a given encoding name

    Args:
        encoding_name: the name of the encoding as str

    Returns:
        an encoding

    """
    try:
        return {
            "directional_encoding": DirectionalEncoding,
            "positional_encoding": PositionalEncoding,
            "direct_encoding": DirectEncoding,
            "spherical_coordinates": SphericalCoordinates
        }[encoding_name]()
    except KeyError:
        raise NotImplementedError(f"The requested encoding {encoding_name} does not exist!")
