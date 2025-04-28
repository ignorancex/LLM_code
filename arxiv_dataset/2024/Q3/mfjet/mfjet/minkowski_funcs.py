"""
Collection of codes calculating Minknowski functionals of shapely Geometry objects.

"""
from typing import Union

import numpy as np
import shapely


def calc_mfs(geom):
    """
    Calculate Minkowski functionals of given geometry.

    Parameters
    ----------
    geom   : shapely.Geometry
        input geometry object

    Returns
    -------
    np.array with shape (3,)
        Minkowski functionals MF[k] of given geometry

        * k=0: Euler characteristic
        * k=1: boundary length
        * k=2: area

    """
    return np.array([
        calc_euler(geom),
        calc_length(geom),
        calc_area(geom),
    ])


def calc_area(geom):
    """
    Calculate area of given geometry.

    Parameters
    ----------
    geom   : shapely.Geometry
        input geometry object

    Returns
    -------
    float
        area of given geometry

    """
    return geom.area

def calc_length(geom):
    """
    calculate boundary length of given geometry.

    Parameters
    ----------
    geom   : shapely.Geometry
        input geometry object

    Returns
    -------
    float
        boundary length of given geometry

    """
    boundary = geom.boundary
    if boundary is None:
        raise ValueError("Boundary is undefined. input geom type: {}".format(type(geom)))
    return boundary.length

def calc_euler(geom: Union[shapely.Polygon, shapely.MultiPolygon]):
    """
    calculate Euler characteristic of given (multi)polygon.

    Parameters
    ----------
    geom   : shapely.Polygon or shapely.MultiPolygon
        input (multi)polygon

    Returns
    -------
    float
        Euler characteristic of given (multi)polygon

    """
    if isinstance(geom, shapely.MultiPolygon):
        euler = sum([calc_euler_poly(poly) for poly in geom.geoms])
    elif isinstance(geom, shapely.Polygon):
        euler = calc_euler_poly(geom)
    else:
        raise TypeError("unsupported type: {}".format(type(geom)))
    return euler

def calc_euler_poly(poly):
    """
    calculate Euler characteristic of given polygon.

    Parameters
    ----------
    poly   : shapely.Polygon
        input polygon object

    Returns
    -------
    float
        Euler characteristic of given polygon

    """
    num_exteriors = 1
    num_interiors = len(poly.interiors)
    return num_exteriors - num_interiors
