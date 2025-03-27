##############################################################################
# Make `print` work the same in all versions of Python, set up numpy,
# matplotlib, and use a nicer set of plot parameters:

import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)


##############################################################################
# Import the packages necessary for coordinates

from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
import astropy.coordinates as coord
import astropy.units as u


from astropy.table import Table
import pandas as pd
import sys

import os



##############################################################################
# The first step is to create a new class, which we'll call
# ``Sagittarius`` and make it a subclass of
# `~astropy.coordinates.BaseCoordinateFrame`:


class Sagittarius(coord.BaseCoordinateFrame):
    """
    A Heliocentric spherical coordinate system defined by the orbit
    of the Sagittarius dwarf galaxy, as described in
        https://ui.adsabs.harvard.edu/abs/2003ApJ...599.1082M
    and further explained in
        https://www.stsci.edu/~dlaw/Sgr/.

    Parameters
    ----------
    representation : `~astropy.coordinates.BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
    Lambda : `~astropy.coordinates.Angle`, optional, must be keyword
        The longitude-like angle corresponding to Sagittarius' orbit.
    Beta : `~astropy.coordinates.Angle`, optional, must be keyword
        The latitude-like angle corresponding to Sagittarius' orbit.
    distance : `~astropy.units.Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.
    pm_Lambda_cosBeta : `~astropy.units.Quantity`, optional, must be keyword
        The proper motion along the stream in ``Lambda`` (including the
        ``cos(Beta)`` factor) for this object (``pm_Beta`` must also be given).
    pm_Beta : `~astropy.units.Quantity`, optional, must be keyword
        The proper motion in Declination for this object (``pm_ra_cosdec`` must
        also be given).
    radial_velocity : `~astropy.units.Quantity`, optional, keyword-only
        The radial velocity of this object.

    """

    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping('lon', 'Lambda'),
            coord.RepresentationMapping('lat', 'Beta'),
            coord.RepresentationMapping('distance', 'distance')]
    }


##############################################################################
# Breaking this down line-by-line, we define the class as a subclass of
# `~astropy.coordinates.BaseCoordinateFrame`. Then we include a descriptive
# docstring.  The final lines are class-level attributes that specify the
# default representation for the data, default differential for the velocity
# information, and mappings from the attribute names used by representation
# objects to the names that are to be used by the ``Sagittarius`` frame. In this
# case we override the names in the spherical representations but don't do
# anything with other representations like cartesian or cylindrical.
#
# Next we have to define the transformation from this coordinate system to some
# other built-in coordinate system; we will use Galactic coordinates. We can do
# this by defining functions that return transformation matrices, or by simply
# defining a function that accepts a coordinate and returns a new coordinate in
# the new system. Because the transformation to the Sagittarius coordinate
# system is just a spherical rotation from Galactic coordinates, we'll just
# define a function that returns this matrix. We'll start by constructing the
# transformation matrix using pre-determined Euler angles and the
# ``rotation_matrix`` helper function:

SGR_PHI = (180 + 3.75) * u.degree # Euler angles (from Law & Majewski 2010)
SGR_THETA = (90 - 13.46) * u.degree
SGR_PSI = (180 + 14.111534) * u.degree

# Generate the rotation matrix using the x-convention (see Goldstein)
D = rotation_matrix(SGR_PHI, "z")
C = rotation_matrix(SGR_THETA, "x")
B = rotation_matrix(SGR_PSI, "z")
A = np.diag([1.,1.,-1.])
SGR_MATRIX = matrix_product(A, B, C, D)


##############################################################################
# Since we already constructed the transformation (rotation) matrix above, and
# the inverse of a rotation matrix is just its transpose, the required
# transformation functions are very simple:

@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.Galactic, Sagittarius)
def galactic_to_sgr():
    """ Compute the transformation matrix from Galactic spherical to
        heliocentric Sgr coordinates.
    """
    return SGR_MATRIX


##############################################################################
# The decorator ``@frame_transform_graph.transform(coord.StaticMatrixTransform,
# coord.Galactic, Sagittarius)``  registers this function on the
# ``frame_transform_graph`` as a coordinate transformation. Inside the function,
# we simply return the previously defined rotation matrix.
#
# We then register the inverse transformation by using the transpose of the
# rotation matrix (which is faster to compute than the inverse):

@frame_transform_graph.transform(coord.StaticMatrixTransform, Sagittarius, coord.Galactic)
def sgr_to_galactic():
    """ Compute the transformation matrix from heliocentric Sgr coordinates to
        spherical Galactic.
    """
    return matrix_transpose(SGR_MATRIX)





##############################################################################
# Now that we've registered these transformations between ``Sagittarius`` and
# `~astropy.coordinates.Galactic`, we can transform between *any* coordinate
# system and ``Sagittarius`` (as long as the other system has a path to
# transform to `~astropy.coordinates.Galactic`). For example, to transform from
# ICRS coordinates to ``Sagittarius``, we would do:

'''
icrs = coord.SkyCoord(280.161732*u.degree, 11.91934*u.degree, frame='icrs')
sgr = icrs.transform_to(Sagittarius)
print(sgr)

##############################################################################
# Or, to transform from the ``Sagittarius`` frame to ICRS coordinates (in this
# case, a line along the ``Sagittarius`` x-y plane):

sgr = coord.SkyCoord(Lambda=np.linspace(0, 2*np.pi, 128)*u.radian,
                     Beta=np.zeros(128)*u.radian, frame='sagittarius')
icrs = sgr.transform_to(coord.ICRS)
print(icrs)
'''




# file_name = '/home/gmedina/Downloads/DESI_RRLs_git/DESI-iron_Gaiadr3_v0.1.csv'

# df = pd.read_csv(file_name)

# print(df.columns)




# icrs = coord.SkyCoord(df['ra']*u.degree, df['dec']*u.degree, frame='icrs')
# sgr = icrs.transform_to(Sagittarius)
# print(sgr)

# print(sgr.Lambda.value, sgr.Beta.value)

# # Lambda: The longitude-like angle corresponding to Sagittarius' orbit.
# # Beta : The latitude-like angle corresponding to Sagittarius' orbit.
# df['stream_longitude_like_Lambda_sgr'] = sgr.Lambda.value
# df['stream_latitude_like_Beta_sgr'] = sgr.Beta.value

# print(df[['source_id','ra','dec','stream_longitude_like_Lambda_sgr','stream_latitude_like_Beta_sgr']])

# # ---------------------------------------------------------------------------------- #

# newname = file_name.replace('.csv','_wSgr.csv')
# print('Saving ', newname, '?')
# print('Save: ', save)

	
# df.to_csv(newname)

# # ---------------------------------------------------------------------------------- #





