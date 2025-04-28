# Use functions dealing with angles, deprojections, etc.

import math

RADIAN_TO_DEGREE = 180.0 / math.pi
DEGREE_TO_RADIAN = math.pi / 180.0


def RectifyPA( angle: float, maxAngle: float ) -> float:
    """Convert angle to lie between [0,maxAngle) degrees.
    (i.e., angle can be >=0, but must be < maxAngle)

        For maxAngle = 360, this just does standard modular arithmetic
        on angles, mapping the angle into [0,360]
        For maxAngle = 180, this maps the angle into [0,180), assuming
        m = 2 symmetry
        For max = 90, this assumes m=4 symmetry (so -30 --> 60, which
        may not be what you are expecting)
    """
    # map angle into [0,360], in case it's < 0 or > 360 to start with
    angle = angle % 360
    while (angle >= maxAngle):
        angle -= maxAngle
    return angle


def projectpa( deltaPA: float, inclination: float ) -> float:
    """Function to calculate a projected position angle, given an input
    (unprojected) position angle.  Position angles are relative to
    disk line-of-nodes."""

    deltaPA_rad = deltaPA * DEGREE_TO_RADIAN
    i_rad = inclination * DEGREE_TO_RADIAN
    deltaPA_proj = math.atan( math.tan(deltaPA_rad) * math.cos(i_rad) )
    return ( deltaPA_proj * RADIAN_TO_DEGREE )


def deprojectr( deltaPA: float, inclination: float, r: float ) -> float:
    """Function to calculate a deprojected length, given an input
    observed position angle (*relative to disk line-of-nodes*, *not*
    straight position angle east of north!) and inclination, both in
    degrees, and an input observed (projected) length r.
    Returns the deprojected length."""

    deltaPA_rad = deltaPA * DEGREE_TO_RADIAN
    i_rad = inclination * DEGREE_TO_RADIAN
    cosi = math.cos(i_rad)
    sindp = math.sin(deltaPA_rad)
    cosdp = math.cos(deltaPA_rad)
    scale = math.sqrt( (sindp*sindp)/(cosi*cosi) + cosdp*cosdp )
    return ( scale * r )


def deprojectpa( deltaPA: float, inclination: float ) -> float:
    """Function to calculate a deprojected position angle, given an input
    observed position angle (*relative to disk line-of-nodes*, *not*
    straight position angle east of north!) and an input inclination, both
    in degrees.  Returns the deprojected position angle, relative to disk
    line-of-nodes, in degrees."""

    deltaPA_rad = deltaPA * DEGREE_TO_RADIAN
    i_rad = inclination * DEGREE_TO_RADIAN
    deltaPA_deproj = math.atan( math.tan(deltaPA_rad) / math.cos(i_rad) )
    return ( deltaPA_deproj * RADIAN_TO_DEGREE )


def deprojectpa_abs( obsPA: float, diskPA: float, inclination: float, 
                    symmetric=True ) -> float:
    """Function to calculate deprojected PA on the sky of a structure, as
    if the galaxy were rotated to face-on orientation.
        obsPA = observed (projected) PA of structure
        diskPA = major axis (line of nodes) of disk
        inclination = inclination of disk (0 = face-on).
    If symmetric=True, then values > 180 are converted to the equivalent
    0--180 values."""

    if (inclination == 0.0):
        dPA = obsPA
    else:
        deltaPA = obsPA - diskPA
        deltaPA_dp = deprojectpa(deltaPA, inclination)
        dPA = deltaPA_dp + diskPA
    if symmetric:
        dPA = RectifyPA(dPA, 180.0)
    return dPA

def minoraxis( structurePA: float, diskPA: float, inclination=0.0 ) -> float:
    """Function to determine the PA of the minor axis of a given structure,
    optionally including the appropriate projection effects if the galaxy is
    inclined.  Returned value is constrained to lie in range 0 <= x < 180."""

    if (inclination == 0.0):
        minoraxis_proj = structurePA + 90.0
    else:
        structurePA_inplane = deprojectpa_abs( structurePA, diskPA, inclination )
        minorPA_inplane = RectifyPA(structurePA_inplane + 90.0, 180.0)
        minoraxis_proj = projectpa(minorPA_inplane - diskPA, inclination)
        minoraxis_proj += diskPA

    return RectifyPA(minoraxis_proj, 180.0)


