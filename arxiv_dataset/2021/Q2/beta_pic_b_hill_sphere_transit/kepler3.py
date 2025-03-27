""" 
This module contains several two-body orbital dynamics algorithms

Intended as a replacement for kepler.py by using the astropy units
package where possible.
"""

import numpy as np
import astropy.units as u
import astropy.constants as c

@u.quantity_input
def kepler_solve(e, M:u.rad, derror=1e-5):
    """Kepler's equation solver using Newtons's method. Slow but robust.

    Args:
        e   eccentricity
        M   Mean anomaly (2pi t / P) since periapse

    Return:
        E   eccentric anomaly
        v   True anomaly

    Written:
        Matthew Kenworthy, 2010

    Text below taken from Terry R. McConnell's code
    at http://barnyard.syr.edu/quickies/kepler.c

    Kepler's equation is the transcendental equation

    E = M + e sin(E)

    where M is a given angle (in radians) and e is a number in the
    range [0,1). Solution of Kepler's equation is a key step in
    determining the position of a planet or satellite in its
    orbit. 

    Here e is the eccentricity of the elliptical orbit and M is
    the "Mean anomaly." M is a fictitious angle that increases at a
    uniform rate from 0 at perihelion to 2Pi in one orbital period. 

    E is an angle at the center of the ellipse. It is somewhat
    difficult to describe in words its precise geometric meaning --
    see any standard text on dynamical or positional astronomy, e.g.,
    W.E. Smart, Spherical Astronomy. Suffice it to say that E
    determines the angle v at the focus between the radius vector to
    the planet and the radius vector to the perihelion, the "true
    anomaly" (the ultimate angle of interest.) Thus, Kepler's equation
    provides the link between the "date", as measured by M, and the
    angular position of the planet on its orbit.

    The following routine implements the binary search algorithm due
    to Roger Sinnott, Sky and Telescope, Vol 70, page 159 (August 1985.)
    It is not the fastest algorithm, but it is completely reliable. 

    you generate the Mean anomaly M convert to the Eccentric anomaly E
    through Kepler and then convert to the true anomaly v

    cos v = (cos E - e) / (1 - e.cos E)
    """

    #   solving  E - e sin(E) - M = 0
    scale = np.pi / 4.

    # first guess of E as M
    E = M
    
    # calculate the residual error
    R = E - (e * np.sin(E))*u.rad

    while True:

        if np.allclose(R.value, M.value, rtol=derror):
            break

        # where M is greater than R, add scale, otherwise subtract
        # scale

        sign = (M > R)

        # sign is 0 or 1
        E = E - (scale * (1 - (2*sign)))*u.rad

        R = E - (e * np.sin(E))*u.rad
        scale = scale / 2.0

    # calaculate the true anomaly from the eccentric anomaly
    # http://en.wikipedia.org/wiki/True_anomaly
    v = 2. * np.arctan2(np.sqrt(1+e) * np.sin(E/2) , np.sqrt(1-e) * np.cos(E/2))

    return E, v

@u.quantity_input
def euler(anode:u.rad, omega:u.rad, i:u.rad):
    """
    Build a 3D Euler rotation matrix of orbital elements

    Args:
        omega (float): the longitude of the periastron (u.angle)
        anode (float): ascending node angle (u.angle)
        i (float): inclination of the orbit (u.angle)

    Returns:
        Mat:   (3x3) rotation matrix 

    Written:
        Matthew Kenworthy, 2017
    
    Taken from "Lecture Notes on Basic Celestial Mechanics" by Sergei
    A. Klioner (2011) celmech.pdf page 15
    """

    can  = np.cos(anode)
    san  = np.sin(anode)
    com  = np.cos(omega)
    som  = np.sin(omega)
    ci   = np.cos(i)
    si   = np.sin(i)

    e1 =  can*com - san*ci*som
    e2 = -can*som - san*ci*com
    e3 =  san*si
    e4 =  san*com + can*ci*som
    e5 = -san*som + can*ci*com
    e6 = -can*si
    e7 =  si*som
    e8 =  si*com
    e9 =  ci

    Mat = np.array([[e1, e2, e3],
                    [e4, e5, e6],
                    [e7, e8, e9]])

    return(Mat)

@u.quantity_input
def kep3d(epoch:u.year, P:u.year, tperi:u.year, a, e, inc:u.deg, omega:u.deg, anode:u.deg, derror=1e-6):
    """
    Calculate the position and velocity of an orbiting body

    Given the Kepler elements for the secondary about the primary
    and in the coordinate frame where the primary is at the origin

    Args:
        epoch (np.array):  epochs to evaluate (u.time)
        P (np.array): orbital period (u.time) 
        tperi (float): epoch of periastron (u.time)
        a (float): semi-major axis of the orbit
        e (float): eccentricity of the orbit
        inc (float): inclination of the orbit  (u.angle)
        omega (float): longitude of periastron (u.angle)
        anode (float): PA of the ascending node (u.angle)

    Returns:
       X,Y, Xs,Ys,Zs, Xsv,Ysv,Zsv

    Output frame has X,Y in computer plotting coordinates
    i.e. X is to the right, increasing (due West)

    Primary body is fixed at the origin.

    X,Y (float): 2D coordinates of in plane orbit with periapse
                 towards the +ve X axis.

    Xs,Ys,Zs (float): The 3D coordinates of the secondary body
        in the Position/velocity coords frame.

    Xsv, Ysv, Zsv (float): The 3D velocity of the secondary body
        in the Position/velocity coords frame.

    Coordinate frames are shown below.

    The 3D axes are NOT the usual right-handed coordinate frame. The
    Observer is located far away on the NEGATIVE Z axis. This is done
    so that +ve Zsv gives positive velocities consistent with most
    astronomers idea of redshift being positive velocity values.


    Sky coords         Computer coords   Position/velocity coords

      North                   Y                +Y    +Z
        ^                     ^                 ^   ^
        |                     |                 |  /
        |                     |                 | /
        |                     |                 |/
        +-------> West        +-------> X       +-------> +X
                                               /
                                              /
                                             /
                                           -Z

    +Y is North, +X is West and +Z is away from the Earth
    so that velocity away from the Earth is positive

    NOTE: Right Ascension is INCREASING to the left, but the
    (X,Y,Z) outputs have RA increasing to the right, as seen
    in the Computer coords. This is done to make plotting easier
    and to remind the user that for the sky coords they need
    to plot (-X,Y) and then relabel the RA/X axis to reflect 
    what is seen in the sky.

    Taken from ``Lecture Notes on Basic Celestial Mechanics''
    by Sergei A. Klioner (2011) page 22
    http://astro.geo.tu-dresden.de/~klioner/celmech.pdf

    Note that mean motion is defined on p.17 and that
    (P^2/a^3) = 4pi^2/kappa^2 and that the document is missing
    the ^2 on the pi.

    Written:
        Matthew Kenworthy, 2017

    """

    # mean motion n
    n = 2 * np.pi / P

    # at epoch = tperi, mean anomoly M is 0
    # 
    # Y = time since epoch periastron
    Y = epoch - tperi

    # mean anomaly M varies smoothly from 0 to 2pi every orbital period
    # convert Y to angle in radians between 0 and 2PI
    Mt = Y / P
    M  = (2 * np.pi * (Mt - np.floor(Mt)))*u.radian

    # calc eccentric anomaly E
    (E,v) = kepler_solve(e, M, derror)

    # calculate position and velocity in the orbital plane
    cE = np.cos(E)
    sE = np.sin(E)
    surde = np.sqrt(1 - (e*e))

    X  = a * (cE - e)
    Y  = a * surde * sE

    Xv = -(a * n * sE) / (1 - e * cE)
    Yv =  (a * n * surde * cE) / (1 - e * cE)

    # calculate Euler rotation matrix to get from orbital plane
    # to the sky plane

    mat = euler(anode, omega, inc)

    # rotate the coordinates from the orbital plane
    # to the sky projected coordinates

    # TODO we lose the dimensionality of X and Y and it
    # needs to be put back artificially
    # problem with building the np.array below and putting the Quantity through
    # the np.dot() routine
    
    (Xe, Ye, Ze)    = np.dot(mat, np.array([X.value,Y.value,0]))
    #blog = np.array([X,Y,np.zeros(X.size)]) * X.unit
    #(Xe, Ye, Ze)    = np.dot(mat, blog)
    (Xev, Yev, Zev) = np.dot(mat, np.array([Xv.value,Yv.value,0]))

    Xs = -Ye * X.unit
    Ys =  Xe * X.unit
    Zs =  Ze * X.unit
    Xsv = -Yev * Xv.unit
    Ysv =  Xev * Xv.unit
    Zsv =  Zev * Xv.unit

    # To transform from the Kep3D code coordinate system to the
    # celestial sphere, where:
    # Y is North, X is West and Z is away from the Earth
    # we have
    #   X,Y,Z,Xdot,Ydot,Zdot = -Ys, Xs, Zs, -Yv, Xv, Zv

    return(X,Y,Xs,Ys,Zs,Xsv,Ysv,Zsv)


@u.quantity_input
def vmaxellip(m1:u.Msun,m2:u.Mjup,P:u.year,e)->u.km/u.s:
    """
    Elliptical maximum velocity

    Args:
        m1: Primary and secondary masses
        P: orbital period
        e: eccentricity [0,1)

    Returns:
        velocity: maximum velocity in elliptical orbit

    >>> import astropy.units as u
    >>> vmaxellip(1.0 *u.M_sun, 1.0 * u.M_earth, 1.0 * u.yr, 0.0)
    <Quantity 29.78490916 km / s>
    """
    mu =  c.G * (m1 + m2)
    c1 = 2 * np.pi * mu / P
    c2 = (1 + e) / (1 - e)

    vmax = np.power(c1, 1./3.) * np.power(c2, 1./2.)
    # http://en.wikipedia.org/wiki/Circular_orbit
    return vmax

@u.quantity_input
def vcirc(m1:u.Msun,m2:u.Mjup,a:u.au)->u.km/u.s:
    """
    Circular orbital velocity of m2 about m1 at distance a

    Args:
        m1, m2: Primary and secondary masses
        a: semimajor axis

    Returns:
        velocity: circular orbital velocity

    >>> import astropy.units as u
    >>> vcirc(1.0 *u.M_sun, 1.0 * u.M_jup, 5.2 * u.au)
    <Quantity 13.06768412 km / s>
    """

    # http://en.wikipedia.org/wiki/Circular_orbit
    mu = c.G * (m1 + m2)
    vcirc = np.power(mu /a, 0.5)
    return vcirc

@u.quantity_input
def Pgivenm1m2vcirc(m1:u.Msun,m2:u.Mjup,vcirc:u.km/u.s)->u.year:
    """
    Circular orbital velocity of m2 about m1 at distance a

    Args:
        m1, m2: primary and secondary masses
        vcirc: velocity of m2 about m2

    Returns:
        P: orbital period

    >>> import astropy.units as u
    >>> Pgivenm1m2vcirc(1.0 *u.M_sun, 1.0 * u.M_jup, 13.1 *u.km/u.s)
    <Quantity 11.76489419 yr>
    """

    # http://en.wikipedia.org/wiki/Circular_orbit

    mu = c.G * (m1 + m2)

    P = 2 * np.pi * mu / np.power(vcirc,3)

    return P

@u.quantity_input
def rhill(m1: u.Msun, m2: u.Mjup, a: u.au)->u.au:
    """
    Hill radius of the secondary m2 orbiting around m1
    
    Args:
        m1, m2: primary and secondary masses
        a: distance between m1 and m2

    Returns:
        rhill: radius of Hill sphere of m2

    >>> import astropy.units as u
    >>> rhill(1.0 * u.M_sun, 1.0 * u.M_jup, 5.2 * u.au)
    <Quantity 0.35489325 AU>
    """

    mu = m2 / (m1 + m2)
    rh = a * np.power(mu/3., 1./3.)
    return rh

@u.quantity_input
def rvel(P:u.year, e, i:u.deg, a:u.au, M:u.deg, omega:u.deg, Mratio)->u.km/u.s:
    """
    Radial velocity of primary star due to secondary

    Args:
        P: period (u.time)
        e: eccentricity
        i: inclination (u.angle)
        M: mean anomaly (u.angle)
        omega: argument of periastron (u.angle)
        Mratio: ratio of primary mass to secondary mass

    Return:
        v: reflex velocity (u.velocity)

    >>> import astropy.units as u
    >>> rvel(12.0*u.year, 0.01, 45*u.deg, 5.2*u.au, 0.0*u.deg, 0.0*u.deg, 1.*u.Mjup/u.Msun)
    <Quantity 9.20952276 km / s>

    """

    (E, v) = kepler_solve(e, M)

    # v_radial_velocity = t1 * t2 / t3

    #       ^
    #       |           G.O.G.
    #     (M1)----a1-----O----------------------a2--------------------(M2)
    #                                                                   |
    #                                                                   v

    # m1.a1 = m2.a2
    #     and
    # a = a1 + a2
    a1 = a / (Mratio + 1)

    t1 = 2 * np.pi * np.sin(i) * a1
    t2 = np.cos(omega + v) + (e * np.cos(omega))
    t3 = P * np.sqrt(1 - e * e)

    vradial = t1 * t2 / t3
    return vradial

@u.quantity_input
def visviva(a:u.au, r:u.au, m1:u.Msun, m2:u.Mjup)->u.km/u.s:
    """
    Vis viva equation for instantaneous velocity

    Args:
        a: semi-major axis (u.length)
        r: instantaneous separation of primary and secondary (u.length)
        m1, m2: primary and secondary masses (u.mass)

    Returns:
        v: instantaneous velocity (u.vel)

    >>> import astropy.units as u
    >>> visviva(1.0*u.au, 1.5*u.au, 1.0 * u.M_sun, 1.0 * u.M_jup)
    <Quantity 17.20440558 km / s>

    """
    # http://en.wikipedia.org/wiki/Elliptic_orbit
    mu = c.G * (m1 + m2)

    con = (2./r) - (1./a)

    return np.sqrt(mu * con)

@u.quantity_input
def semiampl(P:u.year, e, i:u.deg, m1:u.Msun, m2:u.Mjup)->u.m/u.s:
    """
    Calculate semi-amplitude of star due to RV planet

    Args:
        P: period (years)
        e: eccentricity
        i: inclination (u.angle)
        m1, m2: primary and secondary masses (u.mass)

    Returns:
        K: semi-amplitude (u.velocity)

    >>> import astropy.units as u
    >>> semiampl(12.1*u.year, 0.05, 90 * u.deg, 1.0 * u.M_sun, 1.0 * u.M_jup)
    <Quantity 12.40023007 m / s>

    """

    t1 = 2*np.pi*c.G / P
    
    # assumes m1 >> m2
    t2 = m2 * np.sin(i) / np.power(m1, 2./3.)

    t3 = 1. / np.sqrt(1 - (e*e))

    K = np.power(t1,1./3.) * t2 * t3

    return K

@u.quantity_input
def atoP(a:u.au, m1:u.M_sun, m2:u.M_jup)->u.year:
    """
    Calculate period from orbital radius and masses

    Args:
        a: semi-major axis
        m1, m2: Primary and secondary masses

    Returns:
        P: orbital period

    >>> import astropy.units as u
    >>> atoP(1.*u.au, 1.0*u.M_sun, 1.0*u.M_jup)
    <Quantity 0.99954192 yr>
    """

    # a^3/P^2 = (G/4pipi) (m1 + m2)

    const = c.G/(4.*np.pi*np.pi)

    mu = m1 + m2

    P2 = np.power(a,3.)/(const*mu)

    P = np.power(P2, 0.5)
    return P

@u.quantity_input
def Ptoa(P:u.year, m1:u.M_sun, m2:u.M_jup)->u.au:
    """calculate orbital radius from period

    Args:
        P: orbital period
        m1, m2: Primary and secondary masses

    Returns:
        a: semi-major axis

    >>> import astropy.units as u
    >>> Ptoa(11.86*u.year, 1.0*u.M_sun, 1.0*u.M_jup)
    <Quantity 5.20222482 AU>
    """

    # a^3/P^2 = (G/4pipi) (m1 + m2)
    const = c.G / (4.*np.pi*np.pi)
    mu = m1 + m2
    a3 = P*P*const*mu
    aa = np.power(a3, 1./3.)
    
    return aa

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from astropy.visualization import quantity_support
    quantity_support()

    import doctest
    doctest.testmod()

    fig = plt.figure(figsize=(12,4))

    kep = fig.add_subplot(131)

    # solving Keplers equation
    # compare with graphic from http://mathworld.wolfram.com/KeplersEquation.html
    Mtrial = np.linspace(0, 6 * np.pi, 100, endpoint=True) * u.rad

    for e in np.linspace(0, 0.99, 5, endpoint=True):
        (E, v) = kepler_solve(e, Mtrial)
        kep.plot(E, Mtrial)

    plt.xlabel('E')
    plt.ylabel('M(E)')
    plt.title('Keplers Equation')
    plt.axis([0, 3 *np.pi, 0, 3 * np.pi])

    # plot orbit of gamma and
    P = 61.1 * u.year
    t = 1952.1 * u.year
    a = 0.296 * u.au
    e = 0.93
    i = 111.1 * u.deg
    w = 171.15 * u.deg
    anode = 104.15 * u.deg

    axre = fig.add_subplot(132)
    axre.set_xlim(-0.6,0.1)
    axre.set_ylim(-0.6,0.1)

    epochs = np.linspace(1995, 1995+P.value, 101, endpoint='true')
    # draw the orbit
    Xa, Ya, Xsa, Ysa, Zsa, Xva, Yva, Zva = kep3d(epochs*u.year,P,t,a,e,i,w,anode)
    axre.plot(Xsa,Ysa,marker=None,color='blue')

    # label the epochs
    epoch_label = np.array((1995,2000,2005,2010,2012,2013,2015,2020,2025,2030,2040))*u.year
    Xb, Yb, Xsb, Ysb, Zsb, Xvb, Yvb, Zvb = kep3d(epoch_label,P,t,a,e,i,w,anode)
    axre.scatter(Xsb,Ysb,marker='o',color='red')

    for (ide,xn,yn) in zip(epoch_label,Xsb,Ysb):
        axre.text(xn,yn,ide)

    # origin and axes
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    axre.scatter(np.array([0.]),np.array([0.]),color='b')

    plt.title('orbit of gamma Andromeda')

    # testing kep3d()
    # HD 80606b
    P = 111.436 * u.day
    t = 424.857 * u.day
    a = 0.453 * u.au
    e = 0.9340
    i = 89.6 * u.deg
    w = 300.60 * u.deg
    anode = 0.0 * u.deg

    ax7 = fig.add_subplot(133)

    ep2 = np.linspace(800.,800.+P.value, 101) * u.day
    
    X, Y, Xs, Ys, Zs, Xv, Yv, Zv = kep3d(ep2,P,t,a,e,i,w,anode)

    ax7.plot(ep2, Zv.to(u.km/u.s))
    ax7.set_xlabel('Time (HJD-2454000)')
    ax7.set_ylabel('Radial Velocity (km/s)')
    ax7.set_title('RV of HD 80606 star (-ve of planet)')

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.collections import PolyCollection

    P     = 15.9 * u.year # years
    eperi = 2002.3293 * u.year
    a     = 0.1246 # should be in u.arcsec but Quantity barfs on this...
    e     = 0.8831
    i     = 134.87 * u.deg
    anode = 226.53 * u.deg
    w     = 64.98 * u.deg

    # Gillessen 2018 "The orbit of the star S2 about Sgr A*"
    t_orbit = np.linspace(1999,1999+P.value,300) * u.year
    X, Y, Xs, Ys, Zs, Xv, Yv, Zv = kep3d(t_orbit,P,eperi,a,e,i,w,anode)

    t_orbit2 = np.linspace(1999,2010,100) * u.year

    X2, Y2, Xs2, Ys2, Zs2, Xv2, Yv2, Zv2 = kep3d(t_orbit2,P,eperi,a,e,i,w,anode)

    fig6 = plt.figure(figsize=(12,6))

    ax2 = fig6.add_subplot(121,aspect='equal')
    ax3 = fig6.add_subplot(122)

    ax2.plot(Xs, Ys)
    ax2.set_xlim((-0.05,0.075))
    ax2.set_ylim((-0.025, 0.20))
    ax2.set_xlabel('delta RA [arcsec]')
    ax2.set_ylabel('delta Dec [arcsec]')
    ax2.set_title('Figure 3 - Orbit of S2 around GC')
    
    # convert semimajor in arcsec to AU by * 8400 pc to GC
    d_gc = 8400 # distance in pc
    rvels2 = (Zv2*d_gc*u.au).to(u.km/u.s)

    ax3.plot(t_orbit2, rvels2)

    ax3.set_xlabel('t [{}]'.format(t_orbit2.unit))
    ax3.set_ylabel('Radial velocity [{}]'.format(rvels2.unit))
    ax3.set_title('Figure 2 - Radial Velocity of Star S2')

    plt.show()

    fig7 = plt.figure(figsize=(8,8))
    ax5 = fig7.add_subplot(111, projection='3d')

    ax5.plot(Xs, Ys, Zs, marker=None, color='red')

    ax5.scatter(0.0, 0.0, 0.0, marker='o', color='blue')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.set_title('orbit of S2 about Galactic centre')

    h = 0.2
    ax5.set_xlim(-h,h)
    ax5.set_ylim(-h,h)
    ax5.set_zlim(h,-h)

    plt.show()
#    input('press return to continue')

