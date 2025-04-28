# imports and initialization
import astropy as ap
import numpy as np
import pandas as pd
import tempfile as tf
import os
import shutil
from textwrap import dedent as ddent
from astropy.coordinates import Angle
import astropy.units as u

# Creates temporary .eq1 file in epoch folder for OrbFit
def eq1file(el_jpl, tdir):
    '''
    Parameters
    ----------
    el_jpl : `~numpy.ndarray` (N, 18)
        Orbital elements as determined by JPL Horizons.
    tdir : `tf.TemporaryDirectory`
        Temporary directory object, temp directory for file generation
        
    Returns
    ------- 
    temp.eq1: temporary text file to run OrbFit
    '''  
    
    os.makedirs(f'{tdir}/epoch')
    
    with open(f"{tdir}/epoch/temp.eq1", "w") as fp:
        fp.write(ddent(f'''\
        format  = 'OEF2.0'       ! file format
        rectype = 'ML'           ! record type (1L/ML)
        refsys  = ECLM J2000     ! default reference system
        END_OF_HEADER
        temp
        ! Cometary elements: q, e, i, long. node, arg. peric., pericenter time\n'''))
        fp.write(" COM   %.15E  %.15f   %.15f  %.15f  %.15f   %.15f\n" % (el_jpl['q'][0], el_jpl['e'][0], el_jpl['incl'][0], el_jpl['Omega'][0], el_jpl['w'][0], ap.time.Time(el_jpl['Tp_jd'][0], format='jd').mjd))
        fp.write(" MJD     %.9f TDT\n" % (ap.time.Time(el_jpl['datetime_jd'][0], format='jd').mjd))
        fp.write(" MAG  %.3f  %.3f\n" % (el_jpl['H'][0], el_jpl['G'][0]))
        fp.write(ddent(f'''\
        ! Non-grav parameters: model used, actual number in use, dimension
         LSP   0  0    6
        ! RMS    1.68232E-08   5.88900E-08   8.22688E-08   5.34469E-08   7.56890E-08   7.13398E-06
        ! EIG   1.07579E-08   5.31009E-08   5.91864E-08   7.57356E-08   7.86169E-08   1.27486E-07
        ! WEA   0.08651  -0.02289  -0.24412  -0.03357  -0.02078  -0.96480
         COV   2.830210138102556E-16  3.543122024213312E-16 -2.603292682702056E-16
         COV  -4.992042214900484E-18 -1.052690180196314E-18 -7.873861865190710E-14
         COV   3.468027286557968E-15  6.878077752471183E-17  1.886511729787680E-17
         COV   6.689670038864485E-17  1.808279351538482E-14  6.768149177265766E-15
         COV   7.159243161286040E-17  1.248926483233068E-16  1.357728291186093E-13
         COV   2.856568560436748E-15  2.588049637167598E-16  2.529981071526617E-14
         COV   5.728829671236270E-15  1.056596023015451E-14  5.089368128905899E-11
         NOR   8.462990106959648E+15 -9.345934921051774E+14  6.961302078833404E+13
         NOR  -9.766026206616650E+13 -9.148695418092123E+12  1.329006055003970E+13
         NOR   3.921580648140324E+14 -8.566206317904612E+12  1.006265833999790E+13
         NOR  -2.128841368531985E+12 -1.566971456817283E+12  1.567202493495569E+14
         NOR  -7.910041612493922E+11 -2.702958007388599E+12 -3.063965034542373E+11
         NOR   3.541407046591562E+14 -1.551670529664669E+13 -3.253830675316872E+11
         NOR   1.754031538722264E+14 -3.488851624201696E+10  4.175326401599722E+10
         '''))
    
    return


# Writes temporary .fop file for OrbFit
def fopfile(tdir):
    '''
    Parameters
    ----------
    tdir : `tf.TemporaryDirectory`
        Temporary directory object, temp directory for file generation
    
    Returns
    ------- 
    temp.fop: temporary text file to run OrbFit
    '''
        
    with open(f"{tdir}/temp.fop", "w") as fp:
        fp.write(ddent('''\
        ! input file for fitobs
        fitobs.
        ! first arc        .astna0='temp'           ! full name, first arc
                .obsdir0='mpcobs/'         ! directory of observs. file, first arc
                .elefi0='epoch/temp.eq1' ! first arc elements file

        ! second arc
        !        .astnap=''            ! full name, second arc
        !        .obsdirp='mpcobs'     ! directory of observs. file, second arc
        ! bizarre  control;
                .ecclim=     1.9999d0    ! max eccentricity for non bizarre orbit
                .samin=      0.3d0       ! min a for non bizarre orbit
                .samax=      2000.d0     ! max a for non bizarre orbit
                .phmin=      0.001d0     ! min q for non bizarre orbit
                .ahmax=     4000.d0      ! max Q for non bizarre orbit
                .error_model='fcct14'     ! error model
        propag.
                .iast=17            ! 0=no asteroids with mass, n=no. of massive asteroids (def=0)
                .filbe='AST17'      ! name of the asteroid ephemerides file (def='CPV')
                .npoint=600         ! minimum number of data points for a deep close appr (def=100)
                .dmea=0.2d0         ! min. distance for control of close-app. to the Earth only (def=0.1)
                .dter=0.05d0        ! min. distance for control of close-app.
                                    ! to terrestrial planets (MVM)(def=0.1)
                .yark_exp=2.d0      ! A2/r^yark_exp model (def=2)
                .ngr_opt=.TRUE.     ! read options for non-gravitational perturbations from the option file
                .irel=2             ! 0=newtonian 1=gen. relativity, sun 2=gen. rel. all planets
                                    !          (def=0, 1 for NEA, 2 for radar)
                .iaber=2            ! aberration 0=no 1=yes 2=(def=1)
                .ilun=1             ! 0=no moon 1= yes (def=0, 1 for NEA)
                .iyark=3            ! 0=no Yarkovsky, 1=Yark diurnal, 2=Yark seasonal
                                    !    3=secular nongravitational perturbations (including Yark) (def=0)
                .ipa2m=0           ! 0=no drpa2m, 1=yes spherical direct radiation pressure (def=0)
                .det_drp=2          ! how many parameters to solve: 0=none 1=drpa2m 2=dadt 3=both (def=0)
                .det_outgas=0       ! det outgassing for comets

        difcor.

        IERS.
                .extrapolation=.T. ! extrapolation of Earth rotation

        reject.
                .rejopp=.false.    ! reject entire opposition
        '''))
        
    shutil.copyfile("orbfit/lib/AST17.bai", f"{tdir}/AST17.bai")
    shutil.copyfile("orbfit/lib/AST17.bep", f"{tdir}/AST17.bep")

    return


# Writes temporary file to bypass OrbFit's interactive menu
def astfile(start, stop, obs, tdir):
    '''
    Parameters
    ----------
    start : `str`
        Beginning time of integration, UTC.  
    stop : `str`
        End time of integration, UTC.   
    obs : `str`
        Observatory code.
    tdir : `tf.TemporaryDirectory`
        Temporary directory object, temp directory for file generation
        
    Returns
    ------- 
    ast.inp: text file required to bypass OrbFit interactive menu
    '''
    
    with open(f"{tdir}/ast.inp", "w") as fp:
        fp.write(ddent(f'''\
        temp
        6
        6
        {ap.time.Time(start).mjd}
        {ap.time.Time(stop).mjd}
        1
        {obs}
        0
        '''))
        
    return


# Calculates ephemerides using OrbFit
def get_ephem_OrbFit(el_jpl, start, stop, obs):
    '''
    Parameters
    ----------        
    el_jpl : `~numpy.ndarray` (N, 18)
        Orbital elements as determined by JPL Horizons.
    start : `str`
        Beginning time of integration, UTC. 
    stop : `str`
        End time of integration, UTC.  
    obs : `str`
        Observatory code.
        
    Returns
    -------
    coord_OrbFit : `~numpy.ndarray` (N, 2)
        RA and DEC coordinates of ephemderides as determined by OpenOrb, units deg
            RA : first column of coord_OrbFit
            DEC : second column of coord_OrbFit 
            
    mag_OrbFit : `~numpy.ndarray` (N, 1)
        Contains magnitudes of object as determined by OpenOrb, units mags
    '''
    
    with tf.TemporaryDirectory() as tdir:
        eq1file(el_jpl, tdir=tdir)
        fopfile(tdir=tdir)
        astfile(start, stop, obs, tdir=tdir)
        home = os.environ["HOME"]
        os.system(f'(cd "{tdir}" && {home}/bin/fitobs.x < ast.inp)') 
        df = pd.read_fwf(f'{tdir}/temp.eph', skiprows=4, header=None, colspecs=[(20,32),(35,37),(38,40),(41,47),(49,50),(50, 52),(53, 55),(56, 61),(62,67)])

    df["RA"] = Angle((df[1], df[2], df[3]), unit = 'hourangle').degree
    df["DEC"] = Angle((df[5], df[6], df[7]), unit = u.deg)
    df.loc[df[4] == '-', "DEC"] *= -1
    df['Mag'] = df[8]

    coord_OrbFit = np.array([df["RA"], df["DEC"]]) * u.deg
    mag_OrbFit = np.array([df['Mag']]) * u.mag
    
    return coord_OrbFit, mag_OrbFit