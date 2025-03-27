from scipy.optimize import brentq
from scipy.interpolate import interp2d, interp1d 
import numpy as np
import pandas as pd


### BPRP to B-V OR MASS CONVERSIONS
# We used a 1 Gyr PARSEC isochrone to do the conversions, we obtained the isochrone using the amazing CMD 3.6 web app (http://stev.oapd.inaf.it/cmd). We only changed the isochrone Age and the output photometric system, leaving all other parameters as default. For more information, check the files headers in the PARSEC_isochrones folder.
PARSEC_gaia = pd.read_csv('PARSEC_isochrones/PARSEC_iso_1gyr_gaia', comment='#')
PARSEC_BV = pd.read_csv('PARSEC_isochrones/PARSEC_iso_1gyr_BV', comment='#')
PARSEC_gaia = PARSEC_gaia[(PARSEC_gaia['label']<2)] # only used PMS and MS.
PARSEC_BV = PARSEC_BV[PARSEC_BV['label']<2]
bv_color_PARSEC = PARSEC_BV['Bmag']-PARSEC_BV['Vmag']
bprp_color_PARSEC = PARSEC_gaia['G_BPmag']-PARSEC_gaia['G_RPmag']
bprp_to_bv_PARSEC = interp1d(bprp_color_PARSEC, bv_color_PARSEC)
bv_to_bprp_PARSEC =interp1d(bv_color_PARSEC, bprp_color_PARSEC)
bprp_to_mass_PARSEC = interp1d(bprp_color_PARSEC, PARSEC_gaia['Mass'])
mass_to_bprp_PARSEC = interp1d(PARSEC_gaia['Mass'], bprp_color_PARSEC)

### GYROCHRONOLOGY RELATIONS ###
def bpl_model(bprp, age, br=.43): # AKA A19
    '''Inputs Gaia G_BP-G_RP color and Age (in yr), returns P_rot (in days)
    Inputs can be either both arrays of same size, bprp an array and age a float, or both floats
    
    Function from Angus et al. (2019), adapted for this paper.
    '''
    if type(bprp) == float or type(bprp) == np.float64:
        bprp = np.array([bprp])
    logbprp = np.log10(bprp)
    logage = np.log10(age)
    p = [-38.96093821359645, 28.71101008315462, -4.91903824593666, 0.7160561986000329, -
         4.716546365981507, 0.6470642123725334, -13.55890448246179, 0.9361999874258322]
    logp = np.zeros(len(logbprp))
    cool = logbprp >= br
    hot = logbprp < -.25
    warm = (logbprp > -.25) & (logbprp <= br)
    if type(logage) != np.ndarray:
        logp[warm] = np.polyval(p[:5], logbprp[warm]) + p[5] * logage
        logp[cool] = np.polyval(p[6:], logbprp[cool]) + p[5] * logage
    else:
        logp[warm] = np.polyval(p[:5], logbprp[warm]) + p[5] * logage[warm]
        logp[cool] = np.polyval(p[6:], logbprp[cool]) + p[5] * logage[cool]
    return 10**logp


def isochrone2015(BV, age): # AKA A15 
    '''Inputs B-V color and Age (in Myr), returns P_rot (in days)
    
    Equation from Angus et al. (2015)'''
    return age**0.55 * 0.4 * (BV - 0.45)**0.31


ages = np.array([0.1,0.12,0.15,0.2,.22,0.25,0.3,0.4,0.5,0.6,0.7,1,1.5,2,2.5,4, 4.57])
masses = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25]
prots = np.array([[ 9.69,  9.76, 10.94, 14.87, 15.95, 17.1 , 18.24, 19.99, 21.54,
        22.93, 24.18, 27.21, 29.88, 31.11, 31.9 , 33.36, 34.42],
       [ 9.08,  9.27, 11.81, 14.43, 14.99, 15.63, 16.49, 17.8 , 18.76,
        19.53, 20.18, 21.68, 23.19, 24.66, 26.11, 30.52, 32.41],
       [ 8.4 ,  9.89, 11.53, 12.43, 12.76, 13.24, 14.  , 15.3 , 16.18,
        16.69, 17.04, 17.92, 19.47, 21.26, 23.27, 29.82, 32.72],
       [ 8.26,  9.63, 10.38, 11.24, 11.5 , 11.84, 12.28, 12.94, 13.43,
        13.88, 14.3 , 15.61, 17.87, 20.2 , 22.62, 30.29, 33.27],
       [ 8.18,  8.6 ,  9.25,  9.97, 10.16, 10.39, 10.71, 11.22, 11.68,
        12.16, 12.61, 14.12, 16.77, 19.62, 22.62, 31.49, 34.7 ],
       [ 7.38,  7.88,  8.38,  8.83,  8.96,  9.12,  9.38,  9.87, 10.4 ,
        10.92, 11.46, 13.21, 16.3 , 19.5 , 22.77, 31.96, 35.11],
       [ 6.73,  7.15,  7.51,  7.87,  7.97,  8.12,  8.39,  8.92,  9.49,
        10.1 , 10.72, 12.69, 16.12, 19.55, 22.86, 31.74, 34.68],
       [ 6.19,  6.42,  6.64,  6.94,  7.06,  7.24,  7.56,  8.19,  8.86,
         9.55, 10.25, 12.42, 16.05, 19.49, 22.69, 31.01, 33.76],
       [ 5.65,  5.79,  5.97,  6.29,  6.41,  6.61,  6.95,  7.66,  8.39,
         9.14,  9.9 , 12.18, 15.85, 19.21, 22.26, 30.01, 32.56],
       [ 5.14,  5.31,  5.5 ,  5.82,  5.95,  6.15,  6.5 ,  7.22,  7.96,
         8.73,  9.51, 11.82, 15.41, 18.59, 21.46, 28.69, 31.09],
       [ 4.67,  4.81,  5.01,  5.36,  5.51,  5.73,  6.11,  6.87,  7.64,
         8.41,  9.19, 11.42, 14.79, 17.75, 20.4 , 27.15, 29.38],
       [ 4.33,  4.45,  4.64,  4.98,  5.12,  5.33,  5.7 ,  6.45,  7.21,
         7.95,  8.69, 10.79, 13.91, 16.63, 19.06, 25.23, 27.32],
       [ 3.99,  4.1 ,  4.29,  4.62,  4.75,  4.96,  5.31,  5.99,  6.69,
         7.36,  8.02,  9.91, 12.7 , 15.14, 17.31, 22.94, 24.92],
       [ 3.67,  3.78,  3.96,  4.25,  4.36,  4.53,  4.82,  5.4 ,  5.99,
         6.56,  7.12,  8.74, 11.12, 13.19, 15.03, 20.03, 22.02],
       [ 3.35,  3.44,  3.58,  3.81,  3.91,  4.05,  4.28,  4.75,  5.2 ,
         5.63,  6.07,  7.28,  9.11, 10.72, 12.19, 16.67, 18.78],
       [ 3.01,  3.09,  3.21,  3.4 ,  3.46,  3.55,  3.69,  3.94,  4.19,
         4.44,  4.69,  5.45,  6.62,  7.69,  8.74, 13.7 , 13.7 ],
       [ 2.75,  2.77,  2.78,  2.82,  2.84,  2.86,  2.89,  2.97,  3.05,
         3.12,  3.17,  3.35,  3.58,  3.96,  4.69, 10.86, 10.86]])
SL20_interp_mass = interp2d(masses, ages, prots.transpose(), bounds_error=True)
def SL20_model_mass(mass, age): # AKA SL20
    '''Inputs Mass (in M_sun) and Age (in Gyr), returns P_rot (in days)
    
    Function is a 2D interpolation of Table A1 from Spada & Lanzafame (2020)'''

    if type(mass)==np.float64 or type(mass)==float:
        return SL20_interp_mass(mass, age)
    ret = []
    for i in range(mass.shape[0]):
        ret.append(SL20_interp_mass(mass[i], age)[0])
    return np.array(ret)


####################################################################

def solve_isochrone(bprp_1, prot_1, iso='A19'):
    '''Inputs Gaia color G_BP - G_RP and Prot (in days), returns gyrochone age (in yr for A19, Myr for A15, Gyr for SL20)
        All inputs must be numpy arrays of the same size!
    iso (default \'A19\'): str, can be either \'A19\' for the Angus et al (2019) relation,  \'A15\' for the Angus et al (2015) relation, or \'SL20\' for the Spada & Lanzafame (2020) relation.'''
    if iso == 'A19':
        def function(age):
            return bpl_model(bprp_1[i], age) - prot_1[i]
        age_1 = []
        for i in range(bprp_1.shape[0]):
            if prot_1[i] <= 0 or bprp_1[i] <= 10**-0.25:
                print('Cant find a gyrochrone for the {}-th pair! \n'.format(i))
                age_1.append(np.nan)
                print('\r{:3.3f}%'.format(i / bprp_1.shape[0] * 100), end='')
                continue
            res = brentq(function, 1e-10, 1e18)
            age_1.append(res)
            print('\r{:3.3f}%'.format(i / bprp_1.shape[0] * 100), end='')
        age_1 = np.array(age_1)
        return age_1

    elif iso=='A15':
        bv_1 = bprp_to_bv_PARSEC(bprp_1)
        age_1 = []
        def function(x):
            return isochrone2015(bv_1[i], x) - prot_1[i]
        for i in range(bv_1.shape[0]):
            if prot_1[i] <= 0 or bv_1[i] <= 0.45:
                print('Cant find a gyrochrone for the {}-th pair! \n'.format(i))
                age_1.append(np.nan)
                print('\r{:3.3f}%'.format(i / bv_1.shape[0] * 100), end='')
                continue
            res = brentq(function, 1e-10, 1e12)
            age_1.append(res)
            print('\r{:3.3f}%'.format(i / bv_1.shape[0] * 100), end='')
        age_1 = np.array(age_1)
        return age_1
    
    elif iso=='SL20':
        mass_1 = bprp_to_mass_PARSEC(bprp_1)
        age_1 = []
        def function(x):
            return SL20_model_mass(mass_1[i], x) - prot_1[i]
        for i in range(mass_1.shape[0]):
            if mass_1[i]>1.25 or mass_1[i]<0.45 or prot_1[i]<SL20_model_mass(mass_1[i], 0.1) or prot_1[i]>SL20_model_mass(mass_1[i],4.57):
                print('Input parameters not valid for the {}-th pair! \n'.format(i))
                age_1.append(np.nan)
                print('\r{:3.3f}%'.format(i / mass_1.shape[0] * 100), end='')
                continue
            res = brentq(function, 0.1, 4.57)
            age_1.append(res)
            print('\r{:3.3f}%'.format(i / mass_1.shape[0] * 100), end='')
        age_1 = np.array(age_1)
        return age_1
    

def gyro_check(bprp_1, prot_1, bprp_2, prot_2, iso = 'A19'):
    '''Inputs Gaia color G_BP - G_RP and Prot (in days) from both components of a pair, returns DeltaProtGyro (in days)
    All inputs must be numpy arrays of the same size!
    iso (default \'A19\'): str, can be either \'A19\' for the Angus et al (2019) relation,  \'A15\' for the Angus et al (2015) relation, or \'SL20\' for the Spada & Lanzafame (2020) relation.'''
    
    if iso=='A19':
        p = [-38.96093821359645, 28.71101008315462, -4.91903824593666, 0.7160561986000329, -
         4.716546365981507, 0.6470642123725334, -13.55890448246179, 0.9361999874258322]

        logP2 = np.log10(prot_1) + np.polyval(p[:5], np.log10(bprp_2)) - np.polyval(p[:5], np.log10(bprp_1))
        return prot_2 - 10**logP2

    elif iso=='A15':
        c = 0.45
        b = 0.31
        BV_1 = bprp_to_bv_PARSEC(bprp_1)
        BV_2 = bprp_to_bv_PARSEC(bprp_2)
        p2 = prot_1 * ((BV_2 - c)/(BV_1 - c))**b
        return prot_2 - p2
        
    elif iso == 'SL20':
        mass_2 = bprp_to_mass_PARSEC(bprp_2)
        age_1 = solve_isochrone(bprp_1, prot_1, iso=iso)
        ret = []
        for i in range(age_1.shape[0]):
            delta_prot = prot_2[i] - SL20_model_mass(mass_2[i], age_1[i])
            ret.append(delta_prot[0])
        return np.array(ret)

    
def sigma_Delta_P_rot_gyro(bprp_1, prot_1, e_prot_1, bprp_2, e_prot_2, iso='A19'):
    sigma_prot_2_catalog = prop_error_iso(bprp_1, prot_1, e_prot_1, bprp_2, iso=iso)
    sigma_delta = np.sqrt(sigma_prot_2_catalog**2 + e_prot_2**2)
    return sigma_delta

def prop_error_iso(BPRP_1, prot_1, e_prot_1, BPRP_2, iso='A19'): # A15 for Angus et al. (2015) lines, A19 for Angus et al. (2019)
    '''Inputs Gaia color G_BP - G_RP, Prot (in days) and its error (in days), returns error propagated across the gyrochrone (in days)
        All inputs must be numpy arrays of the same size!
    iso (default \'A19\'): str, can be either \'A19\' for the Angus et al (2019) relation,  \'A15\' for the Angus et al (2015) relation, or \'SL20\' for the Spada & Lanzafame (2020) relation.'''

    if iso == 'A15':
        c = 0.45
        sigma_b = 0.3
        BV_1 = bprp_to_bv_PARSEC(BPRP_1)
        BV_2 = bprp_to_bv_PARSEC(BPRP_2)
        age_1 = solve_isochrone(BPRP_1, prot_1, iso=iso)
        P2 = isochrone2015(BV_2, age_1)
        return P2 * np.sqrt(e_prot_1**2/prot_1**2 + sigma_b**2 * np.log((BV_2-c)/(BV_1-c))**2)

    elif iso == 'A19':
        prot_comp = e_prot_1/(prot_1*np.log(10))
        c0 = 0.458
        c1 = 0.0585
        c2 = 0.217
        c3 = 1.508
        c4 = 3.309
        x0 = 0
        x1 = c1*(np.log10(BPRP_2)-np.log10(BPRP_1))
        x2 = c2*(np.log10(BPRP_2)**2-np.log10(BPRP_1)**2)
        x3 = c3*(np.log10(BPRP_2)**3-np.log10(BPRP_1)**3)
        x4 = c4*(np.log10(BPRP_2)**4-np.log10(BPRP_1)**4)
        age_1 = solve_isochrone(BPRP_1, prot_1, iso=iso)
        prot2_exp = bpl_model(BPRP_2, age_1)
        return prot2_exp * np.log(10) * np.sqrt(prot_comp**2 + x0**2 + x1**2 + x2**2 + x3**2 + x4**2)

    elif iso == 'SL20':
        age_plus = solve_isochrone(BPRP_1, prot_1 + e_prot_1, iso='SL20')
        age_minus = solve_isochrone(BPRP_1, prot_1 - e_prot_1, iso='SL20')
        mass_1 = bprp_to_mass_PARSEC(BPRP_1)
        mass_2 = bprp_to_mass_PARSEC(BPRP_2)
        prot_plus = np.array([SL20_model_mass(mass_2[i], age_plus[i])[0] for i in range(age_plus.shape[0])])
        prot_minus = np.array([SL20_model_mass(mass_2[i], age_minus[i])[0] for i in range(age_minus.shape[0])])
        return (prot_plus - prot_minus) / 2
 
    
def solve_x(bprp_1, prot_1, e_prot_1, bprp_2, prot_2, e_prot_2, iso='A19'):
    '''Inputs Gaia color G_BP - G_RP, Prot (in days) and its error (in days), for each component of the pair, returns the pairs x-parameter
        All inputs must be numpy arrays of the same size!
    iso (default \'A19\'): str, can be either \'A19\' for the Angus et al (2019) relation,  \'A15\' for the Angus et al (2015) relation, or \'SL20\' for the Spada & Lanzafame (2020) relation. '''
    
    check_catalog = gyro_check(bprp_1, prot_1, bprp_2, prot_2, iso=iso)
    sigma_delta = sigma_Delta_P_rot_gyro(bprp_1,prot_1, e_prot_1, bprp_2, e_prot_2, iso=iso)
    return abs(check_catalog / sigma_delta)
