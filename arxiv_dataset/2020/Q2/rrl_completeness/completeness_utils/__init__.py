import numpy as np
import healpy as hp
import scipy
import astropy.table
import astropy.coordinates as ac
from astropy import units as u

    
#----------------- Querying utilities ------------------------------------------------------

def query_2d_map(C, coo):
    
    '''
    Query pre-computed 2D completeness tables from Mateu2020 at given coordinates. 
    
    Parameters:
    
        C: pandas DataFrame. Must have the structure: hpix_index, l(deg), b(deg), Completeness
        
        coo: astropy SkyCoord object with line of sight coordinates at which to query C.
        
    Returns:
    
        C[los]
            Input completeness map C at the selected line of sight. 
            
    Examples:
    
        #Read in 2D maps for RRab stars 
        > C = pd.read_csv("maps/completeness2d.faint.rrab.csv")
        #Set l.o.s coords 
        > coo = ac.SkyCoord(l=22.5*u.deg,b=84.0*u.deg,frame='galactic')
        #Query the map at given l.o.s
        > C_los = query_2d_map(C,coo)
        > C_los["PS1_full"]
            0.909091
    '''
        
    #Healpix indices
    npix = C.hpix.size
    nside = hp.npix2nside(npix)

    # convert to theta, phi
    theta = np.radians(90. - coo.galactic.b.degree)
    phi = np.radians(coo.galactic.l.degree)

    # convert to HEALPix indices
    indices = hp.ang2pix(nside, theta, phi)

    return C[C.hpix == indices]
        
def query_3d_map(C, coo, D=None, verbose=False, force_nearest=True):

    '''
    Query pre-computed 3D completeness tables from Mateu2020 at given coordinates. 
    
    Parameters:
    
        C: pandas DataFrame. Must have the structure: hpix_index, l(deg), b(deg), Completeness
        
        coo: astropy SkyCoord object with line of sight (l.o.s.) coordinates at which to query C.
                
        D: None
           The default R=None returns the 3D completeness map at the l.o.s., as a function of distance. 
           To retrieve the correspoding row at one particular distance x*u.kpc, set R=x*u.kpc (R must be in kpc)
        
    Returns:
    
        C[los]
            Input completeness map C at the selected line of sight. Same structure as input map.
            
    Examples:
    
        #Read in 2D maps for RRab stars 
        > C3D = pd.read_csv("maps/completeness3d.gaiadr2.vcsos.rrab.csv",dtype=dict(hpix=np.int))
        #Set l.o.s coords 
        > coo = ac.SkyCoord(l=65.4*u.deg,b=35.2*u.deg,frame='galactic')
        #Query the map at given l.o.s
        > losC=query_3d_map(gaia3D,coo)
        # the ouput DataFrame contains all rows for the corresponding healpix, with completeness C 
        # (with error eC) as a function of distance
        > losC[:2]
                  hpix     l     b       D_o       D_f         C        eC
            1714    43  67.5  30.0  1.189142  2.752091  0.866667  0.232220
            1715    43  67.5  30.0  2.752091  3.949873  0.833333  0.225668
            
        # You can also retrieve it a given distance    
        > losR_C=query_3d_map(Cab3D,coo,D=13.5*u.kpc)
        > losR_C
                  hpix     l     b        D_o        D_f         C        eC
            1733    43  67.5  30.0  13.286508  13.835308  0.933333  0.245251    
        '''
    
    #Healpix indices
    #npix = np.unique(C.hpix).size
    npix = np.max(C.hpix)+1
    nside = hp.npix2nside(npix)

    # convert to theta, phi
    theta = np.radians(90. - coo.galactic.b.degree)
    phi = np.radians(coo.galactic.l.degree)

    # convert to HEALPix indices
    indices = hp.ang2pix(nside, theta, phi)

    mask_index = (C.hpix==indices)  #each index appears more than once, one occurrence per distance interval computed
    
    # distance mask
    if D is not None: 
        if verbose: print(f"Selecting D={D.value}")
        mask_D = (C.D_o<=D.value) & (D.value<=C.D_f)
        if (mask_index & mask_D).any(): 
            return C[mask_index & mask_D]
        else: 
            if force_nearest:
               if (D.value<np.min(C[mask_index].D_o)): return C[mask_index].iloc[0]
               if (D.value>np.max(C[mask_index].D_f)): return C[mask_index].iloc[-1]
            else:   
              print(f"D outside valid distance range [{np.min(C.D_o)},{np.max(C.D_f)}] for hpix")
    else:
        if verbose: print("No D selected")
        return C[mask_index]
        

#----------------- Computing utilities ------------------------------------------------------

def cat2hpx(lon, lat, nside, radec=True):
    #from Daniel Lenz, on StackOverflow: https://stackoverflow.com/questions/50483279/make-a-2d-histogram-with-healpix-pixellization-using-healpy
    """
    Convert a catalogue to a HEALPix map of number counts per resolution
    element.

    Parameters
    ----------
    lon, lat : (ndarray, ndarray)
        Coordinates of the sources in degree. If radec=True, assume input is in the icrs
        coordinate system. Otherwise assume input is glon, glat

    nside : int
        HEALPix nside of the target map

    radec : bool
        Switch between R.A./Dec and glon/glat as input coordinate system.

    Return
    ------
    hpx_map : ndarray
        HEALPix map of the catalogue number counts in Galactic coordinates

    """

    npix = hp.nside2npix(nside)

    if radec:
        #convert to array to avoid type issues with pandas Series
        eq = ac.SkyCoord(np.array(lon), np.array(lat), frame='icrs', unit='deg')
        l, b = eq.galactic.l.value, eq.galactic.b.value
    else:
        l, b = lon, lat

    # convert to theta, phi
    theta = np.radians(90. - b)
    phi = np.radians(l)

    # convert to HEALPix indices
    indices = hp.ang2pix(nside, theta, phi)

    idx, counts = np.unique(indices, return_counts=True)

    # fill the fullsky map
    hpx_map = np.zeros(npix, dtype=np.int64)
    hpx_map[idx] = counts

    return hpx_map

def compute_completeness_map(sc1, sc2, nside=2**2, tol=3.*u.arcsec, verbose=False):

    '''Compute completeness based on stars in common between independent surveys
    
    Parameters:
    ===========
    
    sc1, sc2 : astropy.coordinate.SkyCoord objects 
        Skycoord objects for the two surveys    
    nside : int
        HealPix map nside level (must be a power of 2)
    tol : float with units
        angular tolerance to compute sky-match between the two surveys
    verbose : bool

    Returns:
    ========

    C1_hpx, C2_hpx : healpix completeness maps for sc1 and sc2, respectively

    ...
    '''
    
    idx,sep2d,sep3d=sc1.match_to_catalog_sky(sc2)
    sc1x2=sc1[(sep2d<=tol)]

    if verbose: print("N_s1=%d, N_s2 = %d, N_s1xs2 = %d" % (sc1.size,sc2.size,sc1x2.size))

    #Compute healpix maps    
    s1_hpx=cat2hpx(sc1.ra,sc1.dec, nside, radec=True)
    s2_hpx=cat2hpx(sc2.ra,sc2.dec, nside, radec=True)
    s1x2_hpx=cat2hpx(sc1x2.ra,sc1x2.dec, nside, radec=True)

    #Surveys's completeness 
    C1_hpx=100.*s1x2_hpx/s2_hpx.astype(float)
    C2_hpx=100.*s1x2_hpx/s1_hpx.astype(float)

    return C1_hpx, C2_hpx
  

def fill_with_ecliptic_opposite(C, return_lb=False):
    '''
    Replace values in Healpix map at DEC<-30 deg with opposite field in ecliptic coordinates (ecl_lon,ecl_lat) = (ecl_lon-180,-ecl_lat) [for DEC<-30]
    '''
        
    #Healpix indices
    npix = C.size
    nside = hp.npix2nside(npix)
    hpi = np.arange(npix,dtype=int) #healpix indices for current map 

    #Refill PS1-hole with symmetric region
    l, b = hp.pix2ang(nside, hpi, lonlat=True) 
    sc = ac.SkyCoord(l=l*u.deg, b=b*u.deg, frame='galactic')
    elon, elat = sc.barycentrictrueecliptic.lon, sc.barycentrictrueecliptic.lat
    sc_n = ac.SkyCoord(elon-180*u.deg, -elat, frame='barycentrictrueecliptic')

    ind2replace = sc.icrs.dec.deg<-30 
    replacement_inds = hp.ang2pix(nside, sc_n.galactic.l.value, sc_n.galactic.b.value,
                                  lonlat=True)
    
    newC = np.array(C.copy())
    newC[ind2replace] = C[replacement_inds[ind2replace]]
       
    if return_lb: return newC, l, b
    else: return newC

def completeness_los(los_coo, cat1_coo, cat2_coo, fill_with_ecliptic_opposite=False,
                     los_size=30*u.deg, sky_tol=1*u.arcsec, 
                     r_bin_edges=None, Nmin=100, verbose=False):
    '''
    Compute catalogue completeness along selected line of sight.
    
    Parameters:
    ==========
    
    los_coo:  astropy.coordinates.SkyCoord object 
              containing line of sight coordinates
    
    cat1_coo: astropy.coordinates.SkyCoord object 
              with coordinates for survey 1 objects.
              Distance attribute must exist.
    
    cat2_coo: astropy.coordinates.SkyCoord object with coordinates for survey 1 objects
              Distance attribute must exist.
              
    los_size: radius of selected line of sight (with units)
    
    sky_tol:  tolerance to be used in sky match between cat1 and cat2
    
    r_bin_edges: numpy.array or sequence 
                 Radial bin edges (in same distance units as cat1 and cat2)
                 This keyword overrides Nmin
    
    Nmin: int
          The radial bins width is selected to ensure there are Nmin stars per bin. 
          This gives a constant Poisson noise per bin.
          The last bin is chosen to have a minimum Nmin stars
          
    verbose: bool
             Be chatty
             
    Returns: Completeness estimated for cat2 (based on cross-matches with cat1)
    
     r_bin_edges, C, eC, Ntot_r, Ncommon_r, Cmean
    
     r_bin_edges: array with radial bin edges
     C, eC: arrays with completeness and it's error (for input cat2) 
     Ntot_r, Ncommon_r: total number of stars in cat1 and number of stars in common between cat1 and cat2
     Cmean: mean completeness for cat2
    
    '''
    
    #Original line of sight (l.o.s) coords  
    sco = los_coo

    #If los has DEC<-30, PS1 is not available. substitute chosen l.o.s. by 
    #symmetric field in w.r.t. ecliptic plane   
    if fill_with_ecliptic_opposite and sco.icrs.dec<=-30*u.deg:
      #Get ecliptic coords
      elon, elat = sco.barycentrictrueecliptic.lon, sco.barycentrictrueecliptic.lat
      #Get symmetric field  
      sc_n = ac.SkyCoord(elon-180*u.deg, -elat, frame='barycentrictrueecliptic')
      #SkyCoord object for center of field   
      sco = ac.SkyCoord(sc_n.galactic.l, sc_n.galactic.b, frame="galactic")
      if verbose: 
        print("Field outside PS1 footprint -> getting estimate from symmetric field w.r.t ecliptic plane") 
        #print(f'{lon},{lat} -> {sco.l},{sco.b}')
    
    #Keep catalogue stars in selected l.o.s
    mlos = cat1_coo.separation(sco)<=los_size
    sclos1 = cat1_coo[mlos]
    mlos = cat2_coo.separation(sco)<=los_size
    sclos2 = cat2_coo[mlos]
    
    #Total stars in selected line of sight
    Ntot = sclos1.size #yes, other survey
    
    #In line of sight, select stars in common between ap and g surveys
    try:
        idx,sep2d,sep3d=sclos1.match_to_catalog_sky(sclos2)
        m_common = sep2d<=sky_tol
        Ncommon = m_common.sum() #change to m_common.sum()
    except:
        return ( [0.,0.], 0., 0., 0., 0., 0.)
        
#     m_common = sep2d<=sky_tol
#     Ncommon = m_common.sum() #change to m_common.sum()
         
    #mean Completeness (integrated in r)
    Cmean = Ncommon/Ntot
        
    if r_bin_edges is not None:
        r_bin_edges = r_bin_edges.to(u.kpc).value
    elif Nmin:        
        #Sort by increasing r
        mr = np.argsort(sclos1.distance.kpc)
        clos_D_sort = sclos1[mr].distance.kpc
        #Select bin edges every Nmin stars in total
        r_bin_edges =  clos_D_sort[::Nmin]
        #If fewer than 1/2Nmin stars in last bin, move edge to merge with previous bin into a single one
        if ~(clos_D_sort.size % Nmin > int(0.5*Nmin)): r_bin_edges[-1]=np.max(sclos1.distance.kpc)     
        else: r_bin_edges = np.append(r_bin_edges,sclos1.distance.kpc.max())  

    #Count stars in (irregularly spaced) r bins  
    try:
      Ntot_r = np.histogram(sclos1.distance.kpc, bins=r_bin_edges)[0]
      Ncommon_r = np.histogram(sclos1.distance.kpc[m_common], bins=r_bin_edges)[0]
    except:
      if verbose: print(r_bin_edges)  

    #Compute completeness
    C= Ncommon_r/Ntot_r   #C2 = N1x2/N1
    mz = (Ncommon_r>0)
    eC = np.zeros_like(Ntot_r) #init to deal separately with Ncommon_r==0 avoiding warning
    eC = C*(np.sqrt(Ntot_r)/Ntot_r) #first term is 0 when Ncommon_r==0
    eC[mz] = C[mz]*np.sqrt((np.sqrt(Ncommon_r[mz])/Ncommon_r[mz])**2 
                       + (np.sqrt(Ntot_r[mz])/Ntot_r[mz])**2)

    if verbose: print("Completeness returned for 2nd catalogue (cat2coo)")
    
    return (np.array(r_bin_edges), C, eC, Ntot_r, Ncommon_r, Cmean)
