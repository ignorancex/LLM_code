

    ##########################################
    ##  magnification_module.py             ##
    ##  Maximilian von Wietersheim-Kramsta  ##
    ##  Version 2020.09.22                  ##
    ##########################################

import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import astropy.io.fits as fits
import os
from scipy.optimize import curve_fit
from collections.abc import Iterable
from scipy.signal import find_peaks

def read_MICE2_fits(filename, dilution, mag_limited, mag_limit_band, mag_limit):
    """Reads MICE2 fits files in order to turn relevant data columns into arrays.
    IN:
    - filename (str):       String containing the directory + name of the fits file
    - dilution (bool):      If True, coordinates after magnification are considered and dilution factor is included in analysis of magnification.
    - mag_limited (bool):   If True, the galaxy sample is treated as if it has a definate magnitude limit. In that case, one needs to specify the mag_limit_band and the mag_limit.
    - mag_limit_band (str): String containing the name of the column which contains the band magnitude with which the flux/magnitude limit of the sample has been set.
    - mag_limit (float):    Float specifying the magnitude limit in the band magnitude specifid in mag_limit_band.
    
    OUT:
    - ra (numpy.ndarray):   Right Ascension in radians without magnification (if dilution==False) and with magnification (if dilution==False).
    - dec (numpy.ndarray):  Declination in radians without magnification (if dilution==False) and with magnification (if dilution==False).
    - kappa (numpy.ndarray):Projected matter density
    - z (numpy.ndarray):    Redshift

    Additional outputs may be added by calling additional columns of the fits file. Otherwise, it might be useful to write a new reading function.
    """
    data = fits.open(filename)[1].data
  
    if mag_limited == True:
        if type(mag_limit_band) != str or type(mag_limit) != float:
            raise Exception('Please specify a valid mag_limit_band and a valid mag_limit when mag_limited == True.')
        
        mag   = data[mag_limit_band]
        
        kappa   = data['kappa']
        gamma_1 = data['gamma1']
        gamma_2 = data['gamma2']
        
        kappa   = kappa[mag < mag_limit]
        gamma_1 = gamma_1[mag < mag_limit]
        gamma_2 = gamma_2[mag < mag_limit]
        
        try:
            z = data['z_cgal_v']
            z = z[mag < mag_limit]
        except:
            z = data['z_cgal']
            z = z[mag < mag_limit]
        #In some data sets, the redshift is adjusted according to the objects' peculiar velocity
            
        if dilution == True:
            ra_mag  = data['ra_gal_mag']*np.pi/180.
            ra_mag  = ra_mag[mag < mag_limit]
            dec_mag = data['dec_gal_mag']*np.pi/180.
            dec_mag = dec_mag[mag < mag_limit]
            ra_mag[ra_mag < 0] = ra_mag[ra_mag < 0] + 2*np.pi
            dec_mag = dec_mag + np.pi/2
            return ra_mag, dec_mag, kappa, gamma_1, gamma_2, z
        
        else:  
            ra  = data['ra_gal']*np.pi/180.
            dec = data['dec_gal']*np.pi/180.
            ra  = ra[mag < mag_limit]
            dec =  dec[mag < mag_limit]
            ra[ra < 0] = ra[ra < 0] + 2*np.pi
            dec = dec + np.pi/2
            return ra, dec, kappa, gamma_1, gamma_2, z
        
    else:      
        kappa   = data['kappa']
        gamma_1 = data['gamma1']
        gamma_2 = data['gamma2']
        
        try:
            z = data['z_cgal_v']
        except:
            z = data['z_cgal']
        #In some data sets, the redshift is adjusted according to the objects' peculiar velocity
            
        if dilution == True:
            ra_mag  = data['ra_gal_mag']*np.pi/180.
            dec_mag = data['dec_gal_mag']*np.pi/180.
            ra_mag[ra_mag < 0] = ra_mag[ra_mag < 0] + 2*np.pi
            dec_mag = dec_mag + np.pi/2
            return ra_mag, dec_mag, kappa, gamma_1, gamma_2, z
        
        else:  
            ra  = data['ra_gal']*np.pi/180.
            dec = data['dec_gal']*np.pi/180.
            ra[ra < 0] = ra[ra < 0] + 2*np.pi
            dec = dec + np.pi/2
            return ra, dec, kappa, gamma_1, gamma_2, z

        
def hp_map(ra, dec, kappa, z, nside):
    """Builds HealPix map of the number counts of a given field.
    IN:
    - ra (numpy.ndarray):   Right Acension in radians
    - dec (numpy.ndarray):  Declination in radians
    - kappa (numpy.ndarray):Projected matter density
    - z (numpy.ndarray):    Redshift
    - nside (int):          Resultion of the HealPix pixelation. Must be a power of 2
    
    OUT:
    - m (numpy.ndarray):          HealPix map of the number counts
    - kappa_pix (numpy.ndarray):  Healpix map of the average kappa value
    - npix_index (numpy.ndarray): Indices of the HealPix map which are within the observed field
    """

    #--------------------------------
    #Uncomment the following lines to remove edge pixels OR to only consider edge pixels: (only necessary for tests)
    
    #pix_rad = hp.max_pixrad(nside) #Maximum pixel radius
    #ra_max_diff, ra_min_diff, dec_max_diff, dec_min_diff = abs(ra - max(ra)), abs(ra -  min(ra)), abs(dec - max(dec)), abs(dec - min(dec))
    #Find indices of pixels within one pixel radius of the edge of the field
    #indx1 = np.where(ra_max_diff < pix_rad)[0]
    #indx2 = np.where(ra_min_diff < pix_rad)[0]
    #indx3 = np.where(dec_max_diff < pix_rad)[0]
    #indx4 = np.where(dec_min_diff < pix_rad)[0]

    #obj_to_remove = np.unique(np.concatenate((indx1, indx2, indx3, indx4)))
    #npix_to_remove = np.unique(hp.ang2pix(nside, dec[obj_to_remove], ra[obj_to_remove]))
    #print('Removed {0} pixels at the edges'.format(len(npix_to_remove)))
    #--------------------------------

    npix = hp.ang2pix(nside, dec, ra) #Find pixel index for each object
    npix_count = np.bincount(npix) #Find counts of objects within each pixel

    #--------------------------------
    #Uncomment the following 2 lines to ONLY consider edge pixels: (only necessary for tests)
    #all_ind = np.arange(0, len(npix_count))
    #npix_to_remove = np.setdiff1d(np.union1d(all_ind, npix_to_remove), np.intersect1d(all_ind, npix_to_remove)) 
    #--------------------------------

    #--------------------------------
    #Uncomment the following line to remove edge pixels OR to only consider edge pixels: (only necessary for tests)
    #npix_count[npix_to_remove] = np.zeros(len(npix_to_remove)) #Removes counts from pixels near the edges of the field
    #---------------------------------

    npix_index    = np.nonzero(npix_count)[0] #Find pixels within field (pixels with zero counts are excluded, but can be recovered later if they gain a count)

    m = np.zeros(hp.nside2npix(nside))
    m[npix_index] = npix_count[npix_index] #Fill map of the full sky with field
    kappa_pix = np.zeros(hp.nside2npix(nside))

    for i in range(len(npix)):
        index = np.where(npix == npix[i])
        kappa_pix[npix[i]] = np.mean(kappa[index]) #Determine mean kappa in each pixel
    return m, kappa_pix, npix_index

def hp_map_strong(ra, dec, kappa, gamma_1, gamma_2, z, nside):
    """Builds HealPix map of the number counts of a given field.
    IN:
    - ra (numpy.ndarray):   Right Acension in radians
    - dec (numpy.ndarray):  Declination in radians
    - kappa (numpy.ndarray):Projected matter density
    - z (numpy.ndarray):    Redshift
    - nside (int):          Resultion of the HealPix pixelation. Must be a power of 2
    
    OUT:
    - m (numpy.ndarray):                  HealPix map of the number counts
    - magnification_pix (numpy.ndarray):  Healpix map of the average magnification factor
    - npix_index (numpy.ndarray):         Indices of the HealPix map which are within the observed field
    """
    npix = hp.ang2pix(nside, dec, ra) #Find pixel index for each object
    npix_count = np.bincount(npix) #Find counts of objects within each pixel

    #---------------------------------

    npix_index    = np.nonzero(npix_count)[0] #Find pixels within field (pixels with zero counts are excluded, but can be recovered later if they gain a count)

    m = np.zeros(hp.nside2npix(nside))
    m[npix_index] = npix_count[npix_index] #Fill map of the full sky with field
    magnification_pix = np.zeros(hp.nside2npix(nside))
    
    gamma = np.sqrt(np.power(gamma_1, 2) + np.power(gamma_2, 2))

    for i in range(len(npix)):
        index = np.where(npix == npix[i])
        
        magn = np.power(np.absolute(np.power(1 - kappa[index], 2) - np.power(gamma[index], 2)), -1)
        
        magnification_pix[npix[i]] = np.mean(magn) #Determine mean magnification factor in each pixel
        
        
    return m, magnification_pix, npix_index   
    

def test_field(ra, dec, ramag, decmag, nside, npix_index, dilution):
  """Test function which compares the field of pixels taken from the whole sky by only considering non-zero pixels 
  to the field cut out as a poligon with the extreme data values as the vertices.
  
  If the fields are equivalent, the test prints: 'The field of pixels appears complete!'. In this case, one can be confident that the resolution is not too high.
  
  If the fields do not agree, the test prints: 'N pixels are not shared by the non-zero field and the quadrilateral field. M of these are at the edge and K of these are inside the field'
  In this case, further investigation is necessary:
    - If only pixels near the edge are left out, the currently used selection of non-zero pixels may be more accurate. It may be that the real field is not a true 4-sided polygon such that the polygon disregards or regards too many pixels.
    - If only pixels in the middle of the field are left out, it may be that the polygon would be more accurate. However, this usually means that the resolution selected by the nside is too high.
  
  IN:
    - ra (numpy.ndarray):           Right Acension in radians of the objects within the unmagnified field
    - dec (numpy.ndarray):          Declination in radians of the objects within the unmagnified field
    - ramag (numpy.ndarray):        Right Acension in radians of the objects within the magnified field
    - decmag (numpy.ndarray):       Declination in radians of the objects within the magnified field
    - nside (int):                  Resultion of the HealPix pixelation. Must be a power of 2
    - npix_index (numpy.ndarray):   Indices of the HealPy pixels included in the field at the specified nside (from taking the non-zero pixels)
    - dilution (bool):              If True, coordinates after magnification are considered and dilution factor is included in analysis of magnification.
  """
  #Check whether the field of pixels contains 'all' data
  if dilution == True:
    ra_max = max([max(ra), max(ramag)])
    dec_max = max([max(dec), max(decmag)])
    ra_min = min([min(ra), min(ramag)])
    dec_min = min([min(dec), min(decmag)])
  else:
    ra_max = max(ra)
    dec_max = max(dec)
    ra_min = min(ra)
    dec_min = min(dec)
  
  #Define 4-sided polygon in the sky
  corner1, corner2, corner3, corner4 = hp.ang2vec(dec_max, ra_max), hp.ang2vec(dec_max, ra_min), hp.ang2vec(dec_min, ra_min), hp.ang2vec(dec_min, ra_max)
  corners = np.array([corner1, corner2, corner3, corner4])
  npix_index_check = hp.query_polygon(nside, vertices = corners)

  #Find the indices of the pixels which are considered in the polygon, but were disregarded because their object count = 0
  left_out_pixels = np.setdiff1d(np.union1d(npix_index, npix_index_check), np.intersect1d(npix_index, npix_index_check))
  adjustment = np.where(left_out_pixels == np.intersect1d(left_out_pixels, npix_index))
  left_out_pixels = np.delete(left_out_pixels, adjustment)
  
  if len(left_out_pixels) > 0:
     ra_left, dec_left = hp.pix2ang(nside, left_out_pixels)
     pix_rad = hp.max_pixrad(nside)
     left_out_at_edge, left_out_in_center = [], []
     for i in range(len(ra_left)): #Determine the pixels which are up to one maximum pixel radius away from the edge of the polygon
       if abs(ra_left[i] - ra_max) < pix_rad or abs(ra_left[i] - ra_min) < pix_rad or abs(dec_left[i] - dec_max) < pix_rad or abs(dec_left[i] - dec_min) < pix_rad:
         left_out_at_edge.append(left_out_pixels[i])
         
       else: #All other pixels are considered to be at the centre
         left_out_in_center.append(left_out_pixels[i])
     print('{0} pixels are not shared by the non-zero field and the quadrilateral field. {1} of these are at the edge and {2} of these are inside the field'.format(len(left_out_pixels), len(left_out_at_edge), len(left_out_in_center))) 
  else:
    print('The field of pixels appears complete!')



def compare_mag(ra, dec, kappa, z, ramag, decmag, kappamag, zmag, nside, redshift_range=np.array([None]), dilution = True, tessellation = True, nside_tes = None, mag_limited = False, mag_limit_band = None, mag_limit = None):
    """Compares two populations of objects, one unmagnified and one magnified, by determining the relative count difference within each pixel of a certain HealPix pixelation,
    their respective kappa values and the uncertanties.

    IN:
    - ra (numpy.ndarray):       Right Acension in radians of the objects within the unmagnified field
    - dec (numpy.ndarray):      Declination in radians of the objects within the unmagnified field
    - kappa (numpy.ndarray):    Projected matter density of the objects within the unmagnified field
    - ramag (numpy.ndarray):    Right Acension in radians of the objects within the magnified field
    - decmag (numpy.ndarray):   Declination in radians of the objects within the magnified field
    - kappamag (numpy.ndarray): Projected matter density of the objects within the magnified field
    - nside (int):              Resolution of the HealPix pixelation. Must be a power of 2
    - redshift_range (iterable):n-componenet iterable object specifying the limits of the desired redshift bins.
    - dilution (bool):          If True, coordinates after magnification are considered and dilution factor is included in analysis of magnification.
    - tessellation (bool):      If True, the field will be subdivided into tiles given by HealPix pixels
    - nside_tes (int):          Resolution of the HealPix pixelation used for the tiles. Must be a power of 
    - mag_limited (bool):       If True, the galaxy sample is treated as if it has a definate magnitude limit. In that case, one needs to specify the mag_limit_band and the mag_limit.
    - mag_limit_band (str):     String containing the name of the column which contains the band magnitude with which the flux/magnitude limit of the sample has been set.
    - mag_limit (float):        Float specifying the magnitude limit in the band magnitude specifid in mag_limit_band.
    
    OUT:
    - reldiff_map2unmag (numpy.ndarray):    Array of relative difference between the magnified number counts and the unmagnified number counts for each redshift bin and each tile (dim = no. redshift bins x no. of tiles x no. of pixels in each tile)
    - kappa_reduced (numpy.ndarray):        Array of the mean value of kappa within each pixel, for each redshift bin and each tile (dim = no. redshift bins x no. of tiles x no. of pixels in each tile)
    - uncertainty (numpy.ndarray):          Array of the uncertainty on the relative difference values within each pixel, for each redshift bin and each tile (dim = no. redshift bins x no. of tiles x no. of pixels in each tile)
    """
    
    
    if redshift_range.all() != None and len(redshift_range) != 1:
        if not isinstance(redshift_range, Iterable) or len(redshift_range) < 2:
            raise Exception('Redshift range has to be an iterable object of length > 2.')
    else:
        redshift_range = np.array([min([min(z), min(zmag)]), max([max(z), max(zmag)])])
    
    
    reldiff_list, kappa_list, unc_list = [], [], []
    for j in range(0, len(redshift_range)-1):
        zmin = redshift_range[j]
        zmax = redshift_range[j+1]
        
        ra_j       = ra[(zmin <= z) & (z < zmax)]
        dec_j      = dec[(zmin <= z) & (z < zmax)]
        kappa_j    = kappa[(zmin <= z) & (z < zmax)]
        z_j        = z[(zmin <= z) & (z < zmax)]
        
        ramag_j    = ramag[(zmin <= zmag) & (zmag < zmax)]
        decmag_j   = decmag[(zmin <= zmag) & (zmag < zmax)]
        kappamag_j = kappamag[(zmin <= zmag) & (zmag < zmax)]
        zmag_j     = zmag[(zmin <= zmag) & (zmag < zmax)]

        if tessellation == True:
            if nside_tes != None:
                no_mag_tiles_pix = hp.ang2pix(nside_tes, dec_j, ra_j)
                mag_tiles_pix    = hp.ang2pix(nside_tes, decmag_j, ramag_j)
                common_tiles     = np.unique(np.intersect1d(no_mag_tiles_pix, mag_tiles_pix))
                
                reldiff_list_j, kappa_list_j, unc_list_j = [], [], []
                n = 0
                for i in common_tiles: #Divides pixels into non-overlapping spatial tiles
                    ra_i       = ra_j[no_mag_tiles_pix == i] 
                    dec_i      = dec_j[no_mag_tiles_pix == i]
                    kappa_i    = kappa_j[no_mag_tiles_pix == i]
                    z_i        = z_j[no_mag_tiles_pix == i]
                    ramag_i    = ramag_j[mag_tiles_pix == i]
                    decmag_i   = decmag_j[mag_tiles_pix == i]
                    kappamag_i = kappamag_j[mag_tiles_pix == i]
                    zmag_i     = zmag_j[mag_tiles_pix == i]
                    
                    c_map, kappa_map, npix_index        = hp_map(ra_i, dec_i, kappa_i, z_i, nside)              #Create unmagnified HealPix map
                    mapmag, kappa_mapmag, npix_indexmag = hp_map(ramag_i, decmag_i, kappamag_i, zmag_i, nside)  #Create magnified HealPix map
                    
                    npix_index = np.unique(np.concatenate((npix_index, npix_indexmag), axis=0)) #Take all non-zero pixels from either sample         
                    
                    map_reduced          = c_map[npix_index]
                    mapmag_reduced       = mapmag[npix_index]
                    kappa_mapmag_reduced = kappa_mapmag[npix_index]
                    
                    zero_to_one = np.where(map_reduced == 0)[0] #Find any edge cases
                    one_to_zero = np.where(mapmag_reduced == 0)[0]
                    to_remove   = np.concatenate((zero_to_one, one_to_zero))
                    
                    print('For {3} < z < {4}, in tile {0} with {1} galaxies in {2} pixels'.format(n, len(ra_i), len(map_reduced), round(zmin, 3), round(zmax, 3)))
                    print('Excluded {2} pixels with counts going from 0 to 1 in tile {1}: {0}'.format(np.array2string(zero_to_one, separator = ','), n, len(zero_to_one)))
                    print('Excluded {2} pixels with counts going from 1 to 0 in tile {1}: {0}'.format(np.array2string(one_to_zero, separator = ','), n, len(one_to_zero)))
                    
                    mapmag_reduced = np.delete(mapmag_reduced, to_remove) #Remove edge cases
                    kappa_reduced  = np.delete(kappa_mapmag_reduced, to_remove)
                    map_reduced    = np.delete(map_reduced, to_remove)
                    
                    vec         = np.array([map_reduced, mapmag_reduced])
                    cov         = np.cov(vec)
                    corr        = cov[0][1]/(np.std(vec[0])*np.std(vec[1])) #Determine Pearson correlation factor between samples
                    ratio       = np.divide(mapmag_reduced, map_reduced)
                    uncertainty = np.sqrt(abs(np.multiply(np.divide(mapmag_reduced, map_reduced**2), 1 + ratio - 2*corr*np.sqrt(ratio)))) #Calculate uncertainty for two correlated Poisson distributions
                    
                    reldiff_mag2unmag = np.divide((mapmag_reduced - map_reduced), map_reduced) #Determine relative difference
                    
                    reldiff_list_j.append(reldiff_mag2unmag)
                    kappa_list_j.append(kappa_reduced)
                    unc_list_j.append(uncertainty)
                    n += 1
                
                reldiff_list.append(reldiff_list_j)
                kappa_list.append(kappa_list_j)
                unc_list.append(unc_list_j)
                
            else:
                raise Exception('If tessellation is desired, please specify the HealPix nside for the tessellation.')

        else:
            c_map, kappa_map, npix_index        = hp_map(ra_j, dec_j, kappa_j, z_j, nside) #Create unmagnified HealPix map
            mapmag, kappa_mapmag, npix_indexmag = hp_map(ramag_j, decmag_j, kappamag_j, zmag_j, nside) #Create magnified HealPix map

            npix_index = np.unique(np.concatenate((npix_index, npix_indexmag), axis=0)) #Take all non-zero pixels from either sample

            #test_field(ra_j, dec_j, ramag_j, decmag_j, nside, npix_index, dilution) #May call test function here to find if the field is disregarding any pixels in the field (for suffienctly low resolutions, it should pass the test)

            map_reduced          = c_map[npix_index]
            mapmag_reduced       = mapmag[npix_index]
            kappa_mapmag_reduced = kappa_mapmag[npix_index]

            zero_to_one = np.where(map_reduced == 0)[0] #Find any edge cases
            one_to_zero = np.where(mapmag_reduced == 0)[0]
            to_remove = np.concatenate((zero_to_one, one_to_zero))

            print('Excluded {1} pixels with counts going from 0 to 1: {0}'.format(np.array2string(zero_to_one, separator = ','), len(zero_to_one)))
            print('Excluded {1} pixels with counts going from 1 to 0: {0}'.format(np.array2string(one_to_zero, separator = ','), len(one_to_zero)))

            mapmag_reduced  = np.delete(mapmag_reduced, to_remove) #Remove edge cases
            kappa_reduced_j = np.delete(kappa_mapmag_reduced, to_remove)
            map_reduced     = np.delete(map_reduced, to_remove)

            vec           = np.array([map_reduced, mapmag_reduced])
            cov           = np.cov(vec)
            corr          = cov[0][1]/(np.std(vec[0])*np.std(vec[1])) #Determine Pearson correlation factor between samples
            ratio         = np.divide(mapmag_reduced, map_reduced)
            uncertainty_j = np.sqrt(abs(np.multiply(np.divide(mapmag_reduced, map_reduced**2), 1 + ratio - 2*corr*np.sqrt(ratio)))) #Calculate uncertainty for two correlated Poisson distributions

            reldiff_mag2unmag_j = np.divide((mapmag_reduced - map_reduced), map_reduced) #Determine relative difference
            
            reldiff_list.append(reldiff_mag2unmag_j)
            kappa_list.append(kappa_reduced_j)
            unc_list.append(uncertainty_j)
    
    reldiff_list = np.array(reldiff_list)
    kappa_list   = np.array(kappa_list)
    unc_list     = np.array(unc_list)
    
    return reldiff_list, kappa_list, unc_list



def compare_mag_strong(ra, dec, kappa, gamma_1, gamma_2, z, ramag, decmag, kappamag, gamma_1mag, gamma_2mag, zmag, nside, redshift_range=np.array([None]), dilution = True, tessellation = True, nside_tes = None, mag_limited = False, mag_limit_band = None, mag_limit = None):
    """Compares two populations of objects, one unmagnified and one magnified, by determining the ration between the counts within each pixel of a certain HealPix pixelation,
    their respective magnification factor and the uncertanties.

    IN:
    - ra (numpy.ndarray):       Right Acension in radians of the objects within the unmagnified field
    - dec (numpy.ndarray):      Declination in radians of the objects within the unmagnified field
    - kappa (numpy.ndarray):    Projected matter density of the objects within the unmagnified field
    - z
    - gamma_1
    - gamma_2
    - ramag (numpy.ndarray):    Right Acension in radians of the objects within the magnified field
    - decmag (numpy.ndarray):   Declination in radians of the objects within the magnified field
    - kappamag (numpy.ndarray): Projected matter density of the objects within the magnified field
    - zmag
    - gamma_1mag
    - gamma_2mag
    - nside (int):              Resolution of the HealPix pixelation. Must be a power of 2
    - redshift_range (iterable):n-componenet iterable object specifying the limits of the desired redshift bins.
    - dilution (bool):          If True, coordinates after magnification are considered and dilution factor is included in analysis of magnification.
    - tessellation (bool):      If True, the field will be subdivided into tiles given by HealPix pixels
    - nside_tes (int):          Resolution of the HealPix pixelation used for the tiles. Must be a power of 
    - mag_limited (bool):       If True, the galaxy sample is treated as if it has a definate magnitude limit. In that case, one needs to specify the mag_limit_band and the mag_limit.
    - mag_limit_band (str):     String containing the name of the column which contains the band magnitude with which the flux/magnitude limit of the sample has been set.
    - mag_limit (float):        Float specifying the magnitude limit in the band magnitude specifid in mag_limit_band.
    
    OUT:
    - ratio_list(numpy.ndarray):       Array of the ratio between the magnified number counts and the unmagnified number counts for each redshift bin and each tile (dim = no. redshift bins x no. of tiles x no. of pixels in each tile)
    - magn_list(numpy.ndarray):        Array of the mean value of the magnification within each pixel, for each redshift bin and each tile (dim = no. redshift bins x no. of tiles x no. of pixels in each tile)
    - unc_list (numpy.ndarray):        Array of the uncertainty on the ratios within each pixel, for each redshift bin and each tile (dim = no. redshift bins x no. of tiles x no. of pixels in each tile)
    """
    
    
    if redshift_range.all() != None and len(redshift_range) != 1:
        if not isinstance(redshift_range, Iterable) or len(redshift_range) < 2:
            raise Exception('Redshift range has to be an iterable object of length > 2.')
    else:
        redshift_range = np.array([min([min(z), min(zmag)]), max([max(z), max(zmag)])])
    
    
    ratio_list, magn_list, unc_list = [], [], []
    for j in range(0, len(redshift_range)-1):
        zmin = redshift_range[j]
        zmax = redshift_range[j+1]
        
        ra_j       = ra[(zmin <= z) & (z < zmax)]
        dec_j      = dec[(zmin <= z) & (z < zmax)]
        kappa_j    = kappa[(zmin <= z) & (z < zmax)]
        z_j        = z[(zmin <= z) & (z < zmax)]
        gamma_1_j  = gamma_1[(zmin <= z) & (z < zmax)]
        gamma_2_j  = gamma_2[(zmin <= z) & (z < zmax)]
        
        ramag_j       = ramag[(zmin <= zmag) & (zmag < zmax)]
        decmag_j      = decmag[(zmin <= zmag) & (zmag < zmax)]
        kappamag_j    = kappamag[(zmin <= zmag) & (zmag < zmax)]
        zmag_j        = zmag[(zmin <= zmag) & (zmag < zmax)]
        gamma_1mag_j  = gamma_1mag[(zmin <= zmag) & (zmag < zmax)]
        gamma_2mag_j  = gamma_2mag[(zmin <= zmag) & (zmag < zmax)]

        if tessellation == True:
            if nside_tes != None:
                no_mag_tiles_pix = hp.ang2pix(nside_tes, dec_j, ra_j)
                mag_tiles_pix    = hp.ang2pix(nside_tes, decmag_j, ramag_j)
                common_tiles     = np.unique(np.intersect1d(no_mag_tiles_pix, mag_tiles_pix))
                
                ratio_list_j, magn_list_j, unc_list_j = [], [], []
                n = 0
                for i in common_tiles: #Divides pixels into non-overlapping spatial tiles
                    ra_i          = ra_j[no_mag_tiles_pix == i] 
                    dec_i         = dec_j[no_mag_tiles_pix == i]
                    kappa_i       = kappa_j[no_mag_tiles_pix == i]
                    z_i           = z_j[no_mag_tiles_pix == i]
                    gamma_1_i     = gamma_1_j[no_mag_tiles_pix == i]
                    gamma_2_i     = gamma_2_j[no_mag_tiles_pix == i]
                    ramag_i       = ramag_j[mag_tiles_pix == i]
                    decmag_i      = decmag_j[mag_tiles_pix == i]
                    kappamag_i    = kappamag_j[mag_tiles_pix == i]
                    zmag_i        = zmag_j[mag_tiles_pix == i]
                    gamma_1mag_i  = gamma_1mag_j[mag_tiles_pix == i]
                    gamma_2mag_i  = gamma_2mag_j[mag_tiles_pix == i]
                    
                    c_map, magnification_map, npix_index        = hp_map_strong(ra_i, dec_i, kappa_i, gamma_1_i, gamma_2_i, z_i, nside)                    #Create unmagnified HealPix map
                    mapmag, magnification_mapmag, npix_indexmag = hp_map_strong(ramag_i, decmag_i, kappamag_i, gamma_1mag_i, gamma_2mag_i, zmag_i, nside)  #Create magnified HealPix map
                    
                    npix_index = np.unique(np.concatenate((npix_index, npix_indexmag), axis=0)) #Take all non-zero pixels from either sample         
                    
                    map_reduced         = c_map[npix_index]
                    mapmag_reduced      = mapmag[npix_index]
                    magn_mapmag_reduced = magnification_mapmag[npix_index]
                    
                    zero_to_one = np.where(map_reduced == 0)[0] #Find any edge cases
                    one_to_zero = np.where(mapmag_reduced == 0)[0]
                    to_remove   = np.concatenate((zero_to_one, one_to_zero))
                    
                    print('For {3} < z < {4}, in tile {0} with {1} galaxies in {2} pixels'.format(n, len(ra_i), len(map_reduced), round(zmin, 3), round(zmax, 3)))
                    print('Excluded {2} pixels with counts going from 0 to 1 in tile {1}: {0}'.format(np.array2string(zero_to_one, separator = ','), n, len(zero_to_one)))
                    print('Excluded {2} pixels with counts going from 1 to 0 in tile {1}: {0}'.format(np.array2string(one_to_zero, separator = ','), n, len(one_to_zero)))
                    
                    mapmag_reduced = np.delete(mapmag_reduced, to_remove) #Remove edge cases
                    magn_reduced  = np.delete(magn_mapmag_reduced, to_remove)
                    map_reduced    = np.delete(map_reduced, to_remove)
                    
                    vec         = np.array([map_reduced, mapmag_reduced])
                    cov         = np.cov(vec)
                    corr        = cov[0][1]/(np.std(vec[0])*np.std(vec[1])) #Determine Pearson correlation factor between samples
                    ratio       = np.divide(mapmag_reduced, map_reduced)
                    
                    uncertainty = np.multiply(np.divide(1, map_reduced), np.sqrt(np.absolute(np.multiply(mapmag_reduced, 1 + ratio - 2*corr*np.sqrt(ratio))))) #Calculate uncertainty for two correlated Poisson distributions

                    ratio_mag2unmag = np.divide(mapmag_reduced, map_reduced) #Determine ratio
                    
                    ratio_list_j.append(ratio_mag2unmag)
                    magn_list_j.append(magn_reduced)
                    unc_list_j.append(uncertainty)
                    n += 1
                
                ratio_list.append(ratio_list_j)
                magn_list.append(magn_list_j)
                unc_list.append(unc_list_j)
                
            else:
                raise Exception('If tessellation is desired, please specify the HealPix nside for the tessellation.')

        else:
            c_map, magn_map, npix_index        = hp_map_strong(ra_j, dec_j, kappa_j, gamma_1_j, gamma_2_j, z_j, nside) #Create unmagnified HealPix map
            mapmag, magn_mapmag, npix_indexmag = hp_map_strong(ramag_j, decmag_j, kappamag_j, gamma_1mag_j, gamma_2mag, j, zmag_j, nside) #Create magnified HealPix map

            npix_index = np.unique(np.concatenate((npix_index, npix_indexmag), axis=0)) #Take all non-zero pixels from either sample

            #test_field(ra_j, dec_j, ramag_j, decmag_j, nside, npix_index, dilution) #May call test function here to find if the field is disregarding any pixels in the field (for suffienctly low resolutions, it should pass the test)

            map_reduced          = c_map[npix_index]
            mapmag_reduced       = mapmag[npix_index]
            magn_mapmag_reduced = magn_mapmag[npix_index]

            zero_to_one = np.where(map_reduced == 0)[0] #Find any edge cases
            one_to_zero = np.where(mapmag_reduced == 0)[0]
            to_remove = np.concatenate((zero_to_one, one_to_zero))

            print('Excluded {1} pixels with counts going from 0 to 1: {0}'.format(np.array2string(zero_to_one, separator = ','), len(zero_to_one)))
            print('Excluded {1} pixels with counts going from 1 to 0: {0}'.format(np.array2string(one_to_zero, separator = ','), len(one_to_zero)))

            mapmag_reduced  = np.delete(mapmag_reduced, to_remove) #Remove edge cases
            magn_reduced_j = np.delete(magn_mapmag_reduced, to_remove)
            map_reduced     = np.delete(map_reduced, to_remove)

            vec           = np.array([map_reduced, mapmag_reduced])
            cov           = np.cov(vec)
            corr          = cov[0][1]/(np.std(vec[0])*np.std(vec[1])) #Determine Pearson correlation factor between samples
            ratio         = np.divide(mapmag_reduced, map_reduced)
            uncertainty_j = np.multiply(np.divide(1, map_reduced), np.sqrt(np.absolute(np.multiply(mapmag_reduced, 1 + ratio - 2*corr*np.sqrt(ratio))))) #Calculate uncertainty for two correlated Poisson distributions

            ratio_mag2unmag_j = np.divide(mapmag_reduced, map_reduced) #Determine relative difference
            
            ratio_list.append(ratio_mag2unmag_j)
            magn_list.append(magn_reduced_j)
            unc_list.append(uncertainty_j)
    
    ratio_list   = np.array(ratio_list)
    magn_list    = np.array(magn_list)
    unc_list     = np.array(unc_list)
    
    return ratio_list, magn_list, unc_list


def f(A, x):
  "Linear function for fitting"
  return A*x



def fit_kappa_vs_diff_tiled(reldiff_list, kappa_list, unc_list, nside, nside_tes, redshift_range=np.array([None]), least_squares = True, dilution = True,  mag_limited = False, mag_limit_band = None, mag_limit = None, plots = False, tag = None):
    """Produces a linear fit of the relation of the relative difference of counts vs. the convergence (kappa). Then, estimates the final alpha estimate for each redshift bin by taking the average of the alpha values from each tile.

    IN:
    - reldiff_list (numpy.ndarray):         Relative difference between the magnified number counts and the unmagnified number counts for each redshift bin and each tile (dim = no. redshift bins x no. of tiles x no. of pixels in each tile)
    - kappa_list (numpy.ndarray):           Mean value of kappa within each pixel, for each redshift bin and each tile (dim = no. redshift bins x no. of tiles x no. of pixels in each tile)
    - unc_list (numpy.ndarray):             Uncertainty on the relative difference values within each pixel, for each redshift bin and each tile (dim = no. redshift bins x no. of tiles x no. of pixels in each tile)
    - nside (int):                          Resolution of the HealPix pixelation. Must be a power of 2
    - nside_tes (int):                      Resolution of the HealPix pixelation used for the tiles. Must be a power of 2
    - redshift_range (iterable):            n-componenet iterable object specifying the limits of the desired redshift bins
    - least_squares (bool)                  Type of fitting desired for the linear fit within each tile. If true, least squares fit is applied. If False, applies a weighted fit based on the Poisson noise associated with the counts in each pixel
    - dilution (bool):                      If True, coordinates after magnification are considered and dilution factor is included in analysis of magnification
    - mag_limited (bool):                   If True, the galaxy sample is treated as if it has a definate magnitude limit. In that case, one needs to specify the mag_limit_band and the mag_limit
    - mag_limit_band (str):                 String containing the name of the column which contains the band magnitude with which the flux/magnitude limit of the sample has been set
    - mag_limit (float):                    Float specifying the magnitude limit in the band magnitude specifid in mag_limit_band
    - plots (bool):                         If True, produces and saves figures of the fit in each tile and redshift bin
    - tag (str):                            String used as a distinguishing tag within the figure names. Only necessary, if plots == True
    
    OUT:
    - alpha_list (numpy.ndarray):           Alpha estimates obtained from the linear fit of the data in each tile and redshift bin (dim = no. redshift bins x no. of tiles )
    - alpha_unc_list (numpy.ndarray):       Uncertainty of the alpha estimates obtained from the linear fit of the data in each tile and redshift bin (dim = no. redshift bins x no. of tiles)
    - gof_list (numpy.ndarray):             Goodness of fit the linear fit of the data in each tile and redshift bin (dim = no. redshift bins x no. of tiles). If least_squares == True, all G.o.F.s will equal to 1. (dim = no. redshift bins x no. of tiles)
    - npix_list (numpy.ndarray):            Number of npix pixels contained in each tile within each redshift bin (dim = no. redshift bins x no. of tiles)
    - tiles (numpy.ndarray):                Number ID of each tile contained within each redshift bin (dim = no. redshift bins x no. of tiles)
    - alpha_est_list (numpy.ndarray):       Final alpha estimate for each redshift bin
    - alpha_est_unc_list (numpy.ndarray):   Uncertainty of the final estimate for each redshift bin

    """
    if plots == True and tag == None:
        raise Exception('If diagnostic plots of the the rel. diff. vs. kappa relation in each tile are desired, please provide a tag.')
    
    alpha_list, a_unc_list, gof_list, npix_list, tiles = [], [], [], [], []
    alpha_est_list, alpha_est_unc_list = [],  []
    
    for i in range(0, len(reldiff_list)):
        alpha_list_j, a_unc_list_j, gof_list_j, npix_list_j = [], [], [], []
        
        if redshift_range.all() != None:
            path = 'kappa_vs_diff_plots/z_{0}_to_{1}'.format(round(redshift_range[i], 3), round(redshift_range[i+1], 3))
        else:
            path = 'kappa_vs_diff_plots/All_z'
            
        if plots == True and not os.path.isdir(path):
            os.makedirs(path)
        
        for j in range(0, len(reldiff_list[i])):
            reldiff, kappa, unc = reldiff_list[i][j], kappa_list[i][j], unc_list[i][j]
            if least_squares == True:
                popt, pcov = curve_fit(f, kappa, reldiff)
                gof_list_j.append(1)
            else:
                popt, pcov      = curve_fit(f, kappa, reldiff, sigma = unc) #Fit a straight line to these values
                pull            = np.divide(reldiff - popt[0]*kappa, unc)
                chi_squared     = np.sum(pull**2)
                goodness_of_fit = chi_squared/(len(reldiff)-1)
                
                gof_list_j.append(goodness_of_fit)
    
            perr  = np.sqrt(np.diag(pcov))[0]
            a_unc = perr/2
            
            if dilution == False:
                alpha = popt[0]/2
            else:
                alpha = ((popt[0]/2)) + 1
            
            if plots == True:
                
                plt.errorbar(kappa, reldiff, yerr = unc,  fmt =  'o', ecolor =  'gray', ms = 4, lw = 0.2)
                if least_squares == True:
                    plt.plot(kappa, kappa*popt[0], lw = 0.5, c = 'k', label = r'$\alpha_{{\kappa}} = {a} \pm {b}; \rm{{slope}} = {d} \pm {e}$'.format(a = round(alpha, 2), b = round(a_unc, 2), d = round(popt[0], 2), e = round(perr, 2))) 
                else:
                    plt.plot(kappa, kappa*popt[0], lw = 0.5, c = 'k', label = r'$\alpha_{{\kappa}} = {a} \pm {b}; \rm{{slope}} = {d} \pm {e}; \frac{{\chi^{{2}}}}{{N_{{pixels}}-1}} =$ {c}'.format(a = round(alpha, 2), b = round(a_unc, 2), c = round(goodness_of_fit, 2), d = round(popt[0], 2), e = round(perr, 2))) 
                
                plt.plot(kappa, kappa*(popt[0]+perr), lw = 0.5, c = 'k', ls = 'dotted')
                plt.plot(kappa, kappa*(popt[0]-perr), lw = 0.5, c = 'k', ls = 'dotted') 
                
                plt.ylabel(r'$\frac{N - N_{0}}{N_{0}}$')
                plt.xlabel(r'$\overline{\kappa}$')
                plt.ylim([-1, 1])
                plt.ylim([-0.25, 0.25])
                plt.legend(prop={'size': 12})
                plt.tight_layout(pad=0.8)
                
                if redshift_range.all() == None:    
                    plt.savefig(path + '/{0}_{1}_kappa_vs_countdiff_fit_tiles_nside={4}_tile={2}_dilution={3}.pdf'.format(nside, tag, j, dilution, nside_tes))
                else:
                    plt.savefig(path + '/{0}_{1}_kappa_vs_countdiff_fit_tiles_nside={6}_tile={2}_z_in_{3}-{4}_dilution={5}.pdf'.format(nside, tag, j, round(redshift_range[i], 3), round(redshift_range[i+1], 3), dilution, nside_tes))
                
                plt.close('all')
            
            alpha_list_j.append(alpha)
            a_unc_list_j.append(a_unc)
            npix_list_j.append(len(reldiff))
        
        tiles_j = list(range(0, len(reldiff_list[i])))
        # Remove outlier found in sample (due to inaccurate uncertainties of the galaxy number counts in the MICE2 simulations)
        if nside == 64 and nside_tes == 4 and redshift_range[i] == 0.2 and redshift_range[i+1] == 0.5 and least_squares == True and dilution == True:
            print('Removed outlier tile: 4')
            del alpha_list_j[4]
            del a_unc_list_j[4]
            del gof_list_j[4]
            del npix_list_j[4]
            del tiles_j[4]
        
        alpha_list.append(alpha_list_j)
        a_unc_list.append(a_unc_list_j)
        gof_list.append(gof_list_j)
        npix_list.append(npix_list_j)
        tiles.append(tiles_j)
        
        alpha_list_j = np.array(alpha_list_j)
        a_unc_list_j = np.array(a_unc_list_j)
        gof_list_j   = np.array(gof_list_j)
        npix_list_j  = np.array(npix_list_j)
        tiles_j      = np.array(tiles_j)
        
        weight    = np.divide(1, a_unc_list_j**2)
        alpha_est = np.average(alpha_list_j, weights= weight)
        n_weights = len(np.where(weight != 0)[0])
        alpha_unc = np.sqrt(np.divide(np.sum(np.multiply(weight, (alpha_list_j - alpha_est)**2)), (n_weights - 1)*np.sum(weight)/n_weights))/np.sqrt(len(alpha_list_j))

        print('Final alpha for z between {2} and {3} = {0} pm {1}'.format(round(alpha_est, 3), round(alpha_unc, 3), round(redshift_range[i], 3), round(redshift_range[i+1], 3)))
        
        if plots == True:
            if least_squares == True:
                marker, color_marker = 3.0, 0.0
            else:
                marker, color_marker = 0.0, 3.0
                
            plt.scatter(tiles_j, alpha_list_j, c = gof_list_j, s = color_marker)
            plt.errorbar(tiles_j, alpha_list_j, yerr = a_unc_list_j, fmt =  'o', ecolor =  'gray', ms = marker, lw = 0.1)
            plt.plot(tiles_j, np.repeat(alpha_est, len(tiles_j)), lw = 0.5, c = 'k', label = r'$\alpha = {0} \pm {1}$'.format(round(alpha_est, 2), round(alpha_unc, 2)))
            plt.plot(tiles_j, np.repeat(alpha_est+alpha_unc, len(tiles_j)), lw = 0.5, c = 'k', ls = 'dotted')
            plt.plot(tiles_j, np.repeat(alpha_est-alpha_unc, len(tiles_j)), lw = 0.5, c = 'k', ls = 'dotted')
            
            if least_squares == False:
                plt.colorbar(label = 'G.o.F.')
            
            plt.legend()
            plt.xlabel(r'Tile index')
            plt.ylabel(r'$\alpha$') 
            plt.legend(prop={'size': 12})
            plt.tight_layout(pad=0.8)
            
            if redshift_range.all == None:
                plt.savefig(path + '/{0}_{1}_alpha_per_tile_tiles_nside={3}_ALL_TILES_dilution={2}.pdf'.format(nside, tag, dilution, nside_tes))
            else:
                plt.savefig(path + '/{0}_{1}_alpha_per_tile_tiles_nside={5}_ALL_TILES_z_in_{2}-{3}_dilution={4}.pdf'.format(nside, tag, redshift_range[i], redshift_range[i+1], dilution, nside_tes))
            plt.close('all')
        
        alpha_est_list.append(alpha_est)
        alpha_est_unc_list.append(alpha_unc)
              
    alpha_list     = np.array(alpha_list)
    alpha_unc_list = np.array(a_unc_list)
    gof_list       = np.array(gof_list)
    npix_list      = np.array(npix_list)
    tiles          = np.array(tiles)

    alpha_est_list     = np.array(alpha_est_list)
    alpha_est_unc_list = np.array(alpha_est_unc_list)
    
    return alpha_list, alpha_unc_list, gof_list, npix_list, tiles, alpha_est_list, alpha_est_unc_list


def fit_ratio_vs_magn_tiled(ratio_list, magn_list, unc_list, nside, nside_tes, redshift_range=np.array([None]), least_squares = True, dilution = True,  mag_limited = False, mag_limit_band = None, mag_limit = None, plots = False, tag = None):
    """Produces a linear fit of the relation of the count ratio vs. the log of the magnification factor (mu). Then, estimates the final alpha estimate for each redshift bin by taking the average of the alpha values from each tile.

    IN:
    - ratio_list (numpy.ndarray):           Ratio between the magnified number counts and the unmagnified number counts for each redshift bin and each tile (dim = no. redshift bins x no. of tiles x no. of pixels in each tile)
    - magn_list (numpy.ndarray):            Mean value of the magnification factor within each pixel, for each redshift bin and each tile (dim = no. redshift bins x no. of tiles x no. of pixels in each tile)
    - unc_list (numpy.ndarray):             Uncertainty on the ratios within each pixel, for each redshift bin and each tile (dim = no. redshift bins x no. of tiles x no. of pixels in each tile)
    - nside (int):                          Resolution of the HealPix pixelation. Must be a power of 2
    - nside_tes (int):                      Resolution of the HealPix pixelation used for the tiles. Must be a power of 2
    - redshift_range (iterable):            n-componenet iterable object specifying the limits of the desired redshift bins
    - least_squares (bool)                  Type of fitting desired for the linear fit within each tile. If true, least squares fit is applied. If False, applies a weighted fit based on the Poisson noise associated with the counts in each pixel
    - dilution (bool):                      If True, coordinates after magnification are considered and dilution factor is included in analysis of magnification
    - mag_limited (bool):                   If True, the galaxy sample is treated as if it has a definate magnitude limit. In that case, one needs to specify the mag_limit_band and the mag_limit
    - mag_limit_band (str):                 String containing the name of the column which contains the band magnitude with which the flux/magnitude limit of the sample has been set
    - mag_limit (float):                    Float specifying the magnitude limit in the band magnitude specifid in mag_limit_band
    - plots (bool):                         If True, produces and saves figures of the fit in each tile and redshift bin
    - tag (str):                            String used as a distinguishing tag within the figure names. Only necessary, if plots == True
    
    OUT:
    - alpha_list (numpy.ndarray):           Alpha estimates obtained from the linear fit of the data in each tile and redshift bin (dim = no. redshift bins x no. of tiles )
    - alpha_unc_list (numpy.ndarray):       Uncertainty of the alpha estimates obtained from the linear fit of the data in each tile and redshift bin (dim = no. redshift bins x no. of tiles)
    - gof_list (numpy.ndarray):             Goodness of fit the linear fit of the data in each tile and redshift bin (dim = no. redshift bins x no. of tiles). If least_squares == True, all G.o.F.s will equal to 1. (dim = no. redshift bins x no. of tiles)
    - npix_list (numpy.ndarray):            Number of npix pixels contained in each tile within each redshift bin (dim = no. redshift bins x no. of tiles)
    - tiles (numpy.ndarray):                Number ID of each tile contained within each redshift bin (dim = no. redshift bins x no. of tiles)
    - alpha_est_list (numpy.ndarray):       Final alpha estimate for each redshift bin
    - alpha_est_unc_list (numpy.ndarray):   Uncertainty of the final estimate for each redshift bin

    """
    if plots == True and tag == None:
        raise Exception('If diagnostic plots of the the ratio vs. magnification relation in each tile are desired, please provide a tag.')
    
    alpha_list, a_unc_list, gof_list, npix_list, tiles = [], [], [], [], []
    alpha_est_list, alpha_est_unc_list = [],  []
    
    for i in range(0, len(ratio_list)):
        alpha_list_j, a_unc_list_j, gof_list_j, npix_list_j = [], [], [], []
        
        if redshift_range.all() != None:
            path = 'ratio_vs_magn_plots/z_{0}_to_{1}'.format(round(redshift_range[i], 3), round(redshift_range[i+1], 3))
        else:
            path = 'ratio_vs_magn_plots/All_z'
            
        if plots == True and not os.path.isdir(path):
            os.makedirs(path)
        
        for j in range(0, len(ratio_list[i])):
            ratio, magn, unc = np.log(ratio_list[i][j]), np.log(magn_list[i][j]), np.multiply(unc_list[i][j], np.power(ratio_list[i][j], -1))
            if least_squares == True:
                popt, pcov = curve_fit(f, magn, ratio)
                gof_list_j.append(1)
            else:
                popt, pcov      = curve_fit(f, magn, ratio, sigma = unc) #Fit a straight line to these values
                pull            = np.divide(ratio - popt[0]*magn, unc)
                chi_squared     = np.sum(pull**2)
                goodness_of_fit = chi_squared/(len(ratio)-1)
                
                gof_list_j.append(goodness_of_fit)
    
            perr  = np.sqrt(np.diag(pcov))[0]
            a_unc = perr
            
            if dilution == False:
                alpha = popt[0]
            else:
                alpha = popt[0] + 1
            
            if plots == True:
                
                plt.errorbar(magn, ratio, yerr = unc,  fmt =  'o', ecolor =  'gray', ms = 3, lw = 0.2)
                if least_squares == True:
                    plt.plot(magn, magn*popt[0], lw = 0.5, c = 'k', label = r'$\alpha_{{\mu}} = {a} \pm {b}; \rm{{slope}} = {d} \pm {e}$'.format(a = round(alpha, 2), b = round(a_unc, 2), d = round(popt[0], 2), e = round(perr, 2))) 
                else:
                    plt.plot(magn, magn*popt[0], lw = 0.5, c = 'k', label = r'$\alpha_{{\mu}} = {a} \pm {b}; \rm{{slope}} = {d} \pm {e}; \frac{{\chi^{{2}}}}{{N_{{pixels}}-1}} =$ {c}'.format(a = round(alpha, 2), b = round(a_unc, 2), c = round(goodness_of_fit, 2), d = round(popt[0], 2), e = round(perr, 2))) 
                
                plt.plot(magn, magn*(popt[0]+perr), lw = 0.5, c = 'k', ls = 'dotted')
                plt.plot(magn, magn*(popt[0]-perr), lw = 0.5, c = 'k', ls = 'dotted') 
                
                plt.ylabel(r'$\rm{log}(\frac{N}{N_{0}})$')
                plt.xlabel(r'$\rm{log}(\overline{\mu})$')
                plt.ylim([-0.3, 0.3])
                plt.legend()
                plt.ylim([-0.05, 0.05])
                plt.legend(prop={'size': 12})
                plt.tight_layout(pad=0.8)
                
                if redshift_range.all() == None:
                    plt.savefig(path + '/{0}_{1}_ratio_vs_magn_fit_tiles_nside={4}_tile={2}_dilution={3}.pdf'.format(nside, tag, j, dilution, nside_tes))
                else:
                    plt.savefig(path + '/{0}_{1}_ratio_vs_magn_fit_tiles_nside={6}_tile={2}_z_in_{3}-{4}_dilution={5}.pdf'.format(nside, tag, j, round(redshift_range[i], 3), round(redshift_range[i+1], 3), dilution, nside_tes))
                
                plt.close('all')
            
            alpha_list_j.append(alpha)
            a_unc_list_j.append(a_unc)
            npix_list_j.append(len(ratio))
        
        tiles_j = list(range(0, len(ratio_list[i])))
        # Remove outlier found in sample (due to inaccurate uncertainties of the galaxy number counts in the MICE2 simulations)
        if nside == 64 and nside_tes == 4 and redshift_range[i] == 0.2 and redshift_range[i+1] == 0.5 and least_squares == True and dilution == True:
            print('Removed outlier tile: 4')
            del alpha_list_j[4]
            del a_unc_list_j[4]
            del gof_list_j[4]
            del npix_list_j[4]
            del tiles_j[4]
        
        alpha_list.append(alpha_list_j)
        a_unc_list.append(a_unc_list_j)
        gof_list.append(gof_list_j)
        npix_list.append(npix_list_j)
        tiles.append(tiles_j)
        
        alpha_list_j = np.array(alpha_list_j)
        a_unc_list_j = np.array(a_unc_list_j)
        gof_list_j   = np.array(gof_list_j)
        npix_list_j  = np.array(npix_list_j)
        tiles_j      = np.array(tiles_j)
        
        weight    = np.divide(1, a_unc_list_j**2)
        alpha_est = np.average(alpha_list_j, weights= weight)
        n_weights = len(np.where(weight != 0)[0])
        alpha_unc = np.sqrt(np.divide(np.sum(np.multiply(weight, (alpha_list_j - alpha_est)**2)), (n_weights - 1)*np.sum(weight)/n_weights))/np.sqrt(len(alpha_list_j))

        print('Final alpha for z between {2} and {3} = {0} pm {1}'.format(round(alpha_est, 3), round(alpha_unc, 3), round(redshift_range[i], 3), round(redshift_range[i+1], 3)))
        
        if plots == True:
            if least_squares == True:
                marker, color_marker = 3.0, 0.0
            else:
                marker, color_marker = 0.0, 3.0
                
            plt.scatter(tiles_j, alpha_list_j, c = gof_list_j, s = color_marker)
            plt.errorbar(tiles_j, alpha_list_j, yerr = a_unc_list_j, fmt =  'o', ecolor =  'gray', ms = marker, lw = 0.1)
            plt.plot(tiles_j, np.repeat(alpha_est, len(tiles_j)), lw = 0.5, c = 'k', label = r'$\alpha = {0} \pm {1}$'.format(round(alpha_est, 2), round(alpha_unc, 2)))
            plt.plot(tiles_j, np.repeat(alpha_est+alpha_unc, len(tiles_j)), lw = 0.5, c = 'k', ls = 'dotted')
            plt.plot(tiles_j, np.repeat(alpha_est-alpha_unc, len(tiles_j)), lw = 0.5, c = 'k', ls = 'dotted')
            
            if least_squares == False:
                plt.colorbar(label = 'G.o.F.')
            
            plt.legend()
            plt.xlabel(r'Tile index')
            plt.ylabel(r'$\alpha$') 
            plt.legend(prop={'size': 12})
            plt.tight_layout(pad=0.8)
            
            if redshift_range.all == None:
                plt.savefig(path + '/{0}_{1}_magn_alpha_per_tile_tiles_nside={3}_ALL_TILES_dilution={2}.pdf'.format(nside, tag, dilution, nside_tes))
            else:
                plt.savefig(path + '/{0}_{1}_magn_alpha_per_tile_tiles_nside={5}_ALL_TILES_z_in_{2}-{3}_dilution={4}.pdf'.format(nside, tag, redshift_range[i], redshift_range[i+1], dilution, nside_tes))
            plt.close('all')
        
        alpha_est_list.append(alpha_est)
        alpha_est_unc_list.append(alpha_unc)
              
    alpha_list     = np.array(alpha_list)
    alpha_unc_list = np.array(a_unc_list)
    gof_list       = np.array(gof_list)
    npix_list      = np.array(npix_list)
    tiles          = np.array(tiles)

    alpha_est_list     = np.array(alpha_est_list)
    alpha_est_unc_list = np.array(alpha_est_unc_list)
    
    return alpha_list, alpha_unc_list, gof_list, npix_list, tiles, alpha_est_list, alpha_est_unc_list


def get_magnitude_counts(magnitudes, z, bin_edges, redshift_range=np.array([None]), magnified = True, mag_limited = False, mag_limit = None):
    """Gives the object counts within certain magnitude bins for each redshift bin.
    
    IN:
    - magnitudes (numpy.ndarray):   Magnitudes in a particular band
    - z (numpy.ndarray):            Redshift
    - bin_edges (numpy.ndarray):    Arbitrary limits of the magnitude bins
    - redshift_range (iterable):    n-componenet iterable object specifying the limits of the desired redshift bins
    - magnified (bool):             If True, the galaxy sample is treated as if it has been magnified. Only necessary, if working with a magnitude-limited sample
    - mag_limited (bool):           If True, the galaxy sample is treated as if it has a definate magnitude limit. In that case, one needs to specify the mag_limit
    - mag_limit (float):            Float specifying the magnitude limit in the band magnitude specifid in mag_limit_band
    
    OUT:
    - counts (numpy.ndarray):           Number of object counts in each magnitude bin for each redshift bin (dim = no. redshift bins x no. of magnitude bins)
    - norm_counts (numpy.ndarray):      Normalised distribution of objects in each magnitude bin for each redshift bin (dim = no. redshift bins x no. of magnitude bins)
    - redshift_range (numpy.ndarray):   Updated n-componenet iterable object specifying the limits of the desired redshift bins. If none was given in the input of the function, it will just contain the min and max of z
    """
    
    bool_list  = np.logical_not(np.isnan(magnitudes))
    magnitudes = magnitudes[bool_list]
    z          = z[bool_list]
    
    if mag_limited == True and magnified == True:
        magnitudes = magnitudes[magnitudes < mag_limit]
        z = z[magnitudes < mag_limit]
    
    if redshift_range.all() != None and len(redshift_range) != 1:
        if not isinstance(redshift_range, Iterable) or len(redshift_range) < 2:
            raise Exception('Redshift range has to be an iterable object of length > 2.')
    else:
        redshift_range = np.array([min(z), max(z)])
    
    counts, norm_counts = [], []
    
    for i in range(0, len(redshift_range)-1):
        zmin = redshift_range[i]
        zmax = redshift_range[i+1]
        
        magnitudes_i               = magnitudes[(zmin <= z) & (z < zmax)]
        counts_i, bin_edges_i      = np.histogram(magnitudes_i, bins = bin_edges)
        norm_counts_i, bin_edges_i = np.histogram(magnitudes_i, bins = bin_edges, density =  True)
        
        counts.append(counts_i)
        norm_counts.append(norm_counts_i)
    
    counts = np.array(counts)
    norm_counts = np.array(norm_counts)
    
    return counts, norm_counts, redshift_range
    


def get_alpha_nc(counts, norm_counts, bin_edges, redshift_range=np.array([None]), peak_prominence = 0.005):
    """Gives the alpha values estimated from the gradient of the observed magnitude distributions and the turn off magnitude of the galaxy population.
    
    IN:
    - counts (numpy.ndarray):       Number of object counts in each magnitude bin for each redshift bin (dim = no. redshift bins x no. of magnitude bins)
    - norm_counts (numpy.ndarray):  Normalised distribution of objects in each magnitude bin for each redshift bin (dim = no. redshift bins x no. of magnitude bins)
    - bin_edges (numpy.ndarray):    Arbitrary limits of the magnitude bins
    - redshift_range (iterable):    n-componenet iterable object specifying the limits of the desired redshift bins
    - peak_prominence (float):      Prominence parameter in the scipy.signal.find_peaks function used for peak selection. May have to be adjusted manually depending on the dataset used.
    
    OUT:
    - alpha_nc_list (numpy.ndarray):    Alpha values for each magnitude bin estimated from the gradient of the count distribution (dim = no. redshift bins x no. of magnitude bins)
    - turn_off_list (numpy.ndarray):    Turn-off magnitude in each redshift bin
    - turn_off_unc_list(numpy.ndarray): Uncertainty of the turn-off magnitude in each redshift bin
    
    """
    
    if not isinstance(redshift_range, Iterable) or len(redshift_range) < 2:
        raise Exception('Redshift range has to be an iterable object of length > 2.')
    
    bin_width   = bin_edges[1:] - bin_edges[:-1]
    bin_centers = bin_edges[:-1] + (bin_width)/2
    
    alpha_nc_list, turn_off_list, turn_off_unc_list = [], [], []
    
    for i in range(0, len(redshift_range)-1):
        counts_i          = counts[i]
        norm_counts_i     = norm_counts[i]
        turn_off_index, _ = find_peaks(norm_counts_i, height=0, prominence=(peak_prominence, None))
        turn_off          = float(bin_centers[turn_off_index][-1])
        turn_off_unc      = bin_width[turn_off_index][-1]
        
        grad     = np.gradient(np.log10(counts_i), bin_centers)
        alpha_nc = 2.5*grad
        
        alpha_nc_list.append(alpha_nc)
        turn_off_list.append(turn_off)
        turn_off_unc_list.append(turn_off_unc)
    
    alpha_nc_list     = np.array(alpha_nc_list)
    turn_off_list     = np.array(turn_off_list)
    turn_off_unc_list = np.array(turn_off_unc_list)
    
    return alpha_nc_list, turn_off_list, turn_off_unc_list



def get_calibration(alpha_est_list, alpha_est_unc_list, alpha_nc_list, turn_off_list, turn_off_unc_list, counts_list, bin_edges, redshift_range):
    """Calibrates the range over which the alpha estimates from observations are averaged to produce the best agreement with the alpha estimates from simulations where the underlying convergence is known.
    
    IN:
    - alpha_est_list (numpy.ndarray):       Final alpha estimate for each redshift bin (output from fit_kappa_vs_diff_tiled)
    - alpha_est_unc_list (numpy.ndarray):   Uncertainty of the final estimate for each redshift bin (output from fit_kappa_vs_diff_tiled)
    - alpha_nc_list (numpy.ndarray):        Alpha values for each magnitude bin estimated from the gradient of the count distribution (dim = no. redshift bins x no. of magnitude bins)
    - turn_off_list (numpy.ndarray):        Turn-off magnitude in each redshift bin
    - turn_off_unc_list(numpy.ndarray):     Uncertainty of the turn-off magnitude in each redshift bin
    - counts_list (numpy.ndarray):           Number of object counts in each magnitude bin for each redshift bin (dim = no. redshift bins x no. of magnitude bins)
    - bin_edges (numpy.ndarray):            Arbitrary limits of the magnitude bins used for alpha_nc_list and counts_list
    - redshift_range (iterable):            n-componenet iterable object specifying the limits of the desired redshift bins
    
    OUT:
    - max_overlap_indx_list (numpy.ndarray):        Array index of the alpha estimate from observations in alpha_list which agrees best with the alpha estimate from alpha_est_list for each redshift bin
    - max_overlap_range_list (numpy.ndarray):       Magnitude range averaged over to obtain the alpha estimate from observations in alpha_list which agrees best with the alpha estimate from alpha_est_list for each redshift bin. This is the magnitude range that is used for the calibration of the mock observations
    - max_overlap_alpha_list (numpy.ndarray):       Calibrated alpha estimate from observations in alpha_list which agrees best with the alpha estimate from alpha_est_list for each redshift bin
    - max_overlap_alpha_unc_list (numpy.ndarray):   Uncertainty of the alpha estimate from observations in alpha_list which agrees best with the alpha estimate from alpha_est_list for each redshift bin
    - overlap_list (numpy.ndarray):                 Overlap with the alpha estimate from alpha_est_list for each magnitude range considered in each redshift bin (dim = no. of magnitude bins x no. of redshift bins)
    - ranges_list (numpy.ndarray):                  Magnitude ranges considered for each alpha estimate (dim = no. of magnitude bins x no. of redshift bins)
    - alpha_list (numpy.ndarray):                   Alpha estimates obtained for each mangnitude range considered in each redshift bin (dim = no. of magnitude bins x no. of redshift bins)
    - alpha_unc_list (numpy.ndarray):               Uncertainty of the alpha estimates obtained for each mangnitude range considered in each redshift bin (dim = no. of magnitude bins x no. of redshift bins)
    
    """
    
    if not isinstance(redshift_range, Iterable) or len(redshift_range) < 2:
        raise Exception('Redshift range has to be an iterable object of length > 2.')
    
    bin_width   = bin_edges[1:] - bin_edges[:-1]
    bin_centers = bin_edges[:-1] + (bin_width)/2
    
    max_overlap_indx_list, max_overlap_range_list, max_overlap_alpha_list, max_overlap_alpha_unc_list = [], [], [], []
    ranges_list, alpha_list, alpha_unc_list, overlap_list = [], [], [], []
    
    for j in range(0, len(redshift_range)-1):   
        
        alpha_est     = alpha_est_list[j]
        alpha_est_unc = alpha_est_unc_list[j]
        
        alpha_nc     = alpha_nc_list[j]
        turn_off     = turn_off_list[j]
        turn_off_unc = turn_off_unc_list[j]
        counts       = counts_list[j]
        
        alpha_nc_cl    = alpha_nc[~np.isnan(alpha_nc)]
        bin_centers_cl = bin_centers[~np.isnan(alpha_nc)]
        bin_centers_cl = bin_centers_cl[~np.isinf(alpha_nc_cl)]
        counts_cl      = counts[~np.isnan(alpha_nc)]
        counts_cl      = counts_cl[~np.isinf(alpha_nc_cl)]
        alpha_nc_cl    = alpha_nc_cl[~np.isinf(alpha_nc_cl)]
        
        alpha_list_j, alpha_unc_list_j, overlap_list_j = [], [], []
        
        for i in range(2, len(alpha_nc_cl[bin_centers_cl <= turn_off])):            
            
            near_turnoff_i = alpha_nc_cl[bin_centers_cl <= turn_off][-i:-1]
            counts_i       = counts_cl[bin_centers_cl <= turn_off][-(i+1):]
            
            sigma     = ((2.5*np.log(10)/(2*np.mean(bin_width))) *np.sqrt(np.divide(1, counts_i[:-2]) + np.divide(1, counts_i[2:])))
            weight    = np.divide(1, sigma**2)
            alpha     = np.average(near_turnoff_i, weights = weight)
            n_weights = len(np.where(weight != 0)[0])
            alpha_unc = np.sqrt(np.divide(np.sum(np.multiply(weight, (near_turnoff_i - alpha)**2)), (n_weights - 1)*np.sum(weight)/n_weights))/np.sqrt(len(near_turnoff_i))
                        
            overlap = -(abs(alpha - alpha_est)/alpha_est_unc)
            
            if np.isnan(alpha_unc):
                overlap = np.nan

            alpha_list_j.append(alpha)
            alpha_unc_list_j.append(alpha_unc)
            overlap_list_j.append(overlap)
        
        alpha_list_j = np.array(alpha_list_j)
        alpha_unc_list_j = np.array(alpha_unc_list_j)
        overlap_list_j = np.array(overlap_list_j)
        
        ranges = (np.array(list(range(2, len(alpha_nc_cl[bin_centers_cl <= turn_off])))) -1)*np.mean(bin_width)
        
        ranges_list.append(ranges)
        alpha_list.append(alpha_list_j)
        alpha_unc_list.append(alpha_unc_list_j)
        overlap_list.append(overlap_list_j)
        
        max_overlap_indx = np.argmax(overlap_list_j[~np.isnan(overlap_list_j)])
        
        max_overlap_range     = ranges[~np.isnan(overlap_list_j)][max_overlap_indx]
        max_overlap_alpha     = alpha_list_j[~np.isnan(overlap_list_j)][max_overlap_indx]
        max_overlap_alpha_unc = alpha_unc_list_j[~np.isnan(overlap_list_j)][max_overlap_indx]
        
        max_overlap_indx_list.append(max_overlap_indx)
        max_overlap_range_list.append(max_overlap_range)
        max_overlap_alpha_list.append(max_overlap_alpha)
        max_overlap_alpha_unc_list.append(max_overlap_alpha_unc)
        
    max_overlap_indx_list      = np.array(max_overlap_indx_list)
    max_overlap_range_list     = np.array(max_overlap_range_list)
    max_overlap_alpha_list     = np.array(max_overlap_alpha_list)
    max_overlap_alpha_unc_list = np.array(max_overlap_alpha_unc_list)
    
    overlap_list   = np.array(overlap_list)
    ranges_list    = np.array(ranges_list)
    alpha_list     = np.array(alpha_list)
    alpha_unc_list = np.array(alpha_unc_list)
    
    return max_overlap_indx_list, max_overlap_range_list, max_overlap_alpha_list, max_overlap_alpha_unc_list, overlap_list, ranges_list, alpha_list, alpha_unc_list
    


def get_alpha_obs(alpha_nc_list, turn_off_list, turn_off_unc_list, counts_list, overlap_list, ranges_list, max_overlap_indx_list, bin_edges, redshift_range):
    """Obtains the final alpha estimates for each redshift bin based on the calibration of the mock observations with the simulations.
    
    IN:
    - alpha_nc_list (numpy.ndarray):            Alpha values for each magnitude bin estimated from the gradient of the count distribution (dim = no. redshift bins x no. of magnitude bins)
    - turn_off_list (numpy.ndarray):            Turn-off magnitude in each redshift bin
    - turn_off_unc_list(numpy.ndarray):         Uncertainty of the turn-off magnitude in each redshift bin
    - counts_list (numpy.ndarray):              Number of object counts in each magnitude bin for each redshift bin (dim = no. redshift bins x no. of magnitude bins)
    - overlap_list (numpy.ndarray):             Overlap with the alpha estimate from alpha_est_list for each magnitude range considered in each redshift bin (dim = no. of magnitude bins x no. of redshift bins)
    - ranges_list (numpy.ndarray):              Magnitude ranges considered for each alpha estimate (dim = no. of magnitude bins x no. of redshift bins)
    - max_overlap_indx_list (numpy.ndarray):    Array index of the alpha estimate from observations in alpha_list which agrees best with the alpha estimate from alpha_est_list for each redshift bin
    - bin_edges (numpy.ndarray):                Arbitrary limits of the magnitude bins used for alpha_nc_list and counts_list
    - redshift_range (iterable):                n-componenet iterable object specifying the limits of the desired redshift bins
    
    OUT:
    
    - alpha_obs_list (numpy.ndarray):           Final alpha estimates from observations which have been calibrated with mocks for each redshit bin
    - alpha_obs_unc_list (numpy.ndarray):       Uncertainty of the final alpha estimates from observations which have been calibrated with mocks for each redshit bin
    """
    
    if not isinstance(redshift_range, Iterable) or len(redshift_range) < 2:
        raise Exception('Redshift range has to be an iterable object of length > 2.')
    
    bin_width   = bin_edges[1:] - bin_edges[:-1]
    bin_centers = bin_edges[:-1] + (bin_width)/2
    
    alpha_obs_list, alpha_obs_unc_list = [], []
    
    for j in range(0, len(redshift_range)-1):
        alpha_nc         = alpha_nc_list[j]
        turn_off         = turn_off_list[j]
        turn_off_unc     = turn_off_unc_list[j]
        max_overlap_indx = max_overlap_indx_list[j]
        counts           = counts_list[j]
        ranges           = ranges_list[j]
        
        lim = int(ranges[~np.isnan(overlap_list[j])][max_overlap_indx]/np.mean(bin_width)) + 1
        
        near_turnoff = alpha_nc[bin_centers <= turn_off][-lim:-1]
        counts       = counts[bin_centers <= turn_off][-(lim+1):]
        
        sigma         = ((2.5*np.log(10)/(2*np.mean(bin_width))) *np.sqrt(np.divide(1, counts[:-2]) + np.divide(1, counts[2:])))
        weight        = np.divide(1, sigma**2)
        alpha_obs     = np.average(near_turnoff, weights = weight)
        n_weights     = len(np.where(weight != 0)[0])
        alpha_obs_unc = np.sqrt(np.divide(np.sum(np.multiply(weight, (near_turnoff - alpha_obs)**2)), (n_weights - 1)*np.sum(weight)/n_weights))/np.sqrt(len(near_turnoff))
        
        alpha_obs_list.append(alpha_obs)
        alpha_obs_unc_list.append(alpha_obs_unc)
    
    alpha_obs_list     = np.array(alpha_obs_list)
    alpha_obs_unc_list = np.array(alpha_obs_unc_list)
        
    return alpha_obs_list, alpha_obs_unc_list
        
