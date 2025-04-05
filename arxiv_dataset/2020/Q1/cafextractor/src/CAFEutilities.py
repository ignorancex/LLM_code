import shutil
from astropy.io import fits
import glob
import os
from jdcal import gcal2jd, jd2gcal
import astropy.time
from dateutil import parser
import numpy as np
import ephem
from astropy.time import Time
from PyAstronomy import pyasl
from astropy import coordinates
from astroquery.simbad import Simbad

def ra_from_sec(ra):
	ra = float(ra)
	sign = ' '
	if ra < 0:
		sign = '-' 
		ra *= -1

	hh = ra/3600.
	mm = (hh - int(hh))*60.
	ss = (mm - int(mm))*60.
	shh = str(int(hh))
	smm = str(int(mm))
	sss = str(np.around(ss,2))
	if hh<10:
		shh = '0' + shh
	if mm<10:
		smm = '0' + smm
	if ss<10:
		sss = '0' + sss
	return sign + shh + ':' + smm + ':' + sss


def get_jd(hdr):
	jd = np.float(hdr["JUL-DATE"])
	try:
		if jd < 2400000:
			dt = dateutil.parser.parse(hdr["DATE"])
			time = astropy.time.Time(dt)
			jd = time.jd
	except:
		dt = dateutil.parser.parse(hdr["DATE"])
		time = astropy.time.Time(dt)
		jd = time.jd
	
	return jd

def mjd_fromheader(h):
    """
    return modified Julian date from header
    """
    readout_time = 90.

    secinday = 24*3600.0

    date    = h[0].header['DATE']
    exptime = np.float(h[0].header['EXPTIME'])
    dt = parser.parse(date)
    time = astropy.time.Time(dt)
    jd_fileCreation = time.jd
    
    jd = jd_fileCreation - (exptime/2. + readout_time) / secinday

    return jd

def simbad_query_radec(obname):
	res = Simbad.query_object(obname)
	SimRA 	= tuple(str(res["RA"][0]).split(' '))
	SimDEC 	= tuple(str(res["DEC"][0]).split(' '))

	if len(SimRA) == 2:
		SimRA = (SimRA[0], str(int(np.floor(np.float(SimRA[1])))).zfill(2), str(int(np.float(SimRA[1]) % np.floor(np.float(SimRA[1])) )).zfill(2)   )

	if len(SimDEC) == 2:
		SimDEC = (SimDEC[0], str(int(np.floor(np.float(SimDEC[1])))).zfill(2), str(int(np.float(SimDEC[1]) % np.floor(np.float(SimDEC[1])) )).zfill(2)   )

	c = coordinates.SkyCoord(ra='%sh%sm%ss' % SimRA, 
							 dec='%sd%sm%ss' % SimDEC,
							 frame='icrs')
	
	return c.ra.deg, c.dec.deg


# =====================================================================================

def der_snr(flux):
   
# =====================================================================================
   """
   DESCRIPTION This function computes the signal to noise ratio DER_SNR following the
               definition set forth by the Spectral Container Working Group of ST-ECF,
	       MAST and CADC. 

               signal = median(flux)      
               noise  = 1.482602 / sqrt(6) median(abs(2 flux_i - flux_i-2 - flux_i+2))
	       snr    = signal / noise
               values with padded zeros are skipped

   USAGE       snr = DER_SNR(flux)
   PARAMETERS  none
   INPUT       flux (the computation is unit independent)
   OUTPUT      the estimated signal-to-noise ratio [dimensionless]
   USES        numpy      
   NOTES       The DER_SNR algorithm is an unbiased estimator describing the spectrum 
	       as a whole as long as
               * the noise is uncorrelated in wavelength bins spaced two pixels apart
               * the noise is Normal distributed
               * for large wavelength regions, the signal over the scale of 5 or
	         more pixels can be approximated by a straight line
 
               For most spectra, these conditions are met.

   REFERENCES  * ST-ECF Newsletter, Issue #42:
               www.spacetelescope.org/about/further_information/newsletters/html/newsletter_42.html
               * Software:
	       www.stecf.org/software/ASTROsoft/DER_SNR/
   AUTHOR      Felix Stoehr, ST-ECF
               24.05.2007, fst, initial import
               01.01.2007, fst, added more help text
               28.04.2010, fst, return value is a float now instead of a numpy.float64
   """
   from numpy import array, where, median, abs 

   flux = array(flux)

   # Values that are exactly zero (padded) are skipped
   flux = array(flux[where(flux != 0.0)])
   n    = len(flux)      

   # For spectra shorter than this, no value can be returned
   if (n>4):
      signal = np.nanmedian(flux)

      noise  = 0.6052697 * np.nanmedian(abs(2.0 * flux[2:n-2] - flux[0:n-4] - flux[4:n]))

      return float(signal / noise)  

   else:

      return 0.0

# end DER_SNR -------------------------------------------------------------------------


def observatory_twill(night):
	observatory = ephem.Observer()
	#Obtenemos la fecha de observacion a partir del directorio    
	anio = "20"+night[0:2]
	mes = night[2:4]
	dia = night[4:6]
	# Fijamos la fecha a las 12:00 para que se calcule correctamente el proximo ocaso y crepusculo
	fecha=anio+"/"+mes+"/"+dia+" 12:00"
	# Definimos posicion del telescopio y fecha
	observatory.lat='37.2300'
	observatory.lon='357.4537'
	observatory.date=fecha
	# Definimos astronomical twilight
	observatory.horizon='-18'
	sol=ephem.Sun()
	# Hallamos el twilight para asegurarnos que todos los ficheros se han realizado dentro de ese tiempo
	inicio=observatory.next_setting(sol, use_center=True)
	fin=observatory.next_rising(sol, use_center=True)

	#Hallamos el dia juliano para el twilight
	eveningTw = ephem.julian_date(inicio)
	morningTw = ephem.julian_date(fin)
	return eveningTw,morningTw

def get_closer_frame(date,dateList):
	elements = np.array(range(len(dateList)))
	diff = np.abs(date-dateList)
	return elements[np.where(diff == np.min(diff))[0]]

def get_berv(hdr,RA=-99.99,DEC=-99.99):
	# ===== BERV
	
	try:
		obname = hdr["OBJECT"]
		obname = obname.split(' ')[0]
		if (('KOI' in obname) & ('-' not in obname)): obname = 'KOI-'+obname[3:]
		ra, dec = simbad_query_radec(obname)
		coord_flag = 'Simbad'
	except:
		if RA == -99.99:
			ra  = np.float(hdr["RA"])/60./60. * 15.
			dec = np.float(hdr["DEC"])/60./60.
			coord_flag = 'Telescope'
			print "     --> WARNING: getting RA/DEC from POSTN-RA/POSTN-DE header keywords!: RA="+str(ra)+" DEC="+str(dec)
		else:
			ra  = np.float(RA)	# in degrees 
			dec = np.float(DEC)	# in degrees
			coord_flag = 'User'			
			print "     --> WARNING: getting RA/DEC from provided coordinates!: RA="+str(ra)+" DEC="+str(dec)



	#ra, dec = pyasl.coordsSexaToDeg(ra2000+" "+dec2000)
	# CAHA coordinates
	longitude, latitude, altitude = -2.5461943, 37.2231427 , 2168.
	readout_time = 90. # seconds
	
	#jd = np.float(hdr["JUL-DATE"]) + (np.float(hdr["UT_START"])/3600./24.-12.) + np.float(hdr["EXPTIME"])/2. /3600./24.
	
	# If CAFE1:
	#jd = np.float(hdr["JUL-DATE"]) + ((np.float(hdr["UTC"])/3600.)/24.-0.5) - ((np.float(hdr["EXPTIME"])/2.+ readout_time) /3600./24.)
	
	# If CAFE2:
	jd = np.float(hdr["JUL-DATE"])
	if jd < 2400000:
		dt = dateutil.parser.parse(hdr["DATE"])
		time = astropy.time.Time(dt)
		jd = time.jd

	# Calculate barycentric correction
	berv, hjd = pyasl.helcorr(longitude, latitude, altitude, \
				ra, dec, jd, debug=False)

	return berv, hjd, coord_flag

def jdnight(night):
	time = '20'+night[0:2]+'-'+night[2:4]+'-'+night[4:6]
	t = Time(time)
	return t.jd1 + 0.5
	


# ==========================================
# 	SAVE FINAL REDUCED IMAGE
# ==========================================

def save_final_file(raw_file, x_matrix, cv, type, myHeader=False, Header=""):

	# ===== Read header
	if type != "MasterARC":
		if myHeader == False:
			hdr = fits.getheader(cv.path_red+cv.dir_red+'/'+raw_file)
		else:
			hdr = Header
		
	# ===== Read wavelength file or wavelength matrix
	flux  = fits.ImageHDU(data=np.transpose(x_matrix[1,:,:]), name="FLUX")
	eflux = fits.ImageHDU(data=np.transpose(x_matrix[2,:,:]), name="eFLUX")
	wave  = fits.ImageHDU(data=np.transpose(x_matrix[3,:,:]), name="WAVELENGTH")

	# ===== Add BERV & HJD to header
	if type == "SCI":
		berv, hjd, coord_flag = get_berv(hdr)
		hdr['CAFEX BERV'] = (berv, 'Barycentric Earth Rad. Vel. [km/s]')
		hdr['CAFEX COORDFLAG'] = (coord_flag, 'How target coordinates were computed for BERV')
		hdr['CAFEX HJD']  = (hjd , 'Heliocentric Julian Date [days]')

	# ===== Write file
	
	# Header
	try:
		primary_hdu = fits.PrimaryHDU(header=hdr)
	except:
		primary_hdu = fits.PrimaryHDU()

	# Data
	hdul = fits.HDUList([primary_hdu, wave, flux, eflux])

	# Filename
	rawfilename, file_extension = os.path.splitext(raw_file)
	filename = rawfilename+'_2D.fits'
	#hdul.writeto(cv.redfiles_dir+filename, overwrite=True)
	hdul.writeto(cv.aux_dir+'WCX_'+filename, overwrite=True)




