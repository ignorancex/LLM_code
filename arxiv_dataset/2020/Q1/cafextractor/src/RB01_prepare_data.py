import glob
import os
import shutil
import numpy as np
from astropy.io import fits
from astropy.table import Table

import CAFEutilities
import CAFEx_SetupFile as CS



def copyDirectory(src, dest):
    try:
        shutil.copytree(src, dest)
    # Directories are the same
    except shutil.Error as e:
        print('    --> Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('    --> Directory not copied. Error: %s' % e)
        

def renameRaw(cv):
	dir = cv.path_red+cv.dir_red
	allfiles = glob.glob(dir+"/*.fits")
	dates = []
	# ===== Sort files by date
	for file in allfiles:
		h = fits.open(file)
		dates.append(h[0].header['DATE'])
	
	files = [x for (y,x) in sorted(zip(dates,allfiles))]
	
	# ===== Create new name
	for i,file in enumerate(files):
		h = fits.open(file)
		head = h[0].header		
		# Change ThAr with wrong IMAGETYP set:
		if head['IMAGETYP'] == 'Science': 
			obj = head['OBJECT'].replace(" ","_")
			if 'thar' in str.lower(obj):
				head['IMAGETYP'] = 'Calibration'

		# Proceed with renaming
		if head['IMAGETYP'] == 'Science': 
		    obj = head['OBJECT'].replace(" ","")
		    obj = obj.replace("[std]_","")
		    obj = obj.replace("[std]","")
		    obj = obj.replace("[arc]","")
		    obj = obj.replace(".","_")
		    obj.replace(".","_")
		    newname = obj+'__'+cv.night+'_'+str(i+1).zfill(4)+'.fits'
		elif head['IMAGETYP'] == 'Bias':
			newname = 'bias__'+cv.night+'_'+str(i+1).zfill(4)+'.fits'
		elif head['IMAGETYP'] == 'Flat':
			newname = 'flat__'+cv.night+'_'+str(i+1).zfill(4)+'.fits'
		elif head['IMAGETYP'] == 'Calibration':
			newname = 'arc__'+cv.night+'_'+str(i+1).zfill(4)+'.fits'
		elif head['IMAGETYP'] == 'Test':
			newname = 'test__'+cv.night+'_'+str(i+1).zfill(4)+'.fits'

	# ===== Rename accordingly
		os.rename(file, dir+'/'+newname)

def frames_dict(files,dates,cv,id):
	dates = np.array(dates)
	files = np.array(files)
	IS = np.argsort(dates)
	dates = dates[IS]
	files = files[IS]
	CS.var.set_Orientation(CAFEutilities.jdnight(cv.night))
	
	# If CAFE1:
	#dc = np.array([fits.getdata(f) for f in files]).astype(float)
	
	# If CAFE2:
#	dc = []
# 	for f in files:
# 		ffl = fits.getdata(f)
# 		ffl[:,234] = np.nan
# 		ffl = np.flip(ffl,1)
# 		dc.append(ffl)
# 	dc = np.array(dc).astype(float)

	# Orientation of the CCD (according to grating orientation up/down). Right ==> 2018 on
	if CS.var.orientation == 'right':
		dc = np.array([np.flip(fits.getdata(f),1) for f in files]).astype(float)
	elif CS.var.orientation == 'left':
		dc = np.array([fits.getdata(f) for f in files]).astype(float)
	else:
		print "   ---> ERROR: I found no orientation value in the ReferenceCalib.lis "
		print "				  file for thi date. Please check that your observations fall "
		print " 			  into one of the cafe CCD windows defined in that file. "
		sys.exit()
	
	
	#dc[:,:,234] = np.nan
		
	print "    --> "+id+"...ok"	
	return {'id':id, 'files': files, 'jd':dates, 'dc':dc}



def ClassifyFiles(cv):
	""" Classify files """
	dir = cv.path_red+cv.dir_red
	files = glob.glob(dir+"/*.fits")

	# define output lists
	sci          	= []
	bias         	= []
	flat			= []
	arcs          	= []
	arcs_dates 		= []
	flat_dates 		= []
	bias_dates 		= []
	sci_dates 		= []
	objname        	= []
	exptimes       	= []

	f = open(dir+'/log_'+cv.night,'a')

	# ===== Get the information
	for i,file in enumerate(files):
		h = fits.open(file)
		head = h[0].header		
		# Change ThAr with wrong IMAGETYP set:
		if head['IMAGETYP'] == 'Science': 
			obj = head['OBJECT'].replace(" ","_")
			if 'thar' in str.lower(obj):
				head['IMAGETYP'] = 'Calibration'

		# Proceed with classification:
		if head['IMAGETYP'] == 'Science': 
			sci.append(file)
			obj = head['OBJECT'].replace(" ","_")
			obj = obj.replace("[std]_","")
			obj = obj.replace("[std]","")
			obj = obj.replace(".","_")
			objname.append(obj)
			ra     	= CAFEutilities.ra_from_sec(head['RA'])
			dec  	= CAFEutilities.ra_from_sec(head['DEC'])
			airmass	= np.float(head['AIRMASS'])
			texp   	= np.float(head['EXPTIME'])
			date   	= head['DATE']		    
			stime   = head['UT_START']
			etime   = head['UT_END']
			date    = date[:10]
	# 		    if stime > etime:
	# 				date = yesterday(date)
			hour   = CAFEutilities.ra_from_sec(stime)
			mjd = CAFEutilities.mjd_fromheader(h)
			sci_dates.append( mjd )
			exptimes.append( texp ) 
			filename = file.split('/')[-1]
			line = "%-15s %10s %10s %8.2f %4.2f %8s %11s %s \n" % (obj, ra, dec, texp, airmass, date, hour, filename)
			f.write(line)
		elif head['IMAGETYP'] == 'Bias':
			bias.append(file)
			mjd = CAFEutilities.mjd_fromheader(h)
			bias_dates.append( mjd )
		elif head['IMAGETYP'] == 'Flat':
			flat.append(file)
			mjd = CAFEutilities.mjd_fromheader(h)
			flat_dates.append( mjd )
		elif head['IMAGETYP'] == 'Calibration':
			arcs.append(file)
			mjd = CAFEutilities.mjd_fromheader(h)
			arcs_dates.append( mjd )

	flat = frames_dict(flat,flat_dates,cv,'flat')
	bias = frames_dict(bias,bias_dates,cv,'bias')
	arcs = frames_dict(arcs,arcs_dates,cv,'arcs')
	sci  = frames_dict(sci ,sci_dates,cv,'sci')
	
# 	flat_dates = np.array(flat_dates)
# 	flat = np.array(flat)
# 	IS = np.argsort(flat_dates)
# 	flat_dates = flat_dates[IS]
# 	flat = flat[IS]
# 	flat = {'files': flat, 'jd':flat_dates}
# 	#for i in range(len(flats)):
# 	#	print 'flat',flats[i], flat_ref_dates[i]
# 
# 	bias_dates = np.array(bias_dates)
# 	bias = np.array(bias)
# 	IS = np.argsort(bias_dates)
# 	bias_dates = bias_dates[IS]
# 	bias = bias[IS]
# 	bias = {'files': bias, 'jd':bias_dates}
# 
# 	arc_dates = np.array(arcs_dates)
# 	arcs = np.array(arcs)
# 	IS = np.argsort(arcs_dates)
# 	arcs_dates = arcs_dates[IS]
# 	arcs = arcs[IS]
# 	arcs = {'files': arcs, 'jd':bias_dates}



	



	#for i in range(len(biases)):
	#	print 'bias',biases[i], bias_ref_dates[i]
	f.close()


	return bias,flat, arcs, sci, objname, exptimes
	