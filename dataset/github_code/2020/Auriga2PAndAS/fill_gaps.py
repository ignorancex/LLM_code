import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.table import Table, hstack

import ctypes as C
double_pointer_pointer=C.POINTER(C.POINTER(C.c_double))
double_pointer = C.POINTER(C.c_double)
long_pointer = C.POINTER(C.c_long)

import sys
import os

####################################################
##  To define the position of the Sun and of M31  ##
####################################################
# Define the Solar Frame reference
R_sun=8.1   # kpc  Gravity collaboration 2018
Vcirc=239.0 # km/s  Gravity collaboration 2018
# Motion of the sun Schonrich et al,2010
U_sun=11.1  #km/s
V_sun=12.24 #km/s
W_sun=7.25  #km/s

RA_M31=10.68470833
dec_M31=41.26875000
dist_M31=778.0 # Conn et al., 2012
pm_ra_M31=0.0
pm_dec_M31=0.0
Vrad_M31=-300.0

# Define the circular velocity, the solar distance and the Solar motion (LSR to GSR)
v_sun = coord.CartesianDifferential([U_sun, V_sun+Vcirc, W_sun]*u.km/u.s)
gc_frame = coord.Galactocentric(galcen_distance=R_sun*u.kpc,
										  galcen_v_sun=v_sun,
										  z_sun=0*u.pc)


# Compute the position and the velocity of Andromeda Galaxy
cart_M31 = coord.ICRS(ra=RA_M31*u.degree, dec=dec_M31*u.degree,
							 distance=dist_M31*u.kpc, # from Conn et al 2012
							 pm_ra_cosdec=pm_ra_M31*u.mas/u.yr,
							 pm_dec=pm_dec_M31*u.mas/u.yr,
							 radial_velocity=Vrad_M31*u.km/u.s)
gM31 = cart_M31.transform_to(gc_frame)
x_M31=gM31.x.value
y_M31=gM31.y.value
z_M31=gM31.z.value
vx_M31=gM31.v_x.value
vy_M31=gM31.v_y.value
vz_M31=gM31.v_z.value


###################################
### Save the residual particles ###
###################################
def save(part,infile):

	# Get the name
	x=infile.split(".")

	# Save in fits format
	c0 = fits.Column(name='ID', array=part['ID'], format='D')
	c1 = fits.Column(name='RA', array=part['ra'], format='D')
	c2 = fits.Column(name='Dec', array=part['dec'], format='D')
	c3 = fits.Column(name='xki', array=part['xki'], format='D')
	c4 = fits.Column(name='eta', array=part['eta'], format='D')
	c5 = fits.Column(name='rhelio', array=part['rhelio'], format='D')
	c6 = fits.Column(name='pmra', array=part['pmra'], format='D')
	c7 = fits.Column(name='pmdec', array=part['pmdec'], format='D')
	c8 = fits.Column(name='Vrad', array=part['Vrad'], format='D')
	c9 = fits.Column(name='g', array=part['g'], format='D')
	c10 = fits.Column(name='dg', array=part['dg'], format='D')
	c11 = fits.Column(name='g0', array=part['g0'], format='D')
	c12 = fits.Column(name='i', array=part['i'], format='D')
	c13 = fits.Column(name='di', array=part['di'], format='D')
	c14 = fits.Column(name='i0', array=part['i0'], format='D')
	c15 = fits.Column(name='EBV', array=part['EBV'], format='D')
	c16 = fits.Column(name='nb_field', array=part['nb_field'], format='D')
	c17 = fits.Column(name='x', array=part['x'], format='D')
	c18 = fits.Column(name='y', array=part['y'], format='D')
	c19 = fits.Column(name='z', array=part['z'], format='D')
	c20 = fits.Column(name='Mg', array=part['Mg'], format='D')
	c21 = fits.Column(name='Mi', array=part['Mi'], format='D')
	c22 = fits.Column(name='met_sim', array=part['met_sim'], format='D')
	c23 = fits.Column(name='Acc', array=part['Acc'], format='D')

	t = fits.BinTableHDU.from_columns([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22,c23])
	t.writeto(x[0]+"_gaps.fits",overwrite=True)
	print("Data Saved")




#################
## In polygone ##
#################

def in_poly(x,y,xp,yp):
	
	tiny=1.e-5;
	simag=np.zeros(len(x))

	for j in range(0,len(xp)):
		if (j<(len(xp)-1)):
			xe=xp[j+1]
			xs=xp[j]
			ye=yp[j+1]
			ys=yp[j]
		else:
			xe=xp[0]
			xs=xp[j]
			ye=yp[0]
			ys=yp[j]
		simag+=fimag(x,xs,xe,y,ys,ye)
	
	simag[(np.abs(simag)>tiny)]=1.0
	simag[(np.abs(simag)<=tiny)]=0.0
	return simag

def fimag(x0, xs,xe, y0,  ys,  ye):
	
	top= -(xe-x0) * (ys-y0) + (ye-y0) * (xs-x0)
	bot=  (xe-x0) * (xs-x0) + (ye-y0) * (ys-y0)
	return np.arctan2(top,bot)




#######################
## Compute (xki,eta) ##
#######################
def get_xkieta(ra,dec,raZ,decZ):
	# ra and dec are coordinates of target in radians
	# raZ and decZ are coordinates of tangential point in radians
	
	ra=ra/180.0*np.pi
	dec=dec/180.0*np.pi
	raZ=raZ/180.0*np.pi
	decZ=decZ/180.0*np.pi
	sdecZ=np.sin(decZ)
	sdec=np.sin(dec)
	cdecZ=np.cos(decZ)
	cdec=np.cos(dec)
	radiff=ra-raZ
	sradiff=np.sin(radiff)
	cradiff=np.cos(radiff)
	
	denom=sdec*sdecZ+cdec*cdecZ*cradiff
	
	xki=cdec*sradiff/denom
	eta=(sdec*cdecZ-cdec*sdecZ*cradiff)/denom
	
	return xki*180.0/np.pi,eta*180.0/np.pi



#####################################
## Compute (RA,Dec) from (xki,eta) ##
#####################################
def sla_TP2S(xki, eta, raZ, decZ):
	
	xki=xki/180.0*np.pi
	eta=eta/180.0*np.pi
	raZ=raZ/180.0*np.pi
	decZ=decZ/180.0*np.pi
	sdecZ=np.sin(decZ)
	cdecZ=np.cos(decZ)
	denom=cdecZ-eta*sdecZ
	ra  = (np.arctan2(xki,denom)+raZ)/np.pi*180.0
	dec = np.arctan2(sdecZ+eta*cdecZ,np.sqrt(xki*xki+denom*denom))/np.pi*180.0
	
	ra[(ra<0)]+=360.0
	ra[(ra>360)]-=360.0
	
	return ra,dec



###############################
### Read the generated file ###
###############################
def read_out(infile):
	hdulist = fits.open(infile)
	data_fits=hdulist[1].data
	part= pd.DataFrame()
	# Load position
	part['ID']=data_fits.field(0).byteswap().newbyteorder()
	part['ra']=data_fits.field(1).byteswap().newbyteorder()
	part['dec']=data_fits.field(2).byteswap().newbyteorder()
	part['xki']=data_fits.field(3).byteswap().newbyteorder()
	part['eta']=data_fits.field(4).byteswap().newbyteorder()
	part['rhelio']=data_fits.field(5).byteswap().newbyteorder()
	part['pmra']=data_fits.field(6).byteswap().newbyteorder()
	part['pmdec']=data_fits.field(7).byteswap().newbyteorder()
	part['Vrad']=data_fits.field(8).byteswap().newbyteorder()
	part['g']=data_fits.field(9).byteswap().newbyteorder()
	part['dg']=data_fits.field(10).byteswap().newbyteorder()
	part['g0']=data_fits.field(11).byteswap().newbyteorder()
	part['i']=data_fits.field(12).byteswap().newbyteorder()
	part['di']=data_fits.field(13).byteswap().newbyteorder()
	part['i0']=data_fits.field(14).byteswap().newbyteorder()
	part['EBV']=data_fits.field(15).byteswap().newbyteorder()
	part['nb_field']=data_fits.field(16).byteswap().newbyteorder()
	part['x']=data_fits.field(17).byteswap().newbyteorder()
	part['y']=data_fits.field(18).byteswap().newbyteorder()
	part['z']=data_fits.field(19).byteswap().newbyteorder()
	part['Mg']=data_fits.field(20).byteswap().newbyteorder()
	part['Mi']=data_fits.field(21).byteswap().newbyteorder()
	part['met_sim']=data_fits.field(22).byteswap().newbyteorder()
	part['Acc']=data_fits.field(23).byteswap().newbyteorder()
	hdulist.close()
	
	print("\n********************************")
	print("Data loaded: %d" %(len(part)))
	
	return part





###################################################
### Keep the stars  in the gaps                 ###
###################################################
def keep_gap(part): # To compile gcc -fPIC -fopenmp -shared mask_footprint.c -o mask_footprint.so -lm
	
	part=part.reset_index(drop=True)
	poly=np.loadtxt("donnees/field_limits_actual.dat") # Load the limit of the survey
	# Load the limit of luminosity of each field
	mag_lim=np.loadtxt("donnees/fields_lim_new.csv",delimiter=",")
	
	len_data=len(part)
	len_field=len(poly)
	len_lim=len(mag_lim)
	
	data_xki=(len_data*C.c_double)();data_xki[:]=part['xki']
	data_eta=(len_data*C.c_double)();data_eta[:]=part['eta']
	xki1=(len_field*C.c_double)();xki1[:]=poly[:,2]
	eta1=(len_field*C.c_double)();eta1[:]=poly[:,3]
	xki2=(len_field*C.c_double)();xki2[:]=poly[:,4]
	eta2=(len_field*C.c_double)();eta2[:]=poly[:,5]
	xki3=(len_field*C.c_double)();xki3[:]=poly[:,6]
	eta3=(len_field*C.c_double)();eta3[:]=poly[:,7]
	xki4=(len_field*C.c_double)();xki4[:]=poly[:,8]
	eta4=(len_field*C.c_double)();eta4[:]=poly[:,9]
	nb_field=(len_field*C.c_long)();nb_field[:]=poly[:,0].astype(int)
	lim_g=(len_lim*C.c_double)();lim_g[:]=mag_lim[:,1]
	lim_i=(len_lim*C.c_double)();lim_i[:]=mag_lim[:,2]
	
	
	
	libc=C.CDLL("./pipeline/mask_footprint.so")
	libc.mask.argtypes =[double_pointer,double_pointer,double_pointer,double_pointer,double_pointer,double_pointer,double_pointer,double_pointer,double_pointer,double_pointer,long_pointer,double_pointer,double_pointer,C.c_long,C.c_long,C.c_long]
	libc.mask.restype =double_pointer
	res=libc.mask(data_xki, data_eta, xki1, eta1, xki2,eta2,xki3,eta3,xki4,eta4,nb_field,lim_g,lim_i,len_data,len_field,len_lim)
	res=np.asarray(res[0:3*len_data],dtype=np.float)
	part['nb_field']=res[0:len_data].astype(int)
	part['lim_g']=res[len_data:2*len_data]
	part['lim_i']=res[2*len_data:3*len_data]
	
	
	#print part['lim_g'][0:2],part['lim_i'][0:2]
	

	# Keep also stars in the neigboor of the satured stars
	hip=np.loadtxt("donnees/HIPARCOS.lis")
	xki_hip,eta_hip=get_xkieta(hip[:,1],hip[:,2],RA_M31,dec_M31)
	rcut=(10.0-hip[:,0])/3600.0*50.0

	sel_sat=np.zeros(len(part))
	for i in range(0,len(hip)):
		sel_sat[(np.sqrt((part['xki']-xki_hip[i])**2.0+(part['eta']-eta_hip[i])**2.0)<rcut[i])]=1
	part=part[((part['nb_field']<=0)|(sel_sat==1))] # Keep only the stars in the survey


	part=part.reset_index(drop=True)
	part['ID']=-9999
	print("Stars in the gaps keept: %d" %(len(part)))
	
	return part



####################################
## Fill the gaps in the footprint ##
####################################
def fill_gaps(part):
	
	# Load the PAndAS footprint
	survey=np.loadtxt("donnees/survey_corners.dat")
	
	move=0.15 # Move of 0.15 deg
	
	# Copy the particles and move them
	part_gaps1=part.copy()
	part_gaps1['xki']+=move
	part_gaps1['eta']+=move

	# Check if the particles are in the PAndAS footprint
	sel=in_poly(part_gaps1['xki'],part_gaps1['eta'],survey[:,2],survey[:,3])
	part_gaps1=part_gaps1[(sel==1)]
	
	
	# Doing the previous step it fill the gaps but not in the edges of the survey
	# Thus move stars in opposite direction and select them to be sure that we do'nt fill twice a gap
	sel=in_poly(part['xki'],part['eta'],survey[:,2]+2*move,survey[:,3]+2*move)
	part_gaps2=part[(sel==0)].copy()
	part_gaps2['xki']-=move
	part_gaps2['eta']-=move
	# Check if the particles are in the PAndAS footprint
	sel=in_poly(part_gaps2['xki'],part_gaps2['eta'],survey[:,2],survey[:,3])
	part_gaps2=part_gaps2[(sel==1)]

	# Merge the two catalog
	part_gaps=pd.concat([part_gaps1,part_gaps2], ignore_index=True).copy()
	del(part_gaps1);del(part_gaps2)

	part_gaps=keep_gap(part_gaps)
	
	# Compute the ra,dec of these stars
	part_gaps['ra'],part_gaps['dec']=sla_TP2S(part_gaps['xki'], part_gaps['eta'], RA_M31, dec_M31)


	return part_gaps




#######################
##       MAIN        ##
#######################
if __name__=='__main__':

	infile=sys.argv[1]


	# Read the particles
	part=read_out(infile)
	part_gaps=fill_gaps(part)
	part=pd.concat([part,part_gaps], ignore_index=True).copy()
	save(part,infile)

