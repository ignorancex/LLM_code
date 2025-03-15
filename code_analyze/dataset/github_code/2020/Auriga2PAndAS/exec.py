#!/usr/bin/python
#########################################################################
## Transform Auriga simulation to PAndAS-like mocks                    ##
## Oct. 2018 by Guillaume THOMAS guillaume.thomas@nrc-cnrc.gc.ca       ##
## Copyright (c) 2018 National Research Concil. All rights reserved.   ##
#########################################################################

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.table import Table, hstack
from numpy.linalg import inv


# For using C subroutines
import ctypes as C
double_pointer_pointer=C.POINTER(C.POINTER(C.c_double))
double_pointer = C.POINTER(C.c_double)
long_pointer = C.POINTER(C.c_long) 

import sys
import os


# Nb of section of the file
Nsplit=10




####################################################
##  To define the position of the Sun and of M31  ##
####################################################
# Define the Solar Frame reference
R_sun=8.129   # kpc  Gravity collaboration 2018
Vcirc=229.0 # km/s  Eiler et al. 2019
# Motion of the sun Schonrich et al,2010
U_sun=11.1  #km/s
V_sun=12.24 #km/s
W_sun=7.25  #km/s

RA_M31=10.68470833
dec_M31=41.26875000
dist_M31=778.0 # Conn et al., 2012
pm_ra_M31=0.065 #mas/yr
pm_dec_M31=-0.057 #mas/yr
Vrad_M31=-300.0 # km/s

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

# Extinction coefficient in g and i
Ag=3.793
Ai=2.086

# Limit of magnitude of the referecen field (the one with HSC)
mag_lim=np.loadtxt("donnees/fields_lim_new.csv",delimiter=",")
limref_g=mag_lim[9,1]
limref_i=mag_lim[9,2]
del(mag_lim)





#####################################################################
## Remove regions where there is saturated stars in the foreground ##
#####################################################################
def sat_fg(part):
	hip=np.loadtxt("donnees/HIPARCOS.lis")
	xki_hip,eta_hip=get_xkieta(hip[:,1],hip[:,2],RA_M31,dec_M31)
	rcut=(10.0-hip[:,0])/3600.0*50.0

	for i in range(0,len(hip)):
		part=part[(np.sqrt((part['xki']-xki_hip[i])**2.0+(part['eta']-eta_hip[i])**2.0)>rcut[i])]

	return part




########################
## Add the foreground ##
########################
def foreground(part): # To compile gcc foreground.c -o foreground.x -lm
	os.chdir("./pipeline/foreground/")
	os.system("./foreground.x") # Create the foreground
	read=np.loadtxt("./foreground.asc") # Read the file
	fore=pd.DataFrame()
	
	fore['ID']=-99+np.zeros(len(read))
	fore['xki']=read[:,1]
	fore['eta']=read[:,2]
	fore['ra'],fore['dec']=sla_TP2S(fore['xki'], fore['eta'],10.68470833,41.26875000)
	fore['rhelio']=-99+np.zeros(len(read))
	fore['pmra']=-99+np.zeros(len(read))
	fore['pmdec']=-99+np.zeros(len(read))
	fore['Vrad']=-99+np.zeros(len(read))
	fore['g0']=read[:,3]
	fore['dg']=read[:,4]
	fore['i0']=read[:,5]
	fore['di']=read[:,6]
	fore['EBV']=get_EBV(fore['ra'],fore['dec'],namefile='../../donnees/polecount_dust.fits')
	fore['g']=fore['g0']+Ag*fore['EBV']
	fore['i']=fore['i0']+Ai*fore['EBV']
	fore['nb_field']=read[:,0]


	# Recompute the corresponding uncertainties
	fore['dg']=0.032*np.exp((fore['g']-24.25)/1.1)+0.004
	fore['di']=0.112*np.exp((fore['i']-24.25)/1.13)+0.003

	fore['x']=np.zeros(len(fore))
	fore['y']=np.zeros(len(fore))
	fore['z']=np.zeros(len(fore))
	fore['Mg']=np.zeros(len(fore))
	fore['Mi']=np.zeros(len(fore))
	fore['feh_sim']=np.zeros(len(fore))
	fore['Acc']=np.zeros(len(fore))

	part = pd.concat([part, fore], ignore_index=True)
	part=part.reset_index(drop=True)
	
	os.system("rm ./foreground.asc")
	os.chdir("../..")
	return part


##########################################################
## Keep only the part in the box of Martin et al., 2013 ##
##########################################################
def keep_rgc(part):
	gi_box=np.array([1.6,2.3,0.7,0.4])
	i_box=np.array([23.5,20.9,20.9,23.5])

	sel=in_poly(part['g0']-part['i0'],part['i0'],gi_box,i_box)
	part=part[(sel==1)]

	print("RGB selected: %d" %(len(part)))

	return part



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

###############################
### Read the generated file ###
###############################
def read_out(infile):
	x=infile.split("_")
	hdulist = fits.open("./results/mock_"+x[1]+"_"+x[2]+"_"+x[3]+".fits")
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
	part['feh_sim']=data_fits.field(22).byteswap().newbyteorder()
	part['Acc']=data_fits.field(23).byteswap().newbyteorder()
	hdulist.close()

	print("\n********************************")
	print("Data loaded: %d" %(len(part)))

	return part


###################################
### Save the residual particles ###
###################################
def save(part,nread,infile):

	# Get the name
	x=infile.split("_")

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
	c22 = fits.Column(name='feh_sim', array=part['feh_sim'], format='D')
	c23 = fits.Column(name='Acc', array=part['Acc'], format='D')

	t = fits.BinTableHDU.from_columns([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23])
	if(nread==0):
		t.writeto("./results/mock_"+x[1]+"_"+x[2]+"_"+x[3]+".fits",overwrite=True)

	else:
		t.writeto("./results/tmp.fits",overwrite=True)
		with fits.open("./results/mock_"+x[1]+"_"+x[2]+"_"+x[3]+".fits") as hdul1:
			with fits.open("./results/tmp.fits") as hdul2:
				nrows1 = hdul1[1].data.shape[0]
				nrows2 = hdul2[1].data.shape[0]
				nrows = nrows1 + nrows2
				hdu = fits.BinTableHDU.from_columns(hdul1[1].columns, nrows=nrows)
				for colname in hdul1[1].columns.names:
					hdu.data[colname][nrows1:] = hdul2[1].data[colname]
		hdu.writeto("./results/mock_"+x[1]+"_"+x[2]+"_"+x[3]+".fits",overwrite=True)
	print("Data Saved")


########################################
### Remove stars in saturated region ###
########################################
def saturation(part): # To compile gcc -fPIC -fopenmp -shared density.c -o density.so -lm

	# Sort by xki
	part=part.sort_values(by=['xki'])
	part=part.reset_index(drop=True)

	#print part[['xki','eta']]
	length=len(part)
    
	xki=(length*C.c_double)();xki[:]=part['xki']
	eta=(length*C.c_double)();eta[:]=part['eta']
	g=(length*C.c_double)();g[:]=part['g']
    
	libc=C.CDLL("./pipeline/density.so")
	libc.density.argtypes =[double_pointer,double_pointer,double_pointer,C.c_long]
	libc.density.restype =long_pointer
	flag=libc.density(xki, eta, g, length)
	flag=np.asarray(flag[0:length],dtype=np.int)

	part=part[(flag[:]>0)] # Keep only the stars in the survey
	part=part.reset_index(drop=True)
	print("Saturation done: %d" %(len(part)))
        
	return part


#############################################################
## Remove stars depending of the completness of the survey ##
#############################################################
def completeness(part):
	len_part=len(part)
        
	limref_g=26.03
	limref_i=24.85

	comp_g=np.zeros(len_part)
	comp_i=np.zeros(len_part)

	# Comput the completness at the magnitude of each stars in g and i bands
	comp_g[(part['g']<(26.08+part['lim_g']-limref_g))]=1.0/(1.0+np.exp((part[(part['g']<(26.08+part['lim_g']-limref_g))]['g']-(26.08+part[(part['g']<(26.08+part['lim_g']-limref_g))]['lim_g']-limref_g))/1.04))
	comp_g[(part['g']>=(26.08+part['lim_g']-limref_g))]=1.0/(1.0+np.exp((part[(part['g']>=(26.08+part['lim_g']-limref_g))]['g']-(26.08+part[(part['g']>=(26.08+part['lim_g']-limref_g))]['lim_g']-limref_g))/0.41))

	comp_i[(part['i']<(24.62+part['lim_i']-limref_i))]=0.9/(1.0+np.exp((part[(part['i']<(24.62+part['lim_i']-limref_i))]['i']-(24.62+part[(part['i']<(24.62+part['lim_i']-limref_i))]['lim_i']-limref_i))/1.31))
	comp_i[(part['i']>=(24.62+part['lim_i']-limref_i))]=0.9/(1.0+np.exp((part[(part['i']>=(24.62+part['lim_i']-limref_i))]['i']-(24.62+part[(part['i']>=(24.62+part['lim_i']-limref_i))]['lim_i']-limref_i))/0.50))



	# Random value for reject method in g and i
	valg=np.random.uniform(0.0, 1.0, len_part)
	vali=np.random.uniform(0.0, 1.0, len_part)

	# Keep stars depending of their completeness
	part=part[((comp_g>=valg)&(comp_i>=vali))]
	part=part.reset_index(drop=True)
	print("Completeness applied: %d" %(len(part)))

	return part


#############################
## Get apparent luminosity ##
#############################
def get_appmag(part):
	len_part=len(part)

	# Load the limit of luminosity of each field
	mag_lim=np.loadtxt("donnees/fields_lim_new.csv",delimiter=",") # Nb_field, g_lim, i_lim
	
	# Get apparent magnitude
	part['g']=5.0*np.log10(part['rhelio']*1000.0)-5.0+part['Mg']+Ag*part['EBV']*(1.0+0.1*np.random.normal(0.0, 1.0, len_part))
	part['i']=5.0*np.log10(part['rhelio']*1000.0)-5.0+part['Mi']+Ai*part['EBV']*(1.0+0.1*np.random.normal(0.0, 1.0, len_part))

	# Compute the uncertainties corresponding depending of the position and the magnitude of each stars
	uncert_g=0.032*np.exp((part['g']-24.25-(part['lim_g']-limref_g))/1.1)+0.004
	uncert_i=0.112*np.exp((part['i']-24.25-(part['lim_i']-limref_i))/1.13)+0.003

	# Add random noise to the photometry
	part['g']+=uncert_g*np.random.normal(0.0, 1.0, len_part)
	part['i']+=uncert_i*np.random.normal(0.0, 1.0, len_part)

	# Recompute the corresponding uncertainties
	part['dg']=0.032*np.exp((part['g']-24.25-(part['lim_g']-limref_g))/1.1)+0.004
	part['di']=0.112*np.exp((part['i']-24.25-(part['lim_i']-limref_i))/1.13)+0.003

	print("Apparent magnitude obtained")
	return part



####################
## Get Extinction ##
####################
def get_EBV(ra,dec,namefile='./donnees/polecount_dust.fits'):
	# Transformation of coordinate
	c_icrs = SkyCoord(ra=ra.values*u.degree, dec=dec.values*u.degree, frame='icrs')
	l=c_icrs.galactic.l.degree
	b=c_icrs.galactic.b.degree
	# Read the map of schlegel et al., 98
	EBV=np.zeros(len(ra))
	ebvlist = fits.open(namefile)
	EBV_map= ebvlist[0].data
	ebvlist.close()
	pix_size=0.1 # Degree
	pb=((b+90.0)/pix_size).astype(int)
	pl=((180.0-l)/pix_size).astype(int)
	EBV[:]=EBV_map[pb,pl]
	print("Extinction obtained")
	return EBV


###################################################
### Check if the stars are in the survey or not ###
###################################################
def mask(part): # To compile gcc -fPIC -fopenmp -shared mask_footprint.c -o mask_footprint.so -lm

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

	part=part[(part['nb_field']>0)] # Keep only the stars in the survey
	part=part.reset_index(drop=True)
	print("Mask of the footprint done: %d" %(len(part)))
        
	return part



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




#####################################
### Compute the apparent position ###
#####################################
def get_pos(part,theta):

	# Rotate the disk randomly
	part['xs']=part['x']*np.cos(theta)-part['y']*np.sin(theta)
	part['ys']=part['x']*np.sin(theta)+part['y']*np.cos(theta)
	part['zs']=part['z']
	part['vxs']=part['vx']*np.cos(theta)-part['vy']*np.sin(theta)
	part['vys']=part['vx']*np.sin(theta)+part['vy']*np.cos(theta)
	part['vzs']=part['vz']

	# Transform the orientation to have the disc in the orientation of M31 disc
	RM31 = np.array([[0.7703, 0.3244,0.5490], [-0.6321, 0.5017, 0.5905], [-0.0839, -0.8019, 0.5915]])
	Rinv = inv(np.matrix(RM31))
	
	# Add the position of M31
	part['xp']=Rinv[0,0]*part['xs']+Rinv[0,1]*part['ys']+Rinv[0,2]*part['zs']+x_M31
	part['yp']=Rinv[1,0]*part['xs']+Rinv[1,1]*part['ys']+Rinv[1,2]*part['zs']+y_M31
	part['zp']=-(Rinv[2,0]*part['xs']+Rinv[2,1]*part['ys']+Rinv[2,2]*part['zs'])+z_M31
	part['vxp']=Rinv[0,0]*part['vxs']+Rinv[0,1]*part['vys']+Rinv[0,2]*part['vzs']+vx_M31
	part['vyp']=Rinv[1,0]*part['vxs']+Rinv[1,1]*part['vys']+Rinv[1,2]*part['vzs']+vy_M31
	part['vzp']=-(Rinv[2,0]*part['vxs']+Rinv[2,1]*part['vys']+Rinv[2,2]*part['vzs'])+vz_M31



	# From carthesian to observable position
	cart = coord.Galactocentric(x=part['xp'].values*u.kpc, y=part['yp'].values*u.kpc, z=part['zp'].values*u.kpc,
                v_x=part['vxp'].values*u.km/u.s,v_y=part['vyp'].values*u.km/u.s,v_z=part['vzp'].values*u.km/u.s,
                galcen_distance=(R_sun*u.kpc),
                galcen_v_sun=v_sun,
                z_sun=0*u.pc)
	icrs = cart.transform_to(coord.ICRS)
	part['ra']=(icrs.ra*u.deg).value
	part['dec']=(icrs.dec*u.deg).value
	part['rhelio']=(icrs.distance*u.kpc).value
	part['pmra']=(icrs.pm_ra_cosdec*u.mas/u.yr).value
	part['pmdec']=(icrs.pm_dec*u.mas/u.yr).value
	part['Vrad']=(icrs.radial_velocity*u.km/u.s).value

	# Remove some useless columns
	del(part['xp'],part['yp'],part['zp'],part['vxp'],part['vyp'],part['vzp'])
	del(part['xs'],part['ys'],part['zs'],part['vxs'],part['vys'],part['vzs'])

	#del(part['x'],part['y'],part['z'],part['vx'],part['vy'],part['vz'])

	
	part['xki'],part['eta']=get_xkieta(part['ra'],part['dec'],RA_M31,dec_M31)
	part=part.reset_index(drop=True)

	print("Apparent position computed")

	return part


##################################
## Read the section of the file ##
##################################
def read_file(data,IDmin,IDmax):
	part=pd.DataFrame()
	part['ID']=np.array(data["/Stars/ParticleIDs"][IDmin:IDmax])
	part['x']=np.array(data["/Stars/Coordinates"][IDmin:IDmax,0])*1000.0
	part['y']=np.array(data["/Stars/Coordinates"][IDmin:IDmax,1])*1000.0
	part['z']=np.array(data["/Stars/Coordinates"][IDmin:IDmax,2])*1000.0
	part['vx']=np.array(data["/Stars/Velocity"][IDmin:IDmax,0])
	part['vy']=np.array(data["/Stars/Velocity"][IDmin:IDmax,1])
	part['vz']=np.array(data["/Stars/Velocity"][IDmin:IDmax,2])
	part['feh_sim']=np.array(data["/Stars/metallicity"][IDmin:IDmax])
	part['Mg']=np.array(data["/Stars/AbsoluteMagnitudes/g'"][IDmin:IDmax])
	part['Mi']=np.array(data["/Stars/AbsoluteMagnitudes/i'"][IDmin:IDmax])
	part['Acc']=np.array(data["/Stars/State_form"][IDmin:IDmax])
	print("Size of the section: %d" %(len(part)))
	#part['feh_sim']=np.log10(part['feh_sim']/0.0152)

	return part




#######################
##       MAIN        ##
#######################
if __name__=='__main__':

	# Read the simulation file and obtain the number of line of it
	infile=sys.argv[1]
	data = h5py.File(infile, 'r')
	len_file=len(np.array(data["/Stars/ParticleIDs"]))
	print("Number of initial particles: %d" %(len_file))

	# Split the file in Nslipt sections to prevent the memory errors
	len_read=len_file/Nsplit
	
	# Angle for the random rotation of the disk
	theta=0.0#random.uniform(0, 2.0*np.pi)

	# For each sections
	for nread in range(0,Nsplit):
		print("\n*** Section %d ***" %(nread+1))
		IDmin=int(nread*len_read)
		IDmax=int((nread+1)*len_read)
		if(nread==Nsplit-1):
			IDmax=len_file
			
		# Read the file
		part=read_file(data,IDmin,IDmax)

		# Get the appartent position of each particules
		part=get_pos(part,theta)

		# Remove the stars outside the survey
		part=mask(part)

		# Get the extinction
		part['EBV']=get_EBV(part['ra'],part['dec'])
	
		# Get the apparente magnitude of each stars in g and i
		part=get_appmag(part)
	
		# Remove the stars depending of the completeness of the survey
		part=completeness(part)
	
		# Remove some useless columns
		del(part['lim_g'],part['lim_i'])
	
	
		# Dered the stars
		part['g0']=part['g']-Ag*part['EBV']
		part['i0']=part['i']-Ai*part['EBV']

		part=keep_rgc(part)


		# Save the data
		save(part,nread,infile)

		# Free the memory
		del(part)


	# Remove the tmp file
	os.system("rm ./results/tmp.fits")
	

	# Read the generated file and remove the region where their is saturation
	part=read_out(infile)
	#part=saturation(part)


	# Create the foreground
	part=foreground(part)

	# Remove stars in the regions where there is satured stars in the foreground
	part=sat_fg(part)

	# Select stars in the box of Martin et al., 2013
	part=keep_rgc(part)

	# Save a last time the file
	save(part,0,infile)
			
