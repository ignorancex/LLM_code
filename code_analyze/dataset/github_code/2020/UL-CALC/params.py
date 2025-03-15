# INPUT PARAMETERS (Descriptions at the end)
bane_pth= '/path/to/BANE' 
srcdir	= '/path/to/source/directory'
visname	= '<source>.MS'
imgname = '<source>.IMAGE'
vispath = os.path.join(srcdir, visname)
imgpath = os.path.join(srcdir, imgname)
cluster = '<source-name>'
if cluster != '':
    z = float(Ned.query_object(cluster)['Redshift'])
else:
    z = 0.1
l       = 1000
alpha   = -1.3
ftype	= 'E'

# ESTIMATED PARAMETERS
theta   = (l/cosmo.kpc_proper_per_arcmin(z).value)*60.
cell   	= np.rad2deg(imhead(imgpath)['incr'][1])*3600
hsize   = theta/cell

# FLUX LEVELS
do_fac  = True
flx_fac = [50, 100, 200, 300, 500, 1000, 2000]
flx_lst = []

# CLEAN PARAMETERS
cln_task= 'tclean'
N		= 1000000
isize	= imhead(imgpath)['shape'][0]
csize	= str(cell) + 'arcsec'
weight	= 'briggs'
rbst	= 0.0
grdr	= 'widefield'
wproj	= -1
dcv 	= 'multiscale'
scle	= [0,5,15,30]
thresh_f= 3

# REGION SIZE (in arcsec)
radius	= theta/2.
rms_reg	= 3 * radius

# SMOOTH PARAMETERS
bopts	= 'num_of_beams'
nbeams	= 100
bparams	= (20.0, 20.0, 0.0)
smooth_f= 2

# RECOVERY PARAMETERS
recv_th = 10.0
n_split = 6

do_cntrs= True

####
# bane_pth	= Path to BANE executable
# srcdir	= Source Directory
# visname	= Reference visibility file
# imgname	= Reference image file made from 'visname'
# z 		= Redshift of source
# cluster 	= Cluster name (optional)
# l 		= Size of halo to be injected (kpc)
# alpha 	= Spectral index for frequency scaling (S = k*v^(-a))
# ftype		= Radial profile of halo. Options: (G)aussian, (P)olynomial, (E)xponential (Currently E and G work best)
# theta		= Angular size (in arcsec) for halo (size=l) at redshift z
# x0, y0	= Halo injection position
# cell		= Pixel separation (in arcsec)
# hsize		= Size of halo (in pixels)
# do_fac  = Whether to use flux factors or flux list
# flx_fac	= Flux level factors
# flx_lst   = Manually provided flux list
# cln_task  = Clean task to use ('tclean', 'wsclean')
# N 		= No. of iterations
# csize 	= Cell size
# weight	= Weighting to be used
# dcv		= Deconvolver to use
# scle		= Multi-scale options
# thresh_f	= Cleaning threshold factor
# radius	= Radius of halo (in arcsec)
# rms_reg	= Region from which to estimate rms
# bopts		= Smoothing option ('num_of_beams', 'factor', 'beam')
# nbeams	= No. of synthesized beams in halo
# bparams	= Beam size (bmaj("), bmin("), bpa(deg))
# smooth_f	= Factor to smooth input beam
# recv_th	= Threshold of Excess flux recovery at which to fine tune (in percent)
# n_split   = Number of levels to split during fine tuning (Default: 6)
# do_cntrs  = Whether to do create python images with contours or not
####--------------------------------XXXX------------------------------------####
