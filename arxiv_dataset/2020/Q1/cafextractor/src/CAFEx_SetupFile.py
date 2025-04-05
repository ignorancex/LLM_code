import os
import numpy as np

"""
	Setup file for specific parameters along the CAFE reduction.
"""

# def get_RefArc(jdnight, fref = np.genfromtxt('../ReferenceFrames/RefereceCalibs.lis',dtype=None,names=True,encoding='ascii')):
# 	RefArc		=	fref['ArcRef'][np.max(np.where(fref['Datestart'] < jdnight)[0])]
# 	return RefArc
# 
# def get_RefFlat(jdnight):
# 	RefFlat		=	fref['FlatRef'][np.max(np.where(fref['Datestart'] < jdnight)[0])]
# 	return RefFlat
# 
# def get_OrderProp(jdnight):
# 	y0Nominal_first		=	fref['y0Nominal_first'][np.max(np.where(fref['Datestart'] < jdnight)[0])]
# 	return y0Nominal_first

class variables:
	
	fref = np.genfromtxt('../ReferenceFrames/ReferenceCalibs.lis',dtype=None,names=True,encoding='ascii')
	
	def set_RefArc(self, jdnight, fref=fref):
		self.RefArc = fref['ArcRef'][np.max(np.where(fref['Datestart'] < jdnight)[0])]
	
	def set_RefFlat(self, jdnight, fref=fref):
		self.RefFlat		=	fref['FlatRef'][np.max(np.where(fref['Datestart'] < jdnight)[0])]

	def set_Orientation(self, jdnight, fref=fref):
		self.orientation		=	fref['Orientation'][np.max(np.where(fref['Datestart'] < jdnight)[0])]
		
	def set_OrderProp(self, jdnight, fref=fref):
		thisID = np.max(np.where(fref['Datestart'] < jdnight)[0])
		self.y0Nominal_first	=	np.float(fref['y0Nominal_first'][thisID])
		self.x0_first			=	np.float(fref['x0_first'][thisID])
		self.Nominal_Nord		=	np.int(fref['Nominal_Nord'][thisID])
		self.order0				=	np.int(fref['order0'][thisID])
		self.ordID_5500			=	np.int(fref['ordID_5500'][thisID])



# ==============================
# Basic paths:
if os.path.isdir("/home/pipeline"):
	"""
	I am in CAHA so select its paths
	"""
	root		=	'/'
	raw	        =	'/CAFE_DATA/'
	redfolder	=	'/home/pipeline/CAFE_REDUCED/'
	RefFrames	=	'/home/pipeline/cafe_softwares/cafextractor/ReferenceFrames/'
else:
	"""
	I am NOT in CAHA so select my computer paths
	"""
	root		=	'/Volumes/willyfog/gang5/jlillo/22_RUNS/2019_04_CAHA_2.2_CAFE_CHRONOS/'
	raw			=	'/Volumes/willyfog/gang5/jlillo/22_RUNS/2019_04_CAHA_2.2_CAFE_CHRONOS/00_RAW/'
	redfolder	=	'/Volumes/willyfog/gang5/jlillo/22_RUNS/2019_04_CAHA_2.2_CAFE_CHRONOS/11_REDUCED/'
	RefFrames	=	'/Users/lillo_box/00_Instrumentation/CAFE/CAFExtractor/cafextractor/ReferenceFrames/'

# ==============================


var = variables()


# Range (in days) to group the different calibration frames 
# to perfom the MasterFrame (MasterBias, MasterFlat, MasterArc).
# RB03.
 
bandwidth=	{	'bias': 1.0/24.,
		      	'flat': 1.0/24.,
		      	'arcs': 1.0/24.,
		      	'sci':	1.
		     }

MinMembersBias = 4
MinMembersFlat = 4
MinMembersArcs = 9

# Sigma clipping value. Any frame with a mean above sigclip times the std dev   
# will be rejected to perform the MasterFrame (MasterBias, MasterFlat, MasterArc)
# RB03.

sigclip=	{	'bias': 5.,
		   		'flat': 0.01, # 1% of the mean, just to avoid wrongly classified flats 
		   		'arcs': 5.,
		    }

# ==============================
# RB04: Tracing orders

order_aperture_ampl = 5
order_trace_degree  = 4

#y0Nominal_first = 107.3 # 118.6 # 115.433 #135.9
Nominal_Nord 	= 79	# 84	# Number of orders to be exrtacted
ordID_5500 		= 43			# Order corresponding to 5500A
order0 			= 62	#60		# Order corresponding to first extracted order



# ==============================
# RB05: Wavelength calibraiton
nx	= 3				# X-order of the polynomial for wavelength calibration
nm  = 8				# Y-order of the polynomial for wavelength calibration
#Selected_MasterARC = 'evening' 	# Default MasterArc to use for wavelength calibration
Selected_MasterARC = 0 	# Default MasterArc to use for wavelength calibration


# ==============================

Marsh_alg       = 0
ext_aperture    = 5
NSigma_Marsh    = 5
NCosmic_Marsh   = 5
S_Marsh         = 0.4
N_Marsh         = 3      # grado polinomio 
min_extract_col = 64
max_extract_col = 1984
RO_fl 			= 3.3/20.0
GA_fl 			= 1.0
npools 			= 1
