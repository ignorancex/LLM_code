import os
import scipy
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt

import GLOBALutils
import CAFEutilities
import CAFEx_SetupFile as CS

# ==============================
#	Background image for Flats
# ==============================
def GetBackgroundFlat(o_coeff,Nord,Frames,cv,names):

	frame = []
	for ii,Fr in enumerate(Frames):
		print "    --> Scattered light map for "+names[ii]
		if os.path.isfile(cv.aux_dir+'Background_'+names[ii]+'.fits') == False:
			#print "    ===> Creating a new map for "+names[ii]+"..."
			Centers = np.zeros((Nord[ii],Fr.shape[1]))
			for jj in range(Nord[ii]):
				Centers[jj,:]=scipy.polyval(o_coeff[ii][jj,:],np.arange(len(Centers[jj,:])))
				if np.count_nonzero(o_coeff[ii][jj,:]) == 0: Centers[jj,:] = 2000.
			
			back = GLOBALutils.get_scat(Fr,Centers,span=7)
			bacfile = cv.aux_dir+'Background_'+names[ii]+'.fits'
			hdbac = pyfits.PrimaryHDU( back )
			hdbac.writeto(bacfile)
		else:
			print "       -> I found a saved Background file for "+names[ii]+".fits. Loading the map..."
			back = pyfits.getdata( cv.aux_dir+'Background_'+names[ii]+'.fits' )
		
		frame.append(Fr - back)

	return frame,back


# ====================================
#	Background image for other images
# ====================================
def GetBackgroundFrame(o_coeff,Nord,Frames,Frames_dict,flat_cl,cv,names):

	frame = []
	for ii,Fr in enumerate(Frames):
		print "    --> Scattered light map for "+names[ii]
		if os.path.isfile(cv.aux_dir+'Background_'+names[ii]) == False:
			this = CAFEutilities.get_closer_frame(Frames_dict['jd'][ii],flat_cl['centers'])[0]
			#print "    ===> Creating a new map for "+names[ii]+"..."
			Centers = np.zeros((len(o_coeff[this]),Fr.shape[1]))
			for jj in range(Nord[this]):
				Centers[jj,:]=scipy.polyval(o_coeff[this][jj,:],np.arange(len(Centers[jj,:])))
				if np.count_nonzero(o_coeff[this][jj,:]) == 0: Centers[jj,:] = 2000.
			
			back = GLOBALutils.get_scat(Fr,Centers,span=5)
			bacfile = cv.aux_dir+'Background_'+names[ii]
			hdbac = pyfits.PrimaryHDU( back )
			hdbac.writeto(bacfile)
		else:
			if os.path.isfile(cv.aux_dir+'X_'+names[ii]) == True:
				back = 0.0
			else:
				print "       -> I found a saved Background file for "+names[ii]+". Loading the map..."
				back = pyfits.getdata( cv.aux_dir+'Background_'+names[ii] )
		
		frame.append(Fr - back)

	return frame

# ====================================
#	Background image for other images
# ====================================
def GetBackgroundMasterFrame(o_coeff,Nord,Frames,Frames_dict,flat_cl,cv,names):

	frame = []
	for ii,Fr in enumerate(Frames):
		print "    --> Scattered light map for "+names[ii]
		if os.path.isfile(cv.aux_dir+'Background_'+names[ii]) == False:	
			this = CAFEutilities.get_closer_frame(Frames_dict['centers'][ii],flat_cl['centers'])[0]
			#print "    ===> Creating a new map for "+names[ii]+"..."
			Centers = np.zeros((len(o_coeff[this]),Fr.shape[1]))
			for jj in range(Nord[this]):
				Centers[jj,:]=scipy.polyval(o_coeff[this][jj,:],np.arange(len(Centers[jj,:])))
				if np.count_nonzero(o_coeff[this][jj,:]) == 0: Centers[jj,:] = 2000.
			
			back = GLOBALutils.get_scat(Fr,Centers,span=5)
			bacfile = cv.aux_dir+'Background_'+names[ii]
			hdbac = pyfits.PrimaryHDU( back )
			hdbac.writeto(bacfile)
		else:
			print "       -> I found a saved Background file for "+names[ii]+". Loading the map..."
			back = pyfits.getdata( cv.aux_dir+'Background_'+names[ii] )
		
		frame.append(Fr - back)

	return frame




# ====================================
#	P-matrix of coefficients
# ====================================
def get_P(MasterFlat,o_coeff,Nord,cv, RO_fl, GA_fl):
	"""  """
	P =[]
	for ii,Flat in enumerate(MasterFlat):
		print "    --> MasterFlat #"+np.str(ii)+" order extraction"
		
		print "    ===> Calculating P..."
		if os.path.isfile(cv.aux_dir+'P'+np.str(ii)+'.fits') == False:
			# Determine P
			PP = GLOBALutils.obtain_P(Flat,o_coeff[ii],CS.ext_aperture,RO_fl, GA_fl,\
									CS.NSigma_Marsh, CS.S_Marsh, CS.N_Marsh, \
									CS.Marsh_alg, CS.min_extract_col, CS.max_extract_col,\
									CS.npools)
			# Save P
			hdu = pyfits.PrimaryHDU( PP )
			hdu.writeto( cv.aux_dir+'P'+np.str(ii)+'.fits' )
			P.append(PP)
		else:
			# Load P
			PP = pyfits.getdata( cv.aux_dir+'P'+np.str(ii)+'.fits' )
			P.append(PP)
		
	return np.array(P)


	
# ====================================
#	Extract Orders from FLAT
# ====================================
def ExtractOrders(FrameList,o_coeff,P,Nord,cv,flat_cl, id, already_done=False):
	""" Extract orders from Flat frames"""
	S_flat = []
	for ii,Frame in enumerate(FrameList):
		if already_done == False:
			RON = 3.3
			GAIN = 1.0
			SS_flat  = GLOBALutils.optimal_extraction(Frame,P,o_coeff[ii],\
					   CS.ext_aperture,RON,GAIN,CS.S_Marsh,10*CS.NCosmic_Marsh,\
					   CS.min_extract_col,CS.max_extract_col,CS.npools)
   			SS_flat = np.array(SS_flat)
   			for j in range(Nord[ii]):
   				SS_flat[j,1] = SS_flat[j,1][::-1]
				SS_flat[j,2] = SS_flat[j,2][::-1]

			hdu = pyfits.PrimaryHDU( SS_flat )
			hdu.writeto( cv.aux_dir+'Sflat'+np.str(ii)+'.fits' )
			S_flat.append(SS_flat)
		else:
			SS_flat = pyfits.getdata( cv.aux_dir+'Sflat'+np.str(ii)+'.fits')
			S_flat.append(SS_flat)
		
	return np.array(S_flat)


# ====================================
#	Extract orders from ThAr+Science
# ====================================
def ExtractOrdersFrames(FrameList,Frames_dict,o_coeff,P,Nord,cv,SS_Flat_norm,flat_cl, id, names):
	""" Extract orders from ThAr and Science frames"""
	S_frame = []
	for ii,Frame in enumerate(FrameList):
		already_done = os.path.isfile(cv.aux_dir+'X_'+names[ii])
		if already_done == False:
			this = CAFEutilities.get_closer_frame(Frames_dict['jd'][ii],flat_cl['centers'])[0]
			RON = 3.3
			GAIN = 1.0
			SS_frame  = GLOBALutils.optimal_extraction(Frame,P,o_coeff[this],CS.ext_aperture,RON,GAIN,CS.S_Marsh,10*CS.NCosmic_Marsh, CS.min_extract_col,CS.max_extract_col,CS.npools)
			#SS_frame.append(np.zeros((3,2048)))
			SS_frame = np.array(SS_frame) 
			SS_frame[:,1] = SS_frame[:,1] / SS_Flat_norm[this,:,1,::-1]
			SS_frame[:,2] = SS_frame[:,2] / SS_Flat_norm[this,:,1,::-1]
			for i in range(Nord[this]):
				SS_frame[i,1] = SS_frame[i,1][::-1]
				SS_frame[i,2] = SS_frame[i,2][::-1]
			
			print '       -> '+names[ii]+'...extracted!'

			hdu = pyfits.PrimaryHDU( SS_frame )
			hdu.writeto( cv.aux_dir+'X_'+names[ii] )
			S_frame.append(SS_frame)
			
		else:
			print '       -> '+names[ii]+'... already extracted. Loading it...'
			SS_frame = pyfits.getdata( cv.aux_dir+'X_'+names[ii] )
			S_frame.append(SS_frame)
		
	return np.array(S_frame)

# ====================================
#	Extract orders from Master ThAr
# ====================================
def ExtractOrdersMasterFrames(FrameList,Frames_dict,o_coeff,P,Nord,cv,SS_Flat_norm,flat_cl, id, names):
	""" Extract orders from Master frames"""
	S_frame = []
	for ii,Frame in enumerate(FrameList):
		already_done = os.path.isfile(cv.aux_dir+'X_'+names[ii])
		if already_done == False:
			this = CAFEutilities.get_closer_frame(Frames_dict['centers'][ii],flat_cl['centers'])[0]
			RON = 3.3
			GAIN = 1.0
			SS_frame  = GLOBALutils.optimal_extraction(Frame,P,o_coeff[this],CS.ext_aperture,RON,GAIN,CS.S_Marsh,10*CS.NCosmic_Marsh, CS.min_extract_col,CS.max_extract_col,CS.npools)
			#SS_frame.append(np.zeros((3,2048)))
			SS_frame = np.array(SS_frame)
			SS_frame[:,1] = SS_frame[:,1] / SS_Flat_norm[this,:,1,::-1]
			for i in range(Nord[this]):
				SS_frame[i,1] = SS_frame[i,1][::-1]
				SS_frame[i,2] = SS_frame[i,2][::-1]
			
			print '       -> '+names[ii]+'...extracted!'

			hdu = pyfits.PrimaryHDU( SS_frame )
			hdu.writeto( cv.aux_dir+'X_'+names[ii] )
			S_frame.append(SS_frame)
			
		else:
			SS_frame = pyfits.getdata( cv.aux_dir+'X_'+names[ii] )
			S_frame.append(SS_frame)
		
	return np.array(S_frame)




		
# ==================
#	Normalize flat
# ==================
def NormalizeFlat(X_FrameList,cv, already_done=False):
	X_flat_n = []
	Xnorms   = []
	# Normalize flat field spectra.
	for ii,X_Flat in enumerate(X_FrameList):
		print "    ===> Normalizing Flat..."
		XX_flat_n, XXnorms = GLOBALutils.FlatNormalize_single( X_Flat, mid = int(.5*X_Flat.shape[2]) )
		X_flat_n.append(XX_flat_n)
		Xnorms.append(XXnorms)

	return np.array(X_flat_n), np.array(Xnorms) 

# ==================
#	CTE correction
# ==================
def CTEcorr(FrameList,backList):
	alpha	= 0.056
	beta 	= 0.82
	gamma	= 0.205
	delta	= 3.0
	
	FrameCorr = []
	ymatrix = np.arange(2048)
	
	for Frame,Backg in zip(FrameList,backList):
		CTI = alpha*Frame**(-beta) * np.exp(-gamma*(Backg/Frame)**delta)
		FrameCorr.append( Frame/(1.-CTI)**ymatrix  )

	return np.array(FrameCorr)


# def Backg_and_Extract(o_coeff,Nord,Frames,Frames_dict,flat_cl,cv,names,    P,Nord,S_flat_n, id):
# 	frame_back, backg   = RB05.GetBackgroundFrame(o_coeff,Nord,Frames,Frames_dict,flat_cl,cv,names)
# 	x_frame 			= RB05.ExtractOrdersFrames(frame_back,Frames_dict,o_coeff,P,Nord,cv,S_flat_n,flat_cl, id, names)
# 	return x_frame
		
