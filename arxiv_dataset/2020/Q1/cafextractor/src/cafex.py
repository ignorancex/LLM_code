import sys
import os
import shutil
import numpy as np
import matplotlib
if sys.platform != 'darwin': matplotlib.use('agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord, match_coordinates_sky
from termcolor import colored
import ntpath
import matplotlib.gridspec as gridspec # GRIDSPEC !
import argparse

import RB01_prepare_data as RB01
import RB02_cafe_checks as RB02
import RB03_MasterFrames as RB03
import RB04_TraceOrders as RB04
import RB05_ExtractOrders as RB05
import RB06_WavelengthCalibration as RB06
import RB07_CrossCorr as RB07
import RB08_RadialVelocity as RB08
import RB09_SpecAnalysis as RB09
import RB10_SaveResults as RB10

import CAFEx_SetupFile as CS
import CAFEutilities
import imp

"""
	Pipeline for the upgraded version of the CAFE instrument at Calar Alto
	Observatory: CAFE2.
	
	===== SYNTAX ===== 
	date:	Night to be reduced: YYMMDD
	-SF	:	Setup file location can be modified as --SF path_to_file
	
	===== HISTORY ===== 
	2019/01/21		jlillobox		First version released
	2019/05/16		jlillobox		v0.4 accounting for the changes in the order 
									architecture.
	2019/05/30		jlillobox		v0.5 
									- introducing RV corrections for the SNR-effect,
									- small modifications in header (including CCF_FWHM)
									- option --root from command line instead of SetupFile
	2019/07/26		jlillobox		v0.6 
									- Modifications to automatically account for CAFE windows,
									- New file in ReferenceFrames/ReferenceCalibs.lis
	2019/08/09		jlillobox		v0.7
									- New SNR-RV correction (see measure_RVcorr.py) now for SNR<120
									- Clarification on the reference from id004
	2019/08/12		jlillobox		v0.8
									- Include ThAr correction due to lamp changes
	
"""

class variables:

    def set_night(self, p):
        self.night = p
        print '{0:<30} {1:>}'.format("Night:",p)

    def set_path_root(self, p):
        self.path_root = p
        print '{0:<30} {1:>}'.format("Root path:",p)

    def set_path_raw(self, p):
        self.path_raw = p
        print '{0:<30} {1:>}'.format("Raw data path:",p)

    def set_dir_raw(self, p):
        self.dir_raw = p
        print '{0:<30} {1:>}'.format("Raw data directory:",p)

    def set_dir_ref(self, p):
        self.ref_frames = p
        print '{0:<30} {1:>}'.format("Reference Frame directory:",p)

    def set_path_red(self, p):
        self.path_red = p
        print '{0:<30} {1:>}'.format("Reduced data path:",p)

    def set_dir_red(self, p):
        self.dir_red = p
        print '{0:<30} {1:>}'.format("Reduced data directory:",p)

    def set_redfiles_dir(self, p):
        self.redfiles_dir = p
        print "Final reduced data directory:",p

	self.arc_ref = ''

	self.version = ''


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("date", help="Night date in YYMMDD")
    parser.add_argument("--root", default=None, help="Root folder (folder that contains the 00_RAW folder)")
    parser.add_argument("-SF", "--SF", help="Setup file path", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

	#orig_stdout = sys.stdout
	#flog = open('log_tmp.log', 'w')
	#sys.stdout = flog

	args = cli()

	pipeversion = 'v0.8'	# As in Github repository
	
	print ""
	print "============================================================"
	print "CAFExtractor ("+pipeversion+"): a reduction pipeline to make a good CAFE"
	print "============================================================"
	print "Starting the fun... \n"
	
	
	print "========================"
	print "RB00: Setting paths"
	print "======================== \n"
	print args.root
	if args.root is not None:
		root_fold = args.root
		raw_fold  = root_fold+'/00_RAW/'
		red_fold  = root_fold+'/11_REDUCED/'
	else:
		print "    << Using default root folder set in the SetupFile... >> "
		root_fold = CS.root
		raw_fold  = CS.raw
		red_fold  = CS.redfolder
			
	cv = variables()
	cv.set_path_root(root_fold)
	cv.set_path_raw(raw_fold)
	cv.set_night(args.date)
	cv.set_dir_raw(cv.night)
	cv.set_path_red(red_fold)
	cv.set_dir_red(cv.night)
	
	# Reference frames
	refFrames_dir = CS.RefFrames
	jdnight = CAFEutilities.jdnight(cv.night)
	CS.var.set_RefArc(jdnight)
	CS.var.set_RefFlat(jdnight)
	cv.set_dir_ref(refFrames_dir)
	cv.arc_ref  = refFrames_dir+'/'+CS.var.RefArc  	# arc__160722_0061.fits
	cv.flat_ref = refFrames_dir+'/'+CS.var.RefFlat 	# flat__160722_evening.fits
	cv.aux_dir  = cv.path_red + cv.dir_red + '/auxiliar/'
	cv.redfiles_dir  = cv.path_red + cv.dir_red + '/reduced/'
	cv.version = pipeversion
	
	print " "
	
	print "========================"
	print "RB01: Preparing the data"
	print "======================== \n"
	
	print "+ Duplicating RAW directory..."
	if not os.path.exists(cv.path_red+cv.dir_red):
		RB01.copyDirectory(cv.path_raw+cv.dir_raw, cv.path_red+cv.dir_red)
		if not os.path.exists(cv.aux_dir):
			os.makedirs(cv.aux_dir)
			os.makedirs(cv.redfiles_dir)
		print "    --> "+cv.night+" duplicated!"

		print "+ Re-naming raw files..."
		RB01.renameRaw(cv)
		print "    --> "+cv.night+" renamed!"
	else:
		print "+ Directory already exists. NOT copying the RAW, using the existing files..."

	print "+ Classifying files and creating frames datacubes..."
	bias,flat, arcs, sci, objnames, exptimes = RB01.ClassifyFiles(cv)

	print "    --> "+cv.night+" classified!"
	print " "


	print "========================"
	print "RB02: CAFE checks"
	print "======================== \n"

	print "+ Check CAFE temperatures stability..."
	try:
		Tcoll, Tbenc, Tgrat, Troom = RB02.cafe_temperature(cv,bias,flat,arcs,sci)
		print "    -->", '{0:<30} {1:>10} {2:>10} {3:>4}'.format("Dome temp. delta/variation:",round((np.max(Tbenc)-np.min(Tbenc))*1.e3,0), round(np.std(Tbenc)*1.e3,0),' mK')
		print "    -->", '{0:<30} {1:>10} {2:>10} {3:>4}'.format("CAFE room temp. delta/variation:",round((np.max(Troom)-np.min(Troom))*1.e3,0), round(np.std(Troom)*1.e3,0),' mK')
		print "    -->", '{0:<30} {1:>10} {2:>10} {3:>4}'.format("Collimator temp. delta/variation:",round((np.max(Tcoll)-np.min(Tcoll))*1.e3,0), round(np.std(Tcoll)*1.e3,0),' mK')
		print "    -->", '{0:<30} {1:>10} {2:>10} {3:>4}'.format("Grating temp. delta/variation:",round((np.max(Tgrat)-np.min(Tgrat))*1.e3,0), round(np.std(Tgrat)*1.e3,0),' mK')
		print "    -->", "... done!"
	except:
		print "    --> Temperatures not found in header. Please check headers. Continnuing with the reduction..."

	print "+ Check CAFE preasure stability..."
	try:
		Pres1, Pres2 = RB02.cafe_preasures(cv,bias,flat,arcs,sci)
		print "    -->", '{0:<30} {1:>10} {2:>10} {3:>4}'.format("Preasure delta/variation:",round((np.max(Pres1)-np.min(Pres1)),1), round(np.std(Pres1),1),' mbar')
		print "    -->", '{0:<30} {1:>10} {2:>10} {3:>4}'.format("Diff. preasure delta/variation:",round((np.max(Pres2)-np.min(Pres2))*1.e3,3), round(np.std(Pres2)*1.e3,3),' mmbar')
	except:
		print "    --> Preasures not found in header. Please check headers. Continnuing with the reduction..."
		
	print "+ Calculating arc shifts for night "+cv.night+"..."
	if os.path.isfile(cv.aux_dir+'arc_shifts.npz') == False:
		shifts,intensity = RB02.cafe_shift(cv,arcs)
		np.savez(cv.aux_dir+'arc_shifts',jd=arcs['jd'],shifts=shifts,intensity=intensity)
	else:
		print "    --> I found shifts for this night. Loading..."
		tab = np.load(cv.aux_dir+'arc_shifts.npz',allow_pickle=True)
		shifts,intensity = tab["shifts"], tab["intensity"]
	print "    --> ... done!"
	if 1:
		fig = plt.figure(figsize=(12,8))
		gs = gridspec.GridSpec(3,1, height_ratios=[1.,1.,1.], width_ratios=[1])
		gs.update(left=0.1, right=0.95, bottom=0.08, top=0.93, wspace=0.08, hspace=0.12)
		
		ax0 = plt.subplot(gs[0,:]) 
		plt.plot(arcs['jd'],shifts[:,0]/np.mean(shifts[:,0]),'o',label='Xpos',c='Dodgerblue')
		plt.legend()
		plt.grid(ls=':',c='gray',alpha=0.5)
		
		ax1 = plt.subplot(gs[1,:]) 
		plt.plot(arcs['jd'],shifts[:,1]/np.mean(shifts[:,1]),'o',label='Ypos',c='Tomato')
		plt.legend()
		plt.grid(ls=':',c='gray',alpha=0.5)
		
		ax2 = plt.subplot(gs[2,:]) 
		plt.plot(arcs['jd'],intensity,'o',label='Intens.',c='Limegreen')
		plt.legend()
		plt.grid(ls=':',c='gray',alpha=0.5)
		plt.savefig(cv.aux_dir+'arc_shifts_'+cv.night+'.pdf',bbox_inches='tight')
		plt.close()

 	xshift, yshift = np.mean(shifts,axis=0)[0], np.mean(shifts,axis=0)[1]
	
 	print "+ Checking the 2D shift for each arc frame..."
 	print "    --> Mean shift of the night (x,y) = ",xshift, yshift," pix."
 	print " "
 	if any(np.abs(value) > 5. for value in np.mean(shifts,axis=0)):
 		print colored("    --> WARNING: Large jump in the CCD. Please check if you need to",'red')
 		print colored("                 update static calibrations or modify the threshholds","red")
 	elif any(np.abs(value) > 1. for value in np.mean(shifts,axis=0)):
 		print colored("    --> WARNING: Relatively large jump in the CCD. Please check if you need to",'yellow')
 		print colored("                 update static calibrations or modify the threshholds","yellow")
	else:
 		print colored("    --> OK!: XY shift from reference arc is below 1 pixel!",'green')
		


	print "========================"
	print "RB03: Master frames"
	print "======================== \n"
	
	print "+ BIAS"
	if os.path.isfile(cv.aux_dir+'MasterBias.npz') == False:
		bias_cl    = RB03.FrameClustering(bias,cv)
		MasterBias,ron_b,gain_b = RB03.FrameCombine(bias,bias_cl,cv)
		np.savez(cv.aux_dir+'MasterBias',MasterBias=MasterBias,bias_cl=bias_cl)
	else:
		print "    --> I found a Master Bias file. Loading..."
		tab = np.load(cv.aux_dir+'MasterBias.npz',allow_pickle=True)
		MasterBias = tab['MasterBias']
		bias_cl = tab['bias_cl'].tolist()

	print "+ FLAT"
	if os.path.isfile(cv.aux_dir+'MasterFlat.npz') == False:
		flat_cl = RB03.FrameClustering(flat,cv)
		MasterFlat,ron_f,gain_f = RB03.FrameCombine(flat,flat_cl,cv, biasList=bias_cl, biasFrames=MasterBias)
		np.savez(cv.aux_dir+'MasterFlat',MasterFlat=MasterFlat,flat_cl=flat_cl)
	else:
		print "    --> I found a Master Flat file. Loading..."
		tab = np.load(cv.aux_dir+'MasterFlat.npz',allow_pickle=True)
		MasterFlat = tab['MasterFlat']
		flat_cl = tab['flat_cl'].tolist()
	
	print "+ ARCS"
	if os.path.isfile(cv.aux_dir+'MasterArc.npz') == False:
		arcs_cl = RB03.FrameClustering(arcs,cv)
		MasterArc,ron_a,gain_a = RB03.FrameCombine(arcs,arcs_cl,cv, biasList=bias_cl, biasFrames=MasterBias)
		np.savez(cv.aux_dir+'MasterArc',MasterArc=MasterArc,arcs_cl=arcs_cl)
	else:
		print "    --> I found a Master Arc file. Loading..."
	
	tab = np.load(cv.aux_dir+'MasterArc.npz',allow_pickle=True)
	MasterArc = tab['MasterArc']
	arcs_cl = tab['arcs_cl'].tolist()
	
	arcsb	= RB03.BiasRemove(arcs, MasterBias, bias_cl, cv)

	print "+ SCIENCE"
	sci_cl  = RB03.FrameClustering(sci,cv)
	scib	= RB03.BiasRemove(sci, MasterBias, bias_cl, cv)

	print " "
	
	
	print "===================="
	print "RB04: Order tracing "
	print "==================== \n"
	
	# ==================================================
	print "+ Refining REF. flat first order location..."
	ReferenceFlat_tmp = fits.open(cv.flat_ref)
	ReferenceFlat = [ReferenceFlat_tmp[0].data]
	already_done = os.path.isfile(cv.aux_dir+'traces_ref.npz')
	if already_done == False:
		o_coeff,Nord = RB04.TraceOrders(ReferenceFlat)
		np.savez(cv.aux_dir+'traces_ref',o_coeff=o_coeff,Nord=Nord)
	else:
		print "    --> I found a traces_ref.npz file... loading it!"
		tab = np.load(cv.aux_dir+'traces_ref.npz',allow_pickle=True)
		o_coeff,Nord = tab['o_coeff'], tab['Nord']

	o_coeff,Nord, FirstRefOrder = RB04.SelectOrders(o_coeff,Nord, 0.0, cv) # yshift = 0.0
	print 'orders:',np.shape(o_coeff),Nord
	
	# ==================================================
	print "+ Tracing orders..."
	already_done = os.path.isfile(cv.aux_dir+'traces.npz')
	if already_done == False:
		o_coeff,Nord = RB04.TraceOrders(MasterFlat)
		np.savez(cv.aux_dir+'traces',o_coeff=o_coeff,Nord=Nord)
	else:
		print "    --> I found a traces.npz file... loading it!"
		tab = np.load(cv.aux_dir+'traces.npz',allow_pickle=True)
		o_coeff,Nord = tab['o_coeff'], tab['Nord']
	print 'orders2:',np.shape(o_coeff),Nord
	print "    --> Orders traced!"

	# ==================================================
	print "+ Checking orders traced and selecting the nominal CAFE ones..."
	o_coeff,Nord, Y_of_first_order = RB04.SelectOrders(o_coeff,Nord, yshift, cv, y0Nominal_first = FirstRefOrder)
	print 'orders3:',np.shape(o_coeff),Nord
	
	print " "

	
	print "================================="
	print "RB05: Flat order & normalization "
	print "================================= \n"
	
	ron_f = 3.3/20.
	gain_f = 1.0

	print "+ Creating the scattered light maps for each MasterFlat and substracting..."
	names = ['MasterFlat_'+np.str(ii) for ii in range(flat_cl['nclusters'])]
	MasterFlat_back, backg = RB05.GetBackgroundFlat(o_coeff,Nord,MasterFlat,cv,names)

	print "+ Get P matrix..."
	P = RB05.get_P(MasterFlat_back,o_coeff,Nord,cv, ron_f, gain_f)
	
	print "+ Extracting the MasterFlat orders..."
	if os.path.isfile(cv.aux_dir+'Sflat0.fits') == True:
		done = True
	else:
		done = False
	S_flat = RB05.ExtractOrders(MasterFlat_back,o_coeff,P,Nord,cv,flat_cl,'FLAT',already_done=done)
	
	print "+ Normalizing the extracted MasterFlat..."
	S_flat_n, Snorms = RB05.NormalizeFlat(S_flat,cv)
	
	print "+ Saving Normalized flat spectra..."
	nmaster, _, _, _ = np.shape(S_flat_n)
	for i in range(nmaster):
		primary_hdu = fits.PrimaryHDU()
		tmp = fits.ImageHDU(data=S_flat_n[i,:,1,:], name="nFLAT")
		hdul = fits.HDUList([primary_hdu, tmp])
		hdul.writeto(cv.aux_dir+'X_FlatNorm'+str(i)+'.fits', overwrite=True)
	
	print " "

	
	print "=============================================="
	print "RB05: Arcs+Science - extraction and analysis  "
	print "============================================== \n"

	arc_names = []
	for filename in arcs['files']:
		elem = [pos for pos, char in enumerate(filename) if char == '/']
		arc_names.append(filename[elem[-1]+1:])
	
	MasterArc_names = []
	for ii in range(arcs_cl['nclusters']):
		MasterArc_names.append("MasterArc_"+str(ii)+'.fits')

	print "+ Computing background for arc images..."
	arc_back    	= RB05.GetBackgroundFrame(o_coeff,Nord,arcsb["dc"],arcs,flat_cl,cv,arc_names)
	MasterArc_back 	= RB05.GetBackgroundMasterFrame(o_coeff,Nord,MasterArc,arcs_cl,flat_cl,cv,MasterArc_names)

	print np.shape(arc_back)

	#print "+ CTE correction for ARCs and MasterARC..."
	#arc_back        = RB05.CTEcorr(arc_back, backg)
	#MasterArc_back	 = RB05.CTEcorr(MasterArc_back, MA_backg)
		
	print "+ Extracting arc images..."
	x_arc 		= RB05.ExtractOrdersFrames(arc_back,arcs,o_coeff,P,Nord,cv,S_flat_n,flat_cl, 'ARC', arc_names)

	print "+ Extracting Master Arc images..."
	x_MasterArc 	= RB05.ExtractOrdersMasterFrames(MasterArc_back,arcs_cl,o_coeff,P,Nord,cv,S_flat_n,flat_cl, 'ARC', MasterArc_names)
	
	# Extract SCIENCE images
	sci_names = []
	for filename in sci['files']:
		elem = [pos for pos, char in enumerate(filename) if char == '/']
		sci_names.append(filename[elem[-1]+1:])

	print "+ Computing background for sci images..."
	sci_back  	= RB05.GetBackgroundFrame(o_coeff,Nord,scib["dc"],sci,flat_cl,cv,sci_names)
	
	#print "+ CTE correction for SCIENCE frames..."
	#sci_back        = RB05.CTEcorr(sci_back, backg)

	print "+ Extracting sci images..."
	x_sci = RB05.ExtractOrdersFrames(sci_back,sci,o_coeff,P,Nord,cv,S_flat_n,flat_cl, 'SCI', sci_names)
	
	print "+ Clearing some memory space by removing background images from python..."
	arc_back, sci_back = None, None



	print " "
	print "=========================================="
	print "RB06: MasterArcs - Wavelength calibration "
	print "========================================== \n"
		
	print "+ Wavelength solution for each MasterArc"
	w_MasterArc = RB06.WavelengthCalibration(x_MasterArc,arcs_cl,MasterArc_names,cv,xshift,yshift, 'MasterARC')

	# Test the precision of the wavecal (as suggested by referee)
	#w_arc       = RB06.WavelengthCalibration(x_arc,arcs,arc_names,cv,xshift,yshift, 'ARC')
	#sys.exit()

	print "========================================"
	print "RB07: Arcs - RVs || Science - Attach WC "
	print "======================================== \n"

	print "+ Radial velocity of the Master Arc"
	already_done = os.path.isfile(cv.aux_dir+'MasterArc_RVs.npz')
	if already_done == False:
		print "    --> Estimating RV of MasterArc ..."
		MasterRVs = RB07.ArcRV(w_MasterArc, cv, MasterArc_names)
		np.savez(cv.aux_dir+'MasterArc_RVs',MasterRVs=MasterRVs)
		
		#print "    --> Estimating RV of individual arcs ..."
		#arcRVs = RB07.ArcRV(w_arc, cv, arc_names)		
		#np.savez(cv.aux_dir+'arc_RVs',MasterRVs=MasterRVs,arcRVs=arcRVs)
	else:
		print "    --> ThAr RVs found. Loading values..."
		tab = np.load(cv.aux_dir+'MasterArc_RVs.npz',allow_pickle=True)
		#MasterRVs,arcRVs = tab['MasterRVs'], tab['arcRVs']
		MasterRVs = tab['MasterRVs']


	print "+ Attach drift-corrected wavelength calibration to arcs"
	already_done = os.path.isfile(cv.aux_dir+'arc_RVs.npz')
	if already_done == False:
		w_arc, WCdict_arc = RB07.AttachWC_arcs(x_arc, arcs, arc_names, 				# Individual arc frames
			 		  		                   w_MasterArc, arcs_cl, MasterRVs, cv)	# MasterArc frames	
		RVs, eRVs = [], []
		for dict in WCdict_arc: 
			RVs.append(dict["RVfromMaster"])
			eRVs.append(dict["eRVfromMaster"])
		arcRVs = (RVs, eRVs)
		np.savez(cv.aux_dir+'arc_RVs',arcRVs=arcRVs,WCdict_arc=WCdict_arc,w_arc=w_arc)
	else:
		print "    --> Drift-correct w_arcs and ThAr RVs found. Loading frames..."
		
	tab = np.load(cv.aux_dir+'arc_RVs.npz',allow_pickle=True)
	w_arc, WCdict_arc, arcRVs = tab['w_arc'], tab['WCdict_arc'].tolist(), tab['arcRVs']
	

# 	print "+ Measuring the RV of the individual arcs"
# 	already_done = os.path.isfile(cv.aux_dir+'arc_RVs.npz')
# 	if already_done == False:
# 		print "    --> Estimating RV of individual arcs ..."
# 		arcRVs = RB07.ArcRV(w_arc, cv, arc_names)
# 		np.savez(cv.aux_dir+'arc_RVs',arcRVs=arcRVs,WCdict_arc=WCdict_arc,w_arc=w_arc)
# 	else:
# 		print "    --> ThAr RVs found. Loading values..."
# 		tab = np.load(cv.aux_dir+'arc_RVs.npz')
# 		arcRVs = tab['arcRVs']

	try:
		RB07.plot_ArcRV(sci, arcs, arcRVs, arcs_cl, MasterRVs, cv)
	except: 
		print "---> WARNING: Could not plot the ArcRVs... (probably some are NaN)"
	
	print "+ Attach closest wavelength solution for each Science file and correct from RVs"
	w_sci, WCdicts_sci = RB07.AttachWC(x_sci, sci, sci_names, 
									  w_arc, arcs, arcRVs, arc_names, 
									  w_MasterArc, arcs_cl, MasterRVs, cv)
	
	print "========================================"
	print "RB08: Science - Radial velocity "
	print "======================================== \n"
	
	print "+ Calculating RVs for the science frames..."
	already_done = os.path.isfile(cv.aux_dir+'sci_RVs.npz')
	if already_done == False:
		RVdicts = RB08.ScienceRV(w_sci, cv, sci_names)
		np.savez(cv.aux_dir+'sci_RVs', RVdicts=RVdicts)	
	else:
		print "    --> Science RVs found. Loading dictionaries..."
		tab = np.load(cv.aux_dir+'sci_RVs.npz',allow_pickle=True)
		RVdicts = tab['RVdicts'].tolist()
	
	rvtmp, ervtmp = [], []
	jdtmp = []
	for RVdict in RVdicts:
		jdtmp.append(RVdict["HJD"])
		rvtmp.append(RVdict["RV"])
		ervtmp.append(RVdict["eRV"])

	if 0:
		RVolds = np.array([6.478,6.526, 6.498, 6.530, 6.488])
		rvtmp, ervtmp = [], []
		jdtmp = []
		for RVdict in RVdicts:
			plt.errorbar(RVdict["HJD"],RVdict["RV"],yerr=RVdict["eRV"],fmt='o',c='red')
			jdtmp.append(RVdict["HJD"])
			rvtmp.append(RVdict["RV"])
			ervtmp.append(RVdict["eRV"])
		jdtmp = np.array(jdtmp)
		rvtmp = np.array(rvtmp)
		ervtmp = np.array(ervtmp)
		print np.mean(RVolds), np.mean(rvtmp)
		for i in range(len(RVolds)):
			plt.errorbar(jdtmp[i], RVolds[i]-np.mean(RVolds)+np.mean(rvtmp[0:5]),yerr=0.027,fmt='o',c='blue') #
		plt.show()
		plt.close()


	print "=========================================="
	print "RB09: Science - Spec Norm. & CCF analysis "
	print "========================================== \n"

	print "+ Normalize spectrum and get SNR..."
	NORMdicts = RB09.spec_norm(w_sci, cv, sci_names)

	print "+ Merge orders into a 1D spectrum..."
	MERGEdicts = RB09.merge1D(w_sci, NORMdicts, cv, sci_names)
	

	# Test plots 
	snrtmp = []
	for NORMdict in NORMdicts:
		snrtmp.append(NORMdict["SNR"])
	snrtmp = np.array(snrtmp)
	if 0:
		jdtmp, snrtmp,rvtmp,ervtmp = jdtmp[:-1], snrtmp[:-1],rvtmp[:-1], ervtmp[:-1]
		z = np.polyfit(snrtmp, rvtmp, 1)
		p = np.poly1d(z)
		plt.errorbar(jdtmp,(RVolds-np.mean(RVolds))*1.e3, yerr=27., fmt='s',c='gray',label='Old pipeline',alpha=0.7)
		plt.errorbar(jdtmp,(rvtmp-p(snrtmp))*1.e3, yerr=13., fmt='o',c='red',label='New pipeline',alpha=0.7)
		plt.grid(ls=':')
		plt.axhline(0.,ls='--',c='k')
		plt.ylabel("RV (m/s)")
		plt.xlabel("JD")
		plt.legend()
		plt.ylim(-40,40)
		#plt.errorbar(snrtmp,rvtmp, yerr=ervtmp, fmt='o')
		#plt.errorbar(snrtmp,RVolds-np.mean(RVolds)+np.mean(rvtmp[0:5]), yerr=0.027, fmt='o')
		#plt.savefig('tmp_HD109358.pdf',bbox_inches="tight")
		plt.close()

	if 0:
		plt.errorbar(snrtmp[~np.isnan(rvtmp)],rvtmp[~np.isnan(rvtmp)], yerr=ervtmp[~np.isnan(rvtmp)], fmt='o')
		#plt.errorbar(snrtmp,RVolds-np.mean(RVolds)+np.mean(rvtmp[0:5]), yerr=0.027, fmt='o')
		plt.show()
		plt.close()
		
		print np.std(rvtmp-p(snrtmp)) * 1.e3
		print np.std(RVolds-np.mean(RVolds)) * 1.e3
		print ervtmp*1.e3

	
	print "========================================"
	print "RB10: Saving the FINAL results "
	print "======================================== \n"

	print "+ Saving arc reduced frames..."
	#RB10.save_ArcFrames(arc_names, w_arc, cv, 'ARC')
	RB10.save_ArcFrames(MasterArc_names, w_MasterArc, arcs_cl, cv, 'MasterARC')

	print "+ Saving science reduced frames..."
	RB10.save_SciFrames(sci_names, w_sci, WCdicts_sci, RVdicts, NORMdicts, MERGEdicts, cv)
	
	#sys.stdout = orig_stdout
	#flog.close()

	print " "
	print "========================================"
	print "             THE END! "
	print "         Enjoy your CAFE!"
	print "======================================== \n"
	
	sys.exit()


arguments = sys.argv

#Working directory
self.root_path 	= os.getcwd()

#Raw data
self.rawdir		= arguments[1]
self.rawdir_path = root_path+'/00_RAW/'

#Reduced data
self.redid			= arguments[2]
self.reddir			= arguments[1]
self.reddir_path	= root_path+'/11_REDUCED/'+red_id






sys.exit()


#def RB00_checks():


#RB01_prepare_data()
	
	
	

#def RB02_preliminary():


#def RV03_bias():










