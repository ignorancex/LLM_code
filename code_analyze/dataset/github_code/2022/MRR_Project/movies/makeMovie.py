import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import os
import string
import os
import moviepy.video.io.ImageSequenceClip
from PIL import Image
import glob





def makeMovie_rates(whichRate='intrinsic', fps=.4, duration=300):
	'''
	whichRate = 'intrinsic' or 'observed'
	fps=0.4, frames per second
	duration = duration of the movie 
	'''




	GSMFs = ['Panter et al. (2004) Single', 'Furlong et al. (2015) Single', 'Furlong et al. (2015) Double']
	MZs   = [ 'Langer et al. (2006)'      , 'Langer et al. +offset (2006)', 'Ma et al. (2015)']
	SFRs  = ['Madau et al. (2014)'         ,'Strolger et al. (2004)',     'Madau et al. (2017)']


	MSSFRnameslist = []
	MSSFRnameslist.append('000') # add phenomenological 



	for ind_SFR, SFR in enumerate(SFRs):
		ind_x = ind_SFR + 1
		for ind_GSMF, GSMF in enumerate(GSMFs):
			ind_y = ind_GSMF + 1
			for ind_MZ, MZ in enumerate(MZs):
				ind_z = ind_MZ + 1
				
				MSSFRnameslist.append('%s%s%s'%(ind_x, ind_y, ind_z))



                
  
	image_folder = '../plottingCode/Fig_2/supplementary_material/'

	images = []
	if whichRate=='intrinsic':
		for ind_m, SFRD_model in enumerate(MSSFRnameslist):
			images.append(image_folder +   'Rates_intrinsic_'  + SFRD_model + '.png')
	elif whichRate=='observed':
		for ind_m, SFRD_model in enumerate(MSSFRnameslist):
			images.append(image_folder +   'Rates_observed_'  + SFRD_model + '.png')


	image_files = images
	clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
	clip.write_videofile(image_folder+'movie_'+ 'rates_' + whichRate + '.mp4')


	# make also gif:
 
	# Create the frames
	frames = []
	# imgs = glob.glob("*.png")
	for i in images:
	    new_frame = Image.open(i)
	    frames.append(new_frame)
	 
	# Save into a GIF file that loops forever
	frames[0].save(image_folder+'gif_'+ 'rates_' + whichRate +  '.gif', format='GIF',
	               append_images=frames[1:],
	               save_all=True,
	               duration=duration, loop=0)


	print('done')
	return 



#####

def makeMovie_MBH1(whichRate='intrinsic', fps=.4, duration=300):
	'''
	whichRate = 'intrinsic' or 'observed'
	fps=0.4, frames per second
	duration = duration of the movie 
	'''




	nModels=20 # 
	BPSnameslist = list(string.ascii_uppercase)[0:nModels]



                
  
	image_folder = '/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/plottingCode/extra/intrinsic_weighted_distribution/'

	images = []

	for ind_m, BPSmodel in enumerate(BPSnameslist):
		images.append(image_folder +   'M1_z0__'  + BPSmodel + '.png')



	image_files = images
	clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
	clip.write_videofile(image_folder+'movie_'+ 'M1_' + whichRate + '.mp4')


	# make also gif:
 
	# Create the frames
	frames = []
	# imgs = glob.glob("*.png")
	for i in images:
	    new_frame = Image.open(i)
	    frames.append(new_frame)
	 
	# Save into a GIF file that loops forever
	frames[0].save(image_folder+'gif_'+ 'rates_' + whichRate +  '.gif', format='GIF',
	               append_images=frames[1:],
	               save_all=True,
	               duration=duration, loop=0)


	print('done')
	return 





#####

def makeMovie_Mtotq_FormChannel(whichRate='intrinsic', fps=.4, duration=300):
	'''
	whichRate = 'intrinsic' or 'observed'
	fps=0.4, frames per second
	duration = duration of the movie 
	'''




	nModels=17 # 
	BPSnameslist = list(string.ascii_uppercase)[0:nModels]



                
  
	image_folder = '/Users/floorbroekgaarden/Projects/GitHub/DCO_FormationChannels/plottingCode/Fig_2/vs_Mtot_q/'

	images = []

	for ind_m, BPSmodel in enumerate(BPSnameslist):
		images.append(image_folder +   'Formation_channel_'  + BPSmodel + '_2.png')



	image_files = images
	clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
	clip.write_videofile(image_folder+'movie_'+ 'M1_' + 'BHBH' + '.mp4')


	# make also gif:
 
	# Create the frames
	frames = []
	# imgs = glob.glob("*.png")
	for i in images:
	    new_frame = Image.open(i)
	    frames.append(new_frame)
	 
	# Save into a GIF file that loops forever
	frames[0].save(image_folder+'gif_'+ 'rates_' + 'BHBH' +  '.gif', format='GIF',
	               append_images=frames[1:],
	               save_all=True,
	               duration=duration, loop=0)


	print('done')
	return 





#####

def makeMovie_chi_BH(which_param='chi_BH1', fps=.4, duration=300):
	'''
	whichRate = 'intrinsic' or 'observed'
	fps=0.4, frames per second
	duration = duration of the movie 
	'''




	nModels=17 # 
	BPSnameslist = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',  'P', 'Q', 'R', 'S', 'T'] #list(string.ascii_uppercase)[0:nModels]


	mssfr = '112'
                
  
	image_folder = '/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/plottingCode/extra/lightestBHformsFirst/'

	images = []

	for ind_m, BPSmodel in enumerate(BPSnameslist):
		images.append(image_folder +   'Prob_weight_MRR_vs_q_model_log_'+BPSmodel+ mssfr+ '_'+ which_param +'.png' )



	image_files = images
	clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
	clip.write_videofile(image_folder+'movie_'+ 'M1_' + 'BHBH' + which_param  +  '.mp4')


	# make also gif:
 
	# Create the frames
	frames = []
	# imgs = glob.glob("*.png")
	for i in images:
	    new_frame = Image.open(i)
	    frames.append(new_frame)
	 
	# Save into a GIF file that loops forever
	frames[0].save(image_folder+'gif_'+ 'det_rates_rel_' + 'BHBH' +  which_param + '.gif', format='GIF',
	               append_images=frames[1:],
	               save_all=True,
	               duration=duration, loop=0)


	print('done')
	return 





#####

def makeMovie_chi_eff_BH(which_param='chi_BH1', fps=.4, duration=300):
	'''
	whichRate = 'intrinsic' or 'observed'
	fps=0.4, frames per second
	duration = duration of the movie 
	'''




	nModels=17 # 
	BPSnameslist = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',  'P', 'Q', 'R', 'S', 'T'] #list(string.ascii_uppercase)[0:nModels]


	mssfr = '112'
                
  
	image_folder = '/Users/floorbroekgaarden/Projects/GitHub/MRR_project/1D_spin_histograms/'

	images = []

	for ind_m, BPSmodel in enumerate(BPSnameslist):
		images.append(image_folder +  'Prob_weight_Chi1_vs_Chi2_vs_Total_model_'+BPSmodel+ mssfr+ '_'+ which_param +'.png') 


	image_files = images
	clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
	clip.write_videofile(image_folder+'movie_'+ 'M1_' + 'BHBH' + which_param  +  '.mp4')


	# make also gif:
 
	# Create the frames
	frames = []
	# imgs = glob.glob("*.png")
	for i in images:
	    new_frame = Image.open(i)
	    frames.append(new_frame)
	 
	# Save into a GIF file that loops forever
	frames[0].save(image_folder+'gif_'+ 'det_rates_rel_' + 'BHBH' +  which_param + '.gif', format='GIF',
	               append_images=frames[1:],
	               save_all=True,
	               duration=duration, loop=0)


	print('done')
	return 





makeMovie_intrinsicRates=False
makeMovie_intrinsicM1=False
makeMovie_Mtotq_FormChannels=False
makeMovieChiBH=False
makeMovieChiEffBH=True







# Run rhis using python 3!! 

if makeMovieChiEffBH==True:
	# makeMovie_chi_eff_BH(which_param='chi_eff')
	# makeMovie_chi_eff_BH(which_param='MassRatio')
	makeMovie_chi_eff_BH(which_param='chirpmass')
	makeMovie_chi_eff_BH(which_param='MassBH1')


if makeMovieChiBH==True:
	makeMovie_chi_BH(which_param='chi_BH1')
	makeMovie_chi_BH(which_param='chi_BH2')


if makeMovie_intrinsicRates==True:
	makeMovie_rates(whichRate='intrinsic')


if makeMovie_intrinsicM1==True:
	makeMovie_MBH1(whichRate='intrinsic')

if makeMovie_Mtotq_FormChannels==True: 
	makeMovie_Mtotq_FormChannel(whichRate='intrinsic')

