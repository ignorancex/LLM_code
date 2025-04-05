import shutil
import glob
import os
from astropy.io import fits
import numpy as np
from scipy.stats import binned_statistic,sigmaclip
from sklearn.cluster import MeanShift
from astroML.stats import sigmaG
import matplotlib.pyplot as plt

import CAFEutilities
import CAFEx_SetupFile as CS


# class FrameStruct():
#     def __init__(self, FrameList):
#         self.data = field1
#         self.exptimes = field2
#         self.files = field3


def FrameClustering(frames,cv):

	dates = frames['jd']
	x = dates
	X = np.array(zip(x,np.zeros(len(x))),dtype=float)
	bandwidth = CS.bandwidth[frames['id']] # in hours, range to cluster the bias times.
	ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=False, min_bin_freq=9)
	ms.fit(X)
	labels = ms.labels_
	cluster_centers = ms.cluster_centers_	
	nclusters = len(cluster_centers)
	cl_centers = np.array([cl[0] for cl in cluster_centers])
	clusterIDs = np.array(range(len(cl_centers)))

	if frames['id'] == 'bias': MinMembers = CS.MinMembersBias
	if frames['id'] == 'flat': MinMembers = CS.MinMembersFlat
	if frames['id'] == 'arcs': MinMembers = CS.MinMembersArcs
	
	# Remove clusters with less than 10 members
	if frames['id'] != 'sci':
		for i in range(nclusters):
			group = np.where(labels == i)[0]
			if len(group) < MinMembers:
				labels[group] = -1
				cl_centers[i] = -1
		clusterIDs	= clusterIDs[cl_centers != -1]
		cl_centers 	= cl_centers[cl_centers != -1]
			
	# Create dictionary		
	if frames['id'] == 'arcs':
		eveningTw,morningTw = CAFEutilities.observatory_twill(cv.night)
		eveningID = clusterIDs[np.nonzero(cl_centers < eveningTw)]
		morningID = clusterIDs[np.nonzero(cl_centers > morningTw)]
		res =  {'membership':labels, 
				'centers':cl_centers, 
				'nclusters':len(cl_centers),
				'eveningID':eveningID,
				'morningID':morningID}
	else:
		res =  {'membership':labels, 
				'centers':cl_centers, 
				'nclusters':len(cl_centers)}

	return res


def FrameCombine(frames,frames_cl,cv,biasList='None',biasFrames='None'):
	id = frames['id']
	sigclip = CS.sigclip[id]
	dc = frames['dc'] # np.array(frames['dc']).astype(float)
	MasterFrame = []
	if ((id == 'bias') | (id == 'flat')):
		print "%22s %20s %10s %15s %15s" % ("    --> MasterFrame ID","JD","Nframes","Mean","Deviation")
	if id == 'arcs':
		print "%22s %20s %10s %15s %15s" % ("    --> MasterFrame ID","JD","Nframes","Mean","Deviation")
		
	for k in range(frames_cl['nclusters']):
		members = frames_cl['membership'] == k
		#print dc[0,:,:]
		#MasterFrame, low, upp = sigmaclip(dc[members,:,:],sigclip,sigclip)
		data = dc[members,:,:]
		means = [np.nanmedian(data[ii,:,:]) for ii in range(len(data[:,0,0]))]
		std = [np.nanstd(data[ii,:,:]) for ii in range(len(data[:,0,0]))]
		if id == 'bias':
			ClippedFrames = data[((np.abs(means - np.nanmedian(data)) < sigclip * np.nanstd(data)) & (std < sigclip * np.nanstd(data))),:,:]
			if len(ClippedFrames[:,0,0]) > CS.MinMembersBias:
				MF = np.nanmean(ClippedFrames,axis=0)
				MasterFrame.append(MF)
				print "%22s %20f %10i %15.2f %15.2f" % (np.str(k),frames_cl['centers'][k], len(ClippedFrames[:,0,0]), np.nanmean(MasterFrame[k]),np.nanstd(MasterFrame[k]) )
		if id == 'flat':
			good = np.where(np.abs(means - np.nanmedian(data)) < sigclip * np.nanstd(data))[0]
			ClippedFrames = data[good ,:,:]
			print 
			nframes = len(ClippedFrames[:,0,0])
			# Bias substraction
			for i in range(nframes):
				elem = CAFEutilities.get_closer_frame(frames['jd'][i],biasList['centers'])
				zero = biasFrames[elem]
				ClippedFrames[i,:,:] -= zero[0,:,:]
			if nframes > CS.MinMembersFlat: #len(ClippedFrames[:,0,0])
				MF = np.nanmedian(ClippedFrames,axis=0)
				MasterFrame.append(MF)
				print "%22s %20f %10i %15.2f %15.2f" % (np.str(k),frames_cl['centers'][k], len(ClippedFrames[:,0,0]), np.nanmean(MF),np.nanstd(MF) )
		if id == 'arcs':
#			if (k == frames_cl['eveningID']) |  (k == frames_cl['morningID']):
			ClippedFrames = data[(np.abs(means - np.nanmean(data)) < sigclip * np.nanstd(data) ) ,:,:]
			nframes = len(ClippedFrames[:,0,0])
			# Bias substraction
			for i in range(nframes):
				elem = CAFEutilities.get_closer_frame(frames['jd'][i],biasList['centers'])
				zero = biasFrames[elem]
				ClippedFrames[i,:,:] -= zero[0,:,:]
			if len(ClippedFrames[:,0,0]) > CS.MinMembersArcs:
				MF = np.nanmean(ClippedFrames,axis=0)
				MasterFrame.append(MF)
				print "%22s %20f %10i %15.2f %15.2f" % (np.str(k),frames_cl['centers'][k], len(ClippedFrames[:,0,0]), np.nanmean(MF),np.nanstd(MF) )
			
		nframes = len(ClippedFrames)
	
	RON  = 3.3/np.sqrt(nframes)
	GAIN = 1.0
	return np.array(MasterFrame),RON,GAIN
		
	

def BiasRemove(frames, MasterBias, biasList, cv):
	dc = np.array(frames['dc']).astype(float)
	for i in range(len(dc[:,0,0])):
		elem = CAFEutilities.get_closer_frame(frames['jd'][i],biasList['centers'])
		zero = MasterBias[elem]
		dc[i,:,:] -= zero[0,:,:]
		# Remove bad column X=233
		dc[i,233,:] = np.nan
		
		frames['dc'][i,:,:] = dc[i,:,:]
		
	return frames

def BiasRemoveMaster(frames, MasterBias, biasList, ArcsList, cv):
	dc = np.array(frames).astype(float)
	for i in range(len(dc[:,0,0])):
		elem = CAFEutilities.get_closer_frame(ArcsList['centers'][i],biasList['centers'])
		zero = MasterBias[elem]
		dc[i,:,:] -= zero[0,:,:]
		#frames['dc'][i,:,:] = dc[i,:,:]
		
	return dc








