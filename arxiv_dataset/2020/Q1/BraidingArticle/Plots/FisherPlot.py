#!/usr/bin/python
# Filename: FisherPlot.py

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

##################################################
#Routine to plot the 1- and 2-sigma ellipses for a given Fisher matrix
def plot_FisherMatrix(fiduvec, Matrix, paramnames='', figsize='', debug=0, linewidth=2, fontsize=12, labelsize=12):
	n = np.size(fiduvec)
	assert np.size(np.shape(Matrix)) == 2 , 'Matrix must be a 2D array'
	nx = np.shape(Matrix)[0] ; ny = np.shape(Matrix)[1]
	assert nx == ny , 'Matrix must be square'
	assert nx == n , 'fiduvec and Matrix must have corresponding sizes'

	if paramnames == '':
		paramnames = ['' for x in range(n)]
	if figsize == '':
		figsize = (4*n,3.5*n)


	#invert Fisher Matrix
	iMatrix = np.linalg.inv(Matrix)
	#initialise intervals over which each param will be drawn
	n4interv = 1000
	interv_mat = np.zeros((n4interv,n))
	for i in range(0,n):
		sigmai = np.sqrt(iMatrix[i,i])
		x = np.linspace(-1,1,n4interv)*3.5*sigmai+fiduvec[i]
		interv_mat[:,i] = x
	
	#Do the figure
	fig = plt.figure(figsize=figsize)
	#fig = plt.figure()
	xx = np.linspace(0,1,10)
	
	for c in range(0,n**2):
		i = c % n ; j= c // n
		#diagonal: 1D Gaussian
		if i == j:
			sigmai2 = iMatrix[i,i]
			x = interv_mat[:,i]
			gauss = np.exp(-(x-fiduvec[i])**2/(2.*sigmai2))/np.sqrt(2.*np.pi*sigmai2)
			ax = fig.add_subplot(n, n, c+1)
			ax.set_xlim(x.min(),x.max()) ; ax.set_ylim(0,1.1*gauss.max())
			ax.xaxis.set_tick_params(labelsize=labelsize) ; ax.yaxis.set_tick_params(labelsize=labelsize)
			plt.xlabel(paramnames[i],fontsize=fontsize) ; plt.ylabel('P('+paramnames[i]+')',fontsize=fontsize)
			plt.plot(x,gauss)
		#below diagonal : ellipses at 1/2 sigma
		if i < j:
			x = interv_mat[:,i]
			y = interv_mat[:,j]
			iMat2X2 = np.matrix([[iMatrix[i,i],iMatrix[i,j]],[iMatrix[j,i],iMatrix[j,j]]])
			#Find the eigenvalues of iMat2X2
			tr = iMatrix[i,i]+iMatrix[j,j] ; det = np.linalg.det(iMat2X2)
			discr = tr**2 - 4.*det
			lambda1 = (tr + np.sqrt(discr))/2. ; lambda2 = (tr-np.sqrt(discr))/2.
			if debug == 1:
				print('lambda1/2:',lambda1,lambda2)
				print('sqrt(lambda1/2):',np.sqrt(lambda1),np.sqrt(lambda2))
			#eigenvectors
			if iMatrix[i,j] == 0:
				X1 = [1,0] ; X2 = [0,1]
			else:
				X1 = [1,(lambda1-iMatrix[i,i])/iMatrix[i,j]] ; X1/=np.linalg.norm(X1)
				X2 = [-(lambda1-iMatrix[i,i])/iMatrix[i,j],1] ; X2/=np.linalg.norm(X2)
			#tranfo matrix and invert
			X1 = np.array(X1) ; X2 = np.array(X2)
			P = np.column_stack((X1,X2))
			if debug == 1:
				print('iMat2X2,P:',iMat2X2,P)
				print('lambda1 X1,iMat2X2 # X1', lambda1*X1, iMat2X2 * np.matrix(X1).transpose())
				print('lambda2 X2,iMat2X2 # X2', lambda2*X2, iMat2X2 * np.matrix(X2).transpose())
				print('P # [1,0], P # [0,1], X1, X2', np.dot(P,[1,0]), np.dot(P,[0,1]), X1, X2)
				print('P[0,0],P[0,1],P[1,0],P[1,1]',P[0,0],P[0,1],P[1,0],P[1,1])
			#ellipse in diagonalised basis
			phi = np.linspace(0,1,num=n4interv)*2*np.pi
			ed = np.array([np.cos(phi)*np.sqrt(lambda1),np.sin(phi)*np.sqrt(lambda2)])
			#in original basis
			ellipse = np.array([[P[0,0]*ed[0,:]+P[0,1]*ed[1,:]],[P[1,0]*ed[0,:]+P[1,1]*ed[1,:]]])
			#points at 2 & 3 sigma
			x2 = np.zeros(n4interv) ; y2 = np.zeros(n4interv)
			x3 = np.zeros(n4interv) ; y3 = np.zeros(n4interv)
			x2[:] = ellipse[0,:]*2+fiduvec[i] ; y2[:] = ellipse[1,:]*2+fiduvec[j]
			x3[:] = ellipse[0,:]*3+fiduvec[i] ; y3[:] = ellipse[1,:]*3+fiduvec[j]
			#Do the plot
			ax = fig.add_subplot(n, n, c+1)
			ax.set_xlim(x.min(),x.max()) ; ax.set_ylim(y.min(),y.max())
			ax.xaxis.set_tick_params(labelsize=labelsize) ; ax.yaxis.set_tick_params(labelsize=labelsize)
			plt.xlabel(paramnames[i],fontsize=fontsize) ; plt.ylabel(paramnames[j],fontsize=fontsize)
			plt.plot(x2,y2,color='red')
			plt.plot(x3,y3,color='blue')
			plt.plot(fiduvec[i],fiduvec[j],marker='+',color='black')
	

##################################################
#Routine to oplot the n-sigma ellipses of different Fisher matrices
def oplot_FisherMatrix(fiduvec, dico, paramnames='', figsize='', \
colors='', legitems='', nsigma=2, range1D=2.1, linewidth=2, fontsize=15, labelsize=12, debug=0):
	#dico is a dictionnary with keys 0..nmat-1
	#where dico[i] is the i-th Fisher matrix
	nparams = np.size(fiduvec)
	nmat = len(dico.keys())
	assert list(dico.keys()) == [i for i in range(0,nmat)] , 'Dictionnary must have keys : 0..N_matrices'
	for k in range(0,nmat):
		Matrix = dico[k]
		assert np.size(np.shape(Matrix)) == 2 , str(k)+'-th matrix must be a 2D array'
		nx = np.shape(Matrix)[0] ; ny = np.shape(Matrix)[1]
		assert nx == ny , str(k)+'-th matrix must be square'
		assert nx == nparams , 'fiduvec and '+str(k)+'-th matrix must have corresponding sizes'
	if paramnames == '':
		paramnames = ['' for x in range(0,nparams)]
	if figsize == '':
		figsize = (4*nparams,3.5*nparams)
	if colors == '':
		colors = ['black' for x in range(0,nmat)]
	#Invert Fisher matrices
	dicoinv = {k:0 for k in range(0,nmat)}
	for k in range(0,nmat):
		Matrix = dico[k]
		dicoinv[k] = np.linalg.inv(Matrix)
	#initialise intervals over which each param will be drawn
	n4interv = 1000
	interv_mat = np.zeros((n4interv,nparams))
	sigmaimin_arr = np.zeros(nparams)
	for i in range(0,nparams):
		sigmaimin2 = np.min([(dicoinv[k])[i,i] for k in range(0,nmat)])
		sigmaimax2 = np.max([(dicoinv[k])[i,i] for k in range(0,nmat)])
		sigmaimin_arr[i] = np.sqrt(sigmaimin2)
		x = np.linspace(-1,1,n4interv)*range1D*np.sqrt(sigmaimax2)+fiduvec[i]
		interv_mat[:,i] = x

	#Do the figure
	fig = plt.figure(figsize=figsize)
	xx = np.linspace(0,1,10)
	for c in range(0,nparams**2):
		i = c % nparams ; j= c // nparams
		#diagonal: 1D Gaussian
		if i == j:
			sigmaimin = sigmaimin_arr[i]
			x = interv_mat[:,i]
			ax = fig.add_subplot(nparams, nparams, c+1)
			ax.set_xlim(x.min(),x.max()) ; ax.set_ylim(0,0.44/sigmaimin)
			ax.yaxis.set_tick_params(labelsize=labelsize)
			ax.set_yticklabels( () )
			#h=plt.ylabel('P('+paramnames[i]+')',fontsize=fontsize) ; ax.yaxis.set_label_position("right")
			ax.xaxis.set_tick_params(labelsize=0.)
			if i==(nparams-1):
				plt.xlabel(paramnames[i],fontsize=fontsize)
				ax.xaxis.set_tick_params(labelsize=labelsize)
			for k in range(0,nmat):
				sigmai2 = (dicoinv[k])[i,i]
				gauss = np.exp(-(x-fiduvec[i])**2/(2.*sigmai2))/np.sqrt(2.*np.pi*sigmai2)
				plt.plot(x,gauss,color=colors[k],linewidth=linewidth)
		#below diagonal : ellipses at n sigma
		if i < j:
			x = interv_mat[:,i]
			y = interv_mat[:,j]
			ax = fig.add_subplot(nparams, nparams, c+1)
			ax.set_xlim(x.min(),x.max()) ; ax.set_ylim(y.min(),y.max())
			#ax.xaxis.set_tick_params(labelsize=labelsize) ; plt.xlabel(paramnames[i],fontsize=fontsize)
			ax.xaxis.set_tick_params(labelsize=0.)
			if j==(nparams-1):
				plt.xlabel(paramnames[i],fontsize=fontsize)
				ax.xaxis.set_tick_params(labelsize=labelsize)
			#ax.yaxis.set_tick_params(labelsize=labelsize) ; plt.ylabel(paramnames[j],fontsize=fontsize)
			ax.yaxis.set_tick_params(labelsize=0.)
			if i==0:
				plt.ylabel(paramnames[j],fontsize=fontsize)
				ax.yaxis.set_tick_params(labelsize=labelsize)
			plt.plot(fiduvec[i],fiduvec[j],marker='+',color='black')
			for k in range(0,nmat):
				iMatrix = dicoinv[k]
				iMat2X2 = np.matrix([[iMatrix[i,i],iMatrix[i,j]],[iMatrix[j,i],iMatrix[j,j]]])
				#Find the eigenvalues of iMat2X2
				tr = iMatrix[i,i]+iMatrix[j,j] ; det = np.linalg.det(iMat2X2)
				discr = tr**2 - 4.*det
				lambda1 = (tr + np.sqrt(discr))/2. ; lambda2 = (tr-np.sqrt(discr))/2.
				#eigenvectors
				if iMatrix[i,j] == 0:
					X1 = [1,0] ; X2 = [0,1]
				else:
					X1 = [1,(lambda1-iMatrix[i,i])/iMatrix[i,j]] ; X1/=np.linalg.norm(X1)
					X2 = [-(lambda1-iMatrix[i,i])/iMatrix[i,j],1] ; X2/=np.linalg.norm(X2)
				#tranfo matrix and invert
				X1 = np.array(X1) ; X2 = np.array(X2)
				P = np.column_stack((X1,X2))
				#ellipse in diagonalised basis
				phi = np.linspace(0,1,num=n4interv)*2*np.pi
				ed = np.array([np.cos(phi)*np.sqrt(lambda1),np.sin(phi)*np.sqrt(lambda2)])
				#in original basis
				ellipse = np.array([[P[0,0]*ed[0,:]+P[0,1]*ed[1,:]],[P[1,0]*ed[0,:]+P[1,1]*ed[1,:]]])
				#points at n sigma
				xn = np.zeros(n4interv) ; yn = np.zeros(n4interv)
				xn[:] = ellipse[0,:]*nsigma+fiduvec[i] ; yn[:] = ellipse[1,:]*nsigma+fiduvec[j]
				#Do the plot
				plt.plot(xn,yn,color=colors[k],linewidth=linewidth)
	#plt.tight_layout()

##################################################
#IDL-like rainbow color map, goes from black-violet to red-white
C = np.zeros((3,256))
C[:,0] = [0,0,0] ; C[:,1] = [4,0,3] ; C[:,2] = [9,0,7] ; C[:,3] = [13,0,10] ; C[:,4] = [18,0,14]
C[:,5] = [22,0,19] ; C[:,6] = [27,0,23] ; C[:,7] = [31,0,28] ; C[:,8] = [36,0,32] ; C[:,9] = [40,0,38]
C[:,10] = [45,0,43] ; C[:,11] = [50,0,48] ; C[:,12] = [58,0,59] ; C[:,13] = [61,0,63] ; C[:,14] = [64,0,68]
C[:,15] = [68,0,72] ; C[:,16] = [69,0,77] ; C[:,17] = [72,0,81] ; C[:,18] = [74,0,86] ; C[:,19] = [77,0,91]
C[:,20] = [79,0,95] ; C[:,21] = [80,0,100] ; C[:,22] = [82,0,104] ; C[:,23] = [83,0,109] ; C[:,24] = [84,0,118]
C[:,25] = [86,0,122] ; C[:,26] = [87,0,127] ; C[:,27] = [88,0,132] ; C[:,28] = [86,0,136] ; C[:,29] = [87,0,141]
C[:,30] = [87,0,145] ; C[:,31] = [87,0,150] ; C[:,32] = [85,0,154] ; C[:,33] = [84,0,159] ; C[:,34] = [84,0,163]
C[:,35] = [84,0,168] ; C[:,36] = [79,0,177] ; C[:,37] = [78,0,182] ; C[:,38] = [77,0,186] ; C[:,39] = [76,0,191]
C[:,40] = [71,0,195] ; C[:,41] = [70,0,200] ; C[:,42] = [68,0,204] ; C[:,43] = [66,0,209] ; C[:,44] = [60,0,214]
C[:,45] = [58,0,218] ; C[:,46] = [55,0,223] ; C[:,47] = [46,0,232] ; C[:,48] = [43,0,236] ; C[:,49] = [40,0,241]
C[:,50] = [36,0,245] ; C[:,51] = [33,0,250] ; C[:,52] = [25,0,255] ; C[:,53] = [21,0,255] ; C[:,54] = [16,0,255]
C[:,55] = [12,0,255] ; C[:,56] = [4,0,255] ; C[:,57] = [0,0,255] ; C[:,58] = [0,4,255] ; C[:,59] = [0,16,255]
C[:,60] = [0,21,255] ; C[:,61] = [0,25,255] ; C[:,62] = [0,29,255] ; C[:,63] = [0,38,255] ; C[:,64] = [0,42,255]
C[:,65] = [0,46,255] ; C[:,66] = [0,51,255] ; C[:,67] = [0,55,255] ; C[:,68] = [0,63,255] ; C[:,69] = [0,67,255]
C[:,70] = [0,72,255] ; C[:,71] = [0,84,255] ; C[:,72] = [0,89,255] ; C[:,73] = [0,93,255] ; C[:,74] = [0,97,255]
C[:,75] = [0,106,255] ; C[:,76] = [0,110,255] ; C[:,77] = [0,114,255] ; C[:,78] = [0,119,255] ; C[:,79] = [0,127,255]
C[:,80] = [0,131,255] ; C[:,81] = [0,135,255] ; C[:,82] = [0,140,255] ; C[:,83] = [0,152,255] ; C[:,84] = [0,157,255]
C[:,85] = [0,161,255] ; C[:,86] = [0,165,255] ; C[:,87] = [0,174,255] ; C[:,88] = [0,178,255] ; C[:,89] = [0,182,255]
C[:,90] = [0,187,255] ; C[:,91] = [0,195,255] ; C[:,92] = [0,199,255] ; C[:,93] = [0,203,255] ; C[:,94] = [0,216,255]
C[:,95] = [0,220,255] ; C[:,96] = [0,225,255] ; C[:,97] = [0,229,255] ; C[:,98] = [0,233,255] ; C[:,99] = [0,242,255]
C[:,100] = [0,246,255] ; C[:,101] = [0,250,255] ; C[:,102] = [0,255,255] ; C[:,103] = [0,255,246] ; C[:,104] = [0,255,242]
C[:,105] = [0,255,238] ; C[:,106] = [0,255,225] ; C[:,107] = [0,255,220] ; C[:,108] = [0,255,216] ; C[:,109] = [0,255,212]
C[:,110] = [0,255,203] ; C[:,111] = [0,255,199] ; C[:,112] = [0,255,195] ; C[:,113] = [0,255,191] ; C[:,114] = [0,255,187]
C[:,115] = [0,255,178] ; C[:,116] = [0,255,174] ; C[:,117] = [0,255,170] ; C[:,118] = [0,255,157] ; C[:,119] = [0,255,152]
C[:,120] = [0,255,148] ; C[:,121] = [0,255,144] ; C[:,122] = [0,255,135] ; C[:,123] = [0,255,131] ; C[:,124] = [0,255,127]
C[:,125] = [0,255,123] ; C[:,126] = [0,255,114] ; C[:,127] = [0,255,110] ; C[:,128] = [0,255,106] ; C[:,129] = [0,255,102]
C[:,130] = [0,255,89] ; C[:,131] = [0,255,84] ; C[:,132] = [0,255,80] ; C[:,133] = [0,255,76] ; C[:,134] = [0,255,67]
C[:,135] = [0,255,63] ; C[:,136] = [0,255,59] ; C[:,137] = [0,255,55] ; C[:,138] = [0,255,46] ; C[:,139] = [0,255,42]
C[:,140] = [0,255,38] ; C[:,141] = [0,255,25] ; C[:,142] = [0,255,21] ; C[:,143] = [0,255,16] ; C[:,144] = [0,255,12]
C[:,145] = [0,255,8] ; C[:,146] = [0,255,0] ; C[:,147] = [4,255,0] ; C[:,148] = [8,255,0] ; C[:,149] = [12,255,0]
C[:,150] = [21,255,0] ; C[:,151] = [25,255,0] ; C[:,152] = [29,255,0] ; C[:,153] = [42,255,0] ; C[:,154] = [46,255,0]
C[:,155] = [51,255,0] ; C[:,156] = [55,255,0] ; C[:,157] = [63,255,0] ; C[:,158] = [67,255,0] ; C[:,159] = [72,255,0]
C[:,160] = [76,255,0] ; C[:,161] = [80,255,0] ; C[:,162] = [89,255,0] ; C[:,163] = [93,255,0] ; C[:,164] = [97,255,0]
C[:,165] = [110,255,0] ; C[:,166] = [114,255,0] ; C[:,167] = [119,255,0] ; C[:,168] = [123,255,0] ; C[:,169] = [131,255,0]
C[:,170] = [135,255,0] ; C[:,171] = [140,255,0] ; C[:,172] = [144,255,0] ; C[:,173] = [153,255,0] ; C[:,174] = [157,255,0]
C[:,175] = [161,255,0] ; C[:,176] = [165,255,0] ; C[:,177] = [178,255,0] ; C[:,178] = [182,255,0] ; C[:,179] = [187,255,0]
C[:,180] = [191,255,0] ; C[:,181] = [199,255,0] ; C[:,182] = [203,255,0] ; C[:,183] = [208,255,0] ; C[:,184] = [212,255,0]
C[:,185] = [221,255,0] ; C[:,186] = [225,255,0] ; C[:,187] = [229,255,0] ; C[:,188] = [242,255,0] ; C[:,189] = [246,255,0]
C[:,190] = [250,255,0] ; C[:,191] = [255,255,0] ; C[:,192] = [255,250,0] ; C[:,193] = [255,242,0] ; C[:,194] = [255,238,0]
C[:,195] = [255,233,0] ; C[:,196] = [255,229,0] ; C[:,197] = [255,221,0] ; C[:,198] = [255,216,0] ; C[:,199] = [255,212,0]
C[:,200] = [255,199,0] ; C[:,201] = [255,195,0] ; C[:,202] = [255,191,0] ; C[:,203] = [255,187,0] ; C[:,204] = [255,178,0]
C[:,205] = [255,174,0] ; C[:,206] = [255,170,0] ; C[:,207] = [255,165,0] ; C[:,208] = [255,161,0] ; C[:,209] = [255,153,0]
C[:,210] = [255,148,0] ; C[:,211] = [255,144,0] ; C[:,212] = [255,131,0] ; C[:,213] = [255,127,0] ; C[:,214] = [255,123,0]
C[:,215] = [255,119,0] ; C[:,216] = [255,110,0] ; C[:,217] = [255,106,0] ; C[:,218] = [255,102,0] ; C[:,219] = [255,97,0]
C[:,220] = [255,89,0] ; C[:,221] = [255,85,0] ; C[:,222] = [255,80,0] ; C[:,223] = [255,76,0] ; C[:,224] = [255,63,0]
C[:,225] = [255,59,0] ; C[:,226] = [255,55,0] ; C[:,227] = [255,51,0] ; C[:,228] = [255,42,0] ; C[:,229] = [255,38,0]
C[:,230] = [255,34,0] ; C[:,231] = [255,29,0] ; C[:,232] = [255,21,0] ; C[:,233] = [255,17,0] ; C[:,234] = [255,12,0]
C[:,235] = [255,0,0] ; C[:,236] = [255,0,0] ; C[:,237] = [255,0,0] ; C[:,238] = [255,0,0] ; C[:,239] = [255,0,0]
C[:,240] = [255,0,0] ; C[:,241] = [255,0,0] ; C[:,242] = [255,0,0] ; C[:,243] = [255,0,0] ; C[:,244] = [255,0,0]
C[:,245] = [255,0,0] ; C[:,246] = [255,0,0] ; C[:,247] = [255,0,0] ; C[:,248] = [255,0,0] ; C[:,249] = [255,0,0]
C[:,250] = [255,0,0] ; C[:,251] = [255,0,0] ; C[:,252] = [255,0,0] ; C[:,253] = [255,0,0] ; C[:,254] = [255,0,0]
C[:,255] = [255,255,255]
C = C.T
cmap = matplotlib.colors.ListedColormap(C/255)

# End of FisherPlot.py
