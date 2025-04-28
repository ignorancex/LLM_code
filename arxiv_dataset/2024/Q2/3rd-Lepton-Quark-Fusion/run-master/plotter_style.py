#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
import brewer2mpl
import palettable.tableau

import argparse
from argparse import RawTextHelpFormatter

# plotting tool
# skip to main()

#auxiliary function for text formatting
def initializetex():
	#labels style
	#font.serif         : Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman
	#font.sans-serif    : Helvetica, Avant Garde, Computer Modern Sans serif
	#font.cursive       : Zapf Chancery
	#font.monospace     : Courier, Computer Modern Typewriter
	# plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
	# plt.rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
	plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
	plt.rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
	plt.rc('text', usetex=True)
	#plt.rc('font', family='serif')
	plt.rc('font', family='serif')
	   #r'\usepackage{helvet}',    # set the normal font here
       #r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       #r'\sansmath'               # <- tricky! -- gotta actually tell tex to use it!
	plt.rcParams['text.latex.preamble'] = [
       #r'\usepackage{siunitx}',   # I need upright \micro symbols, but you need...
       #r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       #r'\usepackage{cmbright}'
       r'\usepackage{isomath}'
	   r'\usepackage{slashed}'
	   r'\usepackage{bm}'
	   # needed for bold
	#   r'\DeclareFontFamily{OT1}{cmbr}{\hyphenchar\font45 }'
	#   r'\DeclareFontShape{OT1}{cmbr}{m}{n}{<-9>cmbr8 <9-10>cmbr9 <10-17>cmbr10 <17->cmbr17 }{}'
	#   r'\DeclareFontShape{OT1}{cmbr}{m}{sl}{<-9>cmbrsl8 <9-10>cmbrsl9 <10-17>cmbrsl10 <17->cmbrsl17 }{}'
	#   r'\DeclareFontShape{OT1}{cmbr}{m}{it}{<->ssub*cmbr/m/sl }{}'
	#   r'\DeclareFontShape{OT1}{cmbr}{b}{n}{<->ssub*cmbr/bx/n }{}'
	#   r'\DeclareFontShape{OT1}{cmbr}{bx}{n}{<->cmbrbx10 }{}'
       ]
	#plt.rc('legend',**{'fontsize':12})
	plt.rcParams.update({'font.size':9})
	plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
	plt.rcParams['axes.titlesize'] = 1.2*plt.rcParams['font.size']
	plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
	plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
	plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
	params = {'legend.fontsize': 7,
	'legend.handlelength': 2}
	plt.rcParams['hatch.linewidth'] = 0.5  # previous svg hatch linewidth
	plt.rcParams.update(params)


#figure size optimization
def figsize(fig_dims):

	golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good

	fig_width_in  = 3.5  # figure width in inches
	fig_height_in = fig_width_in * golden_ratio   # figure height in inches
	fig_dims    = [fig_width_in, fig_height_in] # fig dims as a list
	return fig_dims

#plot style
def mystyle(ax1):
	almost_black = '#262626'
	light_grey = np.array([float(248)/float(255)]*3)
	# For remaining spines, thin out their line and change the black to a slightly off-black dark grey
	spines_to_keep = ['bottom', 'left','top','right']
	for spine in spines_to_keep:
		ax1.spines[spine].set_linewidth(0.25)
		ax1.spines[spine].set_color(almost_black)

	# Change the labels to the off-black
	ax1.xaxis.label.set_color(almost_black)
	ax1.yaxis.label.set_color(almost_black)
	ax1.tick_params(which='both',colors=almost_black)

	# Change the axis title to off-black
	ax1.title.set_color(almost_black)

	#max tics
	#my_y_locator = MaxNLocator(6)
	#ax1.yaxis.set_major_locator(my_y_locator)
	#my_x_locator = MaxNLocator(6)
	#ax1.xaxis.set_major_locator(my_x_locator)

	#ax1.xaxis.labelpad = 0
	#ax1.yaxis.labelpad = 0

def mygrid(ax1):
	grid_grey = '#C7C7C7'
	#grid style
	ax1.set_axisbelow(True)
	ax1.grid(color=grid_grey, linestyle='-', linewidth=0.25)

def despine(ax):
	spines_to_remove = ['top', 'right']
	for spine in spines_to_remove:
		ax.spines[spine].set_visible(False)
	almost_black = '#262626'
	spines_to_keep = ['bottom', 'left']
	for spine in spines_to_keep:
		ax.spines[spine].set_linewidth(0.25)
		ax.spines[spine].set_color(almost_black)
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')

#legend style
def mylegstyle(leg):
	almost_black = '#262626'
	light_grey = np.array([float(248)/float(255)]*3)
	#ax1.legend(frameon=False)
	# Change the legend label colors to almost black, too
	texts = leg.get_texts()
	leg.get_frame().set_edgecolor(almost_black)
	for t in texts:
		t.set_color(almost_black)
	rect = leg.get_frame()
	rect.set_facecolor(light_grey)
	rect.set_linewidth(0.0)
	params = {'legend.fontsize': 7,
	'legend.handlelength': 2}
	plt.rcParams.update(params)
	#	'legend.labelspacing': 1,
	#'legend.handleheight': 1}

#main program - plotting
def logxplot_ratio(output,pdfnames,xlabel,ylabel,title,x,y,delta,npdfs):

	#initialization
	initializetex()

	#colours and styles
	almost_black = '#262626'
	light_grey = np.array([float(248)/float(255)]*3)
	dashedlines = [[2,2],[4,2,4,2],[100000,1],[4,1,1,1],[4, 2, 1, 2, 1, 2]]

	set1 = palettable.tableau.TableauMedium_10.mpl_colors
	set1dark = palettable.tableau.Tableau_10.mpl_colors
	hatch = ('//////','\\\\\\\\\\\\',None,None)
	facecolor = ('none','none',set1[2],set1[3])
	alpha = (0.5,0.5,0.7,0.2)
	alphaleg = (None,None,0.7,0.2)

	#write output
	pdf_pages = PdfPages(output)

	fig_dims = []
	fig_dims = figsize(fig_dims)
	fig = plt.figure(figsize=fig_dims)
	lines = []

	ref = y[0]

	for i in range(0,npdfs):

		plt.ylabel(ylabel)
		plt.xlabel(xlabel)
		plt.title(title)

		ax1 = fig.add_subplot(111)
		mystyle(ax1)

		ax1.set_xscale('log')
		#mygrid(ax1)

		#adjust x range
		plt.xlim([x[0],x[-1]])

		yy = y[i]/ref
		ydown = (y[i]-delta[i])
		yup = (y[i]+delta[i])
		yl = np.array(ydown/ref)
		yu = np.array(yup/ref)


		p1, = ax1.plot(x,yy,c=set1dark[i], dashes=dashedlines[i],linewidth=0.25,alpha=1)
		ax1.fill_between(x, yl, yu, color=facecolor[i],linewidth=0.25,edgecolor=set1dark[i],hatch = hatch[i],alpha=alpha[i])
		p2, = [plt.Rectangle((0,0),0,0,fc = facecolor[i],hatch=hatch[i],linewidth=0,edgecolor=set1dark[i],alpha=alphaleg[i] )]

		lines += ((p1,p2),)

		#y_lims = ax1.get_ylim()
		plt.ylim(0.5,1.5)
		plt.ylim(0.85,1.3)

		#legend
		leg = ax1.legend(lines, pdfnames ,loc='best')
		mylegstyle(leg)

		ax1.axhline(y=1,color='#191919',dashes=[1,1],linewidth=0.5,zorder=1)
		#save and produce figure
	pdf_pages.savefig(fig,bbox_inches='tight')

	pdf_pages.close()

def logxplot(output,pdfnames,xlabel,ylabel,title,x,y,delta,npdfs):

	#initialization
	initializetex()

	#colours and styles
	almost_black = '#262626'
	light_grey = np.array([float(248)/float(255)]*3)
	dashedlines = [[2,2],[4,2,4,2],[100000,1],[4,1,1,1],[4, 2, 1, 2, 1, 2]]

	set1 = palettable.tableau.TableauMedium_10.mpl_colors
	set1dark = palettable.tableau.Tableau_10.mpl_colors
	hatch = ('//////','\\\\\\\\\\\\',None,None)
	facecolor = ('none','none',set1[2],set1[3])
	alpha = (0.5,0.5,0.7,0.2)
	alphaleg = (None,None,0.7,0.2)

	#write output
	pdf_pages = PdfPages(output)

	fig_dims = []
	fig_dims = figsize(fig_dims)
	fig = plt.figure(figsize=fig_dims)
	lines = []

	for i in range(0,npdfs):

		plt.ylabel(ylabel)
		plt.xlabel(xlabel)
		plt.title(title)

		ax1 = fig.add_subplot(111)
		mystyle(ax1)

		ax1.set_xscale('log')
		#mygrid(ax1)

		#adjust x range
		plt.xlim([x[0],x[-1]])

		yy = y[i]
		ydown = (y[i]-delta[i])
		yup = (y[i]+delta[i])
		yl = np.array(ydown)
		yu = np.array(yup)

		#plotting
		#p1, = ax1.plot(x,yy,c=set1dark[i],linewidth=0.75,alpha=0.4)
		#p2, = ax1.plot(x,y,c=set1dark[i],linewidth=0.75,alpha=0.4)
		#ax1.fill_between(x, yl, yu, color=set1[i],linewidth=0.25,edgecolor=set1dark[i],alpha=0.3)
		#p2, = [plt.Rectangle((0,0),0,0,fc = set1[i],alpha=0.3,linewidth=0,edgecolor=set1dark[i] )]

		p1, = ax1.plot(x,yy,c=set1dark[i], dashes=dashedlines[i],linewidth=0.25,alpha=1)
		ax1.fill_between(x, yl, yu, color=facecolor[i],linewidth=0.25,edgecolor=set1dark[i],hatch = hatch[i],alpha=alpha[i])
		p2, = [plt.Rectangle((0,0),0,0,fc = facecolor[i],hatch=hatch[i],linewidth=0,edgecolor=set1dark[i],alpha=alphaleg[i] )]

		lines += ((p2,p1),)

		#y_lims = ax1.get_ylim()
		#plt.ylim(-1,6)

		#legend
		leg = ax1.legend(lines, pdfnames ,loc='best')
		mylegstyle(leg)

		#ax1.axhline(y=1,color='#191919',dashes=[1,1],linewidth=0.5,zorder=1)
		#save and produce figure
	pdf_pages.savefig(fig,bbox_inches='tight')

	pdf_pages.close()
