from __future__ import division

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Polygon
from matplotlib.patches import Path
from matplotlib.collections import PatchCollection
import scipy.linalg as sl
import urllib
from io import StringIO
from matplotlib.collections import PolyCollection
import matplotlib.image as mpimg
import shapefile
from PIL import Image
from convert_east_north import convert_list



def Plot_Filled_Map(Values, shpfile, mappingIndex,title='',cmap=cm.seismic,saveDirectory='.', Legend=True,boundaries='UK',features=None, boundary=None, bcol='gray',midpoint=None,blueOnly=False):
    """Plot map of Output Areas coloured according to set of values

           Args: Values - dict - entries of form {OA: value to plot} for each OA.
                 shpfile - shapefile.reader - directory containging the vertices of OAs.
                 mappingIndex - dict - entries of form {OA:index} for each OA
                 title - str - title
                 cmap - colourmap
                 saveDirectory - str - directory for output
                 Legend - True/False - toggle legend
                
        """ 
    # need to change Values type (dict to list)
    list_values = list(Values.values())
    maxVal=np.real(np.max(list_values))
    minVal=np.real(np.min(list_values))
    mid=(maxVal+minVal) * 0.5
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=minVal, vmax=maxVal))
    if midpoint is not None:
        sm = cm.ScalarMappable(cmap=cmap, norm=MidpointNormalize(vmin=minVal, vmax=maxVal,midpoint=midpoint))

    sm._A=[]
    plt.figure("".join(list('Filled_Map'+title)))
    plt.close("".join(list('Filled_Map'+title)))
    fig=plt.figure("".join(list('Filled_Map'+title)),figsize=(8,8))
    
    ax=fig.add_subplot(1,1,1)
    patches=[]
    i=0

    for OA, eValue in Values.items():
        # Use items instead of iteritems
        i+=1
        if i%100 == 0:
            print(i)
            
        patch=(convert_list(shpfile.shape(mappingIndex[OA]).points))
        
        if blueOnly:
            if cmap==cm.seismic:
                if eValue>mid:
                    eValue=mid
            else:
                if eValue<mid:
                    eValue=mid
        c=np.asarray(sm.to_rgba(np.real(eValue)))
        
        c[3]=0.8
        patches.append(Polygon(patch,color=c))


    p=PatchCollection(patches,match_original=True)
    
    ax.add_collection(p)
    if boundary is not None:
        borderlist=np.genfromtxt(boundary)
        border=Polygon(borderlist, linewidth=3.0, facecolor='none', edgecolor=bcol,zorder=2000)

        ax.add_patch(border)


    if boundaries=='UK':
        plt.ylim([15933,660276])
        plt.xlim([117436,674547])
    elif boundaries =='London':
        im= Image.open('backgrounds/London.png')
        ax.imshow(np.asarray(im),origin='upper',aspect=2,extent=[518522,544290,171269,189684],cmap='gray',alpha=0.5,zorder=1000)
        plt.ylim([171269,189684])
        plt.xlim([518522,544290])

    elif boundaries =='Edinburgh':
        im= Image.open('backgrounds/edinburgh-highres.png')
        ax.imshow(np.asarray(im),origin='upper',aspect=2,extent=[-3.39279659953675,-3.00468008876938,55.8594141543381,56.0054426791175],cmap='gray',alpha=0.5,zorder=1000)
        plt.ylim([55.8594141543381,56.0054426791175])
        plt.xlim([-3.39279659953675,-3.00468008876938])


    elif boundaries=='Bristol':
        im= Image.open('backgrounds/bristol-a-transparent.png')
        ax.imshow(np.asarray(im),origin='upper',aspect=2,extent=[-2.70902954207789,-2.45497570029192,51.3974337063777,51.5183538627159],cmap='gray',alpha=0.5,zorder=1000)
        plt.ylim([51.3974337063777,51.5183538627159])
        plt.xlim([-2.70902954207789,-2.45497570029192])
        
    elif boundaries == 'Bristol+Bath':
        im= Image.open('backgrounds/region-a-transparent.png')
        ax.imshow(np.asarray(im),origin='upper',aspect=2,extent=[-2.69163828244996,-2.29907681072277,51.3502984105867,51.5604657566314],cmap='gray',alpha=0.5,zorder=1000)
        plt.ylim([51.3502984105867,51.5604657566314])
        plt.xlim([-2.69163828244996,-2.29907681072277])

    elif boundaries== 'Bedminster':
        im= Image.open('backgrounds/bedminster-zoom-transparent.png')
        ax.imshow(np.asarray(im),origin='upper',aspect=2,extent=[-2.62027258024025,-2.58426925060412,51.4369397655437,51.4501463606738],cmap='gray',alpha=0.5,zorder=1000)
        im=Image.open('backgrounds/CouncilHouses.png')
        ax.imshow(np.asarray(im),origin='upper',aspect=2,extent=[-2.62027258024025,-2.58426925060412,51.4369397655437,51.4501463606738],cmap='gray',alpha=1.0,zorder=1001)
        plt.ylim([51.4369397655437,51.4501463606738])
        plt.xlim([-2.62027258024025,-2.58426925060412])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
    plt.tick_params(
    axis='y',          # changes apply to the y-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft=False)
    if features is not None:
        for feature in features:
            plt.plot(features[feature][1],features[feature][0],'o',markerfacecolor='Y',markeredgecolor='k',markersize=5, zorder=1001,markeredgewidth=2)

    tmpsaveDirectory="".join(list(saveDirectory+'/Filled-'+title))
    if Legend == True:
        cbar=plt.colorbar(sm,fraction=0.01,aspect=100, pad=0.04,orientation='horizontal',ticks=[np.min(Values.values()),0,np.max(Values.values())])
        
    plt.savefig("".join(list(tmpsaveDirectory)),bbox_inches='tight',dpi=300)
    plt.close("".join(list('Filled_Map'+title)))
    return()


