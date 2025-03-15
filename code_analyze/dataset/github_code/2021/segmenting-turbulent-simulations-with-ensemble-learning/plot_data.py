## SOM code for reproducing Bussov & Nattila 2021 image segmentation results

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
#import tensorflow as tf
from matplotlib import colors
import popsom
import pandas as pd
import colorsys

import h5py as h5
import sys, os

from utils_plot2d import read_var
from utils_plot2d import Conf
from utils_plot2d import imshow

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import metrics

import argparse

parser = argparse.ArgumentParser(description='popsom code')
parser.add_argument('--xdim', type=int, dest='xdim', default=10, help='an integer for the accumulator')
parser.add_argument('--ydim', type=int, dest='ydim', default=10, help='an integer for the accumulator')
parser.add_argument('--alpha', type=float, dest='alpha', default=0.5, help='an float for the accumulator')
parser.add_argument('--train', type=int, dest='train', default=10000, help='an integer for the accumulator')

args = parser.parse_args()

def neighbors(arr,x,y,n=3):
        ''' Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]'''
        arr=np.roll(np.roll(arr,shift=-x+1,axis=0),shift=-y+1,axis=1)
        return arr[:n,:n]

if __name__ == "__main__":

        # set-up plotting
        #plt.fig = plt.figure(1, figsize=(4,3.5), dpi=200)
        #fig = plt.figure(1, figsize=(6,6), dpi=300)

        plt.rc('font',  family='sans-serif')
        #plt.rc('text',  usetex=True)
        plt.rc('xtick', labelsize=5)
        plt.rc('ytick', labelsize=5)
        plt.rc('axes',  labelsize=5)


        scaler = StandardScaler()

        #--------------------------------------------------
        #read data
        #--------------------------------------------------
        #build feature matrix
        feature_list = [
                        'rho',
                        'bx',
                        'by',
                        'bz',
                        'ex',
                        'ey',
                        'ez',
                        'jx',
                        'jy',
                        'jz',
                        ]

        conf = Conf()

        fdir = 'sample-raw-data/' # data directory; assume main dir in this sample script

        print("plotting {}".format(fdir))

        xmin = 0.0
        ymin = 0.0
        xmax = 1.0
        ymax = 1.0

        #--------------------------------------------------
        def read_h5_data_size(fdir, lap):
            conf.fields_file   = fdir + 'raw_data_'+str(lap)+'.h5'
            f5F = h5.File(conf.fields_file,'r')

            #read one and check size
            val0 = read_var(f5F, feature_list[0])
            nx,ny = np.shape(val0)

            return nx,ny


        #--------------------------------------------------
        def read_h5_data(fdir, lap):


            conf.fields_file   = fdir + 'raw_data_'+str(lap)+'.h5'
            f5F = h5.File(conf.fields_file,'r')

            #read one and check size
            val0 = read_var(f5F, feature_list[0])
            nx,ny = np.shape(val0)

            #now read all
            x = np.zeros((nx,ny,len(feature_list)))
            for i,var in enumerate(feature_list):
                val = read_var(f5F, var)
                x[:,:,i] = val

            return x


        laps = [6600]

        #re-organize into subimages

        #subimage size
        splitf = 1

        #total size of the complete image in subimages
        splitfx = splitf*len(laps)
        splitfy = splitf

        nx,ny = read_h5_data_size(fdir, laps[0])
        nxs = nx//splitf
        nys = ny//splitf
        data = np.zeros((nxs,nys,splitfx*splitfy,len(feature_list)))

        for ilap, lap in enumerate(laps):
            x = read_h5_data(fdir, lap)

            fea_means = np.zeros(len(feature_list))
            n0 = x[:,:,0]
            bx = x[:,:,1]
            by = x[:,:,2]
            bz = x[:,:,3]
            ex = x[:,:,4]
            ey = x[:,:,5]
            ez = x[:,:,6]
            jx = x[:,:,7]
            jy = x[:,:,8]
            jz = x[:,:,9]

            n0s = np.std(n0)
            bs = np.std(np.sqrt(bx*bx + by*by + bz*bz))
            es = np.std(np.sqrt(ex*ex + ey*ey + ez*ez))
            js = np.std(np.sqrt(jx*jx + jy*jy + jz*jz))

            x[:,:,0] /= n0s

            x[:,:,1] /= bs
            x[:,:,2] /= bs
            x[:,:,3] /= bs

            x[:,:,4] /= es
            x[:,:,5] /= es
            x[:,:,6] /= es

            x[:,:,7] /= js
            x[:,:,8] /= js
            x[:,:,9] /= js

            #--------------------------------------------------
            print("old array img size is {} x {}".format(nx, ny))
            print("new array subimg size is {} x {} with a split of {}".format(nxs, nys, splitf))
            for i in range(len(feature_list)):
                spf = ilap*splitf**2 #global image counter (=splitf^2)


                for spfy in range(splitf):
                        for spfx in range(splitf):

                                ix0 =  spfx*nxs
                                ix1 = (spfx+1)*nxs

                                iy0 = spfy*nys
                                iy1 = (spfy+1)*nys

                                #step ranges one subimg forward
                                data[:,:,spf,i] = x[iy0:iy1, ix0:ix1, i]
                                spf += 1

        #--------------------------------------------------
        # feature engineering
        # re-defining feature_list here
        if True:
                feature_list = [
                    'bperp_2', #1 Picked as best for sheets
                    'jpar_abs',#7
                    'je_par'
                    ]
                data2 = np.zeros((nxs,nys,splitfx*splitfy,len(feature_list)))

                #B_PERP**2
                data2[:,:,:,0] = data[:,:,:,1]**2 + data[:,:,:,2]**2

                #J_PAR**abs
                data2[:,:,:,1] = np.abs(data[:,:,:,9])

                #JE_PAR
                data2[:,:,:,2] = \
                        data[:,:,:,6]*data[:,:,:,9]


        #Before the data hack:
        data = data2

        #--------------------------------------------------

        #plot original images
        if True:
            fig = plt.figure(1, figsize=(6,6*len(laps)), dpi=300)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
            gs = plt.GridSpec(splitfx, splitfy)
            gs.update(hspace = 0.05)
            gs.update(wspace = 0.05)

            axs = []
            for ix in range(splitfx):
                for iy in range(splitfy):
                    axs.append( plt.subplot(gs[ix,iy]) )


            plot_confs = {
            'rho':         {'cmap':'viridis', 'vmin':0.0,  'vmax':15.0}, #0
            'bperp_2':     {'cmap':'viridis', 'vmin':0.0,  'vmax':30.0}, #1
            'bpar_2':      {'cmap':'viridis', 'vmin':0.0,  'vmax':50.0}, #2
            'eperp_2':     {'cmap':'viridis', 'vmin':0.0,  'vmax':40.0}, #3
            'epar_2':      {'cmap':'viridis', 'vmin':0.0,  'vmax':5.0}, #4
            'jperp_abs':   {'cmap':'viridis', 'vmin':0.0,  'vmax':5.0}, #5
            'jpar_abs':    {'cmap':'viridis', 'vmin':0.0,  'vmax':5.0}, #6
            'je':          {'cmap':'PRGn',    'vmin':-2.0, 'vmax':2.0}, #7
            'exb':         {'cmap':'viridis', 'vmin':0.0, ' vmax':0.8}, #8 
            'j':           {'cmap':'PRGn',    'vmin':-2.0, 'vmax':2.0}, #9
            'je_par':      {'cmap':'PRGn',    'vmin':-2.0, 'vmax':2.0}, #9
            'je_perp':     {'cmap':'PRGn',    'vmin':-2.0, 'vmax':2.0}, #10
            'exb_perp':    {'cmap':'Purples', 'vmin':0.0,  'vmax':69.0}, #11
            'exb_par':     {'cmap':'Purples', 'vmin':0.0,  'vmax':28.0}, #12
            'v_drift':     {'cmap': 'viridis','vmin': 0.0, 'vmax':1.0}, #13
            'v_drift_perp':{'cmap': 'viridis','vmin': 0.0, 'vmax':1.0}, #14
            'v_drift_par': {'cmap': 'viridis','vmin': 0.0, 'vmax':1.0}, #15
            'epar':        {'cmap':'RdBu', '   vmin':-2.0, 'vmax':2.0}, #4
            'jpar':        {'cmap':'RdBu',    'vmin':-2.0, 'vmax':2.0}, #6
            }

        if True:
                print("plotting original images")
                #i = 1 #pick one feature from feature_list array

                # loop over features
                for i in range(len(feature_list)): 
                    for spf in range(splitfx*splitfy):
                        ax = axs[spf]

                        ax.axes.get_xaxis().set_visible(False)
                        ax.axes.get_yaxis().set_visible(False)


                        fea = feature_list[i]
                        ops = plot_confs[fea]
                        print("feature {} min {} max {}".format(feature_list[i], np.min(data[:,:,spf,i]), np.max(data[:,:,spf,i])))

                        imshow(ax, data[:,:,spf,i],xmin, xmax, ymin, ymax,
                                cmap=ops['cmap'], vmin=ops['vmin'], vmax=ops['vmax'],
                                clip=None,
                                )

                    fig.savefig('data_img_{}.pdf'.format(feature_list[i]))

        #--------------------------------------------------

        #128,128,64,10 image here
        # subimg len x subimage width x number of subimages x feature_list
        print("orig shape before:", np.shape(data))

        print("creating feature data")
        #shape it into 128*128*64 (pixel-vector combining and flattening all sub-images) x 9 feature_list

        #if true, create the data; else read it from premade file

        #make features into a one long vector of pixels:
        if True:
                x = np.zeros((nxs*nys*splitfx*splitfy, len(feature_list))) #data matrix
                y = np.zeros((nxs*nys*splitfx*splitfy)) #target matrix

                j = 0
                for ix in range(nxs):
                        for iy in range(nys):
                                for isubim in range(splitfx*splitfy):
                                        for ifea in range(len(feature_list)):
                                                x[j, ifea] = data[ix, iy, isubim, ifea]

                                        j += 1

                #add also similarity measure for all features based on nearest neighbors

                #save feature list to file
                f5 = h5.File('data_features_{}.h5'.format(lap), 'w')
                dsetx = f5.create_dataset("features",  data=x)
                dsety = f5.create_dataset("target",  data=y)

                asciilist = [n.encode("ascii", "ignore") for n in feature_list]
                dsetf = f5.create_dataset("names",  data=asciilist)
                f5.close()

        else:
                f5 = h5.File('data_features_{}.h5'.format(lap), 'r')
                x = f5['features'][()]
                y = f5['target'][()]
                feature_list = f5['names'][()]

                feature_list = [n.decode('utf-8') for n in feature_list]
                f5.close()

        print(feature_list)
        print("shape after x:", np.shape(x))


