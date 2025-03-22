## SOM code for reproducing Bussov & Nattila 2021 image segmentation results

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
#import tensorflow as tf
from matplotlib import colors
import popsom.popsom as popsom
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
parser.add_argument('--xdim', type=int, dest='xdim', default=10, help='Map x size')
parser.add_argument('--ydim', type=int, dest='ydim', default=10, help='Map y size')
parser.add_argument('--alpha', type=float, dest='alpha', default=0.5, help='Learning parameter')
parser.add_argument('--train', type=int, dest='train', default=10000, help='Number of training steps')

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
        conf = Conf()

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

        #--------------------------------------------------
        def read_h5_data_size(fdir, lap):
            conf.fields_file   = fdir + 'raw_data_'+str(lap)+'.h5'
            f5F = h5.File(conf.fields_file,'r')
        
            #read one and check size
            val0 = read_var(f5F, feature_list[0])
            nx,ny = np.shape(val0)
        
            return nx,ny

        xmin = 0.0
        ymin = 0.0
        xmax = 1.0
        ymax = 1.0

        fdir = 'sample-raw-data/' # data directory; assume main dir in this sample script
        laps = [6600] # all the data laps to process
        lap = laps[0] # data file number

        #subimage size
        splitf = 1

        #total size of the complete image in subimages
        splitfx = splitf*len(laps)
        splitfy = splitf

        nx,ny = read_h5_data_size(fdir, laps[0])
        nxs = nx//splitf
        nys = ny//splitf


        if True:
                f5 = h5.File('data_features_{}.h5'.format(lap), 'r')
                x = f5['features'][()]
                y = f5['target'][()]
                feature_list = f5['names'][()]

                feature_list = [n.decode('utf-8') for n in feature_list]
                f5.close()

        print(feature_list)
        print("shape after x:", np.shape(x))

        #--------------------------------------------------
        # analyze
        #1. standardize:
        scaler = StandardScaler()
        scaler = MinMaxScaler()

        scaler.fit(x)
        x = scaler.transform(x)


        ##### Using the SOM:

        #       POPSOM SOM:
        attr=pd.DataFrame(x)
        attr.columns=feature_list
        #parser.parse_args
        print("setting dimensions", parser.parse_args())

        print('constructing SOM...')
        m=popsom.map(args.xdim, args.ydim, args.alpha, args.train)

        labels = [str(xxx) for xxx in range(len(x))]
        m.fit(attr,labels)
        m.starburst()

        m.significance()

        #Data matrix with neuron positions:
        data_matrix=m.projection()
        data_Xneuron=data_matrix['x']
        data_Yneuron=data_matrix['y']

        print("Printing Xneuron info")
        print(data_Xneuron)
        print("Printing Xneuron info position 5")
        print(data_Xneuron[4])
        print("Printing Yneuron info")
        print(data_Yneuron)

        #Neuron matrix with centroids:
        umat = m.compute_umat(smoothing=2)
        centrs = m.compute_combined_clusters(umat, False, 0.15) #0.15
        centr_x = centrs['centroid_x']
        centr_y = centrs['centroid_y']

        #create list of centroid _locations
        nx, ny = np.shape(centr_x)

        centr_locs = []
        for i in range(nx):
                for j in range(ny):
                        cx = centr_x[i,j]
                        cy = centr_y[i,j]

                        centr_locs.append((cx,cy))

        unique_ids = list(set(centr_locs))
        print(unique_ids)
        n_clusters = len(unique_ids)
        print("Number of clusters")
        print(n_clusters)

        mapping = {}
        for I, key in enumerate(unique_ids):
                print(key, I)
                mapping[key] = I

        clusters = np.zeros((nx,ny))
        for i in range(nx):
                for j in range(ny):
                        key = (centr_x[i,j], centr_y[i,j])
                        I = mapping[key]

                        clusters[i,j] = I

        print(centr_x)
        print(centr_y)

        print("clusters")
        print(clusters)
        print(np.shape(clusters))

        #TRANSFER RESULT BACK INTO ORIGINAL DATA PLOT
        if True:
                print("plotting SOM cluster images")
                fig = plt.figure(1, figsize=(6,6), dpi=300) #need to hack the number of columns according to image count
                fig.clf()

                fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

                gs = plt.GridSpec(splitfx, splitfy)
                gs.update(hspace = 0.05)
                gs.update(wspace = 0.05)

                #Create N number of colors:
                def get_N_HexCol(N=n_clusters):
                        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
                        hex_out = []
                        for rgb in HSV_tuples:
                                rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
                                hex_out.append('#%02x%02x%02x' % tuple(rgb))
                        return hex_out
                cols=get_N_HexCol(N=n_clusters)
                print("The count of colors, should be same as number of clusters")
                print(len(cols))
                print("Colors", cols)

                cmap = colors.ListedColormap(cols) #should here be len(cols)?
                bounds=np.arange(cmap.N)
                norm = colors.BoundaryNorm(bounds, cmap.N)
                data_back = np.zeros((nxs,nys,splitfx*splitfy))

                xinds = np.zeros(len(data_Xneuron))
                print("shape of xinds:", np.shape(xinds))
                j = 0
                for ix in range(nxs):
                        for iy in range(nys):
                                for isubim in range(splitfx*splitfy):
                                        data_back[ix,iy,isubim] = clusters[data_Xneuron[j], data_Yneuron[j]]
                                        xinds[j] = clusters[data_Xneuron[j], data_Yneuron[j]]
                                        j += 1

                f5 = h5.File('data_som_clusters_{}.h5'.format(lap), 'w')
                dsetx = f5.create_dataset("databack",  data=data_back)
                f5.close()
                print("Done writing the cluster ID file")

                for spf in range(splitfx*splitfy):
                        ax = fig.add_subplot(splitfx,splitfy, spf+1, xticks=[], yticks=[])

                        ax.axes.get_xaxis().set_visible(False)
                        ax.axes.get_yaxis().set_visible(False)

                        ax.clear()
                        ax.minorticks_on()
                        ax.set_xlim(xmin, xmax)
                        ax.set_ylim(ymin, ymax)
                        extent = [ xmin, xmax, ymin, ymax ]
                        mgrid = data_back[:,:,spf].T

                        im = ax.imshow(mgrid, extent=extent)
                        fig.savefig('som_data_transform.png')

        print("done plotting")


        # #PLOTTING:
        #visualize clusters

        x = np.array(x)
        data_back = np.array(data_back)
        if True:
                print("visualizing SOM data")
                #fig2 = plt.figure(2, figsize=(splitfx,splitfy), dpi=400)
                fig2 = plt.figure(2, figsize=(8,8), dpi=400)
                fig2.clf()

                ic = 0
                nfea = len(feature_list)
                for jcomp in range(nfea):
                        for icomp in range(nfea):
                                ic += 1
                                print("i {} j {}".format(icomp, jcomp))

                                #skip covariance with itself
                                if icomp == jcomp:
                                        continue

                                #skip upper triangle
                                if icomp > jcomp:
                                        continue

                                ax = fig2.add_subplot(nfea, nfea, ic)

                                for ki in range(n_clusters):
                                        indxs = np.where(xinds == ki)
                                        #print("len of found pixels:", len(indxs), indxs)
                                        xx = x[indxs, icomp]
                                        yy = x[indxs, jcomp]

                                        xxt = xx[::500]
                                        yyt = yy[::500]

                                        ax.scatter(
                                                xxt,
                                                yyt,
                                                c=cols[ki],
                                                marker='.',
                                                s=0.1,
                                                rasterized=True,
                                                )

                                if False:
                                        #visualize most dissipative points
                                        xx = x[:,icomp]
                                        yy = x[:,jcomp]
                                        zz = y[:]

                                        xxd = xx[np.where(np.abs(zz) > 0.020)]
                                        yyd = yy[np.where(np.abs(zz) > 0.020)]
                                        zzd = zz[np.where(np.abs(zz) > 0.020)]

                                        xxd = xxd[::100]
                                        yyd = yyd[::100]
                                        zzd = zzd[::100]

                                        print("found {} points above threshold".format(len(xxd)))

                                        ax.scatter(xxd,yyd,c=zzd,
                                                                cmap='inferno',
                                                                vmin=-0.015,
                                                                vmax= 0.015,
                                                                marker='.',
                                                                s=0.05,
                                                                alpha=0.1,
                                                                )

                                if jcomp == nfea-1:
                                        ax.set_xlabel('{}'.format(feature_list[icomp]))
                                else:
                                        ax.set_xticks([])

                                if icomp == 0:
                                        ax.set_ylabel('{}'.format(feature_list[jcomp]))
                                else:
                                        ax.set_yticks([])

                fig2.savefig('som_clusters.pdf')
