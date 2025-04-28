#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:31:28 2023

@author: adaskin
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
nq = 4

def savefig_tofile(plt, fname=None):
    plt.savefig(fname+'.eps', dpi='figure', bbox_inches='tight', pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None )
    
    plt.savefig(fname+'.pdf', dpi='figure', bbox_inches='tight', pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None )
    
    plt.savefig(fname+'.png', dpi='figure', bbox_inches='tight', pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None )
    
    #plt.close()
    
    
def get_bin_matrix_onequbit(qubit, m = 16, n = 16): 
    a = np.zeros((m*n,1))
    'scan all states'
    for i in range(0, m*n):   
        'ith state in the vector'
        #for q in range(0,nq): #reading orders n-1, n-2, ... 1,0   
        bit = (i >> qubit) & 1
        if bit == 1:
            a[i] = 1
    
    A = np.reshape(a,(m,n))#.swapaxes(0,1)

    return A       


def get_bin_matrix_twoqubits(qubits, m = 16, n = 16): 
    a = np.zeros((m*n,1))
    nq = int(np.log2(m*n));
    states = np.linspace(0,2**len(qubits)-1,2**len(qubits))
    states= states/2**len(qubits)
    L = len(qubits);
    'scan all states'
    for i in range(0, m*n):   
        'ith state in the vector'
        #for q in range(0,nq): #reading orders n-1, n-2, ... 1,0   
        bits = bin(i)[2:].zfill(nq)
        indx = 0;
        for q in range(L):
            
            if bits[qubits[q]] == '1':
                indx = 2**(L-q-1)+indx
            print(L, bits, q,(L-q-1), indx, qubits[q],nq)
        a[i]=states[indx]        
        
    A = np.reshape(a,(m,n))#.swapaxes(0,1)

    return A   


     
def plot_onequbits(m=4, n=4, fname = None):

    nq = int(np.log2(m*n))
    sqrtnq = int(np.sqrt(nq))
    print("============",sqrtnq, nq)
    fig, ax = plt.subplots(ncols=sqrtnq, nrows = sqrtnq, layout='compressed')
    imcount = 1
    for q in range(0, nq):
        A = get_bin_matrix_onequbit(q, m, n)
        plt.subplot(sqrtnq,sqrtnq, imcount)
        imcount += 1
        plt.imshow(A, origin = 'lower', alpha=0.8,cmap='Blues',extent = [0, m, n, 0],aspect=1)
        plt.xticks(np.arange(n),labels='')
        plt.yticks(np.arange(m),labels='')
       # plt.xlabel('Column')
        #plt.ylabel('Row')
        plt.title('qubits-%d'%(q), fontsize='9')
        # Define grid with axis='y'
        plt.grid()
  #  plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    
    cax = plt.axes((0.81, 0.1, 0.03, 0.8))
    cbar = plt.colorbar(cax=cax, orientation='vertical')
    cbar.set_ticks([0,1])
    cbar.ax.set_yticklabels(['0', '1'])  # horizontal colorbar
    #plt.suptitle('Color Maps of the Qubit States')
    #fig.tight_layout()
    plt.show()
    if fname is not None:
        savefig_tofile(fig, fname)
    
    
def plot_twoqubits(m=4, n=4, fname = None, nr=5, nc=3):

    nq = int(np.log2(m*n))
    #Colour Map using Matrix
    pltcount = int(np.sqrt(nq*(nq-1)/2))
    
    fig, ax = plt.subplots(nrows=nr, ncols=nc, layout='compressed')

    imcount = 1
    for i in range(nq):
        for j in range(i+1,nq):
            q = (i,j)
            A = get_bin_matrix_twoqubits(((q)), m, n)
            plt.subplot(nr, nc,imcount)
            imcount += 1
            plt.imshow(A, origin = 'lower', alpha=0.8,cmap='Blues',extent = [0, m, n, 0],aspect=1)
            plt.xticks(np.arange(n),labels='')
            plt.yticks(np.arange(m),labels='')
           # plt.xlabel('Column')
            #plt.ylabel('Row')
            plt.title('qubits-(%d,%d)'%(q), fontsize='7')
            # Define grid with axis='y'
            plt.grid()
    #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    
    cax = plt.axes((0.7, 0.1, 0.03, 0.8))
    cbar = plt.colorbar(cax=cax,orientation='vertical',pad=0.2)
    cbar.set_ticks([0,1/4,2/4,3/4])
    cbar.ax.set_yticklabels(['00', '01', '10', '11'])  # horizontal colorbar
    #plt.suptitle('Color Maps of the Qubit States')
    #fig.tight_layout()
    plt.show()
    if fname is not None:
        savefig_tofile(fig, fname)
    
if __name__ == "__main__" :
    m = 4
    n = 4
    plot_onequbits(m,n, fname='onequbitcmap')
    #plot_twoqubits(m,n, fname='twoqubitcmap4x4', nr=3, nc=2)
    #plot_twoqubits(8,8, fname='twoqubitcmap8x8', nr=5, nc=3)

    
    