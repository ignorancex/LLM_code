
#import
import numpy as np
import scipy as scp
import scipy.linalg as lng
from numba import njit, prange
import pickle
from matplotlib import pyplot as plt
import sys, os
sys.path.append(os.path.abspath('/feynman/home/dedip/lilas/jl269817/fusion'))
from pyStarlet_master_2D1D import *
import warnings

from ipywidgets import interact, fixed
from tqdm.notebook import  tqdm,trange

def FUSION(X_im,Y_im,kernel2D,kernel1D,first_Z=None,
                    ARF_X=None,ARF_Y=None,ARF_Z=None,i_max=100,alpha=1e-2,
                    spat_reg="Wavelet_2D_1D",mu_D=0.01,rank=None,
                    kmad_var=False,kmin=0.5,kmad=3,J_1D=2,J_2D=3,
                    it_save=100,keep_track=True,para_workers=4,vocal=True,
                    fname="fusion_save",
                         ):

    """
    A function to fuse two hyperspectral images, for X-ray astrophysics.
    ----------------------------------------------
    INPUT
    ----------------------------------------------
    ESSENTIALS
    - X_im: Numpy array. Data with the best SPATIAL resolution (e.g. XMM). Size: (l,M,N),
        where l number of spectral channels, MxN number of pixels.
    - Y_im: Numpy array. Data with the best SPECTRAL resolution (e.g. XRISM) Size: (L,m,n),
        where L number of spectral channels, mxn number of pixels.
    - kernel2D: Numpy array. PSF of Y_im's instrument in the pixel domain.
        Shape must be odd.
    - kernel1D: Numpy array. Spectral response of X_im's instrument in the
        channels domain. Shape must be odd.
    - first_Z: None, or numpy array. If not None, gives a chosen first try for Z.

    - i_max: Integer. Maximum number of iterations
    - alpha: Float. Gradient step

    REGULARISATION OPTIONS
    - spat_reg: String. Options for the regularisation term. Options are:
        * "Wavelet_2D_1D": l1 sparsity under the 2D-1D Starlet transform.
        * "Low_rank_Sobolev": Spectral low rank approximation with Sobolev
                              spatial regularisation.
        * "Low_rank_Wavelet_2D": Spectral low rank approximation with l1 sparsity
                                under the 2D Starlet transform.
        * "Low_rank": Spectral low rank approximation.
        * "None": No regularisation at all.
    - mu_D: Float. Regularisation parameter for the Sobolev regularisation.
    - rank: Integer. The enforced rank in case of low rank approximation.
    - kmad_var: Boolean. If Wavelet2D_1D, if True, this reweights the l1 soft threshold.
    - kmin: Float. The factor of threshold for Wavelet sparsity.
        * If kmad_var and Wavelet2D_1D are active, this is the minimum value of k.
        * If kmad_var is false, or Low_rank_Wavelet_2D, this is the constant value of k.
    - kmad: Float. If kmad_var, maximum threshold for Wavelet sparsity.
    - J_1D: If Wavelet2D_1D is active, number of wavelet scales in the spectral dimension.
    - J_2D: If Wavelet2D_1D or Low_rank_Wavelet_2D are active, Number of wavelet
            scales in the pixel dimensions.

    SAVING
    - it_save: Integer. Saving checkpoint. The result will be saved each it_save iterations.
    - keep_track: Boolean. If True, the values saved every it_save will be kept and returned.
    - para_workers: Integer. Maximum number of workers to use for parallel computation
      for the scipy fast fourier transform.
    - vocal: Boolean. If False, no print commands are called in the function.
    - fname: String. The name with which to save the result as a pickle file.


    ----------------------------------------------
    OUTPUT
    ----------------------------------------------
    - Z:
    if keep_track:
        -

    ----------------------------------------------
    """
    ################# SET UP #################

    l,M,N=X_im.shape
    L,m,n=Y_im.shape
    ker_m,ker_n=kernel2D.shape
    ker_l=kernel1D.shape[0]
    assert ker_m%2==1,  "the shape of kernel2D needs to be odd."
    assert ker_n%2==1,  "the shape of kernel2D needs to be odd."
    assert ker_l%2==1,  "the shape of kernel1D needs to be odd."

    if vocal:
        print(f"X has {l} spectral channels, and {M}x{N} pixels.")
        print(f"Y has {L} spectral channels, and {m}x{n} pixels.")
    Mp,Np=M+ker_m-1,N+ker_n-1 #54,54
    m_p,n_p=m+ker_m-1,n+ker_n-1
    Lp=L+ker_l-1
    l_p=l+ker_l-1

    spat_reg_options=np.array(["Wavelet_2D_1D","Low_rank_Sobolev","Low_rank_Wavelet_2D","Low_rank","None"])
    if not any(spat_reg==spat_reg_options):
        raise ValueError(f"Supported spat_reg options are currently: {spat_reg_options}")

    if (spat_reg[:8] =="Low_rank"):
        lowrank=True
    else:
        lowrank=False

    #NORMALIZING THE KERNEL
    kernel2D_Y=(kernel2D)/(kernel2D).sum()
    kernel1D_X=(kernel1D)/(kernel1D).sum()
    #PADDING THE IMAGE
    def padding(Z,pad_l=ker_l-1,pad_m=ker_m-1,pad_n=ker_n-1):
        #print("pad_m",ker_m-1)
        return np.pad(Z,((pad_l//2,pad_l//2),(pad_m//2,pad_m//2),(pad_n//2,pad_n//2)),mode='reflect')

    def unpad(Z,l=ker_l-1,m=ker_m-1,n=ker_n-1):
        return Z[l//2:-l//2,m//2:-m//2,n//2:-n//2]

    #PADDING THE KERNEL
    #Kernel need to the size of the padded images X and Y, but the conjugates need to be the size of padded Z.
    kernel2D_Y_=np.pad(kernel2D_Y,(((m_p-ker_m)//2, (m_p-ker_m)-(m_p-ker_m)//2),
                               ((n_p-ker_n)//2, (n_p-ker_n)-(n_p-ker_n)//2)),mode='constant')
    kernel2D_Z=np.pad(kernel2D_Y,(((Mp-ker_m)//2, (Mp-ker_m)-(Mp-ker_m)//2),
                                   ((Np-ker_n)//2, (Np-ker_n)-(Np-ker_n)//2)),mode='constant')

    kernel1D_X_=np.pad(kernel1D_X,((l_p-ker_l)//2, (l_p-ker_l)-(l_p-ker_l)//2),mode='constant')
    kernel1D_Z=np.pad(kernel1D_X,((Lp-ker_l)//2, (Lp-ker_l)-(Lp-ker_l)//2),mode='constant')

    # SHIFTING THE KERNEL
    kernel2D_Y=scp.fft.ifftshift(kernel2D_Y_)
    kernel1D_X_=scp.fft.ifftshift(kernel1D_X_)
    kernel2D_Z=scp.fft.ifftshift(kernel2D_Z)
    kernel1D_Z=scp.fft.ifftshift(kernel1D_Z)

    # FFT THE KERNEL
    #kerY_2D_FFT=scp.fft.fft2(kernel2D_Y,workers=para_workers).astype(complex)
    kerZ_2D_FFT=scp.fft.fft2(kernel2D_Z,workers=para_workers).astype(complex)
    kerY_2D_FFT=kerZ_2D_FFT
    kerZ_2D_FFT_ast=np.conjugate(kerZ_2D_FFT).astype(complex)
    kerY_2D_FFT_ast=kerZ_2D_FFT_ast
    #kerX_1D_FFT=scp.fft.fft(kernel1D_X_,workers=para_workers).astype(complex)
    kerZ_1D_FFT=scp.fft.fft(kernel1D_Z,workers=para_workers).astype(complex)
    kerX_1D_FFT=kerZ_1D_FFT
    kerZ_1D_FFT_ast=np.conjugate(kerZ_1D_FFT).astype(complex)
    kerX_1D_FFT_ast=kerZ_1D_FFT_ast

    if ARF_X is not None:
        ARF_X=ARF_X[:,np.newaxis,np.newaxis]
    else:
        ARF_X=1#np.ones((L,M,N))

    if ARF_Y is not None:
        ARF_Y=ARF_Y[:,np.newaxis,np.newaxis]
    else:
        ARF_Y=1#np.ones((L,M,N))

    if ARF_Z is not None:
        ARF_Z=ARF_Z[:,np.newaxis,np.newaxis]
    else:
        ARF_Z=1#np.ones((L,M,N))
    #print(ARF_X.shape,ARF_Y.shape,ARF_Z.shape)

    ARF_XZ=ARF_X/ARF_Z
    ARF_YZ=ARF_Y/ARF_Z

    #REBINNING OPERATOR
    def generate_operator(x,y):
        R_xy=np.zeros((x,y))
        p=y//x
        for i in range(x):
            for j in range(y):
                R_xy[i,j]=(j>=i*p)*(j<=(i+1)*p-1)
        return R_xy,p


    # ACCOUNTING FOR WITH OR WITHOUT REBINNING
    if M!=m or N!=n:
        R_Mm,p_m=generate_operator(m,M)
        R_Nn,p_n=generate_operator(n,N)
        def rebin2D(Z_r,R1=R_Mm,R2=R_Nn):
            return np.tensordot(np.tensordot(Z_r,R1,axes=(-1,1)),R2,axes=(1,1))
        def unbin2D(X_r,R1=R_Mm,R2=R_Nn):
            return np.tensordot(np.tensordot(X_r,R1.T,axes=(-1,1)),R2.T,axes=(1,1))
    else:
        def rebin2D(Z_r,R1=None,R2=None):
            return Z_r
        def unbin2D(X_r,R1=None,R2=None):
            return X_r

    if L!=l:
        R_Ll,p_l=generate_operator(l,L)
        def rebin1D(Z_r,R=R_Ll):
            return np.tensordot(R,Z_r,axes=(-1,0))
        def unbin1D(X_r,R=R_Ll):
            return np.tensordot(R.T,X_r,axes=(-1,0))
    else:
        def rebin1D(Z_r,R=None):
            return Z_r
        def unbin1D(X_r,R=None):
            return X_r

    #only to calculate the cost
    Mask_brightness=((unbin2D(Y_im).sum(0)>10)*(unbin1D(X_im).sum(0)>10))[np.newaxis,:,:]
    #GENERAL SHAPE

    def Z_vectorize(Z,l=L,m=M,n=N):
        """
        Takes Z from hyperspectral size (l,m,n) to vectorized size (l,(m*n))
        """
        return (np.reshape(Z,(l,m*n)))

    def Z_pixelize(Z,l=L,m=M,n=N):
        """
        Takes Z from vectorized size (l,(m*n)) to hyperspectral size (l,m,n)
        """
        return np.reshape(Z,(l,m,n))

    if lowrank:
        if vocal:
            print("Using subspace projection.")
        # PCA TO FIND SUBSPACE
        Y_vec=np.reshape(Y_im/ARF_YZ,(L,m*n))
        #print(Y_vec.shape)
        Y_mean=np.mean(Y_vec)
        Y_pca=Y_vec
        Cov_Y=(Y_pca-Y_pca.mean(1)[:,np.newaxis])@(Y_pca-Y_pca.mean(1)[:,np.newaxis]).T
        S_Y,U_Y=np.linalg.eig(Cov_Y)


        if rank is None:
            percent_sing_val=np.zeros((L-1))
            for i in range(L-1):
                percent_sing_val[i]=100*(S_Y[0:i+1].sum()/S_Y.sum())
            rank=np.where(percent_sing_val>90)[0][0]
            if vocal:
                print("Selected rank: ",rank)
        else:
            if vocal:
                print("Pre-defined rank: ",rank)
        V=U_Y[:,:rank]

        #FUNCTIONS TO GO TOWARDS OR FROM SUBSPACE
        def W_to_Z(W,V=V):
            """
            W size: [rank,(m*n)]
            V size: [channels, rank]
            returns Z in hyperspectral shape
            """
            return Z_pixelize(V@W)#+1e-16

        def Z_to_W(Z,V=V):
            """
            Z size: [channels,m,n]
            V size: [channels, rank]
            returns W in vectorized shape.
            """
            Z_vec=Z_vectorize(Z)
            return V.T@Z_vec#+1e-16

    else:
        if vocal:
            print("No subspace projection.")
        V=1
        def W_to_Z(W,V=V):
            return W
        def Z_to_W(Z,V=V):
            return Z

    #Outside the functions:
    #W always remains in vectorized shape (l,m*n)
    #Z always remains in hyperspectral shape (l,m,n)

    if spat_reg=="Low_rank_Sobolev":
        Mat_m=scp.linalg.convolution_matrix((1,-1),(M),mode='full' )
        Mat_n=scp.linalg.convolution_matrix((1,-1),(N),mode='full' )

        def Diff_Operator(Z,l=rank):

            Z_pad=np.pad(Z,((0,0),(0,1),(0,1)),mode='reflect')
            Diff=np.zeros(Z.shape)
            for k in range(l):
                Diff[k,:,:]=Mat_m.T@(Z_pad[k,:,:]@Mat_n)
            return Diff

        def Diff_Operator_T(Z,l=rank):
            Diff=np.zeros(Z.shape)
            for k in range(l):
                Diff[k,:,:]=((Mat_m@Z[k,:,:])@Mat_n.T)[:-1,:-1]
            return Diff

    def convolve_func(Z,K):
        """
        Calculates the convolution of Z and K.
        Input: Z: Hyperspectral Image (pixel domain)
        K: Kernel (frequency domain)
        """
        #print("Z",np.imag(Z).sum())
        Z_pad=padding(Z)
        #print("Z_pad",np.imag(Z_pad).sum())
        if K.ndim==2:
            conv=scp.fft.ifft2(K[np.newaxis,:,:]*scp.fft.fft2(Z_pad,workers=para_workers),workers=para_workers)
        elif K.ndim==1:
            conv=scp.fft.ifft(K[:,np.newaxis,np.newaxis]*scp.fft.fft(Z_pad,axis=0,workers=para_workers),axis=0,workers=para_workers)
        return np.real(unpad(conv))

    def cost_function(W,B_Y=kerY_2D_FFT,S_X=kerX_1D_FFT,X=X_im,Y=Y_im):
        """
        Calculates the cost.
        Input: Z: hyperspectral image (pixel)
        B_Y: 2D kernel for Y (frequency)
        S_X: 1D kernel for X (frequency)
        X,Y: data sets
        returns: The likelihood (sum of cost),
        and the 3D cost map.
        """
        Z=W_to_Z(W)

        BY_Z=rebin2D(convolve_func(ARF_YZ*Z,B_Y))
        BY_Z=BY_Z*(BY_Z>0)+1e-16
        Cost_Y=(BY_Z-Y*np.log(BY_Z))
        SX_Z=rebin1D(convolve_func(ARF_XZ*Z,S_X))
        SX_Z=SX_Z*(SX_Z>0)+1e-16
        Cost_X=(SX_Z-X*np.log(SX_Z))
        if Cost_Y.shape == Cost_X.shape:
            Total_Cost=Cost_Y+Cost_X
            return (Total_Cost*Mask_brightness).sum(),Total_Cost
        else:
            return (Cost_Y.sum()+Cost_X.sum()),[Cost_Y,Cost_X]



    def grad_function(W,B_Y=kerY_2D_FFT,B_Y_ast=kerY_2D_FFT_ast,
                      S_X=kerX_1D_FFT,S_X_ast=kerX_1D_FFT_ast,
                      X=X_im,Y=Y_im):
        """
        Calculates the grad of the cost function.
        Input: Z: hyperspectral image (pixel)
        B_Y: 2D kernel for Y (frequency)
        S_X: 1D kernel for X (frequency)
        B_Y_ast: conjugate of 2D kernel for Y (frequency)
        S_X_ast: conjugate of 1D kernel for X (frequency)
        X,Y: data sets
        returns: gradient of cost function.
        """
        Z=W_to_Z(W)+1e-16
        #negative_voxel=np.unravel_index(np.argmin(Z),shape=Z.shape)
        #plt.plot(Z[:,negative_voxel[1],negative_voxel[2]])
        #plt.title(f"negative Z {negative_voxel}")
        #plt.show()

        min_Z=np.argmin(Z)
        #grad Y
        #Y_inv_BZ=(1-Y/(convolve_func(Z,B_Y)+1e-16))
        BZY=rebin2D(convolve_func(ARF_YZ*Z+1e-16,B_Y))#+1e-14
        #grad X
        ZSX=rebin1D(convolve_func(ARF_XZ*Z+1e-16,S_X))#+1e-14

        grad_Y=Z_to_W(ARF_YZ*convolve_func(unbin2D(1-Y/(BZY)),B_Y_ast))
        grad_X=Z_to_W(ARF_XZ*convolve_func(unbin1D(1-X/(ZSX)),S_X_ast))
        if spat_reg=="Low_rank_Sobolev":
            Sobolev=Z_vectorize(2*mu_D*Diff_Operator_T(Diff_Operator(Z_pixelize(W,l=rank,m=M,n=N))),l=rank,m=M,n=N)
        else:
            Sobolev=0
        #grad_Y[np.where(np.abs(grad_Y)>1e16)]=1e-16
        #grad_X[np.where(np.abs(grad_X)>1e16)]=1e-16
        grad_full=grad_Y+grad_X+Sobolev
        grad_full[np.where(np.isnan(grad_full))]=1e-16
        return grad_full

    def mad(z):
        """
        Calculates the median absolute deviation.
        """
        return np.median(np.abs(z - np.median(z)))/0.6735
    def prox_l1(x,thresh):
        #print("Under thresh: "+f"{(np.abs(x) <= thresh).sum()}")
        return (x - thresh*np.sign(x))*(np.abs(x) > thresh)


    def reg_2D_lowrank(lowrank_cube,J_2D,kmad=kmin,return_wavelets=False):
        """
        input: lowrank_cube, taille [m*n, rank]

        """
        W_im=np.reshape(np.real(lowrank_cube),(rank,M,N)) #pixelized cube
        cc2D,w2D=np.zeros((rank,M,N)),np.zeros((rank,M,N,J_2D))
        for r in range(rank):
            cc2D[r,:,:],w2D[r,:,:,:]=Starlet_Forward2D(W_im[r,:,:],J=J_2D)
        threshold=kmad*mad(w2D[:,:,:,0])
        for j in range(J_2D):
            w2D[:,:,:,j]=prox_l1(w2D[:,:,:,j],threshold)
        #Reconstructing image
        output_cube=cc2D+w2D.sum(-1)
        output_cube_vec=np.reshape(output_cube,(rank,M*N))
        if return_wavelets:
            return output_cube_vec,{"coarse":cc2D,"ww":w2D,"thresh":threshold}
        else:
            return output_cube_vec

    def reg_2D1D(input_cube,J_1D,J_2D,kmin=kmin,return_wavelets=False):
        def sigmoid(x,x_mean,kmin=0.5):
            z=np.abs(x)-x_mean
            return (3-kmin)/(1+np.exp(z))+kmin
        cc_2D1D,cw_2D1D,wc_2D1D,ww_2D1D=Starlet_Forward2D_1D(np.real(input_cube),J_1D=J_1D,J_2D=J_2D)

        if kmad_var:
            kmad=sigmoid(cc_2D1D,x_mean=np.mean(np.abs(cc_2D1D)),kmin=kmin)
        else:
            kmad=kmin

        threshold=kmad*mad(ww_2D1D[:,:,:,0,0])
        for i in range(J_1D):
            for j in range(J_2D):
                ww_2D1D[:,:,:,j,i]=prox_l1(ww_2D1D[:,:,:,j,i],threshold)
        #Reconstructing image
        output_cube=cc_2D1D+cw_2D1D.sum(axis=3)+wc_2D1D.sum(axis=3)+(ww_2D1D.sum(axis=3)).sum(axis=3)
        if return_wavelets:
            return output_cube,{"coarse":cc_2D1D,"cw":cw_2D1D,"wc":wc_2D1D,"ww":ww_2D1D,"thresh":threshold}
        else:
            return output_cube


    def Response_Operator(W,B_Y=kerZ_2D_FFT,S_X=kerZ_1D_FFT):
        Z_X=rebin1D(convolve_func(ARF_XZ*W_to_Z(W),S_X))
        Z_Y=rebin2D(convolve_func(ARF_YZ*W_to_Z(W),B_Y))
        return Z_X,Z_Y

    def Response_Operator_T(Z_X,Z_Y,B_Y_ast=kerZ_2D_FFT_ast,S_X_ast=kerX_1D_FFT_ast):
        W=Z_to_W(ARF_XZ*convolve_func(unbin1D(Z_X),S_X_ast)
        +ARF_YZ*convolve_func(unbin2D(Z_Y),B_Y_ast))
        return W
#    def Calculate_gradstep(W,X=X_im,Y=Y_im):
#        """
#        Calculate a bound on the Lipschitz constant, ignoring very low values,
#        to obtain a gradient step.
#        """
#        Z_X,Z_Y=Response_Operator(W)
#        def borne_with_Mask(Z_X,Z_Y):
#            return 2*np.max([1/np.min(Z_X[np.where(X_im>2)]),1/np.min(Z_Y[np.where(Y_im>2)])])
#        return 1/borne_with_Mask(Z_X,Z_Y)


    if first_Z is not None:
        Z=first_Z
        W=Z_to_W(Z)
    else:
        if X_im.shape==Y_im.shape:
            if vocal:
                print("no rebinning")
            Z=ARF_Z*(X_im)/ARF_X

        else:
            ratio=(X_im.shape[0])/(Y_im.shape[0])
            if vocal:
                print("with rebinning")
            Z=ratio*unbin1D(X_im)/ARF_XZ
        W=Z_to_W(Z)

    Z_track=[Z]
    mask=Z>0
    Z=Z*mask+1e-16
    #if alpha_auto:
#        alpha=Calculate_gradstep(W)
#        if vocal:
#            print(f"Gradient step= {alpha:2e}")
    cost,Cost_mat=cost_function(W)
    Likelihood=[cost]
    Likelihood_diff=[0]
    Vt=0
    Wavelets=None
    momentum=0
    t=trange(i_max, desc='Loss', leave=True)
    for i in t:
        #GRADIENT DESCENT
        grad=grad_function(W)
        W_new=W-alpha*grad

        #Wavelet Sparsity
        if spat_reg=="Wavelet_2D_1D":
            W_new,Wavelets=reg_2D1D(W_new,J_2D,J_1D,return_wavelets=True)
        elif spat_reg=="Low_rank_Wavelet_2D":
            W_new,Wavelets=reg_2D_lowrank(W_new,J_2D,return_wavelets=True)
        Z_new=W_to_Z(W_new)
        #Non-negativity constraint
        mask=Z_new>0
        Z=Z_new*mask+1e-16
        W=Z_to_W(Z)
        if i%it_save==0:
            Z_track.append(np.real(Z))

        #LIKELIHOOD
        cost,Cost_mat=cost_function(W)
        Likelihood.append(np.real(cost))
        diff=(Likelihood[-1]-Likelihood[-2])/Likelihood[-2]
        Likelihood_diff.append(np.real(diff))

        t.set_description(f"Loss= {cost:.4e}, Difference: {diff:.2e}")
        t.refresh()
        if i%500 ==0:
            if fname is not None:
                with open(fname+'.p',"wb") as f:
                    pickle.dump((np.real(Z),Z_track,Likelihood,Likelihood_diff,Wavelets,alpha),f)

    if fname is not None:
        with open(fname+'.p',"wb") as f:
            pickle.dump((Z,Z_track,Likelihood,Likelihood_diff,Wavelets,alpha),f)
    if vocal:
        plt.plot(Likelihood)
        plt.title("Likelihood")
        plt.show()
        plt.plot(Likelihood_diff)
        plt.title("Likelihood difference")
        plt.show()
    if keep_track:
        return Z,Z_track,Likelihood,Likelihood_diff,Wavelets,alpha
    else:
        return Z,Likelihood
