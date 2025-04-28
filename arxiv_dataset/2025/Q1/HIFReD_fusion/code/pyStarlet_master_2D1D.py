import numpy as np
from copy import deepcopy as dp
import scipy.linalg as lng
import copy as cp


############################################################
################# STARLET TRANSFORM
############################################################

import numpy as np
import warnings


try:
    from numba import njit, prange
    print("Numba imported")
except ModuleNotFoundError:
    warnings.warn("Cannot use Numba. Switch to low performance mode.")

    # Make a decorator that does nothing
    def njit(f):
        return f

    def prange():
        return range


class pystarlet:

    # Choose multi-proc or mono at init

    def __init__(self, parallel=False):
        self.parallel = parallel

    def forward(self,array, J=4):
        """
        Wrapper to compute the Starlet transform of a cube of shape (Nz,Nx,Ny)
        Returns an hypercube of shape (Nz,Nx,Ny,nJ)
        """
        if self.parallel is False:
            hypercube=Starlet_Forward3D(array,J=J)

        else :
            hypercube=Starlet_Forward3D_mp(array,J=J)

        return hypercube


class StarletError(Exception):
    """Common `starlet` module's error."""
    pass


class WrongDimensionError(StarletError):
    """Raised when data having a wrong number of dimensions is given.

    Attributes
    ----------
    msg : str
        Explanation of the error.
    """

    def __init__(self, msg=None):
        if msg is None:
            self.msg = "The data has a wrong number of dimension."



##############################################################################

@njit(parallel=False, fastmath = True)
def get_pixel_value(image, x, y, type_border):

    if type_border == 0:

        #try:
        pixel_value = image[x, y]
        return pixel_value
        #except IndexError as e:
        #    return 0

    elif type_border == 1:

        num_lines, num_col = image.shape    # TODO
        x = x % num_lines
        y = y % num_col
        pixel_value = image[x, y]
        return pixel_value

    elif type_border == 2:

        num_lines, num_col = image.shape    # TODO

        if x >= num_lines:
            x = num_lines - 2 - x
        elif x < 0:
            x = abs(x)

        if y >= num_col:
            y = num_col - 2 - y
        elif y < 0:
            y = abs(y)

        pixel_value = image[x, y]
        return pixel_value

    elif type_border == 3:

        num_lines, num_col = image.shape    # TODO

        if x >= num_lines:
            x = num_lines - 1 - x
        elif x < 0:
            x = abs(x) - 1

        if y >= num_col:
            y = num_col - 1 - y
        elif y < 0:
            y = abs(y) - 1

        pixel_value = image[x, y]
        return pixel_value

    else:
        raise ValueError()


@njit(parallel=False, fastmath = True)
def smooth_bspline(input_image, type_border, step_trou):
    """Apply a convolution kernel on the image using the "à trou" algorithm.

    Pseudo code:

    **convolve(scale, $s_i$):**

    $c_0 \leftarrow 3/8$

    $c_1 \leftarrow 1/4$

    $c_2 \leftarrow 1/16$

    $s \leftarrow \lfloor 2^{s_i} + 0.5 \rfloor$

    **for** all columns $x_i$

    $\quad$ **for** all rows $y_i$

    $\quad\quad$ scale[$x_i$, $y_i$] $\leftarrow$ $c_0$ . scale[$x_i$, $y_i$] + $c_1$ . scale[$x_i-s$, $y_i$] + $c_1$ . scale[$x_i+s$, $y_i$] + $c_2$ . scale[$x_i-2s$, $y_i$] + $c_2$ . scale[$x_i+2s$, $y_i$]

    **for** all columns $x_i$

    $\quad$ **for** all rows $y_i$

    $\quad\quad$ scale[$x_i$, $y_i$] $\leftarrow$ $c_0$ . scale[$x_i$, $y_i$] + $c_1$ . scale[$x_i$, $y_i-s$] + $c_1$ . scale[$x_i$, $y_i+s$] + $c_2$ . scale[$x_i$, $y_i-2s$] + $c_2$ . scale[$x_i$, $y_i+2s$]

    Inspired by Sparse2D mr_transform (originally implemented in *isap/cxx/sparse2d/src/libsparse2d/IM_Smooth.cc* in the
    *smooth_bspline()* function.

    ```cpp
    void smooth_bspline (const Ifloat & Im_in,
                         Ifloat &Im_out,
                         type_border Type, int Step_trou) {
        int Nl = Im_in.nl();  // num lines in the image
        int Nc = Im_in.nc();  // num columns in the image
        int i,j,Step;
        float Coeff_h0 = 3. / 8.;
        float Coeff_h1 = 1. / 4.;
        float Coeff_h2 = 1. / 16.;
        Ifloat Buff(Nl,Nc,"Buff smooth_bspline");

        Step = (int)(pow((double)2., (double) Step_trou) + 0.5);

        for (i = 0; i < Nl; i ++)
        for (j = 0; j < Nc; j ++)
           Buff(i,j) = Coeff_h0 *    Im_in(i,j)
                     + Coeff_h1 * (  Im_in (i, j-Step, Type)
                                   + Im_in (i, j+Step, Type))
                     + Coeff_h2 * (  Im_in (i, j-2*Step, Type)
                                   + Im_in (i, j+2*Step, Type));

        for (i = 0; i < Nl; i ++)
        for (j = 0; j < Nc; j ++)
           Im_out(i,j) = Coeff_h0 *    Buff(i,j)
                       + Coeff_h1 * (  Buff (i-Step, j, Type)
                                     + Buff (i+Step, j, Type))
                       + Coeff_h2 * (  Buff (i-2*Step, j, Type)
                                     + Buff (i+2*Step, j, Type));
    }
    ```

    Parameters
    ----------
    input_image
    type_border
    step_trou

    Returns
    -------

    """

#    input_image = np.asarray(input_image,dtype='float64')

    coeff_h0 = 3. / 8.
    coeff_h1 = 1. / 4.
    coeff_h2 = 1. / 16.

    num_lines, num_col = input_image.shape    # TODO

#    buff = np.zeros(input_image.shape, dtype='float64')
#    img_out = np.zeros(input_image.shape, dtype='float64')

    buff = np.zeros_like(input_image)
    img_out = np.zeros_like(input_image)

    step = int(pow(2., step_trou) + 0.5)

    for i in range(num_lines):
        for j in range(num_col):
            buff[i,j]  = coeff_h0 *    get_pixel_value(input_image, i, j,        type_border)
            buff[i,j] += coeff_h1 * (  get_pixel_value(input_image, i, j-step,   type_border) \
                                     + get_pixel_value(input_image, i, j+step,   type_border))
            buff[i,j] += coeff_h2 * (  get_pixel_value(input_image, i, j-2*step, type_border) \
                                     + get_pixel_value(input_image, i, j+2*step, type_border))

    for i in range(num_lines):
        for j in range(num_col):
            img_out[i,j]  = coeff_h0 *    get_pixel_value(buff, i,        j, type_border)
            img_out[i,j] += coeff_h1 * (  get_pixel_value(buff, i-step,   j, type_border) \
                                        + get_pixel_value(buff, i+step,   j, type_border))
            img_out[i,j] += coeff_h2 * (  get_pixel_value(buff, i-2*step, j, type_border) \
                                        + get_pixel_value(buff, i+2*step, j, type_border))

    return img_out

@njit(parallel=False, fastmath = True)
def get_pixel_value_1D(image, x):
    num_lines = image.shape[0]    # TODO

    if x >= num_lines:
        x = num_lines - 1 - x
    elif x < 0:
        x = abs(x) - 1

    return image[x]



@njit
def mad(z):
    return np.median(np.abs(z - np.median(z)))/0.6735


@njit(parallel=False,fastmath = True)
def smooth_bspline1D(input_image,step_trou):

#    input_image = np.asarray(input_image,dtype='float64')

    coeff_h0 = 3. / 8.
    coeff_h1 = 1. / 4.
    coeff_h2 = 1. / 16.
    #print(input_image.shape)
    num_lines = input_image.shape[0]

    img_out = np.zeros_like(input_image)

    step = int(pow(2., step_trou) + 0.5)

    for i in range(num_lines):
        img_out[i]  = coeff_h0 *    get_pixel_value_1D(input_image, i)
        img_out[i] += coeff_h1 * (  get_pixel_value_1D(input_image, i-step) \
                                  +get_pixel_value_1D(input_image, i+step))
        img_out[i] += coeff_h2 * (  get_pixel_value_1D(input_image, i-2*step) \
                                     + get_pixel_value_1D(input_image, i+2*step))

    return img_out


@njit(parallel=False,fastmath = True)
def Starlet_Forward2D(input_image,J=4):
    """Compute the starlet transform of `input_image`.

    Pseudo code:

    **wavelet_transform(input_image, num_scales):**

    scales[0] $\leftarrow$ input_image

    **for** $i \in [0, \dots, \text{num_scales} - 2]$

    $\quad$ scales[$i + 1$] $\leftarrow$ convolve(scales[$i$], $i$)

    $\quad$ scales[$i$] $\leftarrow$ scales[$i$] - scales[$i + 1$]


    Inspired by Sparce2D mr_transform (originally implemented in *isap/cxx/sparse2d/src/libsparse2d/MR_Trans.cc*)

    ```cpp
    static void mr_transform (Ifloat &Image,
                              MultiResol &MR_Transf,
                              Bool EdgeLineTransform,
                              type_border Border,
                              Bool Details) {
        // [...]
        MR_Transf.band(0) = Image;
        for (s = 0; s < Nbr_Plan -1; s++) {
           smooth_bspline (MR_Transf.band(s),MR_Transf.band(s+1),Border,s);
           MR_Transf.band(s) -= MR_Transf.band(s+1);
        }
        // [...]
    }
    ```

    Parameters
    ----------
    input_image : array_like
        The input image to transform.
    J : int, optional
        The number of scales used to transform `input_image` or in other words
        the number of wavelet planes returned.
    Returns
    -------
    list
        Return a list containing the wavelet planes.

    Raises
    ------
    WrongDimensionError
        If `input_image` is not a 2D array.
    """

#    input_image = np.asarray(input_image,dtype='float64')
    input_image = input_image.copy()
    if input_image.ndim != 2:
        msg = "The data should be a 2D array."
        raise WrongDimensionError(msg)


    # DO THE WAVELET TRANSFORM #############################################

    wavelet_planes_list = []
    wavelet_planes_list.append(input_image)


    for scale_index in range(J):
        previous_scale = wavelet_planes_list[scale_index]

        next_scale = smooth_bspline(previous_scale, 3, scale_index)

        previous_scale -= next_scale

        wavelet_planes_list.append(next_scale)

    coarse = wavelet_planes_list[-1]
    planes = np.zeros((input_image.shape[0],input_image.shape[1],J))

    #Need to do this with a loop as numba doesn't accept np.array(wavelet_planes_list[:-1])
    for i in range(J):
        planes[:,:,i]=wavelet_planes_list[i]

    return coarse, planes  #coarse, all other planes


@njit
def mad(z):
    return np.median(np.abs(z - np.median(z)))/0.6735

@njit(parallel=False)
def FBS_Inpainting(b,mask,kmad=3,J=4,nmax=100,L0=0,perscale=1):

    x = np.zeros_like(b)

    alpha = 1 # path length

    for r in range(nmax):

        # Compute the gradient

        Delta = mask*(mask*x - b)

        # Gradient descent

        x_half = x - alpha*Delta

        # Sparsity constraint

        xp = Starlet_Filter2D(x=x_half,kmad=kmad,J=J,L0=L0,perscale=perscale)

        x = np.copy(xp)

    return x

@njit(parallel=False)
def Starlet_Filter2D(x=0,kmad=3,J=4,L0=0,perscale=1):
    """
    Implementation of a fast 2D filtering using Starlet_Forward2D.
    """

    c,w = Starlet_Forward2D(x,J=J)


    for r in range(J-1):

        if perscale:
            thrd = kmad*mad(w[:,:,r][np.where(x>10)])
        else:
            if r == 0:
                thrd = kmad*mad(w[:,:,r]) # Estimate the threshold in the first scale only

        if L0:
            w[:,:,r] = w[:,:,r]*(np.abs(w[:,:,r]) > thrd)
        else:
            w[:,:,r] = (w[:,:,r] - thrd*np.sign(w[:,:,r]))*(np.abs(w[:,:,r]) > thrd)

    return c + np.sum(w,axis=2) # Sum all planes including coarse scale

@njit(parallel=False)
def Starlet_Forward1D(input_image,J=2):
    # DO THE WAVELET TRANSFORM #############################################
    input_image = input_image.copy()
    wavelet_planes_list = []
    wavelet_planes_list.append(input_image)


    for scale_index in range(J):
        previous_scale = wavelet_planes_list[scale_index]

        next_scale = smooth_bspline1D(previous_scale, scale_index)

        previous_scale -= next_scale

        wavelet_planes_list.append(next_scale)

    coarse = wavelet_planes_list[-1]
    planes = np.zeros((input_image.shape[0],J))

    for i in range(J):
        planes[:,i]=wavelet_planes_list[i]

    return coarse, planes  #coarse, all other planes



@njit(parallel=False)
def Starlet_Filter1D(x=0,kmad=3,J=4,L0=0,perscale=1):
    """
    Implementation of a fast 1D filtering using Starlet_Forward2D.
    """

    c,w = Starlet_Forward1D(x,J=J)


    for r in range(J-1):

        if perscale:
            thrd = kmad*mad(w[:,r])
        else:
            if r == 0:
                thrd = kmad*mad(w[:,r]) # Estimate the threshold in the first scale only

        if L0:
            w[:,r] = w[:,r]*(np.abs(w[:,r]) > thrd)
        else:
            w[:,r] = (w[:,r] - thrd*np.sign(w[:,r]))*(np.abs(w[:,r]) > thrd)

    return c + np.sum(w,axis=1) # Sum all planes including coarse scale


@njit(parallel=False)
def Starlet_Forward2D_1D(input_image,J_1D=3,J_2D=2):

    input_image = input_image.copy()
    l,m,n=input_image.shape
    wavelet_planes=np.zeros((l,m,n,J_1D+1,J_2D+1))
    c_2D=np.zeros((l,m,n))
    cc_2D1D=np.zeros((l,m,n))
    cw_2D1D=np.zeros((l,m,n,J_1D))
    wc_2D1D=np.zeros((l,m,n,J_2D))
    w_2D=np.zeros((l,m,n,J_2D))
    ww_2D1D=np.zeros((l,m,n,J_2D,J_1D))
    #wavelet_planes[:,:,:,0,0]=coarse cales
    for k in range(l):
        im=input_image[k,:,:]
        c_2D[k,:,:],w_2D[k,:,:,:]=Starlet_Forward2D(im,J=J_2D)
    for i in range(m):
        for j in range(n):
            coarse=c_2D[:,i,j]
            cc,cw=Starlet_Forward1D(coarse,J=J_1D)
            cw_2D1D[:,i,j,:]=cw
            cc_2D1D[:,i,j]=cc
            for jj in range(J_2D):
                wave=w_2D[:,i,j,jj]
                wc,ww=Starlet_Forward1D(wave,J=J_1D)
                ww_2D1D[:,i,j,jj,:]=ww
                wc_2D1D[:,i,j,jj]=wc

    coarse_scales=np.zeros(ww_2D1D.shape)
    #coarse_scales[:,:,:,0,:]=

    #for jj in range(J_2D):
#        image=w_2D[:,:,:,jj]
#        for j in range(J_1D):
#            coarse_scales[:,:,:,jj,j]=image-ww_2D1D[:,:,:,jj,j]
#            image=coarse_scales[:,:,:,jj,j]
    return cc_2D1D,cw_2D1D,wc_2D1D,ww_2D1D#,coarse_scales
