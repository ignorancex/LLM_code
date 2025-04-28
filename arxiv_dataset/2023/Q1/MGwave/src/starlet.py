###
#
# This code has been modified from https://github.com/broxtronix/pymultiscale
# See license at bottom
#
###

import numpy as np
import scipy.signal

# ----------------------------  UTIILITY FUNCTIONS ---------------------------

def get_kernel(C1,C2,C3,step):
    KSize = 4*step+1
    KS2 = KSize//2
    kernel = np.zeros((KSize), dtype = np.float32)
    if KSize == 1:
        kernel[0] = 1.0
    else:
        kernel[0] = C1
        kernel[KSize-1] = C1
        kernel[KS2+step] = C2
        kernel[KS2-step] = C2
        kernel[KS2] = C3
    return kernel

def bspline_star(x, step):
    """
    This implements the starlet kernel. Application to different scales is
    accomplished via the step parameter.
    """
    ndim = len(x.shape)
    C1 = 1./16.
    C2 = 4./16.
    C3 = 6./16.
    kernel = get_kernel(C1,C2,C3,step)
    # kernel = get_cont_kernel(step)
    # Based on benchmarks conducted during January 2015, OpenCV has a far faster
    # seperabable convolution routine than scipy does.  We use it for 2D images
    if ndim == 2:
        import cv2
        result = cv2.sepFilter2D(x, cv2.CV_32F, kernelX = kernel, kernelY = kernel)
        return result

    else:
        result = x
        import scipy.ndimage
        for dim in range(ndim):
            result = scipy.ndimage.filters.convolve1d(result, kernel, axis = dim, mode='reflect', cval = 0.0)
    return result

def wavelet_star(x, step, type='diag'):
    ndim = len(x.shape)
    G1 = -1./16.
    G2 = -4./16.
    G3 = 10./16.
    kernela = get_kernel(G1,G2,G3,step)
    kernelb = get_kernel(1/16.,4/16.,6/16.,step)

    # Default = diagonal
    kernelx = kernela
    kernely = kernela
    if (type == 'horiz'):
        kernelx = kernela
        kernely = kernelb
    elif (type == 'vert'):
        kernelx = kernelb
        kernely = kernela
    if ndim == 2:
        import cv2
        result = cv2.sepFilter2D(x, cv2.CV_32F, kernelX = kernelx, kernelY = kernely)
        return result

    else:
        result = x
        import scipy.ndimage
        for dim in range(ndim):
            result = scipy.ndimage.filters.convolve1d(result, kernelx, axis = dim, mode='reflect', cval = 0.0)
    return result

# -----------------------------------------------------------------------------
#                            FUNCTION API
# -----------------------------------------------------------------------------

def starlet_transform(input_image, num_bands = None, gen2 = True):
    '''
    Computes the starlet transform of an image (i.e. undecimated isotropic
    wavelet transform).

    The output is a python list containing the sub-bands. If the keyword Gen2 is set,
    then it is the 2nd generation starlet transform which is computed: i.e. g = Id - h*h
    instead of g = Id - h.

    REFERENCES:
    [1] J.L. Starck and F. Murtagh, "Image Restoration with Noise Suppression Using the Wavelet Transform",
    Astronomy and Astrophysics, 288, pp-343-348, 1994.

    For the modified STARLET transform:
    [2] J.-L. Starck, J. Fadili and F. Murtagh, "The Undecimated Wavelet Decomposition
        and its Reconstruction", IEEE Transaction on Image Processing,  16,  2, pp 297--309, 2007.

    This code is based on the STAR2D IDL function written by J.L. Starck.
            http://www.multiresolutions.com/sparsesignalrecipes/software.html

    '''
    scale = []
    input_image = np.array(input_image)
    if num_bands == None:
        num_bands = int(np.ceil(np.log2(np.min(input_image.shape))) - 3)
        #assert num_bands > 0

    ndim = len(input_image.shape)

    im_in = input_image.astype(np.float32)
    step_trou = 1
    im_out = None
    WT = []

    for band in range(num_bands):
        im_out = bspline_star(im_in, step_trou)
        if gen2:  # Gen2 starlet applies smoothing twice
            WT.append(im_in - bspline_star(im_out, step_trou))
            scale.append(step_trou)
        else:
            #test = im_in - im_out
            WT.append(im_in - im_out)
            scale.append(step_trou)
        im_in = im_out
        step_trou *= 2

    WT.append(im_out)
    scale.append('rest')
    return WT, scale

def inverse_starlet_transform(coefs, gen2 = True):
    '''
    Computes the inverse starlet transform of an image (i.e. undecimated
    isotropic wavelet transform).

    The input is a python list containing the sub-bands. If the keyword Gen2 is
    set, then it is the 2nd generation starlet transform which is computed: i.e.
    g = Id - h*h instead of g = Id - h.

    REFERENCES:
    [1] J.L. Starck and F. Murtagh, "Image Restoration with Noise Suppression Using the Wavelet Transform",
        Astronomy and Astrophysics, 288, pp-343-348, 1994.

    For the modified STARLET transform:
    [2] J.-L. Starck, J. Fadili and F. Murtagh, "The Undecimated Wavelet Decomposition
        and its Reconstruction", IEEE Transaction on Image Processing,  16,  2, pp 297--309, 2007.

    This code is based on the ISTAR2D IDL function written by J.L. Starck.
            http://www.multiresolutions.com/sparsesignalrecipes/software.html
    '''

    # Gen1 starlet can be reconstructed simply by summing the coefficients at each scale.
    if not gen2:
        recon_img = np.zeros_like(coefs[0])
        for i in range(len(coefs)):
            recon_img += coefs[i]

    # Gen2 starlet requires more careful reconstruction.
    else:
        num_bands = len(coefs)-1
        recon_img = coefs[-1]
        step_trou = np.power(2, num_bands - 1)

        for i in reversed(list(range(num_bands))):
            im_temp = bspline_star(recon_img, step_trou)
            recon_img = im_temp + coefs[i]
            step_trou /= 2

    return recon_img

# ----------------- MS-VST Starlet --------------------------


def msvst(im, band):
    ndim = len(im.shape)

    def compute_tau(level, ndim):
        kernel_size = 4*(level+1)+1
        if ndim == 1:
            h_accum = np.zeros(kernel_size)
            h_accum[kernel_size//2] = 1.0   # Create an impulse
        elif ndim == 2:
            h_accum = np.zeros((kernel_size, kernel_size))
            h_accum[kernel_size//2, kernel_size//2] = 1.0   # Create an impulse
        elif ndim == 3:
            h_accum = np.zeros((kernel_size, kernel_size, kernel_size))
            h_accum[kernel_size//2, kernel_size//2, kernel_size//2] = 1.0   # Create an impulse

        h_accum = h_accum.astype(np.float32)
        step_trou = 1
        for i in range(level):
            h_accum = bspline_star(h_accum.copy(), step_trou)
            step_trou *= 2

        return ( np.sum(h_accum), np.sum(np.power(h_accum,2)), np.sum(np.power(h_accum,3)) )

    tau1, tau2, tau3  = compute_tau(band, ndim)
    #print 'band = ', band, '   tau = ', tau1, tau2, tau3
    b = np.sign(tau1) / np.sqrt(np.abs(tau1))
    #    b = 2.0 * np.sqrt(tau1/tau2)
    e = 7.0 * tau2 / (8.0 * tau1) - tau3 / (2.0 * tau2)
    return b * np.sign( im + e ) * np.sqrt( np.abs( im + e ) )


def multiscale_vst_stabilize(input_image, num_bands = None):
    ndim = len(input_image.shape)

    if num_bands == None:
        num_bands = int(np.ceil(np.log2(np.min(input_image.shape))) )

    im_in = input_image.astype(np.float32)
    step_trou = 1
    im_out = None
    coefs = []

    for band in range(num_bands):
        im_out = bspline_star(im_in, step_trou)
        coefs.append(msvst(im_in, band) - msvst(im_out, band+1))
        im_in = im_out
        step_trou *= 2

    coefs.append(im_out)
    return sum(coefs[:-1]) + msvst(coefs[-1], len(coefs)-1)


def msvst_starlet_transform(input_image, num_bands = None, gen2 = True):
    '''
    '''
    scale = []
    input_image = np.array(input_image)
    if num_bands == None:
        num_bands = int(np.ceil(np.log2(np.min(input_image.shape))) -3 )
        assert num_bands > 0

    ndim = len(input_image.shape)

    im_in = input_image.astype(np.float32)
    step_trou = 1
    im_out = None
    WT = []

    for band in range(num_bands):
        im_out = bspline_star(im_in, step_trou)
        if gen2:  # Gen2 starlet applies smoothing twice
            raise NotImplementedError("Gen2 Starlet with MS-VST not yet implemented.")
            # WT.append(msvst(im_in) - (bspline_star(im_out, step_trou))
        else:
            WT.append(msvst(im_in, band) - msvst(im_out, band+1))
            #print ''
            # WT.append((im_in) - (im_out))
        scale.append(step_trou)
        im_in = im_out.copy()
        step_trou *= 2

    WT.append(im_out)
    scale.append('rest')
    return WT,scale


def inverse_msvst_starlet_transform(coefs, gen2 = True):
    '''
    '''
    # Gen1 starlet can be reconstructed simply by summing the coefficients at each scale.
    if not gen2:

        # Reconstruct the image
        recon_img = sum(coefs[:-1]) + msvst(coefs[-1], len(coefs)-1)

        # Apply the normal inverse Anscombe transform to the reconstructed image
        b0 = 1.0
        e0 = 3.0/8.0
        recon_img = np.square(recon_img / b0) - e0
        # print recon_img.min(), recon_img.max()

    # Gen2 starlet requires more careful reconstruction.
    else:
        raise NotImplementedError("Inverse MS-VST Starlet transform not yet implemented.")

    return recon_img

def universal_threshold(X, N = None):
    '''
    Universal threshold based on empirical noise estimation in the
    smallest wavelet coefficients.  See Sparse Signal and
    Image Processing p. 164 by Starck for details.

    You may supply the total number of voxels N, otherwise it is
    computed as N = np.prod(X.shape)

    Returns: (data_median, data_threshold)
    '''
    if N == None:
        N = np.prod(X.shape)

    med = np.median(X)
    sigma_est = 1.4826 * np.median(np.abs(X - med))
    tau = 1# 2*np.sqrt(2)
    return (med, tau*np.sqrt(2*np.log(N))*sigma_est)

def mad_threshold(X, alpha = 0.99):
    '''
    The median absolute deviation (MAD)-based confidence interval is
    similar to the universal threshold, but less conservative and
    user-adjustable.

    Returns: (data_median, data_threshold)
    '''
    import scipy.stats
    multiplier = scipy.stats.norm.interval(alpha, loc=0, scale=1)[1]
    med = np.median(X)
    return (med, multiplier * 1.4826 * np.median(np.abs(X - med)))


# -----------------------------------------------------------------------------
#                         OBJECT-ORIENTED API
# -----------------------------------------------------------------------------

class StarletTransform(object):

    def __init__(self, img_shape, gen2 = True, num_bands = None):
        self.gen2 = gen2
        self.num_bands = num_bands
        self.img_shape = img_shape

        # Run a test tranform to determine the structure of
        # the lists containing the coefficients.  This is used
        # when reconstituting the coefficients from a flattened vector
        # of coefs, and vice versa.
        self.example_coefs = self.fwd(np.zeros(img_shape))

    # ------------- Forward and inverse transforms ------------------

    def fwd(self, data):
        return starlet_transform(data, self.num_bands, self.gen2)

    def inv(self, coefs):
        return inverse_starlet_transform(coefs, self.gen2)

    # --------------------- Utility methods -------------------------

    def num_bands(self, coefs):
        return len(coefs)-1

    def num_coefficients(self):
        return len(self.example_coefs) * np.prod(self.example_coefs[0].shape)

    def num_nonzero_coefficients(self, coefs):
        return sum([ band.nonzero()[0].shape[0] for band in coefs ])

    def coefs_to_vec(self, coefs):
        return np.hstack([vec.ravel(order = 'f') for vec in coefs])

    def vec_to_coefs(self, coef_vec):
        return [np.reshape(vec, self.img_shape, order = 'f') for vec in np.split(coef_vec, len(self.example_coefs))]

    def update(self, coefs, update, alpha):
        '''
        Adds the update (multiplied by alpha) to each set of
        coefficients.
        '''
        assert len(update) == len(coefs)

        update_squared_sum = 0.0;
        for b in range(len(coefs)):
            delta = alpha * update[b]
            coefs[b] += delta
            update_squared_sum += np.square(delta).sum()

        update_norm = np.sqrt(update_squared_sum)
        return (coefs, update_norm)

    def multiplicative_update(self, coefs, numerator, normalization, alpha):
        '''
        Multiplies the update to each set of coefficients, updating
        them in place.
        '''

        # Check arguments
        assert len(coefs) == len(numerator) == len(normalization)
        for b in range(len(coefs)):
            coefs[b] = (coefs[b] * numerator[b]) / (normalization[b] + alpha)
        return coefs

    def set_coefs(self, coefs, value):
        for b in range(len(coefs)):
            coefs[b].fill(value)

    def mean(self, coefs):
        '''
        Compute the average over all starlet coefficients.
        '''
        return np.hstack(coefs[:-1]).mean()
#        n        = sum( [ np.prod(coef.shape) for coef in coefs] )
#        coef_sum = sum( [ coef.sum()          for coef in coefs] )
#        return  coef_sum / n

    # ------------------ Thresholding methods -----------------------

    def threshold_by_band(self, coefs, threshold_func, skip_bands = [], within_axis = None, scaling_factor = None):
        '''
        Threshold each band individually.  The threshold_func() should
        take an array of coefficients (which may be 1d or 2d or 3d),
        and return a tuple: (band_center, band_threshold)

        Note that the low frequency band is left untouched.

        For the sake of speed and memory efficiency, updates to the
        coefficients are performed in-place.
        '''

        for b in range(len(coefs)-1):

            # Skip band?
            if b in skip_bands:
                continue

            if within_axis != None:
                num_planes = np.array(coefs)[b].shape[within_axis]
                for p in range(num_planes):
                    if within_axis == 0:
                        A = coefs[b][p][:][:]
                    elif within_axis == 1:
                        A = coefs[b][:][p][:]
                    else:
                        A = coefs[b][:][:][p]

                    (band_center, band_threshold) = threshold_func(A, b, p)
                    if scaling_factor != None:
                        band_threshold /= scaling_factor

                    # Soft threshold the coefficients
                    idxs = np.where( A > band_threshold )
                    A[idxs] -= band_threshold
                    idxs = np.where( np.abs(A) <= band_threshold )
                    A[idxs] = 0.0
                    idxs = np.where( A < -band_threshold )
                    A[idxs] += band_threshold

            else:
                # Compute the center and threshold.
                (band_center, band_threshold) = threshold_func(coefs[b], b, None)
                if scaling_factor != None:
                  band_threshold *= scaling_factor

                # Soft threshold the coefficients
                idxs = np.where( coefs[b] > band_threshold )
                coefs[b][idxs] -= band_threshold
                idxs = np.where( np.abs(coefs[b]) <= band_threshold )
                coefs[b][idxs] = 0.0
                idxs = np.where( coefs[b] < -band_threshold )
                coefs[b][idxs] += band_threshold

                # Due to their special properties, Gen2 wavelets can be forced
                # to have a strictly positive reconstruction if we zero out all
                # negative coefficients.
                #if self.gen2:
                #    coefs[b][np.where(coefs[b] < 0)] = 0.0

        return coefs


class MsvstStarletTransform(StarletTransform):

    def __init__(self):
        super(MsvstStarletTransform, self).__init__(gen2 = False)

    # ------------- Forward and inverse transforms ------------------

    def fwd(self, data, num_bands = None):
        return msvst_starlet_transform(data, num_bands, gen2 = False)

    def inv(self, coefs):
        return inverse_msvst_starlet_transform(coefs, gen2 = False)


### ---------------------------------------------------------------------------
### ------------------------    LICENSE     -----------------------------------
### ---------------------------------------------------------------------------
# Copyright (c) 2015, Stanford University
#
# All rights reserved.
#
# Redistribution and use in source and binary forms for academic and other
# non-commercial purposes with or without modification, are permitted provided
# that the following conditions are met:
#
#  * Redistributions of source code, including modified source code, must retain
#    the above copyright notice, this list of conditions and the following
#    disclaimer.
#
#  * Redistributions in binary form or a modified form of the source code must
#    reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
#  * Neither the name of The Leland Stanford Junior University, any of its
#    trademarks, the names of its employees, nor contributors to the source code
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# Where a modified version of the source code is redistributed publicly in source
# or binary forms, the modified source code must be published in a freely
# accessible manner, or otherwise redistributed at no charge to anyone requesting
# a copy of the modified source code, subject to the same terms as this agreement.
#
# THIS SOFTWARE IS PROVIDED BY THE TRUSTEES OF THE LELAND STANFORD JUNIOR
# UNIVERSITY ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE LELAND STANFORD JUNIOR
# UNIVERSITY OR ITS TRUSTEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
