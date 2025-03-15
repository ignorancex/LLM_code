import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.restoration as ski
import scipy.ndimage as ndimage

# A method to assist with ploting of images
def display_image(mat, axes=None, cmap=None, hide_axis=True):
    """
    Display a given matrix into Jupyter's notebook
    
    :param mat: Matrix to display
    :param axes: Subplot on which to display the image
    :param cmap: Color scheme to use
    :param hide_axis: If `True` axis ticks will be hidden
    :return: Matplotlib handle
    """
    img = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB) if mat.ndim == 3 else mat
    cmap= cmap if mat.ndim != 2 or cmap is not None else 'gray'
    if axes is None:
        if hide_axis:
            plt.xticks([])
            plt.yticks([])
        return plt.imshow(img, cmap=cmap)
    else:
        if hide_axis:
            axes.set_xticks([])
            axes.set_yticks([])
        return axes.imshow(img, cmap=cmap)
        
# A method to blind estimate a kernel according to the input and output images
def blind_kernel_estimate(Y, X_l, kernel_width, reg_mode=0, reg_weight=10):
    '''
    Operation: estimate the kernel k that minimizes ||Y-X_l*k||**2 (+ reg_weight * ||k||**2)
    Inputs:
        2D images Y and X_l (Gray or multichannel)
        kernel_width (integer > 0, better if even)
        reg_mode (0: no reg, 1: L2 reg)
        reg_weight (weight of the L2 reg term, ignored when reg_mode=0)
    Outputs:
        k of size kernel_width x kernel_width (or kernel_width-1 if it is odd)
    '''

    # Convert inputs to Fourier domain
    X_l_Freq = np.fft.fft2(X_l, axes=[0, 1])
    Y_Freq = np.fft.fft2(Y, axes=[0, 1])

    # Solve for k in Fourier domain (regularization only affects den)
    num = X_l_Freq.conjugate() * Y_Freq
    if reg_mode == 0:
        den = np.abs(X_l_Freq) ** 2  # Fourier transform of X_l transpose * X_l
    elif reg_mode == 1:
        #         reg_term = reg_weight * np.identity(kernel_width)
        #         reg_term_Freq = psf2otf(reg_term, Y.shape[:2])
        #         if X_l_Freq.ndim == 3:
        #             reg_term_Freq = np.repeat(reg_term_Freq[:, :, np.newaxis], X_l_Freq.shape[2], axis=2)
        #         den = reg_term_Freq + np.abs(X_l_Freq)**2 # Fourier transform of [2*reg_weight + X_l transpose * X_l]
        den = reg_weight + np.abs(X_l_Freq) ** 2  # Fourier transform of [2*reg_weight + X_l transpose * X_l]
    k_l_Freq = num / den

    # Get average channel solution if multi-channel
    if k_l_Freq.ndim == 3:
        k_l_Freq = np.mean(k_l_Freq, 2)

    # Convert back to spatial, given the width
    if kernel_width < 1:
        raise ValueError('kernel_width must be a positive integer')

    k_l = otf2psf(k_l_Freq, [kernel_width, kernel_width])

    # Correct the pixel shift for odd width
    if (kernel_width % 2 == 1):
        k_l = k_l[1:, 1:]
        
    k_l[k_l < 0] = 0
    # Normalize to 1
    if(k_l.sum()!=0):
        k_l = k_l / k_l.sum() #With max it identifies the sharp area

    return k_l

# A method the converts an OTF into the PSF of a kernel
def otf2psf(otf, outsize=None):
    '''
    Operation: output the PSF of otf in shape outsize
    Inputs:
        2D otf array
        tuple of the shape of the output psf
    Outputs:
        2d psf array
    '''
    insize = np.array(otf.shape)
    psf = np.fft.ifftn(otf, axes=(0, 1))
    for axis,  axis_size in enumerate(insize):
        psf = np.roll(psf, np.floor(axis_size / 2).astype(int), axis=axis)
    if type(outsize) != type(None):
        insize = np.array(otf.shape)
        outsize = np.array(outsize)
        n = max(np.size(outsize), np.size(insize))
        colvec_out = outsize.flatten().reshape((np.size(outsize), 1))
        colvec_in = insize.flatten().reshape((np.size(insize), 1))
        outsize = np.pad(colvec_out, ((0, max(0, n - np.size(colvec_out))), (0, 0)), mode="constant")
        insize = np.pad(colvec_in, ((0, max(0, n - np.size(colvec_in))), (0, 0)), mode="constant")

        pad = (insize - outsize) / 2
        if np.any(pad < 0):
            print("otf2psf error: OUTSIZE must be smaller than or equal than OTF size")
        prepad = np.floor(pad)
        postpad = np.ceil(pad)
        dims_start = prepad.astype(int)
        dims_end = (insize - postpad).astype(int)
        for i in range(len(dims_start.shape)):
            psf = np.take(psf, range(dims_start[i][0], dims_end[i][0]), axis=i)
    n_ops = np.sum(otf.size * np.log2(otf.shape))
    psf = np.real_if_close(psf, tol=n_ops)
    return psf

# A method to generate Gaussian kernels
def fspecial(shape,sigma):
    """   
    Operation: Returns Gaussian kernel in shape shape and standard deviation sigma equivalent to MATLAB's fspecial('gaussian',[shape],[sigma])
    Inputs:
        tuple of the shape of the output kernel
        float representing the standerd deviation
    Outputs:
        2d Gaussian kernel array
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# A method to evaluate the blur level of a segment
def blurriness(image):
    """   
    Operation: Returns the blur level (in percentage) of the image according to the Crété Method
    Inputs:
        2d image array
    Outputs:
        float representing the blur level of the image
    """
    # blurring the image in both directions 
    #Hv = fspecial('Gaussian',[1, 2*3*1+1],1)
    Bver = ndimage.gaussian_filter(image, [1, 0])
    Bhor = ndimage.gaussian_filter(image, [0, 1])
    #Bver = imfilter(image, Hv, 'symmetric');                        ]

    # computing the edges of the original image and its blurred version in both directions
    D_Fver = np.abs(np.gradient(image, axis=0)); 
    D_Fhor = np.abs(np.gradient(image, axis=1));
    D_Bver = np.abs(np.gradient(Bver, axis=0)); 
    D_Bhor = np.abs(np.gradient(Bhor, axis=1));

    Vver = D_Fver - D_Bver; 
    Vver[Vver < 0]= 0;
    Vhor = D_Fhor - D_Bhor; 
    Vhor[Vhor < 0]= 0;

    s_Fver = np.sum(D_Fver);
    s_Fhor = np.sum(D_Fhor);
    s_Vver = np.sum(Vver);
    s_Vhor = np.sum(Vhor);

    b_Fver = (s_Fver - s_Vver) / s_Fver;
    b_Fhor = (s_Fhor - s_Vhor) / s_Fhor;

    blur = max(b_Fver,b_Fhor);

    blurriness = 1 - blur;
    return blurriness

# A method to get the blur map of an image
def img_blurMap(img, cell_size, step_size, plot=False):
    """   
    Operation: Returns the blur map of img according to segments of cell_size*cell_size with overlap of cell_size-step_size pixels
    Inputs:
        2d image array
        integer of the width of the square segment  resolution to calculate the blurs of (the bigger the value the more smooth the map is, the smaller it is the more detailed it is)
        integer of the distance between each segment (needs to be smaller or equal than cell_size)
        a boolean to plot the blur map or not
    Outputs:
        2d blur map array
    """
    x_end = img.shape[1]-cell_size+1
    y_end = img.shape[0]-cell_size+1

    blurLvl = np.zeros_like(img,dtype=np.float64)
    aveMap = np.zeros_like(img,dtype=np.float64)

    for stride_x in range(0,x_end,step_size):
        xStart = stride_x
        xStop  = stride_x+cell_size
        for stride_y in range(0,y_end,step_size):
            yStart = stride_y
            yStop  = stride_y+cell_size     
            blur = blurriness(img[yStart:yStop,xStart:xStop]) # calculate segment's blur level
            blurLvl[yStart:yStop,xStart:xStop] = blurLvl[yStart:yStop,xStart:xStop] + blur
            aveMap[yStart:yStop,xStart:xStop] = aveMap[yStart:yStop,xStart:xStop] + 1

    blurLvl_norm = blurLvl/aveMap # normalise blur levels map
    
    if plot:
        plt.figure(figsize=((16,16)))
        plt.title('Blur Level Map')            
        plt.xticks([])
        plt.yticks([])
        plt.imshow(blurLvl_norm, cmap='gray', vmin=0, vmax=1)
    
    return blurLvl_norm

# A method to restore image using the Blind Estimated Gaussian Kernels method
def img_deblur_gaussian(img, blurMap, k_size, step_size, coef=50, plot=False):
    """   
    Operation: Returns the result of the deblurring img according to blurMap 
    Inputs:
        2d image array
        2d blur map array of the image
        integer width of the square Gaussian kernel 
        integer of the distance between each segment (needs to be smaller or equal than k_size)
        a float coefficient to determine the linear ratio between the blur level and the kernel's standard deviation
        a boolean to plot the blur map or not
    Outputs:
        2d deblured image array
    """
    x_end = img.shape[1]-k_size+1
    y_end = img.shape[0]-k_size+1
    deblur = np.zeros_like(img,dtype=np.float64)
    aveMap = np.zeros_like(img,dtype=np.float64)

    for stride_x in range(0,x_end,step_size):
        xStart = stride_x
        xStop  = stride_x+k_size
        xCentre= stride_x+k_size//2
        for stride_y in range(0,y_end,step_size):
            yStart = stride_y
            yStop  = stride_y+k_size  
            yCentre= stride_y+k_size//2
            
            b = img[yStart:yStop,xStart:xStop]
            g = fspecial((k_size,k_size),coef*blurMap[yCentre,xCentre]) # generate a Gaussian kernel accordiing to bler map value
            d = ski.wiener(b,g,balance=0.1,clip=False) # deblur using Wiener deconvolution
            deblur[yStart:yStop,xStart:xStop] = deblur[yStart:yStop,xStart:xStop] + d
            aveMap[yStart:yStop,xStart:xStop] = aveMap[yStart:yStop,xStart:xStop] + 1

    # Normalize the deblur to be between 0 and the max value of the img
    deblur_norm = deblur/aveMap
    deblur_norm = deblur_norm - deblur_norm.min()
    deblur_norm = (deblur_norm * img.max()/deblur_norm.max()).astype(np.uint8)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(20, 20))
        display_image(img, axes=ax[0])
        ax[0].set_title('Input Image')
        display_image(deblur_norm, axes=ax[1])
        ax[1].set_title('Deblured')
        
    return deblur_norm

# A method to generate a kernel lookup table
def kernel_lookup_table(imgList, blurMap, sharp, k_size, step_size, resolution=3):
    """   
    Operation: Returns a dictionary with the kernels lookup table {blurLevel: kernel}
    Inputs:
        list of 2d images arrays to create a kernel lookup table from
        list of 2d blur maps arrays correlating to the images list
        2d image array of the sharp image correlated to the images list
        integer width of the square kernels
        integer of the distance between each segment (needs to be smaller or equal than k_size)
        integer of the resoltion of the blur level decimal depth
    Outputs:
        dictionary of the kernels lookup table {blurLevel: kernel}
    """
    lookupTable = {}
    for i, img in enumerate(imgList):
        x_end = img.shape[1]-k_size+1
        y_end = img.shape[0]-k_size+1
        for stride_x in range(0,x_end,step_size):
            xStart = stride_x
            xStop  = stride_x+k_size
            xCentre= stride_x+k_size//2
            for stride_y in range(0,y_end,step_size):
                yStart = stride_y
                yStop  = stride_y+k_size  
                yCentre= stride_y+k_size//2
                
                current_k = blind_kernel_estimate(img[yStart:yStop,xStart:xStop],sharp[yStart:yStop,xStart:xStop],k_size,reg_mode=1,reg_weight = 5e6) # blind estimate kernel from sharp image
                blurLvl = blurMap[i][yCentre,xCentre].round(resolution) # check pixel location blur level to resolution depth
                # add kernel to lookup table
                if blurLvl in lookupTable.keys():
                    lookupTable[blurLvl].append(current_k) 
                else: 
                    lookupTable[blurLvl] = [current_k]
    # average all kernels in each blur level entry
    for blur in lookupTable:
        k_mean = np.float32(0.0)
        for k in lookupTable[blur]:
            k_mean = k_mean + k
        k_mean = k_mean / len(lookupTable[blur])
        lookupTable[blur] = k_mean
        
    ###########################################################################
    # Future work: implement interpolation to cover missing dictionary values #
    ###########################################################################
        
    lookupTable = dict(sorted(lookupTable.items())) # sort lookup table
        
    return lookupTable
    
# A method to restore image using the Kernels Lookup Table method
def img_deblur_lookupTable(img, blurMap, lookupTable, k_size, step_size, resolution=3, plot=False):
    """   
    Operation: Returns the result of the deblurring img according to the lookup table 
    Inputs:
        2d image array
        2d blur map array of the image
        dictionary of lookup table {blurLevel: kernel}
        integer width of the square kernel 
        integer of the distance between each segment (needs to be smaller or equal than k_size)
        integer of the resoltion of the blur level decimal depth
        a boolean to plot the blur map or not
    Outputs:
        2d deblured image array
    """
    x_end = img.shape[1]-k_size+1
    y_end = img.shape[0]-k_size+1
    deblur = np.zeros_like(img,dtype=np.float64)
    aveMap = np.zeros_like(img,dtype=np.float64)

    for stride_x in range(0,x_end,step_size):
        xStart = stride_x
        xStop  = stride_x+k_size
        xCentre= stride_x+k_size//2
        for stride_y in range(0,y_end,step_size):
            yStart = stride_y
            yStop  = stride_y+k_size  
            yCentre= stride_y+k_size//2
            
            b = img[yStart:yStop,xStart:xStop]
            blur = blurMap[yCentre,xCentre].round(resolution)
            while not blur in lookupTable: # taking closest blur level that exists in the lookup table
                blur = blur + 10**(-resolution) # can be removed if all values covered by interpolation
            g = lookupTable[blur] # taking corelating kernel to blur local level
            d = ski.wiener(b,g,balance=0.1,clip=False) # deblur using Wiener deconvolution
            deblur[yStart:yStop,xStart:xStop] = deblur[yStart:yStop,xStart:xStop] + d
            aveMap[yStart:yStop,xStart:xStop] = aveMap[yStart:yStop,xStart:xStop] + 1

    # Normalize the deblur to be between 0 and the max value of the img
    deblur_norm = deblur/aveMap
    deblur_norm = deblur_norm - deblur_norm.min()
    deblur_norm = (deblur_norm * img.max()/deblur_norm.max()).astype(np.uint8)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(20, 20))
        display_image(img, axes=ax[0])
        ax[0].set_title('Input Image')
        display_image(deblur_norm, axes=ax[1])
        ax[1].set_title('Deblured')

    return deblur_norm
