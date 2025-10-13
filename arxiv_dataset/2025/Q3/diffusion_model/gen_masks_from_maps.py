import numpy as np, sys, os, warnings
import scipy.ndimage as ndimage

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    xgrid, ygrid = indices(data.shape)
    x = (xgrid*data).sum()/total
    y = (ygrid*data).sum()/total
    col = data[:, int(y)]
    width_x = sqrt(abs((arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = sqrt(abs((arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: ravel(gaussian(*p)(*indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

def fitting_func(p, p0, xgrid, ygrid, tmap, lbounds = None, ubounds = None, fixed = None, return_fit = 0):
    if hasattr(fixed, '__len__'):
        p[fixed] = p0[fixed]

    if hasattr(lbounds, '__len__'):
        linds = abs(p)<abs(lbounds)
        if len(linds)>0: return tmap
        p[linds] = lbounds[linds]

    if hasattr(ubounds, '__len__'):
        uinds = abs(p)>abs(ubounds)
        if len(uinds)>0: return tmap
        p[uinds] = ubounds[uinds]

    def gaussian(p, xp, yp):
        if len(p)>6:
            wx, wy = p[4], p[5]
            rota = np.radians(p[6])
            xp = np.radians(xp/60.) * np.cos(rota) - np.radians(yp/60.) * np.sin(rota)
            yp = np.radians(xp/60.) * np.sin(rota) + np.radians(yp/60.) * np.cos(rota)

            xp = np.degrees(xp) * 60.
            yp = np.degrees(yp) * 60.
        else:
            wx = wy = p[4]
        ##print(p); sys.exit()
        g = p[0]+p[1]*np.exp( -(((p[2]-xp)/wx)**2+ ((p[3]-yp)/wy)**2)/2.)

        return g

    if not return_fit:
        return np.ravel(gaussian(p, xgrid, ygrid) - tmap)
    else:
        return gaussian(p, xgrid, ygrid)

def get_lxly(flatskymapparams):

    """
    returns lx, ly based on the flatskymap parameters
    input:
    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    output:
    lx, ly
    """

    nx, ny, dx, dx = flatskymapparams
    dx = np.radians(dx/60.)

    lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx ), np.fft.fftfreq( ny, dx ) )
    lx *= 2* np.pi
    ly *= 2* np.pi

    return lx, ly

def cl_to_cl2d(el, cl, flatskymapparams):

    """
    converts 1d_cl to 2d_cl
    inputs:
    el = el values over which cl is defined
    cl = power spectra - cl

    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    output:
    2d_cl
    """
    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)

    cl2d = np.interp(ell.flatten(), el, cl).reshape(ell.shape) 

    return cl2d

def get_mask_using_gaussian_fitting(nonpeak_mask, mul_width_by_factor = 2, ini_height = 0., ini_amp = 1., ini_rot = 0., ini_blob_size_in_pixels = 10., use_elliptical_gaussian = False, perform_apod = True):
        
    #fit Gaussian to the tmap to get the centroid
    ny, nx = nonpeak_mask.shape
    x = y = np.arange(nx)
    xgrid, ygrid = np.meshgrid(x,y)
    #predict initial parameters
    height = ini_height #np.std(nonpeak_mask) #overall floor
    amp = ini_amp #np.max(nonpeak_mask) #peak
    #ellipse rotation
    rot = ini_rot
    #width
    blob_size_in_pixels = ini_blob_size_in_pixels
    wx = wy = blob_size_in_pixels

    non_zero_yinds, non_zero_xinds = np.where( nonpeak_mask == 1)
    howmany = len(non_zero_yinds)

    final_fit_arr = []
    total_mask = np.zeros( nonpeak_mask.shape )
    for cntr, (x,y) in enumerate( zip(non_zero_xinds, non_zero_yinds) ):
        if not (cntr<10 or cntr>howmany-10): continue
        #if not (cntr>howmany-10): continue
        #print(cntr, howmany, x, y)

        #centres
        #y_cen_ini, x_cen_ini = np.unravel_index(np.argmax(tmap), tmap.shape)
        x_cen_ini, y_cen_ini = x, y
        if use_elliptical_gaussian:
            p0 = [height, amp, x_cen_ini, y_cen_ini, wx, wy, rot]
        else:
            p0 = [height, amp, x_cen_ini, y_cen_ini, wx]

        #bounds - not used currently
        ##lbounds = np.asarray( [height*3., amp*3., x_cen - 10, y_cen - 10, wx/2] )
        ##ubounds = np.asarray( [height*3., amp*3., x_cen + 10, y_cen + 10, wx*2.] )
        p1, success = optimize.leastsq(fitting_func, p0[:], args=(p0, xgrid, ygrid, nonpeak_mask))#, lbounds, ubounds))
        final_fit_arr.append( p1 )
        nonpeak_mask_fit = fitting_func(p1,p1,xgrid,ygrid, nonpeak_mask,return_fit = 1)

        '''
        subplot(121); imshow(nonpeak_mask); colorbar()
        subplot(122); imshow(nonpeak_mask_fit); colorbar(); show(); 
        #sys.exit()
        '''

        #create a mask based on the centres and the widths
        x_fit, y_fit, x_width = p1[2:]
        width_for_mask = x_width * mul_width_by_factor
        rad_grid = np.hypot( xgrid - x_fit, ygrid - y_fit ) 
        inds_to_mask = np.where( rad_grid <= width_for_mask )
        curr_mask = np.zeros( nonpeak_mask.shape )
        curr_mask[inds_to_mask] = 1.
        total_mask = total_mask + curr_mask

    final_mask = np.ones( nonpeak_mask.shape )
    final_mask[total_mask != 0] = 0.
    '''
    subplot(121); imshow(nonpeak_mask); colorbar()
    subplot(122); imshow(total_mask); colorbar(); show(); 
    #sys.exit()
    '''
    if perform_apod:
        pix = 1
        radius = (nx * pix)/10.
        npix_cos = int(radius/pix)
        ker=np.hanning(npix_cos)
        ker2d=np.asarray( np.sqrt(np.outer(ker,ker)) )

        final_mask=ndimage.convolve(final_mask, ker2d)
        final_mask/=final_mask.max()
    return final_mask
################################################################################################################
################################################################################################################
################################################################################################################
import numpy as np, os, sys, glob, healpy as H
from pylab import *
import scipy.optimize as optimize

mask_threshold_sigma_units = 0.3 #1 #5 ##10.
threshold_to_pick_average_blobs = 0.6
mul_width_by_factor = 2.5
beam_val = 20. #am
which_tsz = 'agora' #'ddpm'

start, end =int(sys.argv[1]), int(sys.argv[2])

if which_tsz == 'agora':
    fname = 'data/low_pass/2mjy/cut_maps_RES_256_ANG_X_6.0 deg_2mJy_lp_tsz3_no_pickle.npy'

tsz_dic = np.load(fname, allow_pickle=True)
op_folder = 'tsz_masks/%s/widthx%g/' %(which_tsz, mul_width_by_factor)
pl_folder = '%s/plots/' %(op_folder)
if not os.path.exists( op_folder ): os.system('mkdir -p %s' %(op_folder))
if not os.path.exists( pl_folder ): os.system('mkdir -p %s' %(pl_folder))
total_sims = 50 ##len( tsz_dic )

for sim_no in range(start, end ):
    print(sim_no)
    opfname = '%s/sim%s.npy' %(op_folder, sim_no)
    if os.path.exists( opfname ): continue

    tszmap = tsz_dic[sim_no][:, :, 0]
    ###tszmap = tszmap - np.mean(tszmap)

    #--------------------
    #--------------------
    #get locations closer to the mean
    mean_val = np.mean(tszmap)
    inds = np.where( abs(tszmap - mean_val) <= mask_threshold_sigma_units * np.std(tszmap) )
    nonpeak_mask = np.zeros( tszmap.shape )
    nonpeak_mask[inds] = 1.
    ##subplot(121); imshow(nonpeak_mask); colorbar()
    ##subplot(122); imshow(tszmap); colorbar(); show(); sys.exit()
    #--------------------
    #--------------------

    #--------------------
    #--------------------
    #smooth the non-peak mask
    #--- get beam
    ny, nx = tszmap.shape
    dx = 1.4 #am
    lmax = 10000
    bl = H.gauss_beam( np.radians(beam_val/60.), lmax = lmax - 1)
    el = np.arange(len(bl))
    mapparams = [ny, nx, dx, dx]
    bl_2d = cl_to_cl2d(el, bl, mapparams)
    #---- apply beam to the nonpeak_mask
    #subplot(131); imshow(nonpeak_mask); colorbar()
    #subplot(132); imshow(tszmap); colorbar()
    nonpeak_mask = np.fft.ifft2( np.fft.fft2( nonpeak_mask ) * bl_2d ).real
    #subplot(133); imshow(nonpeak_mask); colorbar(); show(); sys.exit()
    #--------------------
    #--------------------


    #--------------------
    #--------------------
    #threshhold this
    ##subplot(121); imshow(nonpeak_mask); colorbar()
    nonpeak_mask[nonpeak_mask<threshold_to_pick_average_blobs] = 0.
    nonpeak_mask[nonpeak_mask>threshold_to_pick_average_blobs] = 1.
    ##subplot(122); imshow(nonpeak_mask); colorbar(); show(); sys.exit()

    #--------------------
    #--------------------
    #fit Gaussian and get the centroids
    new_mask = get_mask_using_gaussian_fitting(nonpeak_mask, mul_width_by_factor = mul_width_by_factor, ini_height = 0., ini_amp = 1., ini_rot = 0., ini_blob_size_in_pixels = 10., use_elliptical_gaussian = False)
    np.save( opfname, new_mask )

    #plot
    clf()
    cmap = cm.hot
    vmin, vmax = -10, 0. ##-10., 0. ##-3, 2.
    subplot(131); imshow(tszmap, vmin = vmin, vmax = vmax, cmap = cmap); colorbar(); title(r'tSZ')
    subplot(132); imshow(nonpeak_mask); colorbar(); title(r'blobs close to mean', fontsize = 8)
    subplot(133); imshow(new_mask); colorbar(); title(r'Created mask')
    plname = '%s/sim%s.png' %(pl_folder, sim_no)
    savefig(plname, dpi = 200.)
    #sys.exit()
    continue

    clf()
    cmap = cm.hot
    vmin, vmax = -10, 0. ##-10., 0. ##-3, 2.
    figure(figsize = (8.5, 4))
    subplot(131); imshow(nonpeak_mask); colorbar(); title(r'Non-peak mask')
    subplot(132); imshow(tszmap, vmin = vmin, vmax = vmax, cmap = cmap); colorbar(); title(r'tSZ ')
    subplot(133); imshow(tszmap * nonpeak_mask, vmin = vmin, vmax = vmax, cmap = cmap); colorbar(); title(r'Combined')
    plname = 'plots/tsz_%s.png' %(sim_no)
    savefig(plname, dpi = 200.)
    close()

sys.exit()

delay_val = 100
cmd = 'convert -delay %s -loop 0 plots/*.png plots/tsz_picking_avg_blobs.gif' %(delay_val)
os.system( cmd )
sys.exit()

#masked map
#tszmap = tszmap * source_mask

#gradient
x_grad, y_grad = np.gradient( tszmap )
mag_grad = np.sqrt( x_grad**2 + y_grad**2. )

#filter


#fft
tszmap_fft = np.fft.fft2( tszmap )




