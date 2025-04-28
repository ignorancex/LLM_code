import sys, os, h5py, bz2, pylab as plt, importlib, itertools, numpy as np
from tqdm.auto import tqdm, trange
from collections import OrderedDict
import seaborn
import plotting

def mm_to_pix(s, n_pix=2000, det_length=412):
    s_pix = s.copy()
    s_pix[:,1:] += det_length/2
    s_pix[:,1:] *= n_pix/det_length
    return s_pix


###########################################################################
###
### interactive spot plots 
###
###########################################################################

def scatter_spots(ax, spots, colorbar=False, colorbar_label='lambda [A]', **kw):
    cb = ax.scatter(spots[:,1], spots[:,2], **kw)
    if colorbar:
        ax.get_figure().colorbar(cb, ax=ax, label=colorbar_label)

    
def scatter_spots_obs_mod_per_angle_interactive(s_obs, inds_obs, s_mod, inds_mod, spot_mod_assign, ia, ig=None, p_lam=None, omegas=None, color_obs='k', lambda_lims=[0.4, 7], lam_vals=False, **kw):
    
    nx, ny = 1, 2; fig, ax = plt.subplots(nx, ny, figsize=(ny * 8, nx * 6), squeeze=False); axc=ax[0,0]; axl=ax[0,:];
    
    lims=[-210, 210]
    om = omegas[ia] if omegas is not None else -99
    
    select0o = (inds_obs[1]==0) & (inds_obs[0]==ia)  
    select1o = (inds_obs[1]==1) & (inds_obs[0]==ia)
    scatter_spots(ax[0,0], s_obs[select0o], marker='x', color=color_obs, s=100, label='peaks', **kw)
    scatter_spots(ax[0,1], s_obs[select1o], marker='x', color=color_obs, s=100, label='peaks', **kw)    
        
    
    grain_ids = np.unique(i_grn_mod) if ig == None else [ig]
    colors = seaborn.color_palette('tab20', len(grain_ids))

    for j, i_grn_ in enumerate(grain_ids):

        select0m = (inds_mod[1]==0) & (inds_mod[0]==ia) & (inds_mod[2]==i_grn_)   & (spot_mod_assign>-1)
        select1m = (inds_mod[1]==1) & (inds_mod[0]==ia) & (inds_mod[2]==i_grn_)   & (spot_mod_assign>-1)
        
        if p_lam is not None:
            colors0 = p_lam[select0m]
            colors1 = p_lam[select1m]  
        else:
            colors0 = colors1 = colors[j]

        scatter_spots(ax[0,0], s_mod[select0m], marker='o', c=colors0, s=100, zorder=0, cmap='Spectral_r', vmin=lambda_lims[0], vmax=lambda_lims[1], colorbar=True, label=f'model grain {grain_ids[j]}')
        scatter_spots(ax[0,1], s_mod[select1m], marker='o', c=colors1, s=100, zorder=0, cmap='Spectral_r', vmin=lambda_lims[0], vmax=lambda_lims[1], colorbar=True, label=f'model grain {grain_ids[j]}')    
        
        select0m = (inds_mod[1]==0) & (inds_mod[0]==ia) & (inds_mod[2]==i_grn_)   & (spot_mod_assign<0)
        select1m = (inds_mod[1]==1) & (inds_mod[0]==ia) & (inds_mod[2]==i_grn_)   & (spot_mod_assign<0)
        
        if p_lam is not None:
            colors0 = p_lam[select0m]
            colors1 = p_lam[select1m]  
        else:
            colors0 = colors1 = colors[j]

        scatter_spots(ax[0,0], s_mod[select0m], marker='d', c=colors0, s=100, zorder=0, cmap='Spectral_r', vmin=lambda_lims[0], vmax=lambda_lims[1], colorbar=False, label=f'model outliers grain {grain_ids[j]}')
        scatter_spots(ax[0,1], s_mod[select1m], marker='d', c=colors1, s=100, zorder=0, cmap='Spectral_r', vmin=lambda_lims[0], vmax=lambda_lims[1], colorbar=False, label=f'model outliers grain {grain_ids[j]}')    
        
        if lam_vals:
            for i in range(np.count_nonzero(select0m)):
                ax[0,0].text(x=s_mod[select0m][i,1], y=s_mod[select0m][i,2], s=f'{colors0[i]:2.2f}')
            for i in range(np.count_nonzero(select1m)):
                ax[0,1].text(x=s_mod[select1m][i,1], y=s_mod[select1m][i,2], s=f'{colors1[i]:2.2f}')
            
    ax[0,0].set_title(f'reflection   om={om:3.0f} deg')
    ax[0,1].set_title(f'transmission om={om:3.0f} deg')


    for a in ax.ravel():
        a.legend(loc='upper right')
        a.grid(True)
        a.set(xlim=lims, ylim=lims)



def scatter_spots_assignment_per_angle_interactive(s_obs, inds_obs, s_mod, inds_mod, spot_mod_assign, ia, omegas=None, color_obs='k', **kw):
    
    nx, ny = 1, 2; fig, ax = plt.subplots(nx, ny, figsize=(ny * 8, nx * 6), squeeze=False); axc=ax[0,0]; axl=ax[0,:];
        
    om = omegas[ia] if omegas is not None else -99
    lims=[-210, 210]
    
    select0o = (inds_obs[1]==0) & (inds_obs[0]==ia)  
    select1o = (inds_obs[1]==1) & (inds_obs[0]==ia)
    scatter_spots(ax[0,0], s_obs[select0o], marker='x', color=color_obs, s=100, label='peaks', **kw)
    scatter_spots(ax[0,1], s_obs[select1o], marker='x', color=color_obs, s=100, label='peaks', **kw)    
        
    
    grain_ids = np.unique(inds_mod[2])
    colors = seaborn.color_palette('tab20', len(grain_ids))

    select0m = (inds_mod[1]==0) & (inds_mod[0]==ia) & (spot_mod_assign>-1)
    select1m = (inds_mod[1]==1) & (inds_mod[0]==ia) & (spot_mod_assign>-1)
    colors0 = inds_mod[2][select0m]
    colors1 = inds_mod[2][select1m]

    scatter_spots(ax[0,0], s_mod[select0m], marker='o', c=colors0, s=100, zorder=0, cmap='tab20', colorbar=True, colorbar_label='grain id')
    scatter_spots(ax[0,1], s_mod[select1m], marker='o', c=colors1, s=100, zorder=0, cmap='tab20', colorbar=True, colorbar_label='grain id')    
    
    select0m = (inds_mod[1]==0) & (inds_mod[0]==ia) & (spot_mod_assign<0)
    select1m = (inds_mod[1]==1) & (inds_mod[0]==ia) & (spot_mod_assign<0)
    colors0 = inds_mod[2][select0m]
    colors1 = inds_mod[2][select1m]
    
    scatter_spots(ax[0,0], s_mod[select0m], marker='d', c=colors0, s=100, zorder=0, cmap='tab20', colorbar=False, colorbar_label='grain id')
    scatter_spots(ax[0,1], s_mod[select1m], marker='d', c=colors1, s=100, zorder=0, cmap='tab20', colorbar=False, colorbar_label='grain id')    
    
    ax[0,0].set_title(f'reflection   om={om:3.0f} deg')
    ax[0,1].set_title(f'transmission om={om:3.0f} deg')


    for a in ax.ravel():
        a.legend(loc='upper right')
        a.set(xlim=lims, ylim=lims)
        a.grid(True)

def scatter_spots_assignment_per_grain_interactive(s_obs, inds_obs, s_mod, inds_mod, s2g_mod_assign, s2g_obs_assign, ig, p_lam=None, omegas=None, color_obs='k', plot_all_bkg=False):
    
    nx, ny = 1, 2; fig, ax = plt.subplots(nx, ny, figsize=(ny * 8, nx * 6), squeeze=False); axc=ax[0,0]; axl=ax[0,:];
        
    lims=[-210, 210]
    step_deg = omegas[1]-omegas[0]
    
    select0o = (inds_obs[1]==0) & (s2g_obs_assign==ig)
    select1o = (inds_obs[1]==1) & (s2g_obs_assign==ig)
    scatter_spots(ax[0,0], s_obs[select0o], marker='x', color=color_obs, s=100, label='peaks (matched)')
    scatter_spots(ax[0,1], s_obs[select1o], marker='x', color=color_obs, s=100, label='peaks (matched)')    

    if plot_all_bkg:
        select0o = (inds_obs[1]==0)
        select1o = (inds_obs[1]==1)
        scatter_spots(ax[0,0], s_obs[select0o], marker='x', color='lightgray', s=100, label='peaks (all)', zorder=-100)
        scatter_spots(ax[0,1], s_obs[select1o], marker='x', color='lightgray', s=100, label='peaks (all)', zorder=-100)    
        
    
    grain_ids = np.unique(inds_mod[2])
    colors = seaborn.color_palette('tab20', len(grain_ids))

    select0m = (inds_mod[1]==0) & (s2g_mod_assign>-1) & (inds_mod[2]==ig)
    select1m = (inds_mod[1]==1) & (s2g_mod_assign>-1) & (inds_mod[2]==ig)
    # colors0 = inds_mod[0][select0m] * step_deg
    # colors1 = inds_mod[0][select1m] * step_deg
    colors0 = p_lam[select0m]
    colors1 = p_lam[select1m]

    scatter_spots(ax[0,0], s_mod[select0m], marker='o', c=colors0, s=100, zorder=0, cmap='Spectral_r', colorbar=True, colorbar_label='projection angle')
    scatter_spots(ax[0,1], s_mod[select1m], marker='o', c=colors1, s=100, zorder=0, cmap='Spectral_r', colorbar=True, colorbar_label='projection angle')    
    
    select0m = (inds_mod[1]==0) & (s2g_mod_assign<0) & (inds_mod[2]==ig)
    select1m = (inds_mod[1]==1) & (s2g_mod_assign<0) & (inds_mod[2]==ig)
    # colors0 = inds_mod[0][select0m] * step_deg
    # colors1 = inds_mod[0][select1m] * step_deg
    colors0 = p_lam[select0m]
    colors1 = p_lam[select1m]
    
    scatter_spots(ax[0,0], s_mod[select0m], marker='d', c=colors0, s=100, zorder=0, cmap='Spectral_r', colorbar=False, colorbar_label='projection angle')
    scatter_spots(ax[0,1], s_mod[select1m], marker='d', c=colors1, s=100, zorder=0, cmap='Spectral_r', colorbar=False, colorbar_label='projection angle')    
    
    ax[0,0].set_title(f'reflection   ')
    ax[0,1].set_title(f'transmission ')


    for a in ax.ravel():
        a.legend(loc='upper right')
        a.set(xlim=lims, ylim=lims)
        a.grid(True)

###########################################################################
###
### loading and others
###
###########################################################################

        
def load_sample(fname):

    with h5py.File(fname, 'r') as f:
        spot_loss = np.array(f['spot_loss'])
        s_mod = np.array(f['assignments/global']['s'])
        i_det_mod = np.array(f['assignments/global/i_det'])[:,0]
        i_ang_mod = np.array(f['assignments/global/i_ang'])[:,0]
        i_grn_mod = np.array(f['assignments/global/i_grn'])[:,0]
        i_hkl_mod = np.array(f['assignments/global/i_hkl'])[:,0]
        inds_mod = (i_ang_mod, i_det_mod, i_grn_mod, i_hkl_mod)

        i_ang_obs = np.array(f['i_ang_obs'])[:,0]
        i_det_obs = np.array(f['i_det_obs'])[:,0]
        inds_obs = (i_ang_obs, i_det_obs)

        p_lam = np.array(f['assignments/global/p_lam'])
        s2g_mod_assign = np.array(f['assignments/global/spot_mod_assign'])
        s2g_obs_assign = np.array(f['assignments/global/spot_obs_assign'])
        s2s_mod_assign = np.array(f['assignments/global/s2s_mod_assign'])
        s2s_obs_assign = np.array(f['assignments/global/s2s_obs_assign'])
        s_obs = np.array(f['s_lab_noisy'])

    return spot_loss, s_obs, s_mod, inds_mod, inds_obs, p_lam, s2g_mod_assign, s2g_obs_assign, s2s_mod_assign, s2s_obs_assign


def print_nspots_per_det(i_det, assign, tag=''):

    n_assigned = np.count_nonzero((assign>=-1))
    n_spots = len(i_det)

    print('spots={} total      n_spots={:>6d} assigned {:>6d} [{:4.2f}%]'.format(tag, n_spots, n_assigned, n_assigned/n_spots*100))
    for i in [0,1]:
        n_assigned = np.count_nonzero((i_det == i) & (assign>=-1))
        n_spots = np.count_nonzero(i_det == i)
        print('spots={} detector={} n_spots={:>6d} assigned {:>6d} [{:4.2f}%]'.format(tag, i, n_spots, n_assigned, n_assigned/n_spots*100))


def plot_grain_stats(s2s_mod_assign, s2s_obs_assign, s2g_mod_assign, s2g_obs_assign, s_obs, s_mod, inds_mod, inds_obs):
    
    select = s2s_mod_assign.copy()
    mask = select<0
    select[mask] = 0
    s_obs_match = s_obs[select]
    l2_mod = np.linalg.norm(s_obs_match-s_mod, axis=1)
    l2_mod[mask]=-99999

    print(f'n_grains={len(np.unique(inds_mod[2]))}')

    n_outliers_obs = np.count_nonzero(s2g_obs_assign<0)
    n_outliers_mod = np.count_nonzero(s2g_mod_assign<0)
    
    n_grains = len(np.unique(s2g_obs_assign))-1
    
    spot_obs_inlier = s2g_obs_assign[s2g_obs_assign>=0]
    spot_mod_inlier = s2g_mod_assign[s2g_mod_assign>=0]
    l2_mod_inlier = l2_mod[s2g_mod_assign>=0]
    grain_ids = np.unique(spot_obs_inlier)
    n_matched_spots_per_grain = np.array([np.count_nonzero(spot_obs_inlier==g) for g in np.unique(spot_obs_inlier)])
    n_model_spots_per_grain = np.array([np.count_nonzero(inds_mod[2]==g) for g in grain_ids])
    median_l2_per_grain = np.array([ np.median(l2_mod_inlier[spot_mod_inlier==g]) for g in np.unique(spot_mod_inlier) ] )
    mean_l2_per_grain = np.array([ np.mean(l2_mod_inlier[spot_mod_inlier==g]) for g in np.unique(spot_mod_inlier) ] )
    
    nx, ny = 1, 2; fig, ax = plt.subplots(nx, ny, figsize=(ny * 8, nx * 6), squeeze=False); axc=ax[0,0]; axl=ax[0,:];
    
    axl[0].plot(grain_ids, n_matched_spots_per_grain, 'd', label='matched')
    axl[0].plot(grain_ids, n_model_spots_per_grain, 's', label='model')
    
    axl[0].set(xlabel='grain id', ylabel='number of spots', ylim=[0,None])

    axl[1].plot(grain_ids, mean_l2_per_grain, 'd', label='mean')
    axl[1].plot(grain_ids, median_l2_per_grain, 's', label='median')
    axl[1].set(xlabel='grain id', ylabel='l2 residual norm per grain [mm]')    
    axl[0].legend()
    axl[1].legend()



def plot_spot_statistics(fname, print_stats=False):

    with h5py.File(fname, 'r') as f:
        obj = np.array(f['objects'])

    ang = np.unique(obj['ang_deg'])
    n_per_ang_det0 = np.array([np.count_nonzero((obj['i_ang']==i)&(obj['i_det']==0)) for i in range(len(ang))])
    n_per_ang_det1 = np.array([np.count_nonzero((obj['i_ang']==i)&(obj['i_det']==1)) for i in range(len(ang))])

    nx, ny = 1, 1; fig, ax = plt.subplots(nx, ny, figsize=(ny * 8, nx * 6), squeeze=False); axc=ax[0,0]; axl=ax[0,:];
    axl[0].plot(n_per_ang_det0, 'o-', label='reflection')
    axl[0].plot(n_per_ang_det1, 'o-', label='transmission')
    axl[0].set(xlabel='projection id', ylabel='n detected spots')
    for a in ax.ravel(): a.legend()

    if print_stats:
        for i in range(len(ang)):
            print(i, n_per_ang_det0[i], n_per_ang_det1[i])

def plot_spot_loss(fname, n_max=None, xscale='linear', yscale='linear'):

    with h5py.File(fname, 'r') as f:
        spot_loss = np.array(f['spot_loss'])        

    spot_loss_diff = lambda spot_loss: spot_loss[1:]/spot_loss[:-1]
    nx, ny = 1, 4; fig, ax = plt.subplots(nx, ny, figsize=(ny * 8, nx * 6), squeeze=False); axc=ax[0,0]; axl=ax[0,:];
    from scipy.interpolate import interp1d
    from scipy.interpolate import UnivariateSpline
    
    x = np.arange(len(spot_loss))
    y = spot_loss - np.min(spot_loss)
    y = y / np.max(y)
    
        # axl[0].plot(spot_loss, 'x')
    min_y = 1e-3

    axl[0].plot(spot_loss, 'x')
    axl[1].plot(y, 'x')
    axl[2].plot(np.diff(y, n=1), 'x')
    axl[2].axhline(min_y)
    axl[2].axhline(-min_y)
    # axl[1].plot(np.diff(y, n=1), '-')
    # n_max_proto = np.nonzero(np.abs(np.diff(y, n=1))<min_y)[0][0]+1
    # print(n_max_proto)

    axl[3].plot(spot_loss_diff(spot_loss), 'x')


    axl[0].set(xlabel='prototype index', ylabel='SPOT loss', xlim=[0,n_max], xscale=xscale, yscale=yscale)
    axl[1].set(xlabel='prototype index', ylabel='SPOT loss (normalized)', xlim=[0,n_max], xscale=xscale, yscale=yscale)
    axl[2].set(xlabel='prototype index', ylabel='SPOT loss frac diff', xlim=[0,n_max], xscale=xscale, yscale=yscale)
    axl[3].set(xlabel='prototype index', ylabel='SPOT loss diff', xlim=[0,n_max], xscale=xscale, yscale=yscale)
    for a in axl: a.grid(True)



###########################################################################
###
### eyeballing
###
###########################################################################

def get_imgs(list_i, det='fwd', fname='/Users/tomaszk/Archive/projects/211129_laue3d/data//transfer_152646_files_f1654478/{prefix}/CoNiGa_ExtrHT_a/CoNiGa_ExtrHT_a_{det}_clean/CoNiGa_ExtrHT_a_{det}_clean_{im:04d}.tif'):
    
    im_step = 1
    from PIL import Image
    imgs = []

    for i in list_i:
        im = im_step*i
        
        if '.h5/' in fname:
            
            fname_h5 = fname.split('.h5/')[0] + '.h5'
            dataset = fname.split('.h5/')[1].format(det=det, im=im)
            with h5py.File(fname_h5, 'r') as f:
                img = np.array(f[dataset])

        elif fname.endswith('.tif'):
            fname = fname.format(det=det, im=im)# = f'{prefix}/CoNiGa_ExtrHT_a/CoNiGa_ExtrHT_a_{det}_clean/CoNiGa_ExtrHT_a_{det}_clean_{im:04d}.tif'
            print(fname)
            with Image.open(fname) as im:
                img = np.array(im, dtype=float)

        imgs += [img]
        
    return imgs

def get_objs(oms, om, spots):

    select = oms==om
    s_ = spots[select]
    objects = dict(x=s_[:,1], y=s_[:,2])
    return objects

def get_objs_se(om, spots_se):
    
    select = (spots_se['ang_deg'] == om) & (spots_se['i_det'] == 1)
    s_ = spots_se[select]
    objects = dict(x=s_['x'], y=s_['y'])
    return objects

def get_objs_mpd(s_det, i_ang, i_det, i_grn, a, d):

    select = (i_ang == a) & (i_det == d)
    print(np.count_nonzero(select))
    s_ = s_det[select]
    g_ = i_grn[select]
    i_ = np.nonzero(select)[0]
    objects = dict(x=s_[:,1], y=s_[:,2], g=g_, i=i_)
    return objects

def plot_ds9_multi(imgs, objects=None, radius=10, lw=2, regs=True):
    
    from astropy.io import fits
        
    for i, im in enumerate(imgs):

        fits.writeto(f'temp{i}.fits', im, overwrite=True)

        with open(f'temp{i}.reg', 'w') as f:
            
            if regs:
                
                obj = objects[i]

                for j in range(len(obj['x'])):
    
                    line = 'point {} {} # color={} point={} width={}'.format(obj['x'][j],  obj['y'][j],  objects[i]['c'][j], objects[i]['m'][j], lw)
                    f.write(line+'\n')

                    line = 'circle {} {} {} # color={} width={}'.format(obj['x'][j],  obj['y'][j], radius,  objects[i]['c'][j], lw)
                    f.write(line+'\n')

                    line = 'text {} {} # text='.format(obj['x'][j]+3,  obj['y'][j]+3) + '{' + 'gr='+str(obj['g'][j]) + ' id='+str(obj['i'][j]) + '}' + ' color={}'.format(objects[i]['c'][j])
                    f.write(line+'\n')

                    
    cmd = 'ds9 -cmap rainbow'
    for i in range(len(imgs)):
        cmd += f' temp{i}.fits '
        if regs:
            cmd += f' -regions temp{i}.reg '
        
    cmd += ' -lock frame image'
    print(cmd)
    os.system(cmd)
        

def eyeball_angle(i, s_obs, inds_obs, s_mod, inds_mod, s2g_mod_assign, s2g_obs_assign,  lw=2, radius=20, om_step=10, det='back', fname='/Users/tomaszk/Archive/projects/211129_laue3d/data//transfer_152646_files_f1654478/CoNiGa_ExtrHT_a/CoNiGa_ExtrHT_a_{det}_clean/CoNiGa_ExtrHT_a_{det}_clean_{im:04d}.tif', add_image='/Users/tomaszk/Archive/projects/211129_laue3d/data/FeGaRwImages/FeGa10A_back.tif'):

    imgs = get_imgs([i], det=det, fname=fname)
    if det=='fwd':
        imgs = [np.fliplr(im) for im in imgs]
    om = i*om_step
    
    objects0 = get_objs_mpd(s_obs, i_ang=inds_obs[0], i_det=inds_obs[1], i_grn=s2g_obs_assign, a=i, d=0 if det=='back' else 1)
    objects1 = get_objs_mpd(s_mod, i_ang=inds_mod[0], i_det=inds_mod[1], i_grn=s2g_mod_assign, a=i, d=0 if det=='back' else 1)
    obj_all = [objects0, objects1]
    colors = ['red', 'blue', 'cyan']
    markers = ['x', 'cross']
    
    objects = dict(x=np.concatenate([o['x'] for o in obj_all]), 
                   y=np.concatenate([o['y'] for o in obj_all]), 
                   c=np.concatenate([[colors[i]]*len(o['x']) for i, o in enumerate(obj_all)]),
                   m=np.concatenate([[markers[i]]*len(o['x']) for i, o in enumerate(obj_all)]),
                   g=np.concatenate([o['g'] for o in obj_all]),
                   i=np.concatenate([o['i'] for o in obj_all]))

    if add_image is not None:
        from PIL import Image
        with Image.open(add_image) as im:
            img = np.array(im, dtype=float)
        imgs.append(img)
        objects_use = [objects, objects]

    else:
        objects_use = [objects]        

    plot_ds9_multi(imgs, objects=objects_use, radius=radius, regs=True, lw=lw)

###########################################################################
###
### 3D sample plots
###
###########################################################################

def load_grain_params(fname):


    with h5py.File(fname, 'r') as f:

        a_est = np.array(f['assignments/global/a'])
        x_est = np.array(f['assignments/global/x'])

    return a_est, x_est

def plotly_scatter3d_sample(a_est, x_est, a_max=0.2, alpha=1, suptitle='', plot_size=500):

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]]) 

    s = 5
    c = np.round(a_est/a_max*255)
    
    d = go.Scatter3d(x=x_est[:,0],
                     y=x_est[:,1],
                     z=x_est[:,2],
                     mode='markers',
                     # marker=dict(size=s*2, color='dodgerblue', opacity=alpha))
                     marker=dict(size=s*2, color=c, opacity=alpha))
    fig.add_trace(d, row=1, col=1)


    # tight layout
    fig.update_layout(margin=dict(l=0.1, r=0.1, b=0.1, t=0.1), 
                      width=plot_size*1.2,
                      height=plot_size,
                      scene=dict(xaxis_title='position x [mm]', yaxis_title='position y [mm]', zaxis_title='position z [mm]'), 
                      title={'text':suptitle, 'xanchor': 'center', 'yanchor': 'top', 'y':0.9, 'x':0.5})
    
    
    fig.show()   

    


###########################################################################
###
### Segmentation plotting
###
###########################################################################


def plot_detections(ax, objects, data_sub, marker_color='black', **kw):
    
    from matplotlib.patches import Ellipse
    pcolormesh_cbar(ax, data_sub, **kw)

    for i in range(len(objects['x'])):

        e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                    width=6*objects['semimajor_sigma'][i],
                    height=6*objects['semiminor_sigma'][i],
                    angle=objects['orientation'][i])

        e.set_facecolor('none')
        e.set_edgecolor(marker_color)
        ax.add_artist(e)
        
    ax.scatter(objects['x'], objects['y'], marker='.', color=marker_color, s=1)
    
def pcolormesh_cbar(ax, im, nskip=10, **kw):

    a = np.arange(0, im.shape[0], nskip)
    cb = ax.pcolormesh(a, a, im[::nskip,::nskip], **kw)
    ax.get_figure().colorbar(cb, ax=ax)
     
def load_segmentation(fname):
    
    with h5py.File(fname, 'r') as f:
        objects = np.array(f['objects'])
    print(f'loaded {fname} with {len(objects)} spots')
    return objects

def load_tiff_image(fname):
    
    from PIL import Image
    im = np.array(Image.open(fname))
    return im

def plot_single_projection(path_spots, path_img_b, path_img_f, ia=0, figscale=1, lims=[0,4000],  **kw):

    def load_h5_image(path_img):

        fname_h5 = path_img.split('.h5/')[0] + '.h5'
        dataset = path_img.split('.h5/')[1].format(ia=ia)
        with h5py.File(fname_h5, 'r') as f:
            img = np.array(f[dataset])

        return img

    
    objects = load_segmentation(path_spots)


    if '.h5/' in path_img_b:

        img_b = load_h5_image(path_img_b)
        img_f = load_h5_image(path_img_f)


    elif path_img_b.endwith('.tif'):

        img_b = load_tiff_image(path_img_b.format(ia))
        img_f = load_tiff_image(path_img_f.format(ia))

    
    nx, ny = 1, 2; fig, ax = plt.subplots(nx, ny, figsize=(ny * 8 * figscale, nx * 6 * figscale), squeeze=False); axc=ax[0,0]; axl=ax[0,:];
    
    select = (objects['i_ang']==ia) & (objects['i_det']==0)
    plot_detections(axl[0], objects[select], img_b, **kw)
    axl[0].set(title=f'reflection, projection {ia}', xlim=lims, ylim=lims)

    select = (objects['i_ang']==ia) & (objects['i_det']==1)
    plot_detections(axl[1], objects[select], img_f, **kw)
    axl[1].set(title=f'transmission, projection {ia}', xlim=lims, ylim=lims)






###########################################################################
###
### Plotly versions
###
###########################################################################

def plotly_grain_stats(s2s_mod_assign, s2s_obs_assign, s2g_mod_assign, s2g_obs_assign, s_obs, s_mod, inds_mod, inds_obs, plot_size=500):
    
    select = s2s_mod_assign.copy()
    mask = select<0
    select[mask] = 0
    s_obs_match = s_obs[select]
    l2_mod = np.linalg.norm(s_obs_match-s_mod, axis=1)
    l2_mod[mask]=-99999

    print(f'n_grains={len(np.unique(inds_mod[2]))}')

    n_outliers_obs = np.count_nonzero(s2g_obs_assign<0)
    n_outliers_mod = np.count_nonzero(s2g_mod_assign<0)
    
    n_grains = len(np.unique(s2g_obs_assign))-1
    
    spot_obs_inlier = s2g_obs_assign[s2g_obs_assign>=0]
    spot_mod_inlier = s2g_mod_assign[s2g_mod_assign>=0]
    l2_mod_inlier = l2_mod[s2g_mod_assign>=0]
    grain_ids = np.unique(spot_obs_inlier)
    n_matched_spots_per_grain = np.array([np.count_nonzero(spot_obs_inlier==g) for g in np.unique(spot_obs_inlier)])
    n_model_spots_per_grain = np.array([np.count_nonzero(inds_mod[2]==g) for g in grain_ids])
    median_l2_per_grain = np.array([ np.median(l2_mod_inlier[spot_mod_inlier==g]) for g in np.unique(spot_mod_inlier) ] )
    mean_l2_per_grain = np.array([ np.mean(l2_mod_inlier[spot_mod_inlier==g]) for g in np.unique(spot_mod_inlier) ] )
    
    # nx, ny = 1, 2; fig, ax = plt.subplots(nx, ny, figsize=(ny * 8, nx * 6), squeeze=False); axc=ax[0,0]; axl=ax[0,:];
    
    # axl[0].plot(grain_ids, n_matched_spots_per_grain, 'd', label='matched')
    # axl[0].plot(grain_ids, n_model_spots_per_grain, 's', label='model')
    
    # axl[0].set(xlabel='grain id', ylabel='number of spots', ylim=[0,None])

    # axl[1].plot(grain_ids, mean_l2_per_grain, 'd', label='mean')
    # axl[1].plot(grain_ids, median_l2_per_grain, 's', label='median')
    # axl[1].set(xlabel='grain id', ylabel='l2 residual norm per grain [mm]')    
    # axl[0].legend()
    # axl[1].legend()

    import plotly.graph_objects as go
    import plotly.subplots as sp

    nx, ny = 1, 2
    fig = sp.make_subplots(rows=nx, cols=ny, subplot_titles=("Number of Spots", "L2 Residual Norm per Grain [mm]"))

    fig.add_trace(go.Scatter(x=grain_ids, y=n_matched_spots_per_grain, mode='markers', marker=dict(symbol='diamond'), name='matched'), row=1, col=1)
    fig.add_trace(go.Scatter(x=grain_ids, y=n_model_spots_per_grain, mode='markers', marker=dict(symbol='square'), name='model'), row=1, col=1)

    fig.add_trace(go.Scatter(x=grain_ids, y=mean_l2_per_grain, mode='markers', marker=dict(symbol='diamond'), name='mean'), row=1, col=2)
    fig.add_trace(go.Scatter(x=grain_ids, y=median_l2_per_grain, mode='markers', marker=dict(symbol='square'), name='median'), row=1, col=2)

    fig.update_xaxes(title_text='grain id', row=1, col=1)
    fig.update_yaxes(title_text='number of spots', range=[0, None], row=1, col=1)

    fig.update_xaxes(title_text='grain id', row=1, col=2)
    fig.update_yaxes(title_text='l2 residual norm per grain [mm]', row=1, col=2)

    fig.update_layout(showlegend=True, 
                      legend=dict(x=0.5, y=-0.1, orientation='h'),
                      width=ny*plot_size,
                      height=nx*plot_size)
    fig.show()



def plotly_scatter_spots_assignment_per_angle(s_obs, inds_obs, s_mod, inds_mod, spot_mod_assign, ind_angle, p_lam, marker_size=10, omegas=None, color_obs='darkgrey',  cmap='tab10', plot_size=500, **kw):


    import plotly.graph_objects as go
    import plotly.subplots as sp


    om = omegas[ind_angle]
    det_ids = np.unique(inds_obs[1])
    nx, ny = 1, len(det_ids)
    fig = sp.make_subplots(rows=1, cols=ny, subplot_titles=([f'detector={d} angle={om}' for d in det_ids]))

    grain_ids = np.unique(inds_mod[2])
    colors = seaborn.color_palette(cmap, np.max(inds_mod[2])+1).as_hex()

    max_x = max( max(np.abs(s_obs[:,1])), max(np.abs(s_mod[:,1])))
    max_y = max( max(np.abs(s_obs[:,2])), max(np.abs(s_mod[:,2])))

    for di in det_ids:

        for gi, ind_grain in enumerate(grain_ids):

            select_m = (inds_mod[1]==di) & (inds_mod[0]==ind_angle) & (inds_mod[2]==ind_grain) & (spot_mod_assign>-1)

            s_ = s_mod[select_m]
            p_ = p_lam[select_m]
            m_ = inds_mod[3][select_m]

            hovertext = [f'x={s__[1]: 2.3f} y={s__[2]: 2.3f} lam={p__: 2.3f} hkl=[{m__}] gi={gi} ' for s__,p__,m__ in zip(s_, p_, m_)]

            fig.add_trace(go.Scatter(x=s_[:,1], y=s_[:,2], 
                          mode='markers', 
                          visible=True,
                          marker=dict(symbol='circle-dot', color=colors[ind_grain], size=marker_size), 
                          name=f'Model {ind_grain: 3d}, Detector {di}, Spots {np.count_nonzero(s_)}',
                          hovertext=hovertext,
                          hoverinfo='text'), 
                          row=1, 
                          col=di+1)


            fig.update_xaxes(title_text='x [mm]', row=1, col=1)
            fig.update_yaxes(title_text='y [mm]', row=1, col=1)
            fig.update_layout({f'xaxis{di+1}': {'range': [-max_x, max_x]}, f'yaxis{di+1}': {'range': [-max_y, max_y]}})


    for di in det_ids:


        select_o = (inds_obs[1]==di) & (inds_obs[0]==ind_angle)  
        s_ = s_obs[select_o]
        fig.add_trace(go.Scatter(x=s_[:,1], y=s_[:,2], 
                      mode='markers', 
                      marker=dict(symbol='x-thin-open', color=color_obs, size=marker_size, line=dict(color='MediumPurple', width=0)), 
                      name=f'Detected ({np.count_nonzero(s_)} spots)'), 
                      row=1, 
                      col=di+1)
    
    buttons = []
    buttons.append(dict(method='update',
                        label='Show all',
                        visible=True,
                        args=[{'visible':[True for x in fig.data]}],
                        ))

    buttons.append(dict(method='update',
                        label='Show none',
                        visible=True,
                        args=[{'visible':['legendonly' for x in fig.data]}],
                        ))

    # fig.update_layout()
    um = [{'buttons':buttons, 'direction': 'd'}]
    fig.update_layout(showlegend=True,
                      legend=dict(
                            yanchor="top",
                            y=0.9,
                            xanchor="left",
                            x=1.01),
                      width=ny*plot_size*1.2,
                      height=nx*plot_size,
                      title_text=f'Spots for angle {om} deg',
                      updatemenus=[dict(
                                        type='buttons',
                                        direction='right',
                                        xanchor='left',
                                        x=1.01,
                                        y=1.0,
                                        showactive=True,
                                        buttons=buttons)],)


    fig.show()
    

def plotly_scatter_spots_assignment_per_grain(s_obs, inds_obs, s_mod, inds_mod, spot_mod_assign, spot_obs_assign, ind_grain, p_lam, marker_size=10, omegas=None, color_obs='darkgrey', cmap='tab10', plot_size=500,  **kw):
    # print('lol')


    import plotly.graph_objects as go
    import plotly.subplots as sp


    
    det_ids = np.unique(inds_obs[1])
    nx, ny = 1, len(det_ids)
    fig = sp.make_subplots(rows=1, cols=ny, subplot_titles=([f'detector={d}' for d in det_ids]))

    grain_ids = np.unique(inds_mod[2])
    angle_ids = np.unique(inds_mod[0])
    colors = np.array(seaborn.color_palette(cmap, max(angle_ids)+1).as_hex())

    max_x = max( max(np.abs(s_obs[:,1])), max(np.abs(s_mod[:,1])))
    max_y = max( max(np.abs(s_obs[:,2])), max(np.abs(s_mod[:,2])))

    for di in det_ids:

        for ai, ind_angle in enumerate(angle_ids):


            select_m = (inds_mod[1]==di) & (inds_mod[2]==ind_grain) &  (inds_mod[0]==ind_angle) & (spot_mod_assign>-1)

            sm_ = s_mod[select_m]
            pm_ = p_lam[select_m]
            mm_ = inds_mod[3][select_m]
            hm_ = [f'x={s__[1]: 2.3f} y={s__[2]: 2.3f} lam={p__: 2.3f} hkl=[{m__}] rot={omegas[ind_angle]} ' for s__,p__,m__ in zip(sm_, pm_, mm_)]

            select_o = (inds_obs[1]==di)  &  (inds_obs[0]==ind_angle)  & (spot_obs_assign==ind_grain)
            so_ = s_obs[select_o]
            ho_ = [f'x={s__[1]: 2.3f} y={s__[2]: 2.3f} rot={omegas[ind_angle]} deg ' for s__ in so_]

            s_ = np.vstack([sm_, so_])
            c_ = [colors[ind_angle]]*len(sm_) + [color_obs]*len(so_)
            hovertext = hm_ + ho_

            symbol = ['circle-dot']*len(sm_) + ['x-thin-open']*len(so_)


            fig.add_trace(go.Scatter(x=s_[:,1], y=s_[:,2], 
                          mode='markers', 
                          visible=True,
                          marker=dict(symbol=symbol, color=c_, size=marker_size), 
                          name=f'Rotation {omegas[ind_angle]: 3d} deg, Detector {di}, Spots {np.count_nonzero(select_m)}/{np.count_nonzero(select_o)}',
                          hovertext=hovertext,
                          hoverinfo='text'), 
                          row=1, 
                          col=di+1)


            fig.update_xaxes(title_text='x [mm]', row=1, col=di)
            fig.update_yaxes(title_text='y [mm]', row=1, col=di)
            fig.update_layout({f'xaxis{di+1}': {'range': [-max_x, max_x]}, f'yaxis{di+1}': {'range': [-max_y, max_y]}})




    buttons = []
    buttons.append(dict(method='update',
                        label='Show all',
                        visible=True,
                        args=[{'visible':[True for x in fig.data]}],
                        ))

    buttons.append(dict(method='update',
                        label='Show none',
                        visible=True,
                        args=[{'visible':['legendonly' for x in fig.data]}],
                        ))

    um = [{'buttons':buttons, 'direction': 'down'}]
    fig.update_layout(showlegend=True,
                      xaxis_range=[-max_x, max_x],
                      yaxis_range=[-max_y, max_y],
                      legend=dict(
                            yanchor="top",
                            y=0.9,
                            xanchor="left",
                            x=1.01),
                      width=ny*plot_size*1.2,
                      height=nx*plot_size,
                      title_text=f'Spots for grain {ind_grain}',
                      updatemenus=[dict(
                                        type='buttons',
                                        direction='right',
                                        xanchor='left',
                                        x=1.01,
                                        y=1.0,
                                        showactive=True,
                                        buttons=buttons)],)


    fig.show()
    

def plotly_spot_loss(fname, n_max=None, xscale='linear', yscale='linear', plot_size=500):
    
    import h5py
    import numpy as np
    from plotly.subplots import make_subplots
    import plotly.express as px
    import plotly.graph_objects as go

    with h5py.File(fname, 'r') as f:
        spot_loss = np.array(f['spot_loss'])
        a_est = np.array(f['assignments/global/a'])
        n_grains = len(a_est)

    spot_loss_diff = lambda spot_loss: spot_loss[1:] / spot_loss[:-1]
    x = np.arange(len(spot_loss))
    y = spot_loss - np.min(spot_loss)
    y = y / np.max(y)

    min_y = 1e-3

    nx, ny = 1, 2;
    fig = make_subplots(rows=nx, cols=ny) 

    # Create subplots
    fig.add_trace(go.Scatter(x=x, y=spot_loss, mode='markers', name=''), row=1, col=1)
    fig.add_trace(go.Scatter(x=x+2, y=spot_loss_diff(spot_loss), mode='markers', name=''), row=1, col=2)
    for i in range(ny):
        fig.add_vline(x=n_grains, row=1, col=i+1)

    fig.update_layout(
        width=ny*plot_size,
        height=nx*plot_size,
        title_text='SPOT Loss Analysis',
        showlegend=False
    ) 

    for i in range(ny):
        fig.update_xaxes(title_text="Number of grains", row=1, col=i+1)
    fig.update_yaxes(title_text="SPOT loss", row=1, col=1)
    fig.update_yaxes(title_text="SPOT loss fractional difference", row=1, col=2)

    # # Show the plot
    fig.show()






# def plot_spot_loss_plotly(fname, n_max=None, xscale='linear', yscale='linear'):    

#     import h5py
#     import numpy as np
#     import plotly.express as px
#     import plotly.graph_objects as go

#     with h5py.File(fname, 'r') as f:
#         spot_loss = np.array(f['spot_loss'])

#     spot_loss_diff = lambda spot_loss: spot_loss[1:] / spot_loss[:-1]
#     x = np.arange(len(spot_loss))
#     y = spot_loss - np.min(spot_loss)
#     y = y / np.max(y)

#     min_y = 1e-3

#     fig = go.Figure()

#     # Create subplots
#     fig.add_trace(go.Scatter(x=x, y=spot_loss, mode='markers', name='SPOT loss'))
#     fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='SPOT loss (normalized)'))
#     fig.add_trace(go.Scatter(x=x[1:], y=np.diff(y, n=1), mode='markers', name='SPOT loss frac diff'))
#     fig.add_trace(go.Scatter(x=x, y=spot_loss_diff(spot_loss), mode='markers', name='SPOT loss diff'))

#     # Add horizontal lines
#     fig.add_shape(go.layout.Shape(
#         type="line",
#         x0=0,
#         x1=n_max if n_max else len(spot_loss),
#         y0=min_y,
#         y1=min_y,
#         line=dict(color="red", dash="dash"),
#     ))
#     fig.add_shape(go.layout.Shape(
#         type="line",
#         x0=0,
#         x1=n_max if n_max else len(spot_loss),
#         y0=-min_y,
#         y1=-min_y,
#         line=dict(color="red", dash="dash"),
#     ))

#     # Update axis properties
#     fig.update_xaxes(title_text='prototype index', range=[0, n_max] if n_max else None, type=xscale)
#     fig.update_yaxes(title_text='SPOT loss', type=yscale)

#     # Set layout options
#     fig.update_layout(
#         width=ny * 8,
#         height=nx * 6,
#         title_text='SPOT Loss Analysis',
#         legend=dict(x=0, y=1),
#         grid=dict(yaxis=dict(showgrid=True, zeroline=False, showline=True)),
#     )

#     # Show the plot
#     fig.show()

# Example usage:
# plot_spot_loss('your_file_name.h5', n_max=100, xscale='linear', yscale='linear')

