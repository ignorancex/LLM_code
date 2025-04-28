import numpy as np
import pandas as pd
from exorings3 import ellipse, ring_patch, PathPatch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.patches import Circle
import matplotlib.animation as pltani
#from IPython import display
import astropy.units as U
import time as T
from os.path import isfile
import pickle
import paths

#Look and make sure data is available in the same directory
path_binned_data = paths.data / 'a21js_Vgf_avgsqrt_bin9days.csv'

a21js_gf_bin = pd.read_csv(f"{path_binned_data}")

##Assign real_data columns into appropriate variables
real_lc_MJD = a21js_gf_bin.iloc[:, 0]
real_lc_flux = a21js_gf_bin.iloc[:, 1]
real_lc_fluxerr = a21js_gf_bin.iloc[:, 2]

#Limb Darkening
def lin_ld(u, r, step = 100):
    """
    u = limb darkening factor
    r = star radius (px)
    step = number of steps to divide between 0 and r
    """
    f_norm = [] #define empty array to store flux values

    R = np.linspace(0, r, step) #define r values to evaluate.
    R_norm = R / r

    #limb darkening function (linear) --> already normalized through R_norm
    for z in R_norm:
        ld = 1 - (u * (1 - np.sqrt(1 - z ** 2)))
        f_norm.append(ld)
        pass
    return f_norm, R_norm


# # Convert Time to Position (x, in px)
def t_to_x(t, v, t0 = None):
    """
    t = epoch (list)
    v = velocity (px/day)
    t0 = time when (x_ring - x_star) = 0 [days]
    """
    
    if t0 == None:
        t0 = np.mean(t)
    x = -(t - t0) * v
    return x


# # Draw Star as Vector
# 
# Using limb darkening profile and cmap
def star_vector(xy, ax, r_star, u, step = 100):
    """
    xy: position of star center in the format of (x, y)
    ax: axis to draw figure
    r_star: star radius, in pixels
    u: limb darkening coefficient, from 0 to 1
    step: number of steps to calculate limb darkening
    """
    
    #get limb darkening
    f_norm, r_norm = lin_ld(u, r_star, step)
    
    #mapping f_norm to colour
    minima = min(f_norm)
    maxima = max(f_norm)
    
    norm = Normalize(vmin = minima, vmax = maxima, clip = True) #normalize list to [0, 1]
    
    mapper = cm.ScalarMappable(norm = norm, cmap = cm.YlOrRd_r) #map normalized f_norm to colourmap
    
    #draw artist (figure)
    for x in range(len(r_norm)):
        r = r_star * r_norm[x]
        starArtist = ax.add_patch(Circle(xy, radius = r, zorder = -20-x, color = mapper.to_rgba(f_norm[x])))
        pass
    
    return starArtist

#function to create star model
def star(grid_yx, r_star, u, step = 100):
    """
    grid_yx: grid to draw star [float, float]
    r_star: radius of the star in pixels [float]
    u: limb darkening factor between 0 - 1 [float]
    step: number of steps to divide between 0 and r_star --> for lin_ld func to work [int]
    plot: generate plot of the star created [True/False]
    """
    #create grid
    Grid_star = np.mgrid[:grid_yx[0], :grid_yx[1]]
    
    #centering star
    yc = (((len(Grid_star[0])) - 1) / 2)
    xc = (((len(Grid_star[0][0])) - 1) / 2)
    Grid_star[0] -= yc
    Grid_star[1] -= xc
    
    #create star mask
    r = np.sqrt(((Grid_star[1])**2 + (Grid_star[0])**2))
    
    #acquire limb darkening profile
    f_norm, rad_norm = lin_ld(u, r_star, step = step)
    
    #denormalizing rad
    rad_denorm = rad_norm * r_star
    
    #create similar object to r
    star_ld = np.zeros_like(r)
    
    #apply masking and assign values
    for _ in reversed(range(len(rad_denorm))): #must be reversed because the order of masking is from the largest to the smallest  
        #create star mask
        star = (r < rad_denorm[_])
        star_ld[star] = f_norm[_]
        pass
    
    total_flux = np.sum(star_ld)
    
    return total_flux, star_ld, Grid_star[0]


# # Create Rings/Disks (Old Functions)
# 
# 1 ring is created by using 2 disks with the inner disk is transparent
def onedisk(grid_yx, a, i, phi, tr, x_disk = 0., y_disk = 0., plot = False):
    """
    grid_yx: grid to draw disk [float, float]
    a: semimajor axis of the disk [float]
    i: inclination of the disk [float]
    phi: rotation of the disk [float]
    tr: transmissivity of the disk, between 0 - 1 [float]
    x_disk, y_disk: center position of the disk [float][float]
    
    """
    #create grid
    Grid_disk = np.mgrid[:grid_yx[0], :grid_yx[1]]
    
    #Centering coordinate
    #(xc, yc) = (0, 0) at the center of the grid
    xc = (((len(Grid_disk[0][0])) - 1) / 2)
    yc = (((len(Grid_disk[0])) - 1) / 2)
    
    #positioning disk based on the new center
    Grid_disk[0] -= (yc + y_disk)
    Grid_disk[1] -= (xc + x_disk)
    
    #create disk
    im1 = ellipse(Grid_disk, i, phi)
    
    #create disk mask
    disk = (im1 < a) 
    
    #apply transmissivity value
    disk_tr = np.ones_like(im1)
    disk_tr[disk] = tr
    
    if plot == True:
        plt.figure(figsize = (10, 5))
        plt.imshow(disk_tr, origin = 'lower', vmin = 0, vmax = 1)
        plt.colorbar(location = 'right')
        pass

    return disk_tr


# ## 1b. Combine Star + Disk

# In[9]:


def onedisk_model_stat(grid_yx, star_flux, star_model, a, i, phi, tr, x_disk, y_disk):
    """
    star_flux: total flux if star not eclipsed [float]
    star_model: meshgrid of star model [obj]
    
    Notes:
    Both star_flux and star_model need to be calculated using star function
    """
    #create disk
    disk_model = onedisk(grid_yx, a, i, phi, tr, x_disk, y_disk)
    
    #combine star and disk
    star_disk = star_model * disk_model
    
    #calculate normalized combined flux
    f_norm = np.sum(star_disk)/star_flux
        
    return f_norm, star_model, disk_model 


# ## 1c. Create Moving Disk and Calculate Flux
def onedisk_model_mov(a, i, phi, tr, y_disk, v, t0, t = real_lc_MJD, r_star = 15, u = 0.5, step = 100, 
                      t_end = 0, t_step = 0, save_plot = False, plot_title = False):
    """
    grid_yx: grid to draw model in px [float, float]
    r_star: radius of star in px [float]
    u: limb darkening factor between 0 and 1 [float]
    step: limb darkening model step [int]
    a: semimajor axis of disk [float]
    i: inclination of disk [float]
    phi: rotation of disk [float]
    tr: transmissivity, from 0 (opaque) to 1 (transparent)
    y_disk: distance between disk center and star center in px [float]
    t: time in days [array]
    v: disk velocity in px/day [float]
    t0: time when (xc_disk - xc_star) = 0 [float]
    """
    grid_yx = [32., 32.]
    
    #create star
    star_flux, star_model, g = star(grid_yx, r_star, u, step)
    
    #convert t to x     
    if t_end != 0:
        #additional time for prediction when the eclipse ends
        t_pred = np.arange(max(t)+t_step, t_end+t_step, t_step)
        #concatenate t and t_pred
        t_tot = np.concatenate((t, t_pred))
        pass
    else:
        t_tot = t
    
    x = t_to_x(t_tot, v, t0)
    
    #set empty list to hold values
    flux = []
    mjd = t_tot
    resi = []
    
    #create star+ring
    j = 0
    
    for x_disk in x:
        f_norm, __, ___ = onedisk_model_stat(grid_yx, star_flux, star_model, a, i, phi, tr, x_disk, y_disk)
        
        flux.append(f_norm)
        
        #calculate residue
        if t_tot[j] <= max(t):
            f_res = f_norm - real_lc_flux[j]
            resi.append(f_res)
            j += 1
            pass
        else:
            pass
    
    
    #calculate chi-squared value
    n_chi2 = ((flux[:len(t)] - real_lc_flux)**2) / (real_lc_fluxerr ** 2)
    chi2 = np.sum(n_chi2)
    
    print('-' * 40)
    print(f'Calculated chi-squared value: {chi2:.2f}')
    print('-' * 40)
    
    ##PLOTTING
    fig2, (ax_1, ax_2) = plt.subplots(nrows = 2, figsize = (8, 5), gridspec_kw = {'height_ratios': [1.5, 0.5]}, sharex = True)

    #plot bg real data
    real_data = ax_1.errorbar(x = real_lc_MJD, y = real_lc_flux, 
                                yerr = real_lc_fluxerr, c = 'r', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.5, label = 'real observation')

    #plot model scatter plot
    model = ax_1.plot(mjd[:len(t)], flux[:len(t)], color = 'blue', linewidth = 3, alpha = 0.5, label = 'model', zorder = 5)
    ax_1.set_xlim((min(mjd)-10), (max(mjd)))
    ax_1.set_ylim((min(flux) - 0.025), 1.05)
    ax_1.set_ylabel('Normalized Flux')
#     ax_1.set_xlabel('MJD [days]')
    ax_1.tick_params(direction = 'in')
    # ax_1.set_title(f'2 Rings Model \n a_ring1 = {a_ring}px, incl. = {i_test}deg, rot. = {phi_test}deg, trans. 1 = {tr*100}%, \n a_ring2 = {a_ring2} px, trans. 2 = {tr2*100}%, \n R_star = {r_star} px, ring_y_offset = {yc_ring} px, cons. x-axis = {c} px, u = {u}')

    #plot residuals
    y0 = ax_2.hlines(0, xmin = (min(mjd)-50), xmax = (max(mjd)+50), colors = 'grey', linestyles = '--')
    resids = ax_2.errorbar(x = real_lc_MJD, y = resi, yerr = real_lc_fluxerr, c = 'b', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.5)
    ax_2.set_xlim((min(mjd)-10), (max(mjd)))
    ax_2.set_xlabel('MJD [HJD-2400000.5 days]')
    ax_2.set_ylabel('Residuals')
    ax_2.tick_params(direction = 'in')
    
    #Predict end of eclipse based on predicted flux
    if t_end != 0:
        l = len(t)
        while flux[l] < 1.0:
            l += 1
            if l == len(t_tot):
                print('=' * 40)
                print('Predicted end of eclipse not reached! Use longer additional time!')
                print('=' * 40)
                break
            pass
        print(f'Forecasted end of eclipse is around MJD = {t_tot[l]:.3f} days.')
        print('-' * 40)
    
        #Predict eclipse length
        k = 0
        while flux[k] == 1.0:
            k += 1
            pass
        print(f'Eclipse started at around MJD = {t_tot[k]:.3f} days.')
        ecl_l = t_tot[l] - t_tot[k]
        print(f'Predicted eclipse length is around {ecl_l:.3f} days (data resolution: {t_step} days).')
        print('-' * 40)

    if t_end != 0:
        pred = ax_1.plot(t_pred, flux[(len(t)):], color = 'blue', linestyle = '--', label = 'predicted data')
        x1 = ax_1.vlines(t_tot[l], ymin = (min(flux) - 0.05), ymax = 1.05, colors = 'grey', linestyles = '-', label = f'Pred. end of eclipse \n(resolution: {t_step} days)')
        if plot_title == True:
            ax_1.set_title(f'r_ring_outer = {a1}px, $i$ = {i}$^\circ$, $\phi$ = {phi}$^\circ$, tr._ring_outer = {tr1}%, w_ring_outer = {w1}px, \nr_ring_inner = {a3}px, tr._ring_inner = {tr3}%, w_ring_inner = {w3}px, \ny_ring = {y_disk}px, $v$ = {v}px/day, $t_0$ = {t0}days \nPred. end of eclipse = {t_tot[l]:.3f} days, t_resol. (bin) = {t_step} days, \ntr_star = {r_star} px, limb dark. coef. = {u}, mgrid size = [200, 600]')
            pass
        pass
    else:
        if plot_title == True:
            ax_1.set_title(f'r_ring_outer = {a1}px, $i$ = {i}$^\circ$, $\phi$ = {phi}$^\circ$, tr._ring_outer = {tr1}%, w_ring_outer = {w1}px, \nr_ring_inner = {a3}px, tr._ring_inner = {tr3}%, w_ring_inner = {w3}px, \ny_ring = {y_disk}px, $v$ = {v}px/day, $t_0$ = {t0}days \ntr_star = {r_star}px, limb dark. coef. = {u}, mgrid size = [200, 600]')
            pass
        pass
    
    #set legend and tight layout
    ax_1.legend()
    ax_1.grid(alpha = 0.5)
    plt.tight_layout()
    
    if save_plot == True:
        plt.savefig(f"onedisk_fit.png", dpi = 300, facecolor = 'white')
    
    #plt.show()
    
    return mjd, flux, resi, fig2


# ## 2a. Create 2 Disks
# 
# **Notes**
# 
# To create 1 ring, set tr2 = 1.0
def twodisk(grid_yx, a1, i, phi, tr1, w, tr2, x_disk = 0., y_disk = 0., plot = False):
    """
    [input]:
    
    grid_yx: grid to draw disk [float, float]
    a1: semimajor axis of the disk [float]
    i: inclination of the disk [float]
    phi: rotation of the disk [float]
    tr1: transmissivity of the disk, between 0 - 1 [float]
    w: ring width [float]
    x_disk, y_disk: center position of the disk [float][float]
    plot: plot the ring model; default: False [boolean]
    ==========================================================
    [return]:
    
    disk_tr: disk model (obj)
    """
    #create grid
    Grid_disk = np.mgrid[:grid_yx[0], :grid_yx[1]]
    
    #Centering coordinate
    #(xc, yc) = (0, 0) at the center of the grid
    xc = (((len(Grid_disk[0][0])) - 1) / 2)
    yc = (((len(Grid_disk[0])) - 1) / 2)
    
    #positioning disk based on the new center
    Grid_disk[0] -= (yc + y_disk)
    Grid_disk[1] -= (xc + x_disk)
    
    #create disk
    im1 = ellipse(Grid_disk, i, phi)
    
    #create disk mask
    disk1 = (im1 < a1)
    disk2 = (im1 < (a1 - w))
    
    #apply transmissivity value
    disk_tr = np.ones_like(im1)
    disk_tr[disk1] = tr1
    disk_tr[disk2] = tr2
    
    if plot == True:
        plt.figure(figsize = (10, 5))
        plt.imshow(disk_tr, origin = 'lower', vmin = 0, vmax = 1)
        plt.colorbar(location = 'right')
        pass

    return disk_tr


# ## 2b. Combine Star + Disks
#function to create 2 disks static model
def twodisk_model_stat(grid_yx, star_flux, star_model, a1, i, phi, tr1, w, tr2, x_disk, y_disk):
    """
    star_flux: total flux if star not eclipsed [float]
    star_model: meshgrid of star model [obj]
    
    Notes:
    Both star_flux and star_model need to be calculated using star function
    """
    #create disk
    disk_model = twodisk(grid_yx, a1, i, phi, tr1, w, tr2, x_disk, y_disk)
    
    #combine star and disk
    star_disk = star_model * disk_model
    
    #calculate normalized combined flux
    f_norm = np.sum(star_disk)/star_flux
        
    return f_norm, star_model, disk_model 


# ## 2c. Create Moving Disks and Calculate Flux
def twodisk_model_mov(a1, i, phi, tr1, w, tr2, y_disk, v, t0, t = real_lc_MJD, 
                      r_star = 15, u = 0.5, step = 100, t_end = 0, t_step = 0,
                      save_plot = False, plot_title = False):
    """
    grid_yx: grid to draw model in px [float, float]
    r_star: radius of star in px [float]
    u: limb darkening factor between 0 and 1 [float]
    step: limb darkening model step [int]
    a1: semimajor axis of disk 1 (outer disk) [float]
    i: inclination of disk [float]
    phi: rotation of disk [float]
    tr1: transmissivity of disk 1, from 0 (opaque) to 1 (transparent) [float]
    tr2: transmissivity of disk 2, from 0 (opaque) to 1 (transparent) [float]
    w: width of ring in px [float]
    tr2: transmissivity of disk 2, from 0 (opaque) to 1 (transparent) [float]
    y_disk: distance between disk center and star center in px [float]
    t: time in days [array]
    v: disk velocity in px/day [float]
    t0: time when (xc_disk - xc_star) = 0 [float]
    save_plot: save the light curve plot; default: False [boolean]
    plot_title: print the title of the plot; default: False. If True and `save_plot` = True, the title will also be saved 
    alongside the plot. [boolean]
    ===============================================================
    [return]:
    
    mjd: array of MJD
    flux: array of normalized flux
    resi: array of residuals
    """
    grid_yx = [32., 32.]
    
    #create star
    star_flux, star_model, grid = star(grid_yx, r_star, u, step)
    
    #convert t to x     
    if t_end != 0:
        #additional time for prediction when the eclipse ends
        t_pred = np.arange(max(t)+t_step, t_end+t_step, t_step)
        #concatenate t and t_pred
        t_tot = np.concatenate((t, t_pred))
        pass
    else:
        t_tot = t
    
    x = t_to_x(t_tot, v, t0)
    
    #set empty list to hold values
    flux = []
    mjd = t_tot
    resi = []
    
    #create star+ring
    j = 0
    
    for x_disk in x:
        f_norm, __, ___ = twodisk_model_stat(grid_yx, star_flux, star_model, a1, i, phi, tr1, w, tr2, x_disk, y_disk)
        
        flux.append(f_norm)
        
        #calculate residue
        if t_tot[j] <= max(t):
            f_res = f_norm - real_lc_flux[j]
            resi.append(f_res)
            j += 1
            pass
        else:
            pass
    
    n_chi2 = ((flux[:len(t)] - real_lc_flux)**2) / (real_lc_fluxerr ** 2)
    chi2 = np.sum(n_chi2)
    
    print('-' * 40)
    print(f'Calculated chi-squared value: {chi2:.2f}')
    print('-' * 40)
    
    ##PLOTTING
    fig2, (ax_1, ax_2) = plt.subplots(nrows = 2, figsize = (8, 5), gridspec_kw = {'height_ratios': [1.5, 0.5]}, sharex = True)

    #plot bg real data
    real_data = ax_1.errorbar(x = real_lc_MJD, y = real_lc_flux, 
                                yerr = real_lc_fluxerr, c = 'r', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.5, label = 'real observation')

    #plot model scatter plot
    model = ax_1.plot(mjd[:len(t)], flux[:len(t)], color = 'blue', linewidth = 3, alpha = 0.5, label = 'model', zorder = 5)
    ax_1.set_xlim((min(mjd)-10), (max(mjd)))
    ax_1.set_ylim((min(flux) - 0.025), 1.05)
    ax_1.set_ylabel('Normalized Flux')
    ax_1.tick_params(direction = 'in')

    #plot residuals
    y0 = ax_2.hlines(0, xmin = (min(mjd)-50), xmax = (max(mjd)+50), colors = 'grey', linestyles = '--')
    resids = ax_2.errorbar(x = real_lc_MJD, y = resi, yerr = real_lc_fluxerr, c = 'b', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.5)
    ax_2.set_xlim((min(mjd)-10), (max(mjd)))
    ax_2.set_xlabel('MJD [HJD-2400000.5 days]')
    ax_2.set_ylabel('Residuals')
    ax_2.tick_params(direction = 'in')

    #Predict end of eclipse based on predicted flux
    if t_end != 0:
        l = len(t)
        while flux[l] < 1.0:
            l += 1
            if l == len(t_tot):
                print('=' * 40)
                print('Predicted end of eclipse not reached! Use longer additional time!')
                print('=' * 40)
                break
            pass
        print(f'Forecasted end of eclipse is around MJD = {t_tot[l]:.3f} days.')
        print('-' * 40)
    
        #Predict eclipse length
        k = 0
        while flux[k] == 1.0:
            k += 1
            pass
        print(f'Eclipse started at around MJD = {t_tot[k]:.3f} days.')
        ecl_l = t_tot[l] - t_tot[k]
        print(f'Predicted eclipse length is around {ecl_l:.3f} days (data resolution: {t_step} days).')
        print('-' * 40)
    
    if t_end != 0:
        pred = ax_1.plot(t_pred, flux[(len(t)):], color = 'blue', linestyle = '--', label = 'predicted data')
        x1 = ax_1.vlines(t_tot[l], ymin = (tr1 - 0.05), ymax = 1.05, colors = 'grey', linestyles = '-', label = f'Pred. end of eclipse \n(resolution: {t_step} days)')
        if plot_title == True:
            ax_1.set_title(f'r_ring = {a1}px, $i$ = {i}$^\circ$, $\phi$ = {phi}$^\circ$, tr. = {tr1}%, \nw_ring = {w}px, y_ring = {y_disk} px, $v$ = {v} px/day, $t_0$ = {t0} days \nPred. end of eclipse = {t_tot[l]:.3f} days, t_resol. (bin) = {t_step} days, \ntr_star = {r_star} px, limb dark. coef. = {u}, mgrid size = [200, 600]')
            pass
        pass
    else:
        if plot_title == True:
            ax_1.set_title(f'r_ring = {a1}px, $i$ = {i}$^\circ$, $\phi$ = {phi}$^\circ$, tr. = {tr1}%, \nw_ring = {w}px, y_ring = {y_disk} px, $v$ = {v} px/day, $t_0$ = {t0} days \ntr_star = {r_star} px, limb dark. coef. = {u}, mgrid size = [200, 600]')
            pass
        pass
    
    #set legend and tight layout
    ax_1.legend()
    ax_1.grid(alpha = 0.5)
    plt.tight_layout()
    
    if save_plot == True:
        plt.savefig(f"twodisk_fit.png", dpi = 300, facecolor = 'white')
    
    #plt.show()
    
    return mjd, flux, resi, fig2


# ## 3a. Create 4 Disks (2 Rings)
# 
# **Notes**
# 
# Here, to create 2 rings, set tr2 and tr4 = 1.0
def fourdisk(grid_yx, a1, i, phi, tr1, w1, tr2, a3, tr3, w3, tr4, x_disk = 0., y_disk = 0., plot = False):
    """
    grid_yx: grid to draw disk [float, float]
    a: semimajor axis of the disk [float]
    i: inclination of the disk [float]
    phi: rotation of the disk [float]
    tr: transmissivity of the disk, between 0 - 1 [float]
    w: ring width [float]
    x_disk, y_disk: center position of the disk [float][float]
    
    The numbering might cause confusion, but basically, the outer ring is numbered 1 
    while the inner ring is numbered 3
    """
    #create grid
    Grid_disk = np.mgrid[:grid_yx[0], :grid_yx[1]]
    
    #Centering coordinate
    #(xc, yc) = (0, 0) at the center of the grid
    xc = (((len(Grid_disk[0][0])) - 1) / 2)
    yc = (((len(Grid_disk[0])) - 1) / 2)
    
    #positioning disk based on the new center
    Grid_disk[0] -= (yc + y_disk)
    Grid_disk[1] -= (xc + x_disk)
    
    #create disk
    im1 = ellipse(Grid_disk, i, phi)
    
    #create disk mask
    disk1 = (im1 < a1) #outermost
    disk2 = (im1 < (a1-w1)) #inner 1
    disk3 = (im1 < (a3)) #inner 2
    disk4 = (im1 < (a3-w3)) #inner 3
    
    #apply transmissivity value
    disk_tr = np.ones_like(im1)
    disk_tr[disk1] = tr1
    disk_tr[disk2] = tr2
    disk_tr[disk3] = tr3
    disk_tr[disk4] = tr4
    
    if plot == True:
        plt.figure(figsize = (10, 5))
        plt.imshow(disk_tr, origin = 'lower', vmin = 0, vmax = 1)
        plt.colorbar(location = 'right')
        pass

    return disk_tr


# ## 3b. Combine Star + Rings
def fourdisk_model_stat(grid_yx, star_flux, star_model, a1, i, phi, tr1, w1, tr2, a3, tr3, w3, tr4, x_disk, y_disk):
    """
    star_flux: total flux if star not eclipsed [float]
    star_model: meshgrid of star model [obj]
    
    Notes:
    Both star_flux and star_model need to be calculated using star function
    """
    #create disk
    disk_model = fourdisk(grid_yx, a1, i, phi, tr1, w1, tr2, a3, tr3, w3, tr4, x_disk, y_disk)
    
    #combine star and disk
    star_disk = star_model * disk_model
    
    #calculate normalized combined flux
    f_norm = np.sum(star_disk)/star_flux
        
    return f_norm, star_model, disk_model 


# ## 3c. Create Moving Rings and Calculate Flux
def fourdisk_model_mov(a1, i, phi, tr1, w1, a3, tr3, w3, y_disk, v, t0, t = real_lc_MJD, 
                       tr2 = 1.0, tr4 = 1.0, r_star = 50, u = 0.5, step = 100, t_end = 0, t_step = 0, 
                       plot = False, save_plot = False, plot_title = False):
    """
    a1: semimajor axis of disk 1 (ring 1; outermost) [float]
    i: inclination of disk [float]
    phi: rotation of disk [float]
    tr1: transmissivity of disk 1, from 0 (opaque) to 1 (transparent) [float]
    w1: width of disk 1 (ring 1; outermost) in px [float]
    a3: semimajor axis of disk 3 (ring 2; inner) [float]
    tr3: transmissivity of disk 3, from 0 (opaque) to 1 (transparent) [float]
    w3: width of disk 3 (ring 2; inner) in px [float]
    y_disk: distance between disk center and star center in px [float]
    t: time in days [array]
    v: disk velocity in px/day [float]
    t0: time when (xc_disk - xc_star) = 0 [float]
    r_star: radius of star in px [float]
    u: limb darkening factor between 0 and 1 [float]
    step: limb darkening model step [int]
    t_end: MJD when the model should finish, must be >= 0. If set to 0, no additional MJD will be calculated after the last MJD data [float]
    t_step: MJD step after the last MJD data towards t_end [int]
    plot: whether to plot the light curve and model or not; default: False [boolean]
    save_plot: whether to save the plot or not; default: False [boolean]
    plot_title: whether to display the plot title or not; default: False [boolean]
    """
    #create grid
    gridx = float(r_star * 2 + 2)
    gridy = gridx
    
    grid_yx = [gridy, gridx]
    
    #create star
    star_flux, star_model, g = star(grid_yx, r_star, u, step)
    
    #convert t to x     
    if t_end != 0:
        #additional time for prediction when the eclipse ends
        t_pred = np.arange(max(t)+t_step, t_end+t_step, t_step)
        #concatenate t and t_pred
        t_tot = np.concatenate((t, t_pred))
        pass
    else:
        t_tot = t
    
    x = t_to_x(t_tot, v, t0)
    
    #set empty list to hold values
    flux = []
    mjd = t_tot
    resi = []
    
    #create star+ring
    j = 0
    
    for x_disk in x:
        f_norm, __, ___ = fourdisk_model_stat(grid_yx, star_flux, star_model, a1, i, phi, tr1, w1, tr2, a3, tr3, w3, tr4, x_disk, y_disk)
        
        flux.append(f_norm)
        
        #calculate residue
        if t_tot[j] <= max(t):
            f_res = f_norm - real_lc_flux[j]
            resi.append(f_res)
            j += 1
            pass
        else:
            pass
    
    
    #calculate chi-squared value
    n_chi2 = ((flux[:len(t)] - real_lc_flux)**2) / (real_lc_fluxerr ** 2)
    chi2 = np.sum(n_chi2)
    
    #calculate reduced chi2 value
    red_chi2 = chi2 / len(real_lc_flux)    
    
    print(f'Calculated chi-squared value: {chi2:.2f}')
    print(f'Calculated reduced chi-squared value: {red_chi2:.2f}')
    print('-' * 40)
    
    #Predict end of eclipse based on predicted flux
    if t_end != 0:
        #Predict transit start
        k = 0
        while flux[k] == 1.0:
            k += 1
            pass
        print(f'Transit started at around MJD = {t_tot[k]:.3f} days.')
        
        #Predict transit end 
        l = len(t)
        token = 0 #to check if the break condition was fulfilled. Break condition fulfilled --> token = 1 (see below)
        while flux[l] < 1.0:
            l += 1
            if l == len(t_tot):
                print('=' * 40)
                print('Predicted end of transit not reached! Use longer additional time!')
                print('=' * 40)
                token = 1
                break
            pass
        
        if token != 1:
            print(f'Forecasted end of transit is around MJD = {t_tot[l]:.3f} days.')
            ecl_l = t_tot[l] - t_tot[k]
            print(f'Predicted transit length is around {ecl_l:.3f} days\n(data resolution past last MJD data: {t_step} days).')
            print('=' * 40)
    
    if plot == True:
        ##PLOTTING
        fig2, (ax_1, ax_2) = plt.subplots(nrows = 2, figsize = (8, 5), gridspec_kw = {'height_ratios': [1.5, 0.5]}, sharex = True)

        #plot bg real data
        real_data = ax_1.errorbar(x = real_lc_MJD, y = real_lc_flux, 
                                    yerr = real_lc_fluxerr, c = 'r', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.5, label = 'real observation')

        #plot model scatter plot
        model = ax_1.plot(mjd[:len(t)], flux[:len(t)], color = 'blue', linewidth = 3, alpha = 0.5, label = 'model', zorder = 5)
        ax_1.set_xlim((min(mjd)-10), (max(mjd)))
        ax_1.set_ylim((min(flux) - 0.025), 1.05)
        ax_1.set_ylabel('Normalized Flux')
    #     ax_1.set_xlabel('MJD [days]')
        ax_1.tick_params(direction = 'in')
        # ax_1.set_title(f'2 Rings Model \n a_ring1 = {a_ring}px, incl. = {i_test}deg, rot. = {phi_test}deg, trans. 1 = {tr*100}%, \n a_ring2 = {a_ring2} px, trans. 2 = {tr2*100}%, \n R_star = {r_star} px, ring_y_offset = {yc_ring} px, cons. x-axis = {c} px, u = {u}')

        #plot residuals
        y0 = ax_2.hlines(0, xmin = (min(mjd)-50), xmax = (max(mjd)+50), colors = 'grey', linestyles = '--')
        resids = ax_2.errorbar(x = real_lc_MJD, y = resi, yerr = real_lc_fluxerr, c = 'b', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.5)
        ax_2.set_xlim((min(mjd)-10), (max(mjd)))
        ax_2.set_xlabel('MJD')
        ax_2.set_ylabel('Residuals')
        ax_2.tick_params(direction = 'in')

        if t_end != 0:
            pred = ax_1.plot(t_pred, flux[(len(t)):], color = 'blue', linestyle = '--', label = 'predicted data')
            if token != 1:
                x1 = ax_1.vlines(t_tot[l], ymin = (min(flux) - 0.05), ymax = 1.05, colors = 'grey', linestyles = '-', label = f'Pred. end of eclipse \n(resolution: {t_step} days)')
            if plot_title == True:
                ax_1.set_title(f'r_ring_outer = {a1}px, $i$ = {i}$^\circ$, $\phi$ = {phi}$^\circ$, tr._ring_outer = {tr1}%, w_ring_outer = {w1}px, \nr_ring_inner = {a3}px, tr._ring_inner = {tr3}%, w_ring_inner = {w3}px, \ny_ring = {y_disk}px, $v$ = {v}px/day, $t_0$ = {t0}days \nPred. end of eclipse = {t_tot[l]:.3f} days, t_resol. (bin) = {t_step} days, \ntr_star = {r_star} px, limb dark. coef. = {u}, mgrid size = [200, 600]')
                pass
            pass
        else:
            if plot_title == True:
                ax_1.set_title(f'r_ring_outer = {a1}px, $i$ = {i}$^\circ$, $\phi$ = {phi}$^\circ$, tr._ring_outer = {tr1}%, w_ring_outer = {w1}px, \nr_ring_inner = {a3}px, tr._ring_inner = {tr3}%, w_ring_inner = {w3}px, \ny_ring = {y_disk}px, $v$ = {v}px/day, $t_0$ = {t0}days \ntr_star = {r_star}px, limb dark. coef. = {u}, mgrid size = [200, 600]')
                pass
            pass

        #set legend and tight layout
        ax_1.legend()
        ax_1.grid(alpha = 0.5)
        plt.tight_layout()

        if save_plot == True:
            plt.savefig(f"tworings_fit.png", dpi = 300, facecolor = 'white')

        #plt.show()
        pass
    else:
        pass
    
    return mjd, flux, resi, chi2

# #test if function is working
print('Check if specific moving model for 2 rings is working properly')
print('See the functions for the values used')
print('=' * 40)

mjd, flux, resi, _ = fourdisk_model_mov(780, 81.5, 7.5, 0.99, 205, 450.811, 0.66, 
                                        135.543, 80, 0.35, 60480.801, t_end = 62500, 
                                        t_step = 10, r_star = 50, plot = False)

# # Create Rings/Disks (New Function)
# 
# **Notes**
# 
# This function supports the creation of unlimited number of rings/disks.
# 
# Also, I'm not sure if this function can be used for emcee (MCMC). Because some variables accept
# list/array and not a single value (like int of float).

def disk_ring(grid_yx, a, w, tr, i, phi, x_disk = 0., y_disk = 0., plot = False):
    """
    grid_yx: grid to draw disk [float, float]
    a: semimajor axis of the disk (just measuring the outermost edge of the largest disk) [float]
    i: inclination of the disk [float]
    phi: rotation of the disk [float]
    tr: transmissivity of the disk, between 0 - 1. number of tr must be equal to number of (w + 1) [list]
    w: ring width [list]
    x_disk, y_disk: center position of the disk [float][float]
    
    Notes:
    The way the size of the inner disks/rings are calculated is as follows. The variable "a" will measure the 
    size of the outermost disk/ring. The subsequent sizes of the inner disks/rings are calculated by subtracting
    "a" with the "w" values. Suppose "a" is 150 and "w" is [20, 30], then the outer disk/ring will have a
    measured width of 20. The inner disk/ring will start from 130-20 = 110 and has a width of 30. The most inner 
    disk will start from 110-30 = 80.
    """
    #create grid
    Grid_disk = np.mgrid[:grid_yx[0], :grid_yx[1]]
    
    #Centering coordinate
    #(xc, yc) = (0, 0) at the center of the grid
    xc = (((len(Grid_disk[0][0])) - 1) / 2)
    yc = (((len(Grid_disk[0])) - 1) / 2)
    
    #positioning disk based on the new center
    Grid_disk[0] -= (yc + y_disk)
    Grid_disk[1] -= (xc + x_disk)
    
    #create disk
    im1 = ellipse(Grid_disk, i, phi)
    
    #creating sim object like the disk mask
    disk_tr = np.ones_like(im1)
    
    #creating the other disk masks
    for _ in range(len(w)):
        if _ == 0:
            disk = (im1 < a)
            disk_tr[disk] = tr[_]
            disk = (im1 < (a - w[_])) #first inner disk
            disk_tr[disk] = tr[(_+1)]
            a = a - w[_]
            pass
        else:
            disk = (im1 < (a - w[_])) #subsequent inner disks
            disk_tr[disk] = tr[(_+1)]
            a -= w[_]
            pass
    
    if plot == True:
        plt.figure(figsize = (10, 5))
        plt.imshow(disk_tr, origin = 'lower', vmin = 0, vmax = 1)
        plt.colorbar(location = 'right')
        plt.grid(alpha = 0.7, ls = '--')
        pass

    return disk_tr


# ## 4b. Combine Star and Disk/Ring
def disk_ring_stat(grid_yx, star_flux, star_model, a, w, tr, i, phi, x_disk, y_disk, plot = False):
    """
    star_flux: total flux if star not eclipsed [float]
    star_model: meshgrid of star model [obj]
    
    Notes:
    Both star_flux and star_model need to be calculated using star function
    """
    #create disk
    disk_model = disk_ring(grid_yx, a, w, tr, i, phi, x_disk, y_disk, plot = False)
    
    #combine star and disk
    star_disk = star_model * disk_model
    
    #calculate normalized combined flux
    f_norm = np.sum(star_disk)/star_flux
    
    if plot == True:
        plt.figure(figsize = (10, 5))
        plt.imshow(star_model + disk_model, origin = 'lower', vmin = 0, vmax = 2)
        plt.colorbar(location = 'right')
        plt.show()
        
    return f_norm, star_model, disk_model 


# ## 4c. Create Moving Rings and Calculate Flux
def disk_ring_mov(a, w, tr, i, phi, y_disk, v, t0, t = real_lc_MJD, 
                  r_star = 50, u = 0.5, step = 100, t_end = 0, t_step = 0,
                  plot = False, save_plot = False, plot_title = False):
    """
    a: semimajor axis of outermost disk [float]
    i: inclination of disks [float]
    phi: rotation of disks [float]
    tr: transmissivity of disks, from 0 (opaque) to 1 (transparent); order from outermost [float]
    w: width of disk in px; order from outermost disk [float]
    y_disk: distance between disk center and star center in px [float]
    v: disk velocity in px/day [float]
    t0: time when (xc_disk - xc_star) = 0 [float]
    t: time in days [array]
    r_star: radius of star in px [float]
    u: limb darkening factor between 0 and 1 [float]
    step: limb darkening model step [int]
    t_end: MJD when the model should finish, must be >= 0. If set to 0, no additional MJD will be calculated after the last MJD data [float]
    t_step: MJD step after the last MJD data towards t_end [int]
    plot: whether to plot the light curve and model or not; default: False [boolean]
    save_plot: whether to save the plot or not; default: False [boolean]
    plot_title: whether to display the plot title or not; default: False [boolean]
    """
    #create grid
    gridx = float(r_star * 2 + 2)
    gridy = gridx

    grid_yx = [gridy, gridx]
    
    #create star
    star_flux, star_model, g = star(grid_yx, r_star, u, step)
    
    #convert t to x     
    if t_end != 0:
        #additional time for prediction when the eclipse ends
        t_pred = np.arange(max(t)+t_step, t_end+t_step, t_step)
        #concatenate t and t_pred
        t_tot = np.concatenate((t, t_pred))
        pass
    else:
        t_tot = t
    
    x = t_to_x(t_tot, v, t0)
    
    #set empty list to hold values
    flux = []
    mjd = t_tot
    resi = []
    
    #create star+ring
    j = 0
    
    for x_disk in x:
        f_norm, __, ___ = disk_ring_stat(grid_yx, star_flux, star_model, a, w, tr, i, phi, x_disk, y_disk, plot = False)        
        flux.append(f_norm)
        
        #calculate residue
        if t_tot[j] <= max(t):
            f_res = f_norm - real_lc_flux[j]
            resi.append(f_res)
            j += 1
            pass
        else:
            pass
    
    
    #calculate chi-squared value
    n_chi2 = ((flux[:len(t)] - real_lc_flux)**2) / (real_lc_fluxerr ** 2)
    chi2 = np.sum(n_chi2)
    
    #calculate reduced chi2 value
    red_chi2 = chi2 / len(real_lc_flux)
    
    print(f'Calculated chi-squared value: {chi2:.2f}')
    print(f'Calculated reduced chi-squared value: {red_chi2:.2f}')
    print('-' * 40)

    #Predict end of eclipse based on predicted flux
    if t_end != 0:
        #Predict transit start
        k = 0
        while flux[k] == 1.0:
            k += 1
            pass
        print(f'Transit started at around MJD = {t_tot[k]:.3f} days.')
        
        #Predict transit end 
        l = len(t)
        token = 0 #to check if the break condition was fulfilled. Break condition fulfilled --> token = 1 (see below)
        while flux[l] < 1.0:
            l += 1
            if l == len(t_tot):
                print('=' * 40)
                print('Predicted end of transit not reached! Use longer additional time!')
                print('=' * 40)
                token = 1
                break
            pass
        
        if token != 1:
            print(f'Forecasted end of transit is around MJD = {t_tot[l]:.3f} days.')
            ecl_l = t_tot[l] - t_tot[k]
            print(f'Predicted transit length is around {ecl_l:.3f} days\n(data resolution past last MJD data: {t_step} days).')
            print('=' * 40)   
    
    if plot == True:
        ##PLOTTING
        fig2, (ax_1, ax_2) = plt.subplots(nrows = 2, figsize = (8, 5), gridspec_kw = {'height_ratios': [1.5, 0.5]}, sharex = True)

        #plot bg real data
        real_data = ax_1.errorbar(x = real_lc_MJD, y = real_lc_flux, 
                                    yerr = real_lc_fluxerr, c = 'r', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.5, label = 'real observation')

        #plot model scatter plot
        model = ax_1.plot(mjd[:len(t)], flux[:len(t)], color = 'blue', linewidth = 3, alpha = 0.5, label = 'model', zorder = 5)
        ax_1.set_xlim((min(mjd)-10), (max(mjd)))
        ax_1.set_ylim((min(flux) - 0.025), 1.05)
        ax_1.set_ylabel('Normalized Flux')
    #     ax_1.set_xlabel('MJD [days]')
        ax_1.tick_params(direction = 'in')
        # ax_1.set_title(f'2 Rings Model \n a_ring1 = {a_ring}px, incl. = {i_test}deg, rot. = {phi_test}deg, trans. 1 = {tr*100}%, \n a_ring2 = {a_ring2} px, trans. 2 = {tr2*100}%, \n R_star = {r_star} px, ring_y_offset = {yc_ring} px, cons. x-axis = {c} px, u = {u}')

        #plot residuals
        y0 = ax_2.hlines(0, xmin = (min(mjd)-50), xmax = (max(mjd)+50), colors = 'grey', linestyles = '--')
        resids = ax_2.errorbar(x = real_lc_MJD, y = resi, yerr = real_lc_fluxerr, c = 'b', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.5)
        ax_2.set_xlim((min(mjd)-10), (max(mjd)))
        ax_2.set_xlabel('MJD')
        ax_2.set_ylabel('Residuals')
        ax_2.tick_params(direction = 'in')

        if t_end != 0:
            pred = ax_1.plot(t_pred, flux[(len(t)):], color = 'blue', linestyle = '--', label = 'predicted data')
            if token != 1:
                x1 = ax_1.vlines(t_tot[l], ymin = (min(flux) - 0.05), ymax = 1.05, colors = 'grey', linestyles = '-', label = f'Pred. end of eclipse \n(resolution: {t_step} days)')
            if plot_title == True:
                ax_1.set_title(f'r_ring_outer = {a1}px, $i$ = {i}$^\circ$, $\phi$ = {phi}$^\circ$, tr._ring_outer = {tr1}%, w_ring_outer = {w1}px, \nr_ring_inner = {a3}px, tr._ring_inner = {tr3}%, w_ring_inner = {w3}px, \ny_ring = {y_disk}px, $v$ = {v}px/day, $t_0$ = {t0}days \nPred. end of eclipse = {t_tot[l]:.3f} days, t_resol. (bin) = {t_step} days, \ntr_star = {r_star} px, limb dark. coef. = {u}, mgrid size = [200, 600]')
                pass
            pass
        else:
            if plot_title == True:
                ax_1.set_title(f'r_ring_outer = {a1}px, $i$ = {i}$^\circ$, $\phi$ = {phi}$^\circ$, tr._ring_outer = {tr1}%, w_ring_outer = {w1}px, \nr_ring_inner = {a3}px, tr._ring_inner = {tr3}%, w_ring_inner = {w3}px, \ny_ring = {y_disk}px, $v$ = {v}px/day, $t_0$ = {t0}days \ntr_star = {r_star}px, limb dark. coef. = {u}, mgrid size = [200, 600]')
                pass
            pass

        #set legend and tight layout
        ax_1.legend()
        ax_1.grid(alpha = 0.5)
        plt.tight_layout()

        if save_plot == True:
            plt.savefig(f"tworings_fit.png", dpi = 300, facecolor = 'white')

        #plt.show()
        pass
    else:
        pass
    
    return mjd, flux, resi, chi2

#Check if function is working properly
print('Check if generalized disk number function is working properly')
print('See function for used values')
print('=' * 40)

#parameters
a = 782
i = 81.5
phi = 7.5
w = [205, 129, 135.5] #width of disk 1 (outermost), 2, 3. Width of disk 4 is essentially its radius.
tr = [0.99, 1.0, 0.66, 1.0] #transmissivity of disk 1 (outermost), 2, 3, and 4. 

#call function
mjd_, fl_, res_, _= disk_ring_mov(a = a, w = w, tr = tr, i = i, phi = phi, y_disk = 80, 
                                  v = 0.35, t0 = 60489, t_end = 62000, t_step = 9, plot = False)


# # Produce Animation using Vectors
# ## Combine Star and Ring Vectors

# ## Function to Change px to $R_*$
def px_to_r(x):
    """
    x = px value to convert
    """
    
    R_star = 5*U.solRad
    r_star = 50
    sol_px = R_star / (r_star *U.pixel)

    x_ticklab_sc = ((x * U.pixel) * sol_px / R_star).value
    x_ticklab_sc_str = ['%.2f' % X for X in x_ticklab_sc]
    
    return x_ticklab_sc


# In[24]:


def r_to_px(x):
    """
    x = r to convert
    real_r_star = real star radius
    """
    
    R_star = 5 * U.solRad
    r_star = 50
    sol_px = R_star / (r_star * U.pixel)
    
    x_px_ticklab_sc = ((R_star / sol_px) * x * U.pixel).value
    x_px_ticklab_sc_str = ['%.2f' % X for X in x_px_ticklab_sc]
    
    return x_px_ticklab_sc

#function to create vectors of disks (rings) --> small edit from exorings3

def draw_rings_vector(r, tau, xcen, incl, phi, p, ringcol='red', xrang=20., yrang=20.):
    ycen = 0.0

    p.set_xlim(xcen-xrang, xcen+xrang)
    p.set_ylim(ycen-yrang, ycen+yrang)

    for i in np.arange(0, r.size):
        if i == 0:
            rin = 0
            rout = r[0]
        else:
            rin = r[i-1]
            rout = r[i]

        ttau = 1 - tau[i]

        path = ring_patch(rin, rout, incl, phi, ([xcen, 0]))
        pnew = PathPatch(path, facecolor=ringcol, ec='none', alpha=ttau, zorder=-10)
        p.add_patch(pnew)
        pass
    
    return pnew


# ## Produce Animation
def animov(a, w, tr, i, phi, y_disk, v, t0, t_add, t_step, 
           r_star = 15, real_r_star = 5, u = 0.5, step = 100,
           t_real = real_lc_MJD, fl_real = real_lc_flux, flerr_real = real_lc_fluxerr, 
           xyrang = [1200., 600.], nticks = 8, save = False):
    """
    Create animation of the model
    ------
    
    
    Parameters:
    a = outermost semimajor axis of the disk (largest disk size), in px.
    w = width of every disk(s), in px. List, order from outermost to innermost. 
    Include except the last disk's width because practically it doesn't have any width.
    So if there are 4 disks, w[0] is the width of disk 1 (outermost), w[1] is for disk 2, w[2] is for disk 3, and none for disk 4.
    tr = transmissivity of each disk, between 0 (opaque) to 1 (transparent). List, order from outermost disk. Total number of
    tr is as many as the number of disk (pay attention to the difference from listing w values; see above).
    i = inclination, in degrees. Assuming all disks have the same inclination.
    phi = disk rotation, CCW, in degrees. Assuming all disks have the same phi value.
    y_disk = disk y-axis position of the center, in px. Assume all disks have the same center coordinate. 
    v = disk velocity, in px/day
    t0 = time when center of disk and star align
    t_add = number of additional days
    t_step = steps for additional days
    r_star = star model radius in px.
    real_r_star = star radius in solar radii.
    u = limb darkening profile, from 0 to 1.
    step = limb darkening profile step.
    t_real = real lightcurve MJD data.
    fl_real = real lightcurve flux data.
    flerr_real = real lightcurve flux error data.
    xyrang = x-range and y-range for ring vector canvas ([float, float])
    nticks = number of ticks for the new axes for the lower plot (int)
    
    """
    #time calculation
    tstart = T.time()
    
    #invert param disks (invert order)
    #invert radii
    A = a #dummy outermost radius
    r_disk = [A]
    for W in w:
        r_disk.append(A - W)
        A -= W
        pass
    r_disk.reverse()

    #invert tr
    tr_disk = tr[::-1]

    #convert to array
    r_disk = np.array(r_disk)
    tr_disk = np.array(tr_disk)

    #set total time
    t_tot = np.arange(min(t_real), max(t_real)+t_add+t_step, t_step)
    # t_tot = np.arange(60000, 61000+t_add+t_step, t_step)

    #convert t to x
    x = t_to_x(t_tot, v, t0)

    #create model star
    grid_yx = [float((r_star*2)+2), float((r_star*2)+2)]
    star_flux, star_model, g = star(grid_yx, r_star, u, step)

    #create empty list
    flux = []

    #create figure
    fig, (ax1, ax2) = plt.subplots(nrows = 2, figsize = (15, 10), gridspec_kw = {'height_ratios': [1.0, 1.0]})
    real_data = ax1.errorbar(x = t_real, y = fl_real, yerr = flerr_real, c = 'r', markersize = '6', 
                             elinewidth = 0.5, fmt = '.', alpha = 0.5, label = 'real observation')
    disk_ = [draw_rings_vector(r = r_disk, tau = tr_disk, xcen = 100, incl = i, phi = phi, p = ax2, xrang = 600., yrang = 200.)] #create disk patch
    star_ = [star_vector(xy = (100, 25), ax = ax2, r_star = 15, u = 0.5)] #create star patch
    lc, = ax1.plot([], lw = 3, c = 'blue', zorder = 5)

    def init():
        ax2.clear()
        return disk_[0], star_[0], lc,

    def animate(j):
        plt.cla()
        f_norm, __, ___ = disk_ring_stat(grid_yx, star_flux, star_model, a,
                                         w, tr, i, phi, x[j], y_disk, plot = False)
        flux.append(f_norm)
        lc.set_data(t_tot[:j], flux[:j])

        ax1.set_xlim((min(t_tot)-10), (max(t_tot)+10))
        ax1.grid(alpha = 0.1, ls = '--')
        ax1.set_xlabel('MJD', fontsize = 16)
        ax1.set_ylabel('Normalized Flux', fontsize = 16)
        ax1.tick_params(labelsize = 16, direction = 'in')

        disk_[0] = draw_rings_vector(r = r_disk, tau = tr_disk, xcen = x[j], incl = i, phi = phi, p = ax2, xrang = 1200., yrang = 400.) #create star patch
        star_[0] = star_vector(xy = (-0, -y_disk), ax = ax2, r_star = 50, u = 0.5) #create star patch
        ax2.add_patch(disk_[0])
        ax2.add_patch(star_[0])

#       ax2.set_xlim(-(xyrang[0]), (xyrang[0]+1))
        ax2.set_xlim(-(xyrang[0]/2), (xyrang[0]/2))
        ax2.set_ylim(-(xyrang[1]), (xyrang[1]))
#ax2.set_ylim(-(xyrang[1]/2), (xyrang[1]+1))
        ax2.grid(alpha = 0.1, ls = '--')

        #change tickmarks
        #scaling to R_*
        R_star = real_r_star*U.solRad
        sol_px = R_star / (r_star * U.pixel) #in solrad/px

        #set figure limit
        xlim = ax2.get_xlim()
        ylim = ax2.get_ylim()

        #obtain new ticks for y-axis
        y_ticklab = np.arange(min(ylim), max(ylim), ((max(ylim)-min(ylim))/nticks)) #create how many ticks
        y_ticklab_sc = ((y_ticklab * U.pixel) * sol_px / R_star).value #in solrad
        y_ticklab_sc_str = ['%.2f' % Y for Y in y_ticklab_sc] #obtain string labels for tick marks
        y_ticklab_sc_str[0] = ' ' #set the lowest tick label to empty so it will not go over tick from plot 2 when merged

        #obtain new ticks for secondary x-axis
        x_ticklab = np.arange(min(xlim), max(xlim), ((max(xlim)-min(xlim))/nticks)) #create how many ticks
        x_ticklab_sc = ((x_ticklab * U.pixel) * sol_px / R_star).value #in solrad
        x_ticklab_sc_str = ['%.2f' % X for X in x_ticklab_sc] #obtain string labels for tick marks

        #now changing the axes
        #y-axis
        ax2.set_yticks(y_ticklab)
        ax2.set_yticklabels(y_ticklab_sc_str)
        ax2.set_ylabel(r'$R_*$', fontsize = 16)
        ax2.tick_params(labelsize = 16, direction = 'in')

        #x-axis
        ax2.set_xticks(x_ticklab)
        ax2.set_xticklabels(x_ticklab_sc_str)
        ax2.set_xlabel(r'$R_*$', fontsize = 16)
        ax2.tick_params(labelsize = 16, direction = 'in')

        ax2.set_aspect('equal')
        return disk_[0], star_[0], lc,

    anim = pltani.FuncAnimation(fig, animate, frames = len(x), init_func=init, interval = 200, blit = True)

    if save == False:
        vid = anim.to_html5_video()
        html = display.HTML(vid)
        display.display(html)
        pass
    else:
        vidname = input('Video name?')
        writer = pltani.writers['ffmpeg']
        FFwriter = writer(fps = 12, metadata = {'artist': 'Me'}, bitrate = 1800)
        anim.save(f'{vidname}.mp4', writer = FFwriter, dpi = 300)
        pass

    plt.close()
    
    tend = T.time()
    processtotal = tend - tstart
    print('Done!')
    print('-' * 40)
    print(f'Total elapsed time: {processtotal:.2f}s')


# In[27]:


# #test animation
# animov(a = 4174.354, w = [135, 95, 96.5], tr = [0.99, 1, 0.74, 1], i = 84.2, phi = 5, y_disk = 27.5, v = 0.27, t0 = 60145, t_add = 1000, t_step = 9, save = True)


# # *emcee* Functions
# 
# The functions below supports the run for `emcee`. Edit the numbers according to the needs. 

# ## Log Likelihood Function
#log_likelihood for func moving ring, stat star (Recommended)
def log_likelihood(theta, t, f, ferr):
    """
    t is mjd from the observation
    f is the flux from observation
    ferr is the flux error from observation
    """
    a1, i, phi, tr1, w1, a3, tr3, w3, y_disk, v, t0 = theta #variables for emcee to play with
    mjd, fl, _, _ = fourdisk_model_mov(a1, i, phi, tr1, w1, a3, tr3, w3, y_disk, v, t0)
    
    #calculate all chi squared value
    n_chi2 = ((f - fl)**2) / (ferr ** 2) 
    return -0.5 * np.sum(n_chi2) #return summed chi squared value; -0.5 is for emcee

# ## Prior Function
# 
# 1. `a1` (semimajor axis) must be positive
# 2. `i` (inclination) must be from 0 to not including 90 degrees
# 3. `phi` (rotation) between -90 and 90 *or* between 0 and 180 (only take into account 2 quadrants because other than that is the same)
# 4. `tr1` (transmissivity) must be from 0 (opaque) to 1 (transparent)
# 5. `w` (ring width) must be between 0 and `a1` (basically cannot be negative _and_ greater than ring size)
# 6. `y_disk` ('b' / impact parameter) greater than including 0
# 7. `v` must be greater than 0
# 8. `t0` must be between the start of eclipse and around the largest MJD value used in the model/real data (upper limit can be beyond the largest MJD value available, because `t0` doesn't have to be during the eclipse, especially if the ring is huge, rotated, and the center of the ring is above/below the center of the star)

def log_prior(theta):
    a1, i, phi, tr1, w1, a3, tr3, w3, y_disk, v, t0 = theta
    
    if a1 > 0 and 0 <= i < 90 and 0 <= phi <= 180 and 0 <= tr1 <= 1 and 0 < w1 < a1 and 0 < a3 < (a1 - w1) and 0 < tr3 < 1 and 0 < w3 < a3 and y_disk >= 0 and v > 0 and 58000 <= t0 <= 64000:
        return 0.0 #return 0.0 if all conditions satisfied
    return -np.inf #returns -inf if one or more conditions violated --> raise error


# ## Log Probability Function

# In[ ]:


#log_prob using log_likelihood
def log_prob_glob1(theta):
    """
    lp is log prior
    t is the mjd of the model to be "fitted"
    f is the flux of the model to be "fitted"
    ferr is the flux_err of the model to be "fitted"
    """
    lp = log_prior(theta) 
    t = real_lc_MJD
    f = real_lc_flux
    ferr = real_lc_fluxerr
    
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, f, ferr)

