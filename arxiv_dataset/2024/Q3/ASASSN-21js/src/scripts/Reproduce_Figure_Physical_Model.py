from Main_Functions import *
import pickle
import paths

#Import random 200 models from pickle file
path = paths.data / 'rand200_mcmc_20240208_backup_2e6.pkl'

with open(path, 'rb') as f:
    rand200 = pickle.load(f)
    pass

print('Models loaded successfully.')


# In[21]:


def mcmc_rand200(flat_samples = None, prev_rand = None, samp_size = 200, chi2_thres = 1000, r_star = 15,
                 t = real_lc_MJD, t_end = 64000, t_step = 10, pred_end = 61249.206, nogap_thresh = 0.1, 
                 save_plot = False, save_data = False):
    """
    flat_samples: flattened samples from emcee. Use to generate new batch of n_samples based on samp_size.
                  Input is the variable containing the flattened emcee result. Both flat_samples and prev_rand cannot be of the same Boolean state (both None or both not None)
    prev_rand: filename of previous randomized 200 models run. Input is the variable containing previously generated randomized samples. 
               Both flat_samples and prev_rand cannot be of the same Boolean state (both None or both not None)
    
    ***
    If prev_rand is not None, all arguments below except for t will be ignored
    ***
    
    samp_size: number of model to sample randomly; default = 200
    chi2_thres: set chi2 threshold to search for the lowest chi2; default = 1000
    r_star: star radius, in px. Change according to the star radius used in emcee. Default = 15 px
    t: MJD data from observation
    t_end: end MJD day for forecasting. Must be greater than pred_end! If only want to 
           show models for the whole MJD observation data, set t_end = 0. Default = 64000 days
    t_step: additional MJD day step. Default = 10 days
    pred_end: predicted transit end day, taken from the best fit model. Default = 61249.206
    nogap_thresh: threshold to consider the ring system has no gap. In unit of R_star. To be converted into px
                  by multiplying it with the variable r_star, which is the representation of how many pixels is 1 R_star. 
                  Default = 0.1 R_star
    """

    if prev_rand is None:
        #select random 200 results from emcee
        inds = np.random.randint(len(flat_samples), size=samp_size)
        inds_size = len(inds)
        
        #array to save to pickle
        MJD = []
        FLUX = []
        FLUXERR = []
        RESI = []
        STAT = [] #status, if the sample has gap less, the same, or more than threshold
        CHI2 = [] #to hold chi2 values
        PAR = [] #to hold ring parameters of the lowest chi2
        
        if t_end != 0:
            while t_end <= max(t):
                print(f't_end needs to be either 0 or more than {max(t)}! Please put in the correct value!')
                t_end_input = input()
                t_end = float(t_end_input)
                pass
            else:
                pass
            while t_step == 0:
                print(f't_add needs to be more than 0!')
                t_step_input = input()
                t_step = int(t_step_input)
                pass
            else:
                pass
            pass

        last = flat_samples[inds[-1]]
        a1, i, phi, tr1, w1, a3, tr3, w3, y_disk, v, t0 = last
        dif = (a1 - w1 - a3)
        
        #calculate threshold value
        thresh = nogap_thresh * r_star #threshold for no gap in px
        print(f'The threshold we consider to not have a gap is less than {thresh:.2f} px')
        print(f'The threshold we consider to not have a gap is less than {nogap_thresh:.2f} stellar radii.')
        print('=' * 40)

        #Pick 2 samples: one bigger and one smaller than the threshold
        if (dif >= thresh):
            for k in inds:
                sample_ = flat_samples[k]
                a1, i, phi, tr1, w1, a3, tr3, w3, y_disk, v, t0 = sample_
                dif = (a1 - w1 - a3)
                print(f'nextdif: {dif}')
                if (dif < thresh):
                    print('Obtained sample with gap width below threshold')
                    print('-' * 40)
                    break
                else:
                    pass
                pass
            pass
        else:
            for k in inds:
                sample_ = flat_samples[k]
                a1, i, phi, tr1, w1, a3, tr3, w3, y_disk, v, t0 = sample_
                dif = (a1 - w1 - a3)
                print(f'nextdif: {dif}')
                if (dif >= thresh):
                    print('Obtained sample with gap width above or the same as threshold')
                    print('-' * 40)
                    break
                else:
                    pass
                pass
            pass

        #set parameters of plot
        fig, (ax_1, ax_2) = plt.subplots(nrows = 2, figsize = (15, 10), gridspec_kw = {'height_ratios': [1.5, 0.5]}, sharex = 'all')

        #plot real data
        real_data = ax_1.errorbar(t, real_lc_flux, real_lc_fluxerr,
                                  fmt=".k", capsize=0, alpha = 1.0, label = 'observed', zorder = 0)

        #making sure t_comp is correct, whether it contains additional MJD or not
        if t_end != 0:
            #uncomment IF want to set array of t for complete light curve (to see how samples from MCMC for after the end of data)
            t_add = np.arange(t[(len(t)-1)]+1, t_end, t_step)
            t_comp = np.concatenate((t, t_add))
            pass
        else:
            t_comp = t

        #plot all 200 random results
        for ind in inds:
            if ind == inds[inds_size-1]:
                sample = flat_samples[ind]
                a1, i, phi, tr1, w1, a3, tr3, w3, y_disk, v, t0 = sample #set according to sampled variables in MCMC
                mjd_, flux_, resi_, chi2 = fourdisk_model_mov(a1, i, phi, tr1, w1, a3, tr3, w3, y_disk, v, t0, 
                                                          r_star = r_star, t_end = t_end, t_step = t_step)

                #appending to every list
                MJD.append(mjd_)
                FLUX.append(flux_)
                RESI.append(resi_)
                CHI2.append(chi2)

                #plot residuals
                #DON'T FORGET TO CHANGE xmin, xmax, x, y, and yerr if EITHER USING t = model_1ring_mjd OR t = t_comp!!
                y0 = ax_2.hlines(0, xmin = (min(t_comp)-100), xmax = (max(t_comp)+100), colors = 'grey', linestyles = '--', alpha = 0.5)
                resids = ax_2.errorbar(x = t, y = resi_, yerr = real_lc_fluxerr, c = 'k', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.01)
                ax_2.set_ylabel('Residuals', fontsize = 16)

                #plot model
                dif = (a1 - w1 - a3)
                if (dif <= thresh):
                    stat = f"{thresh}-0"
                    STAT.append(stat)
                    model = ax_1.plot(mjd_, flux_, "#255FE4", alpha=0.35, label = fr"model ring gap $\leq$ {nogap_thresh} $R_*$")
                    pass
                else:
                    stat = f"{thresh}-1"
                    STAT.append(stat)
                    model = ax_1.plot(mjd_, flux_, "#D0E635", alpha=0.35, label = fr'model ring gap > {nogap_thresh} $R_*$')
                    pass

                #see if chi2 is lower than chi2 threshold
                if chi2 <= chi2_thres:
                    sample_lo = flat_samples[ind]
                    a1_lo, i_lo, phi_lo, tr1_lo, w1_lo, a3_lo, tr3_lo, w3_lo, y_disk_lo, v_lo, t0_lo = sample_lo

                    chi2_thres = chi2
                    pass
                else:
                    pass
                pass
            elif ind == k:
                sample = flat_samples[ind]
                a1, i, phi, tr1, w1, a3, tr3, w3, y_disk, v, t0 = sample #set according to sampled variables in MCMC
                mjd_, flux_, resi_, chi2 = fourdisk_model_mov(a1, i, phi, tr1, w1, a3, tr3, w3, y_disk, v, t0, 
                                                          r_star = r_star, t_end = t_end, t_step = t_step)

                #append to every list
                MJD.append(mjd_)
                FLUX.append(flux_)
                RESI.append(resi_)
                CHI2.append(chi2)

                #plot residuals
                #DON'T FORGET TO CHANGE xmin, xmax, x, y, and yerr if EITHER USING t = model_1ring_mjd OR t = t_comp!!
                y0 = ax_2.hlines(0, xmin = (min(t_comp)-100), xmax = (max(t_comp)+100), colors = 'grey', linestyles = '--', alpha = 0.5)
                resids = ax_2.errorbar(x = t, y = resi_, yerr = real_lc_fluxerr, c = 'k', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.01)

                #plot model
                dif = (a1 - w1 - a3)
                if (dif <= thresh):
                    stat = f"{thresh}-0"
                    STAT.append(stat)
                    model = ax_1.plot(mjd_, flux_, "#255FE4", alpha=0.35, label = fr'model ring gap $\leq$ {nogap_thresh} $R_*$')
                    pass
                else:
                    stat = f"{thresh}-1"
                    STAT.append(stat)
                    model = ax_1.plot(mjd_, flux_, "#D0E635", alpha=0.35, label = fr'model ring gap > {nogap_thresh} $R_*$')
                    pass

                #see if chi2 is lower than chi2 threshold
                if chi2 <= chi2_thres:
                    sample_lo = flat_samples[ind]
                    a1_lo, i_lo, phi_lo, tr1_lo, w1_lo, a3_lo, tr3_lo, w3_lo, y_disk_lo, v_lo, t0_lo = sample_lo

                    chi2_thres = chi2
                    pass
                else:
                    pass
                pass
            else:
                sample = flat_samples[ind]
                a1, i, phi, tr1, w1, a3, tr3, w3, y_disk, v, t0 = sample
                mjd_, flux_, resi_, chi2 = fourdisk_model_mov(a1, i, phi, tr1, w1, a3, tr3, w3, y_disk, v, t0, 
                                                          r_star = r_star, t_end = t_end, t_step = t_step)

                #append to every list
                MJD.append(mjd_)
                FLUX.append(flux_)
                RESI.append(resi_)
                CHI2.append(chi2)

                #plot model
                dif = (a1 - w1 - a3)
                if (dif <= thresh):
                    stat = f"{thresh}-0"
                    STAT.append(stat)
                    model = ax_1.plot(mjd_, flux_, "#255FE4", alpha=0.35)
                    pass
                else:
                    stat = f"{thresh}-1"
                    STAT.append(stat)
                    model = ax_1.plot(mjd_, flux_, "#D0E635", alpha=0.35)
                    pass

                #see if chi2 is lower than chi2 threshold
                if chi2 <= chi2_thres:
                    sample_lo = flat_samples[ind]
                    a1_lo, i_lo, phi_lo, tr1_lo, w1_lo, a3_lo, tr3_lo, w3_lo, y_disk_lo, v_lo, t0_lo = sample_lo

                    chi2_thres = chi2
                    pass
                else:
                    pass
                pass
            pass

            #plot residuals
            #DON'T FORGET TO CHANGE xmin, xmax, x, y, and yerr if EITHER USING t = model_1ring_mjd OR t = t_comp!!
            y0 = ax_2.hlines(0, xmin = (min(t_comp)-100), xmax = (max(t_comp)+100), colors = 'grey', linestyles = '--', alpha = 0.5)
            resids = ax_2.errorbar(x = t, y = resi_, yerr = real_lc_fluxerr, c = 'k', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.01)

            pass

        #plot the model with the lowest chi2 value
        mjd_, flux_, _, _ = fourdisk_model_mov(a1_lo, i_lo, phi_lo, tr1_lo, w1_lo, a3_lo, tr3_lo, w3_lo, y_disk_lo, v_lo, t0_lo,
                                              t_end = t_end, t_step = t_step, r_star = r_star)
        model = ax_1.plot(mjd_, flux_, "#FF0031", alpha=0.8, zorder = 201, lw = 3, label = r"lowest $X^2$ value")
        
        #append for saving
        PAR = [a1_lo, i_lo, phi_lo, tr1_lo, w1_lo, a3_lo, tr3_lo, w3_lo, y_disk_lo, v_lo, t0_lo]

        if t_end != 0:
            if t_end > pred_end:
                #change x value for ax_1.vlines based on predicted end of the transit!
                ax_1.vlines(pred_end, 0.5, 1.5, colors = 'k', ls = '--')
                textlim = pred_end + 100
                
                if textlim >= max(mjd_):
                    ax_1.text((pred_end - 50), 0.85, f"Pred. end of eclipse \nusing manual best fit model \nMJD = {pred_end} days",
                              rotation = 90, verticalalignment = 'center', fontsize = 14) #set x and y value to fit the plot
                    pass
                else:
                    ax_1.text((pred_end + 50), 0.85, f"Pred. end of eclipse \nusing manual best fit model \nMJD = {pred_end} days", 
                              rotation = 90, verticalalignment = 'center', fontsize = 14) #set x and y value to fit the plot
                    pass
                pass
            else:
                pass
            pass
        else:
            pass

        #set lims
        ax_1.set_ylim(0.76, 1.05)
        ax_1.legend(loc = 'lower left', fontsize = 16)
        ax_1.set_ylabel("Normalized Flux", fontsize = 16)
        ax_1.tick_params(direction = 'in', labelsize = 16)
        #ax_1.grid()

        ax_2.tick_params(direction = 'in', labelsize = 16)

        plt.xlim(min(mjd_), max(mjd_))
        plt.xlabel("MJD", fontsize = 16)
        plt.subplots_adjust(hspace = 0)

        if save_plot == True:
            #change file name for saving the plot
            plt.savefig(f'plot_{filename}.png', dpi = 300, facecolor= 'w')
            pass

        if save_data == True:
            #save randomly picked to pickle
            saved = pd.DataFrame((MJD, FLUX, RESI, CHI2, STAT, PAR)) 
            #note: PAR contains the ring parameters of the model with the lowest chi2 value. I put
            #it here to have just one file instead of separate files. But, I allocate the first row
            #of the last column (element (0, 5)) to hold the whole parameter list. To take the value
            #again, simply call that element and assign to the correct variables you have prepared
            #(This is for when wanting to draw the rings for graph_clean_2)
            saved.to_pickle(f'rand{samp_size}_{filename}.pkl') #change filename if needed
#        plt.show()
        
        return mjd_, flux_, PAR
    
    else:
        dumT = prev_rand.T #transpose dataset
        
        #assign to respective vars
        MJD = dumT[0]
        FLUX = dumT[1]
        RESI = dumT[2]
        CHI2 = dumT[3]
        STAT = dumT[4]
        PAR = dumT[:11][5]
        
        #obtain lowest chi2
        m = np.array(CHI2).argmin()
        
        n = range(len(STAT))
        last = STAT.iloc[-1][-1]

        for G in n:
            if STAT.iloc[G][-1] != last:
                break
            else:
                pass
        
        ##PLOT
        #set figure
        fig, (ax_1, ax_2) = plt.subplots(nrows = 2, figsize = (15, 10), gridspec_kw = {'height_ratios': [1.5, 0.5]}, sharex = 'all')
        
        #real data
        real_data = ax_1.errorbar(t, real_lc_flux, real_lc_fluxerr, fmt=".k", capsize=0, alpha = 1.0, label = 'observed', zorder = 0)
        
        #dashed line for residual
        y0 = ax_2.hlines(0, xmin = (min(MJD[0])-100), xmax = (max(MJD[0])+100), colors = 'grey', linestyles = '--', alpha = 0.5)
        
        #models
        #get characters for gap threshold and change to float
        n = range(len(STAT[0]))
        for N in n:
            if STAT[0][N] != "-":
                pass
            else:
                thresh = float(STAT[0][:N]) #in pixels
                nogap_thresh = thresh / r_star #in units of r_star
                pass
            pass
        
        print(f'The threshold we consider to not have a gap is less than {thresh:.2f} px')
        print(f'The threshold we consider to not have a gap is less than {nogap_thresh:.2f} stellar radii.')
        print('-' * 40)
        
        for M in range(len(MJD)):
            if M == (len(MJD)-1):
                if STAT[M][-1] == '0':
                    model = ax_1.plot(MJD[M], FLUX[M], "#255FE4", alpha=0.35, label = fr'model ring gap $\leq$ {nogap_thresh} $R_*$')
                    resids = ax_2.errorbar(x = t, y = RESI[M], yerr = real_lc_fluxerr, c = 'k', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.01)
                    #ax_2.set_xlim((min(t_comp)-50), (max(t_comp)+50))
                    pass
                else:
                    model = ax_1.plot(MJD[M], FLUX[M], "#D0E635", alpha=0.35, label = fr"model ring gap > {nogap_thresh} $R_*$")
                    resids = ax_2.errorbar(x = t, y = RESI[M], yerr = real_lc_fluxerr, c = 'k', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.01)
                    pass
                pass
            if M == G:
                if STAT.iloc[M][-1] == '0':
                    model = ax_1.plot(MJD[M], FLUX[M], "#255FE4", alpha=0.35, label = fr'model ring gap $\leq$ {nogap_thresh} $R_*$')
                    resids = ax_2.errorbar(x = t, y = RESI[M], yerr = real_lc_fluxerr, c = 'k', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.01)
                    #ax_2.set_xlim((min(t_comp)-50), (max(t_comp)+50))
                    pass
                else:
                    model = ax_1.plot(MJD[M], FLUX[M], "#D0E635", alpha=0.35, label = fr"model ring gap > {nogap_thresh} $R_*$")
                    resids = ax_2.errorbar(x = t, y = RESI[M], yerr = real_lc_fluxerr, c = 'k', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.01)
                    pass
                pass
            else:
                if STAT[M][-1] == '0':
                    model = ax_1.plot(MJD[M], FLUX[M], "#255FE4", alpha=0.35)
                    resids = ax_2.errorbar(x = t, y = RESI[M], yerr = real_lc_fluxerr, c = 'k', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.01)
                    #ax_2.set_xlim((min(t_comp)-50), (max(t_comp)+50))
                    pass
                else:
                    model = ax_1.plot(MJD[M], FLUX[M], "#D0E635", alpha=0.35)
                    resids = ax_2.errorbar(x = t, y = RESI[M], yerr = real_lc_fluxerr, c = 'k', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.01)
                    pass
                pass
        
        #plot lowest chi2
        model_lochi2 = ax_1.plot(MJD[m], FLUX[m], "#FF0031", alpha=0.8, zorder = 201, lw = 3, label = r"lowest $X^2$ value")
        
        #plot predicted end of transit
        if FLUX[0][-1] != 1.0:
            print('Contained model(s) do not reach predicted end of transit! Rerun the model using longer additional MJD!')
            pass
        else:
            ax_1.vlines(pred_end, 0.5, 1.5, colors = 'k', ls = '--')
            textlim = pred_end + 100
            if textlim >= max(MJD[0]):   
                ax_1.text((pred_end - 50), 0.85, f"Pred. end of eclipse \nusing manual best fit model \nMJD = {pred_end} days", 
                          rotation = 90, verticalalignment = 'center', fontsize = 14) #set x and y value to fit the plot
                pass
            else:
                ax_1.text((pred_end + 50), 0.85, f"Pred. end of eclipse \nusing manual best fit model \nMJD = {pred_end} days", 
                          rotation = 90, verticalalignment = 'center', fontsize = 14) #set x and y value to fit the plot
                pass
            pass
        
        #set lims
        ax_1.set_ylim(0.76, 1.05)
        ax_1.legend(loc = 'lower left', fontsize = 16)
        ax_1.set_ylabel("Normalized Flux", fontsize = 16)
        ax_1.tick_params(direction = 'in', labelsize = 16)
        #ax_1.grid()

        ax_2.tick_params(direction = 'in', labelsize = 16)

        plt.xlim(min(MJD[0]), max(MJD[0]))
        plt.xlabel("MJD", fontsize = 16)
        plt.subplots_adjust(hspace = 0)

        if save_plot == True:
            #change file name for saving the plot
            plt.savefig(f'plot_{filename}.png', dpi = 300, facecolor= 'w')
            pass
        
        return MJD[m], FLUX[m], PAR


# In[22]:


mjd_lo, flux_lo, par = mcmc_rand200(prev_rand = rand200, r_star = 50, t = real_lc_MJD, t_end = 62500, t_step = 9, 
                                    pred_end = 61249.206, save_plot = False, save_data = False)


# In[23]:


#assign ring parameters
#because the 2 rings are created by drawing 4 disks with 2 of them transparent, that is why
#ring_r and ring_tr have 4 values instead of only two.

ring_r = np.array([(par[5]-par[7]), par[5], (par[0] - par[4]), par[0]]) #order from inner to outer
ring_tr = np.array([1.0, par[6], 1.0, par[3]]) #order from inner to outer
ring_tr_scale = np.array([1, ring_tr[1] * 0.95, 1, ring_tr[3] * 0.95]) #times 0.95 to get better contrast of the almost transparent ring
ring_xcen = 0

i_lo = par[1]
phi_lo = par[2]

v_lo = par[9]
t0_lo = par[10]


# In[24]:


#assign star positions
t_stamp = [58341.827, 58658.756,
           59179.690, 59529.344,
           60919.206, 61529.206]

#convert t_stamp into x positions
x_star = []
for t in t_stamp:
    x_star.append(t_to_x(t, v_lo, t0_lo))

#rearrange x positions (because of how the grid is displayed)
x_star = [i * -1 for i in x_star] #times -1 because the plot's 0 is in the middle

#set star center
y_star = -par[8]

#set star model radius
r_star = 50


# In[25]:


#Plotting ring and star models, MCMC results (200 random results), real observation, and residuals.
#create plot canvas
fig1, (ax_1, ax_2, ax_3) = plt.subplots(nrows = 3, figsize = (15, 10), 
                                                  gridspec_kw = {'height_ratios': [0.9, 1.65, 0.25]})

#using my code (incorporating limb darkening profile)
for I in range(len(x_star)):
    vec_star = star_vector((x_star[I], y_star), ax_1, 50, 0.5)

rings = draw_rings_vector(ring_r, ring_tr_scale, ring_xcen, incl = i_lo, phi = phi_lo, 
                              p = ax_1, xrang = 2400., yrang = 600., ringcol = 'red') #using true tau values

ax_1.set_aspect('auto')

#Change ticklabels to R_*
xlim = ax_1.get_xlim()
ylim = ax_1.get_ylim()

#put vertical lines to mark mjd for the star at specific timestamps
for _ in x_star:
    ax_1.vlines(_, min(ylim), 0-r_star+y_star, alpha = 0.25, ls = '--', color = 'k')
    pass

#change both axes
#prepare changing by rescaling ro R_*
R_star = 3.7 * U.solRad
sol_px = R_star / (r_star * U.pixel) #in solrad/px

#obtain new ticks for y-axis
y_ticklab = np.arange(min(ylim), max(ylim), 100) #create how many ticks
y_ticklab_sc = ((y_ticklab * U.pixel) * sol_px / R_star).value #in solrad
y_ticklab_sc_str = ['%.2f' % Y for Y in y_ticklab_sc] #obtain string labels for tick marks
y_ticklab_sc_str[0] = ' ' #set the lowest tick label to empty so it will not go over tick from plot 2 when merged

#obtain new ticks for secondary x-axis
x_ticklab = np.arange(min(xlim), max(xlim), 100) #create how many ticks
x_ticklab_sc = ((x_ticklab * U.pixel) * sol_px / R_star).value #in solrad
x_ticklab_sc_str = ['%.2f' % X for X in x_ticklab_sc] #obtain string labels for tick marks

#now changing the axes
#y-axis
ax_1.set_yticks(y_ticklab)
ax_1.set_yticklabels(y_ticklab_sc_str)
ax_1.set_ylabel(r'$R_*$', fontsize = 16)
ax_1.tick_params(labelsize = 16, direction = 'in')

#set secondary axis
secax = ax_1.secondary_xaxis('top', functions = (px_to_r, r_to_px))
secax.set_xlabel(r'$R_*$', fontsize = 16)
secax.tick_params(direction = 'in') #I don't know but if used, the ticks disappear
secax.tick_params(labelsize = 16)

#plot 2 --> light curve
t = real_lc_MJD #put mjd values into variable for simplicity

#plot bg real data
real_data = ax_2.errorbar(x = real_lc_MJD, y = real_lc_flux, 
                          yerr = real_lc_fluxerr, c = 'k', 
                          markersize = '6', elinewidth = 0.5, 
                          fmt = '.', alpha = 0.5, label = 'observation', 
                          zorder = 0)

#plot model
ax_2.plot(mjd_lo, flux_lo, color = 'r', linewidth = 2, alpha = 1, label = r'lowest $X^2$', zorder = 200)
ax_2.set_xlim((min(mjd_lo)-10), 62500)
ax_2.set_ylim((min(flux_lo) - 0.025), 1.05)
ax_2.set_ylabel('Normalized Flux', fontsize = 16)
ax_2.tick_params(direction = 'in', labelsize = 16)

#plot mcmc run for 11 parameters with 75 px gap
#load the pkl file from the mcmc_rand200 function
#change file path and pkl file name to read the pkl file properly
dumT = rand200.T #transpose dataset

#assign to respective vars
MJD = dumT[0]
FLUX = dumT[1]
RESI = dumT[2]
CHI2 = dumT[3]
STAT = dumT[4]

#plot mcmc run
#search for gap that is bigger/smaller than the threshold
n = range(len(STAT))
last = STAT.iloc[-1][-1]

for G in n:
    if STAT.iloc[G][-1] != last:
        break
    else:
        pass

#obtain lowest chi2
m = np.array(CHI2).argmin()

#models
#get characters for gap threshold and change to float
n = range(len(STAT[0]))
for N in n:
    if STAT[0][N] != "-":
        pass
    else:
        thresh = float(STAT[0][:N])
        nogap_thresh = thresh / r_star # in units of r_star
        pass
    pass

for M in range(len(MJD)):
    if M == (len(MJD)-1):
        if STAT[M][-1] == '0':
            model = ax_2.plot(MJD[M], FLUX[M], "#255FE4", alpha=0.35, label = fr'model ring gap $\leq$ {nogap_thresh} $R_*$')
            resids = ax_3.errorbar(x = t, y = RESI[M], yerr = real_lc_fluxerr, c = 'k', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.01)
            #ax_2.set_xlim((min(t_comp)-50), (max(t_comp)+50))
            pass
        else:
            model = ax_2.plot(MJD[M], FLUX[M], "#D0E635", alpha=0.35, label = fr"model ring gap > {nogap_thresh} $R_*$")
            resids = ax_3.errorbar(x = t, y = RESI[M], yerr = real_lc_fluxerr, c = 'k', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.01)
            pass
        pass
    if M == G:
        if STAT[M][-1] == '0':
            model = ax_2.plot(MJD[M], FLUX[M], "#255FE4", alpha=0.35, label = fr'model ring gap $\leq$ {nogap_thresh} $R_*$')
            resids = ax_3.errorbar(x = t, y = RESI[M], yerr = real_lc_fluxerr, c = 'k', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.01)
            #ax_2.set_xlim((min(t_comp)-50), (max(t_comp)+50))
            pass
        else:
            model = ax_2.plot(MJD[M], FLUX[M], "#D0E635", alpha=0.35, label = fr"model ring gap > {nogap_thresh} $R_*$")
            resids = ax_3.errorbar(x = t, y = RESI[M], yerr = real_lc_fluxerr, c = 'k', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.01)
            pass
        pass
    else:
        if STAT[M][-1] == '0':
            model = ax_2.plot(MJD[M], FLUX[M], "#255FE4", alpha=0.35)
            resids = ax_3.errorbar(x = t, y = RESI[M], yerr = real_lc_fluxerr, c = 'k', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.01)
            #ax_2.set_xlim((min(t_comp)-50), (max(t_comp)+50))
            pass
        else:
            model = ax_2.plot(MJD[M], FLUX[M], "#D0E635", alpha=0.35)
            resids = ax_3.errorbar(x = t, y = RESI[M], yerr = real_lc_fluxerr, c = 'k', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.01)
            pass
        pass

#draw lines for the timestamps of the star
for tstamp in t_stamp:
    ax_2.vlines(tstamp, min(fl_)-0.025, 1.05, alpha = 0.25, color = 'k', ls = '--')
    pass

#create dummy plot to obtain secondary axis
ax_4 = ax_2.twiny()
ax_4.plot(mjd_[:len(t)], fl_[:len(t)], alpha = 0)
ax_4.set_xlim((min(mjd_)-10), 62500)
ax_4.tick_params(direction = 'in')

#legends
ax_2.legend(fontsize = 16)

#plot 3 --> residuals
ax_3.hlines(0, xmin = (min(mjd_)-50), xmax = 62500, colors = 'grey', linestyles = '--')
ax_3.errorbar(x = real_lc_MJD, y = resi, yerr = real_lc_fluxerr, c = 'k', markersize = '6', 
             elinewidth = 0.5, fmt = '.', alpha = 0.5)
ax_3.set_xlim((min(mjd_)-10), 62500)
ax_3.set_xlabel('MJD', fontsize = 16)
ax_3.set_ylabel('Residuals', fontsize = 16)
ax_3.tick_params(direction = 'in', labelsize = 16)

#set dummy plot for secondary x-axis
ax_5 = ax_3.twiny()
ax_5.errorbar(x = real_lc_MJD, y = resi, yerr = real_lc_fluxerr, fmt = '.', alpha = 0)
ax_5.set_xlim((min(mjd_)-10), 62500)
ax_5.tick_params(direction = 'in')

#convert primary x-axis for plot 1 (finishing)
#rescale primary x-axis into mjd (CAUTION: I decouple the value of the pixel from the axis for this one!)
xlo, xhi = ax_2.get_xlim() #get x limit from middle plot as source for axis conversion
#search for nearest round thousand number (like 62000, 61000, etc.)
for xhimod in np.arange(xhi, xlo, -1):
    modres = xhimod % 1000
    if modres == 0:
        break
    else:
        pass

px_mjd_scale = (xhi-xlo)/max(xlim) #get mjd to px ratio
# x_mjd = px_mjd_scale * np.arange(0, max(xlim)+1, 1) + xlo #convert every mjd into px value
# x_mjd = px_mjd_scale * np.arange(xlo, max(xhi)+1, 1)
x_tick_mjd = np.arange(xhimod, xlo, -1000) #create ticks with the same spacing as the other plots' mjd axis
x_tick_mjd_str = ['%.0f' % X for X in x_tick_mjd[::-1]] #get the string, inverse the order (because the default is from big to low)
x_tickmark_mjd_px = t_to_x(x_tick_mjd, v_lo, t0_lo) #convert the same spacing mjd ticks to px again
x_tickmark_mjd_px = 0-x_tickmark_mjd_px #position the ticks against the 0-point (midway on x-axis)
x_tickmark_mjd_px = x_tickmark_mjd_px[::-1] #reverse the order, so the value is from low to high

#primary x-axis
ax_1.set_xticks(x_tickmark_mjd_px)
ax_1.set_xticklabels(x_tick_mjd_str) #check if the positions and labels are correct
ax_1.tick_params(direction = 'in', color = 'k') #direction of ticks inward
ax_1.set(xlabel=None)

#set primary axis limit to match the middle plot's x-axis
xlim_low = -(t_to_x(xlo, v_lo, t0_lo))
xlim_hi = -(t_to_x(xhi, v_lo, t0_lo))
ax_1.set_xlim(xlim_low, xlim_hi)

#mark center of rings
xlim = ax_1.get_xlim()
ax_1.vlines(0, max(ylim), 0, colors = 'black', ls = '--', alpha = 0.2)
ax_1.hlines(0, min(xlim), 0, colors = 'black', ls = '--', alpha = 0.2)
ax_1.scatter(0, 0, s = 5, c = 'r', marker = 'o', label = 'ring system center')
ax_1.legend(fontsize = 16, loc = 'upper left')

#merge all plots
plt.setp(ax_1.get_xticklabels(), visible = False)
plt.setp(ax_1.get_yticklabels()[::2], visible = False)
plt.setp(ax_2.get_xticklabels(), visible = False)
plt.setp(ax_2.get_yticklabels()[-1], visible = False)
plt.setp(ax_4.get_xticklabels(), visible = False)
plt.setp(ax_5.get_xticklabels(), visible = False)

plt.subplots_adjust(hspace = 0)

#save figure
plt.savefig(paths.figures / f'asassn21js_rand200_lightcurve_model.pdf', dpi = 300, facecolor = 'w')

