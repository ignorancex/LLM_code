import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import scipy.stats as stats
import math as math
import uncertainties as un
import astropy.units as u
import astropy.constants as C
from scipy.stats import norm
import paths

#load raw data
a21js_asassn = pd.read_csv(paths.data/'ASASSN_lc_new230124.csv')
#a21js_asassn = pd.read_csv(paths.data/'609886184506-light-curves.csv')
#convert to MJD
MJD = a21js_asassn['HJD'] - 2400000.5
a21js_asassn.insert(1, 'MJD', MJD) #insert new column titled 'MJD'

#check any weird magnitude values
mag_min = np.amin(a21js_asassn['mag'])
mag_max = np.amax(a21js_asassn['mag'])

#clean weird magnitude values
i = np.where(a21js_asassn['mag'] == mag_max)
a21js_asassn = a21js_asassn.drop(index = i[0]) 

#Masking based on filter
mask_V = a21js_asassn['Filter'] == 'V' #creates a boolean filter, TRUE if the value of 'Filter' for the row is 'V'
mask_g = a21js_asassn['Filter'] == 'g' #same, but for 'Filter' = 'g'

a21js_asassn_V = a21js_asassn[mask_V] #applying mask to the data
a21js_asassn_g = a21js_asassn[mask_g]

t_V = a21js_asassn_V['MJD']
t_g = a21js_asassn_g['MJD']

#Masking error bars based on threshold

#mag_V
err_mask_V = a21js_asassn_V['mag_err'] < 0.030
a21js_asassn_V_clean = a21js_asassn_V[err_mask_V]

#mag_g
err_mask_g = a21js_asassn_g['mag_err'] < 0.028
a21js_asassn_g_clean = a21js_asassn_g[err_mask_g]

#flux_
err_flux_V = a21js_asassn_V['flux_err'] < 0.755
a21js_asassn_V_clean_f = a21js_asassn_V[err_flux_V]

#flux_g
err_flux_g = a21js_asassn_g['flux_err'] < 0.50
a21js_asassn_g_clean_f = a21js_asassn_g[err_flux_g]


# In[8]:


#discarding far data points
# MJD<59100
data_mask_lo59100 = a21js_asassn_g_clean_f[(a21js_asassn_g_clean_f['MJD'] < 59100)]

#create histogram
plt.hist(data_mask_lo59100['flux(mJy)'], bins = 30, density = True)

#make normal distribution to fit
mu, sig = norm.fit(data_mask_lo59100['flux(mJy)'])

#create PDF (Prob. Dist. Func.)
xmin, xmax = plt.xlim() #obtain x limits of the histogram
x = np.linspace(xmin, xmax, 100)
pdf = norm.pdf(x, mu, sig)

#print 3sig for both wings (99.7%)
threesigl = mu-3*sig
threesigr = mu+3*sig

print(f'Value of mu + (3-sigma) = {threesigr}')
print(f'Value of mu - (3-sigma) = {threesigl}')

#see histogram to determine which values to trim
plt.plot(x, pdf, 'k', linewidth = 2)
ymin, ymax = plt.ylim() #obtain y limits of the histogram
plt.vlines(threesigl, ymin, ymax, 'r')
plt.vlines(threesigr, ymin, ymax, 'r')
plt.ylabel('count')
plt.xlabel('flux (mJy)')
plt.ylim(0, 0.9)
plt.grid(alpha = 0.5);

#trimming values
data_mask_lo59100 = data_mask_lo59100[(data_mask_lo59100['flux(mJy)'] > threesigl) * (data_mask_lo59100['flux(mJy)'] < threesigr)]

# MJD>59100
data_mask_hi59100 = a21js_asassn_g_clean_f[(a21js_asassn_g_clean_f['MJD'] >= 59100) * (a21js_asassn_g_clean_f['MJD'] <= 59850)] #we use '>=' here because in the 'low' region we don't include MJD = 59100 days.

#create histogram
plt.hist(data_mask_hi59100['flux(mJy)'], bins = 30, density = True)

#make normal distribution to fit
mu, sig = norm.fit(data_mask_hi59100['flux(mJy)'])

#create PDF (Prob. Dist. Func.)
xmin, xmax = plt.xlim() #obtain x limits of the histogram
x = np.linspace(xmin, xmax, 100)
pdf = norm.pdf(x, mu, sig)

#print 3sig for both wings (99.7%)
threesigl = mu-3*sig
threesigr = mu+3*sig

print(f'Value of mu + (3-sigma) = {threesigr}')
print(f'Value of mu - (3-sigma) = {threesigl}')

#see histogram to determine which values to trim
plt.plot(x, pdf, 'k', linewidth = 2)
ymin, ymax = plt.ylim() #obtain y limits of the histogram
plt.vlines(threesigl, ymin, ymax, 'r')
plt.vlines(threesigr, ymin, ymax, 'r')
plt.ylabel('count')
plt.xlabel('flux (mJy)')
plt.ylim(0, 0.51)
plt.grid(alpha = 0.5);

#trimming values
data_mask_hi59100 = data_mask_hi59100[(data_mask_hi59100['flux(mJy)'] < threesigr) * (data_mask_hi59100['flux(mJy)'] > threesigl)]


# In[10]:


# MJD>59850
late_obs_mask = a21js_asassn_g_clean_f[a21js_asassn_g_clean_f['MJD'] > 59850]

#create histogram
plt.hist(late_obs_mask['flux(mJy)'], bins = 30, density = True)

#make normal distribution to fit
mu, sig = norm.fit(late_obs_mask['flux(mJy)'])

#create PDF (Prob. Dist. Func.)
xmin, xmax = plt.xlim() #obtain x limits of the histogram
x = np.linspace(xmin, xmax, 100)
pdf = norm.pdf(x, mu, sig)

#print 3sig for both wings (99.7%)
threesigl = mu-3*sig
threesigr = mu+3*sig

print(f'Value of mu + (3-sigma) = {threesigr}')
print(f'Value of mu - (3-sigma) = {threesigl}')

#see histogram to determine which values to trim
plt.plot(x, pdf, 'k', linewidth = 2)
ymin, ymax = plt.ylim() #obtain y limits of the histogram
plt.vlines(threesigl, ymin, ymax, 'r')
plt.vlines(threesigr, ymin, ymax, 'r')
plt.ylabel('count')
plt.xlabel('flux (mJy)')
plt.ylim(0, 0.83)
plt.grid(alpha = 0.5);

#trimming
late_obs_mask = late_obs_mask[(late_obs_mask['flux(mJy)'] > threesigl) * (late_obs_mask['flux(mJy)'] < threesigr)]


# In[11]:


#concatenate
data_comp = pd.concat([data_mask_lo59100, data_mask_hi59100, late_obs_mask]).sort_values(by = 'MJD').reset_index().drop(columns = ['index'])


# In[12]:


#trim for V filter data
#Use histogram to know the values to trim
plt.hist(a21js_asassn_V_clean_f['flux(mJy)'], bins = 20, density = True)

#make normal distribution to fit
mu, sig = norm.fit(a21js_asassn_V_clean_f['flux(mJy)'])

#create PDF (Prob. Dist. Func.)
xmin, xmax = plt.xlim() #obtain x limits of the histogram
x = np.linspace(xmin, xmax, 100)
pdf = norm.pdf(x, mu, sig)

#print 3sig for both wings (99.7%)
threesigl = mu-3*sig
threesigr = mu+3*sig

print(f'Value of mu + (3-sigma) = {threesigr}')
print(f'Value of mu - (3-sigma) = {threesigl}')

#see histogram to determine which values to trim
plt.plot(x, pdf, 'k', linewidth = 2)
ymin, ymax = plt.ylim() #obtain y limits of the histogram
plt.vlines(threesigl, ymin, ymax, 'r')
plt.vlines(threesigr, ymin, ymax, 'r')
plt.xlabel('flux (mJy)')
plt.ylabel('count')
plt.ylim(0, 0.75)
plt.grid(alpha = 0.5);

#trimming
V_data_masked = a21js_asassn_V_clean_f[(a21js_asassn_V_clean_f['flux(mJy)'] < threesigr) * (a21js_asassn_V_clean_f['flux(mJy)'] > threesigl)]


# In[13]:


#Masking for overlapping MJD
t_up = 58350
t_low = 58175

#compute the mean value in the masked region
mean_V = np.mean(V_data_masked[(V_data_masked['MJD'] > t_low) * (V_data_masked['MJD'] < t_up)]["mag"])
mean_g = np.mean(data_comp[(data_comp['MJD'] > t_low) * (data_comp['MJD'] < t_up)]["mag"])

mean_Vf = np.mean(V_data_masked[(V_data_masked['MJD'] > t_low) * (V_data_masked['MJD'] < t_up)]["flux(mJy)"])
mean_gf = np.mean(data_comp[(data_comp['MJD'] > t_low) * (data_comp['MJD'] < t_up)]["flux(mJy)"])


# In[14]:


#normalizing data for each filter
a21js_V_norm = V_data_masked['mag'] - mean_V
a21js_g_norm = data_comp['mag'] - mean_g
a21js_V_enorm = V_data_masked['mag_err']
a21js_g_enorm = data_comp['mag_err']

a21js_Vf_norm = V_data_masked['flux(mJy)'] / mean_Vf
a21js_gf_norm = data_comp['flux(mJy)'] / mean_gf
a21js_Vf_enorm = V_data_masked['flux_err'] / mean_Vf
a21js_gf_enorm = data_comp['flux_err'] / mean_g


# In[15]:


#create dataframe of normalized data
data = {'MJD':data_comp['MJD'],
       'flux':a21js_gf_norm,
       'flux_err':a21js_gf_enorm}
a21js_gf_clean = pd.DataFrame(data)


# In[16]:


#collect and create clean dataset of V flux data
data_Vnorm = {'MJD': V_data_masked['MJD'], 
              'flux': a21js_Vf_norm,
              'flux_err': a21js_Vf_enorm}

a21js_Vf_clean = pd.DataFrame(data_Vnorm)


# In[17]:


#concatenate V and g flux datasets
a21js_Vgf_clean = pd.concat([a21js_Vf_clean, a21js_gf_clean]).sort_values(by = 'MJD').reset_index().drop(columns = ['index'])


# In[18]:


def binning(data, step = 5):
    """
    This function splits the data into bins with **equal widths (equal time lengths)**
    
    [inputs]
    data: array of values to be binned, containing [MJD, value, value_err]
    step: bin width (time-width)
    """
    start_time = data.iloc[0, 0]
    end_time = data.iloc[len(data)-1, 0]
    
    upbin = start_time + step
    
    mjd_temp = []
    flux_temp = []
    std_temp = []
    binned = []
    bin_size = []
    
    print('===============[HISTORY LOG]=================')
    print(f'current start_time: {start_time}')
    for i in range(len(data)):
        if data.iloc[i, 0] <= upbin:
            # Check whether current data_MJD has lower value than
            # current upper bin boundary. As long as it is the case,
            # data will be appended to the temp arrays. Also, as it 
            # is the case, the (data_MJD > upper_bin_boundary) condition
            # will NEVER be satisfied and program will just proceed to the outer loop.
            
            mjd_temp.append(data.iloc[i, 0])
            flux_temp.append(data.iloc[i, 1])
            std_temp.append(data.iloc[i, 2])
            
            if data.iloc[i, 0] == end_time:
                # This is the last 'gate' to prevent the last couple
                # of data to just sit in their bin and not being processed
                # as if this is the case (without the gate), then the
                # following (data_MJD > upper_bin_boundary) condition will NEVER be 
                # satisfied.
                
                print(f'datum in bin: {len(mjd_temp)}')
                bin_size.append(len(mjd_temp))
                mjd_mean = np.average(mjd_temp)
                #flux_mean = np.average(flux_temp)
                
                #getting error for each bin
                std_temp2 = np.array(std_temp)**2
                p = np.sum(1/std_temp2)
                std_fin = np.sqrt(1/p) #new error

                fl_mean = np.array(flux_temp)/std_temp2
                a = np.sum(fl_mean)/p

                flux_mean = a
                std_mean = std_fin
                        
                bin_temp = [mjd_mean, flux_mean, std_mean, len(mjd_temp)]
                binned.append(bin_temp)
                
                print('-' * 30)
                print(f'Total number of bins: {len(binned)} bins')
                print('=============== END OF FUNCTION ===============')
                
        elif data.iloc[i, 0] > upbin:
            # When the current data_MJD has value higher than the
            # upper boundary of the bin, we need to stop appending
            # and process everything inside the temp arrays. However,
            # the current data still need to be appended to the NEXT
            # BIN, but it should satisfy a condition first where
            # the data is WITHIN the boundary of the new bin. If not,
            # proceed to the 'while' loop to get new bin boundaries.
            
            print(f'datum in bin: {len(mjd_temp)}')
            bin_size.append(len(mjd_temp))
            mjd_mean = np.average(mjd_temp)
            #flux_mean = np.average(flux_temp)

            #getting error for each bin
            if len(std_temp) == 1:
                print('Caution! Only 1 data in bin! Cannot perform division by 0')
            
            std_temp2 = np.array(std_temp)**2
            p = np.sum(1/std_temp2)
            std_fin = np.sqrt(1/p) #new error

            fl_mean = np.array(flux_temp)/std_temp2
            a = np.sum(fl_mean)/p

            flux_mean = a
            std_mean = std_fin
            
            bin_temp = [mjd_mean, flux_mean, std_mean, len(mjd_temp)]
            binned.append(bin_temp)

            mjd_temp = []
            flux_temp = []
            std_temp = []
            
            while ((data.iloc[i, 0] - start_time) > step):
                # moving bin boundary so that the current data MJD
                # is within the boundary of the bin, determined
                # by the fact that (data_MJD - bin_low_bound) < step
                # should be satisfied
                
                start_time = upbin
                upbin = start_time + step
                print(f'new start_time: {start_time}')
            
            # If (data_MJD - bin_low_bound) < step, meaning that the data
            # is now WITHIN the bin boundary, proceed to next step. 
            # However, the current data then needs to be appended first
            # as it now satisifies the requrement above.
            
            print('=====')
            print(f'current start_time: {start_time}')
            mjd_temp.append(data.iloc[i, 0])
            flux_temp.append(data.iloc[i, 1])
            std_temp.append(data.iloc[i, 2])
            
            if data.iloc[i, 0] == data.iloc[len(data)-1, 0]:
                datbin = [mjd_temp, flux_temp, std_temp, len(mjd_temp)]
                binned.append(datbin)
                print('-' * 30)
                print('Be careful as the last bin only contains 1 data')
                print('-' * 30)
                print(f'Total number of bins: {len(binned)} bins')
                print('=============== END OF FUNCTION ===============')
    
    #convert to dataframe
    binned_df = pd.DataFrame(binned)
    binned_df.columns = ['MJD', 'flux', 'flux_err', 'num_data']
    
    #raise warning if there is condition met
    one_dat = 0
    for k in range(len(binned_df.flux_err)):
        if (binned_df.flux_err[k] == np.inf) | (binned_df.flux_err[k] == np.nan):
            one_dat += 1
    
    if one_dat != 0:
        print(f'Caution! There are {one_dat} bin(s) with only 1 data, resulting in std_mean = inf! \nPlease use different bin size!')
        
    return binned_df, bin_size


# In[19]:


#binning
a21js_Vgf_binned, s3 = binning(a21js_Vgf_clean, step = 9)


# In[ ]:


#save as csv file
a21js_Vgf_binned.to_csv(paths.data/'a21js_Vgf_avgsqrt_bin9days.csv', index = False)

