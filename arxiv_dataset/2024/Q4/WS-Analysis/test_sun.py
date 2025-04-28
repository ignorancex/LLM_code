# Copyright (c) 2024, Markus Gaug
# All rights reserved.
# 
# This source code is licensed under the GNU General Public License v3.0 found in the
# LICENSE file in the root directory of this source tree. 
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from plot_helper import plot_profile
from filter_helper import *

h5file_long = 'WS2003-23_long.h5'

df = pd.read_hdf(h5file_long)

times_S1 = np.array(['2010-10-11 09:30', '2010-10-11 13:00'])
#times_S2 = np.array(['2010-10-12 12:00', '2010-10-12 16:00'])
times_S3 = np.array(['2014-02-04 14:00', '2014-02-04 15:40'])
times_S4 = np.array(['2014-11-20 12:25', '2014-11-20 13:10'])
times_S5 = np.array(['2020-03-24 09:00', '2020-03-24 13:00'])
times_S6 = np.array(['2020-04-17 10:40', '2020-04-17 11:40'])
times_S = np.array([times_S1, times_S3, times_S4, times_S5, times_S6])

masks_S = [ ((df.index > '2010-10-11 09:30') & (df.index < '2010-10-11 13:00')),  # MAGIC mirros focusing sun light on WS
          ((df.index > '2009-03-30 16:00') & (df.index < '2009-03-30 17:00')), # sudden unexplained drop in temperature, followed by format errors later on
          ((df.index > '2014-02-04 14:10') & (df.index < '2014-02-04 15:10')), # sudden temperature rise from direction of LST-1
          ((df.index > '2014-11-20 12:34') & (df.index < '2014-11-20 12:52')), # sudden unexplained rise in temperature by 4 deg during 10 minutes
          ((df.index > '2020-03-24 09:00') & (df.index < '2020-03-24 13:00')) ] # focusing of MAGIC mirrors on WS?

times_N1 = np.array(['2009-03-25 00:00','2009-03-25 05:20'])
times_N2 = np.array(['2010-12-22 03:50','2010-12-22 04:30'])
times_N = np.array([times_N1, times_N2])

masks_nT = [ ((df.index > '2009-03-25 00:00') & (df.index < '2009-03-25 02:40')), # sudden unexplained rise in temperature by 8 deg in 5 minutes
          ((df.index > '2010-12-19 00:30') & (df.index < '2010-12-19 01:40')), # sudden unexplained rise in temperature by 8 deg in 5 minutes, introduction of new data format
          ((df.index > '2010-12-22 03:30') & (df.index < '2010-12-22 04:40')) ] # sudden unexplained rise in temperature by 8 deg in 5 minutes

fig, ax = plt.subplots(2,1)

for times in times_S:
    print ('mask', times[0], ', ', times[1])
    mask = ((df.index > str(times[0])) & (df.index < str(times[1])))
    ax[0].plot(df.loc[mask,'sun_alt'],df.loc[mask,'temperature'],'.',label=times[0]+'-'+times[1])
    ax[1].plot(df.loc[mask,'sun_az'],df.loc[mask,'temperature'],'.',label=times[0]+'-'+times[1])
    ax[1].arrow(85.,50.,0.,-5.)
    ax[1].arrow(175.,50.,0.,-5.)
    ax[1].arrow(240.,50.,0.,-5.)
    ax[0].legend(loc='best', fontsize=7)
    ax[1].legend(loc='best', fontsize=7)    
    ax[0].set_xlabel('Sun altitude (º)')
    ax[1].set_xlabel('Sun azimuth angle (º)')    
    ax[0].set_ylabel('Temperature (ºC)')
    ax[1].set_ylabel('Temperature (ºC)')    

plt.tight_layout()
plt.savefig('SunAltAz.pdf')
plt.show()

plt.clf()

mask = (df['sun_alt']>5)
plot_profile(df.loc[mask,'sun_alt'],df.loc[mask,'Tgradient5'],nbins=50)
plt.savefig('SunAlt_Tgradient5.pdf')
plt.clf()
plot_profile(df.loc[mask,'sun_az'],df.loc[mask,'Tgradient5'],nbins=50)
plt.savefig('SunAz_Tgradient5.pdf')

plt.clf()

mask = ((df['sun_alt']>5) & (df['temperature_reliable']==True))
plot_profile(df.loc[mask,'sun_alt'],df.loc[mask,'Tgradient5'],nbins=50)
plt.savefig('SunAlt_Tgradient5_reliable.pdf')
plt.clf()
plot_profile(df.loc[mask,'sun_az'],df.loc[mask,'Tgradient5'],nbins=50)
plt.savefig('SunAz_Tgradient5_reliable.pdf')

plt.clf()

ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(212)

ax = [ax1, ax2]

wind = 'windGust'

for i, times in enumerate(times_N):
    print ('mask', times[0], ', ', times[1])
    mask = ((df.index > str(times[0])) & (df.index < str(times[1])))
    dff = df[mask]
    ax[i].plot(dff.index,dff['temperature'],'.',label=times[0]+'-'+times[1],color='b')
    ax2 = ax[i].twinx()
    ax2.plot(dff.index,dff['windGust'],'.', color='orange')
    ax2.set_ylabel('Wind Gust Speed (km/h)',color='orange')
    ax2.tick_params(labelcolor='orange')    
    ax3.plot(dff['windGust'],dff['temperature'],'.',label=times[0]+'-'+times[1])
    ax[i].legend(loc='best', fontsize=7)
    ax3.legend(loc='best', fontsize=7)    
    ax[i].set_xlabel('Time')
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[i].tick_params(labelsize=7)    
    ax3.set_xlabel('Wind Gust Speed (km/h)')    
    ax[i].set_ylabel('Temperature (ºC)',color='b')
    ax3.set_ylabel('Temperature (ºC)')    

plt.tight_layout()    
plt.savefig('Humidity.pdf')
plt.show()

df = Filtre_BrokenStation(df,False)
df = df[~(Filtre_Temperatures(df, False))]

mask = ((df['Tgradient10R']>0.32) & (df['temperature_reliable']==True))
print ('Strong Tgradients: ', df.loc[mask,'temperature'])
mask = ((df['Tgradient10R']<-0.4) & (df['temperature_reliable']==True))
print ('Negative Tgradients: ', df.loc[mask,'temperature'])
mask = ((df['Rgradient10R']<-4.5) & (df['temperature_reliable']==True)  & (df['humidity_reliable']==True))
print ('Negative Rgradients: ', df.loc[mask,'humidity'])
