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
import pandas as pd
import astropy.coordinates as coord
import astropy.units as u
from astroplan import Observer
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np

def NightLength(year,month,sun_horizon=-18,moon_horizon=0,max_moon_phase=0.95*180*u.deg):

    Roque_Muchachos = Observer.at_site('Roque de los Muchachos')

    mdays = [0., 31., 28.25, 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.]

    length_tot = 0

    # loop over days of month
    for d in range(int(mdays[month])):
        if (d < 1):
            continue
        date_string  = '{:4d}-{:02d}-{:02d} 07:00:00'.format(year,month,d)
        #print ('date_string: ',date_string)
        time = Time(date_string,format='iso', scale='utc')     
        sun_rise = Roque_Muchachos.sun_rise_time(time, which="nearest",horizon=sun_horizon*u.deg)
        sun_set  = Roque_Muchachos.sun_set_time(time, which='previous',horizon=sun_horizon*u.deg)

        moon_set  = Roque_Muchachos.moon_set_time(sun_set, which='next',horizon=moon_horizon*u.deg)
        moon_rise = Roque_Muchachos.moon_rise_time(sun_set, which='next',horizon=moon_horizon*u.deg)
        if moon_rise.masked:    # bug in astroplan, see https://github.com/astropy/astroplan/issues/261
            moon_phase = np.pi*u.rad
        else:
            moon_phase = Roque_Muchachos.moon_phase(moon_rise)
            
        if (moon_phase > max_moon_phase):
            # phase=pi is “new”, phase=0 is “full”, do not subtract anything here
            print ('moon phase: ',moon_phase)
        elif (moon_set > moon_rise):
            # case when moon rises during the night
            if (moon_rise < sun_rise):
                length_tot -=  sun_rise-moon_rise
        elif (moon_rise > moon_set):
            # case when moon sets during the night
            if (moon_set < sun_rise):
                # subtract only part of the night
                length_tot -= moon_set-sun_set
            else:
                # subtract full night
                length_tot -= sun_rise-sun_set
        #print ('sun rise: {0.iso} '.format(sun_rise))
        #print ('sun set:  {0.iso} '.format(sun_set))
        #print ('moon rise: {0.iso} '.format(moon_rise))
        #print ('moon set:  {0.iso} '.format(moon_set))
        #print ('Length of night: ',sun_rise-sun_set)
        #print ('Moon set Tdff: ',(moon_set-sun_set))
        #print ('Moon rise Tdff: ',(moon_rise-sun_set))
        length_tot += sun_rise-sun_set

    if (month == 2):
        return length_tot.value * 28.25/28        
    return length_tot.value

def NightLengths_hour(year,sun_horizon=-18,moon_horizon=0,max_moon_phase=0.95*180*u.deg):

    L = []
    
    for m in range(13):
        if m == 0:
            continue
        L.append(NightLength(year,m,sun_horizon=sun_horizon,moon_horizon=moon_horizon,max_moon_phase=max_moon_phase)*24)

    return L

        
def AltAzSun(df):
    dates = df.index
    Alt_sun = []
    Az_sun = []

    Roque_Muchachos = coord.EarthLocation(lon = -17.8907 * u.deg, lat = 28.7619 * u.deg)
    my_time = Time(dates)
    sun = coord.get_sun(my_time)
    altaz = coord.AltAz(location=Roque_Muchachos, obstime=my_time)
    Alt_sun = coord.get_sun(my_time).transform_to(altaz).alt.degree
    Az_sun = coord.get_sun(my_time).transform_to(altaz).az.degree

    df.insert(0, 'sun_alt', Alt_sun)
    df.insert(1, 'sun_az', Az_sun)
    
    return df

def plot_anual_sol(df):
    plt.figure(figsize = (10,5))
    plt.scatter(df['sun_alt'], df['temperature'], s = 0.3)
    plt.xlabel('Zenith Angle (º)')
    plt.ylabel('Temperature (ºC)')
    plt.savefig('Angle_zenital_Sol.png', bbox_inches = 'tight')
    plt.show()
    
    plt.figure(figsize = (10,5))
    plt.scatter(df['sun_az'], df['temperature'], s = 0.3)
    plt.xlabel('Azimuth Angle (º)')
    plt.ylabel('Temperature (ºC)')
    plt.savefig('Angle_azimutal_Sol.png', bbox_inches = 'tight')
    plt.show()
    
