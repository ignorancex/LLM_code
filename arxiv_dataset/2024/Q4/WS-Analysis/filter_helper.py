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
import numpy as np

def Filtre_BrokenStation(df, inverted=False):
    if (inverted):
        df = df[(  (df.index > '2010-11-26 15:55') & (df.index < '2010-11-30 17:00') # installation new weather stations
                 | (df.index > '2007-03-19 02:00') & (df.index < '2007-03-28 00:00') # broken station
                 | (df.index > '2011-01-26 15:30') & (df.index < '2011-01-26 15:50') # installation after repair
               )]
                 #| (df.index > '2007-06-01 00:00') & (df.index < '2007-06-16 00:00'))]
    else:
        df = df[(df.index < '2010-12-19 00:30') | (df.index > '2010-12-19 01:40')] # exchange of WS, introduction of new data format
        df = df[(df.index < '2010-11-26 15:58') | (df.index > '2010-11-30 17:00')] # installatino new weather station
        df = df[(df.index < '2007-03-19 02:00') | (df.index > '2007-03-28 00:00')]
        df = df[(df.index < '2009-03-28 02:00') | (df.index > '2009-03-28 10:00')] # sudden unexplained swaps in temperature by 9 deg in 2 minutes
        df = df[(df.index < '2009-03-30 16:00') | (df.index > '2009-03-30 18:10')] # sudden unexplained drop in temperature, followed by format errors later on
        df = df[(df.index < '2011-01-26 15:30') | (df.index > '2011-01-26 15:50')] # installation after repair
        df = df[(df.index < '2019-11-14 13:00') | (df.index > '2019-11-16 16:20')] # broken station, wrong data  until installation of a new one
        df = df[(df.index < '2020-03-24 09:00') | (df.index > '2020-03-24 13:00')] # cannot be explained by focusing of MAGIC mirrors on WS, too sudden rise and fall, probable malfunctioning of the sensor
        df = df[(df.index < '2020-04-17 10:40') | (df.index > '2020-04-17 11:40')] # sudden unexplained rise in temperature with 5 measurements of zero's in the middle
                
        df = df[(df['humidity'] > -9998)]
        #df = df[(df.index < '2007-06-01 00:00') | (df.index > '2007-06-16 00:00')]
    return df

def Filtre_Temperatures(df, inverted=False):
    if (inverted):
        df = df[  ((df.index > '2010-10-11 09:30') & (df.index < '2010-10-11 13:00') # sun reflected by mirrors
                 | (df.index > '2020-03-24 09:00') & (df.index < '2020-03-24 13:00') # sudden jumps in temperature 
                 )]
                 #| (df.index > '2007-06-01 00:00') & (df.index < '2007-06-16 00:00'))]
    else:
        mask = (((df.index < '2010-10-11 09:30') | (df.index > '2010-10-11 13:00')) # MAGIC mirros focusing sun light on WS
                & ((df.index < '2009-03-25 00:00') | (df.index > '2009-03-25 05:20')) # sudden unexplained rise in temperature by 8 deg in 5 minutes
                & ((df.index < '2010-12-22 03:50') | (df.index > '2010-12-22 04:30')) # sudden unexplained rise in temperature by 8 deg in 5 minutes
                & ((df.index < '2014-11-20 12:25') | (df.index > '2014-11-20 13:10')) # sudden unexplained rise in temperature by 4 deg during 10 minutes
                & ((df.index < '2014-02-04 14:00') | (df.index > '2014-02-04 15:40'))) # sudden temperature rise from direction of LST-1

        #df = df[(df.index < '2010-11-26 15:50') | (df.index > '2010-11-30 17:00')] # installation new weather stations
        #df = df[(df.index < '2010-10-11 09:30') | (df.index > '2010-10-11 13:00')]
        #df = df[(df.index < '2009-03-25 00:00') | (df.index > '2009-03-25 02:40')] # sudden unexplained rise in temperature by 8 deg in 5 minutes
        #df = df[(df.index < '2009-03-28 02:00') | (df.index > '2009-03-28 10:00')] # sudden unexplained swaps in temperature by 9 deg in 2 minutes
        #df = df[(df.index < '2009-03-30 16:00') | (df.index > '2009-03-30 17:00')] # sudden unexplained drop in temperature, followed by format errors later on
        #df = df[(df.index < '2010-12-19 00:30') | (df.index > '2010-12-19 01:40')] # sudden unexplained rise in temperature by 8 deg in 5 minutes, introduction of new data format
        #df = df[(df.index < '2010-12-22 03:30') | (df.index > '2010-12-22 04:40')] # sudden unexplained rise in temperature by 8 deg in 5 minutes
        #df = df[(df.index < '2014-02-04 14:10') | (df.index > '2014-02-04 15:10')] # sudden temperature rise from direction of LST-1
        #df = df[(df.index < '2014-11-20 12:34') | (df.index > '2014-11-20 12:52')] # sudden unexplained rise in temperature by 4 deg during 10 minutes
        #df = df[(df.index < '2020-03-24 09:00') | (df.index > '2020-03-24 13:00')]

    return mask

def Filtre_Pressure(df, inverted=False):
    if (inverted):
        df = df[(  ((df.index > '2010-12-21 15:25') & (df.index < '2010-12-21 16:54'))  # sudden increase to 1013 mbar
                 #| ((df.index > '2004-02-17 12:00') & (df.index < '2004-02-21 12:00')) # ice storm, OK
               )]
        #df = df[(  (df.index > '2010-12-21 15:25') & (df.index < '2010-12-21 16:54') )]# sudden increase to 1013 mbar
        #df = df[ df['pressure'] < 765]
        #df = df[ ( df['humidity'] < 0.) | ( df['humidity'] > 100.) ]
    else:
        df = df[(   ((df.index < '2010-12-21 15:25') | (df.index > '2010-12-21 16:54')) # sudden increase of pressure up to 1013 mbar
                    # | ((df.index < '2004-02-17 12:00') | (df.index > '2004-02-21 12:00')) # ice storm, OK 
                    #| ((df.index < '2005-03-01 00:00') | (df.index > '2005-03-05 12:00')) # new installed station, but probably broken barometer, station was taken off again and put back one day later, probably an ice storm, should be OK
               )] # sudden increase to 1013 mbar
        #df = df[df.pressure < 800]
    return df

def Filtre_Wind(df, inverted=False):
    if (inverted):
        #df = df[ (df['windSpeedCurrent'] < -100) ] # | (df['windSpeedAverage'] < -100) ]
        df = df[ (df['windSpeedCurrent'] == 0) & (df['temperature'] < 1.0) & (df.index < '2007-03-20 00:00') ]  # frozen anemometers
    else:
        #df = df[ (df['windSpeedCurrent'] > -100) ] # & (df['windSpeedAverage'] > -100) ]
        df = df[ ~((df['windSpeedCurrent'] == 0) & (df['temperature'] < 1.0) & (df.index < '2007-03-20 00:00')) ]  # frozen anemometers
    return df

def Filtre_Humidity(df, inverted=False):
    if (inverted):
        #df = df[ (df['humidity'] < 0) ] # | (df['windSpeedAverage'] < -100) ]
        df = df[ (df['humidity'] >= 0) ] # & (df['windSpeedAverage'] > -100) ]
        df = df[(  (df.index > '2020-01-01 00:00') & (df.index < '2023-01-16 14:00') )] # gradual increase of humidity, compared with other stations
    else:
        #df = df[ (df['humidity'] >= 0) ] # & (df['windSpeedAverage'] > -100) ]
        mask = (((df.index < '2014-11-24 00:00') | (df.index > '2015-01-27 15:29')))  # gradual increase of humidity, compared with other stations
                #& ((df.index < '2020-01-01 00:00') | (df.index > '2023-01-16 14:00'))) # gradual increase of humidity, compared with other stations
                #                & ((df.index < '2014-11-30 15:55') | (df.index > '2014-11-30 19:00'))                
                #                & ((df.index < '2014-12-09 03:01') | (df.index > '2014-12-13 01:30'))
                #                & ((df.index < '2014-12-14 01:19') | (df.index > '2014-12-14 01:38'))
                #                & ((df.index < '2015-01-14 19:19') | (df.index > '2015-01-14 21:21'))
                #                & ((df.index < '2015-01-15 01:15') | (df.index > '2015-01-15 02:38'))
                #                & ((df.index < '2015-01-23 19:08') | (df.index > '2015-01-25 14:05')))
        
    return mask
