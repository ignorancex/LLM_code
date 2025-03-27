# Measurement program Kun Zuo & Vincent Mourik
# Last updated: May 2012

# We use this script for transport measurements, both DC and AC,
# as a function of 1 or 2 other variables. Its well suited to measure 
# with 1 or 2 Keithleys and/or 1 or 2 Lockins in 2 or 4 terminal geometries.
# If you are a first time user, we recommend scrolling down to the 
# 'initialization' part of the script, we put some usefull comments there.
# Comments/improvements are appreciated.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.



from numpy import pi, random, arange, size, array, sin, cos, linspace, sinc, sqrt
from time import time, sleep
from shutil import copyfile
from os import mkdir
from os.path import exists
from lib.file_support.spyview import SpyView


import qt
import timetrack
import sys
import numpy as np
import data as d
import traces
import shutil
import os


keithley1 = qt.instruments.get('keithley1')
keithley2 = qt.instruments.get('keithley2')
lockin1 = qt.instruments.get('lockin1')
lockin2 = qt.instruments.get('lockin2')
#cryocon = qt.instruments.get('cryocon')
#opto = qt.instruments.get('opto')
magnet = qt.instruments.get('magnet')
magnetX = qt.instruments.get('magnetX')
mw = qt.instruments.get('mw')
ivvi = qt.instruments.get('ivvi')


    
class majorana():
    
    def __init__(self): 
        self.filename=filename
        self.generator=d.IncrementalGenerator(qt.config['datadir']+'\\Yifan_Jiang\\LG07172019MYJ\\'+self.filename,1);
    
    
    # Function generates data file, spyview file and copies the pyton script.
    def create_data(self,x_vector,x_coordinate,x_parameter,y_vector,y_coordinate,y_parameter,z_vector,z_coordinate,z_parameter):
        qt.Data.set_filename_generator(self.generator)
        data = qt.Data(name=self.filename)
        data.add_coordinate(x_parameter+' ('+x_coordinate+')',
                            size=len(x_vector),
                            start=x_vector[0],
                            end=x_vector[-1]) 
        data.add_coordinate(y_parameter+' ('+y_coordinate+')',
                            size=len(y_vector),
                            start=y_vector[0],
                            end=y_vector[-1]) 
        data.add_coordinate(z_parameter+' ('+z_coordinate+')',
                            size=len(z_vector),
                            start=z_vector[0],
                            end=z_vector[-1])
        data.add_value('lockin 1')                    # from lock-in 1 for dI measurement
        data.add_value('Keithley 1')            	            # Read out of Keithley 1 for I_dc		
        data.add_value('Keithley 2')                    # from lock-in 2 for dV measurement		
        #data.add_value('processed dIdV')            	            # processed dIdV	
        data.add_value('lockin 2') 
        data.create_file()                                  # Create data file
        SpyView(data).write_meta_file()                     # Create the spyview meta.txt file
        #traces.copy_script(sys._getframe().f_code.co_filename,data._dir,self._filename+str(self._generator._counter-1))# Copy the python script into the data folder
        traces.copy_script(sys._getframe().f_code.co_filename,data._dir,self.filename+str(self.generator._counter-1))
        return data
    
    
    # Function reads out relevant data
    def take_data(self,dacx,x):
        
        ivvi.set(dacx,x)                                                    # Set specified dac to specified value, has to be done here because of delays needed for Lockin measurements
        qt.msleep(Twait) 
        #L1 = lockin1.get_R()                                                 # Read out Lockin1		
        K1 = 0
        K2 = 0
        L1 = 0
        L2 = 0
        for i in range(0, average):
            K1 += keithley1.get_readlastval()
            K2 += keithley2.get_readlastval()
            L1 += lockin1.get_R()
            L2 += lockin2.get_R()
            qt.msleep(tau)
        K1 = K1/average
        K2 = K2/average
        L1 = L1/average
        L2 = L2/average
        
        # K1 = keithley1.get_readlastval()                                     # Read out Keithley1   
        #K2 = keithley2.get_readlastval()                                     # Read out Keithley2 		
        #value1 = K1- x* RC           # calculate current change (in 2e^2/h) if sample conductance=0.1*2e^2/h                                 #  for amp=1M, R_in=9.17KOhm; for amp=10M, R_in=22.5KOhm; for amp=100M, R_in=182KOhm;;; R_in=11.9KOhm for output*100mV, amp=1M	
        datavalues = [L1,K1,K2,L2]
		#datavalues = [L1,K1,K2,value1]
        #datavalues = [value1,K1,0,0,0]		

        return datavalues
        qt.msleep(0.1)                                                     # Keep GUI responsive
     
    ################ 1D scans #####################    
    
    # 1D sweep of a single dac
    def _single_dac_sweep(self,xname,dacx,xstart,xend,xsteps):
        qt.mstart()
        
        # Create sweep vectors
        x_vector = linspace(xstart,xend,xsteps+1)
        y_vector = [0]
        z_vector = [0]
        
        data = self.create_data(x_vector,xname,dacx,y_vector,'none','y_parameter',z_vector,'none','z_parameter')                                # create data file, spyview metafile, copy script
        
        for x in x_vector:

            datavalues = self.take_data(dacx,x) 
			# Go to next sweep value and take data                                                                                                          # Read out mixing chamber temperature
            data.add_data_point(x,0,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])               		       											        # write datapoint into datafile
        
        data.new_block()
        data._write_settings_file()                                                                                                             # Overwrite the settings file created at the beginning, this ensures updating the sweep variable with the latest value
        data.close_file()
        qt.mend()
    

        
    #################### 2D scans ####################
    def _2dacs_sweep(self,xname,dacx,xstart,xend,mname,dacm,mstart,mend,steps):
        qt.mstart()
                
        # Create sweep vectors
        x_vector = linspace(xstart,xend,steps+1)
        y_vector = [0]
        z_vector = [0]
        
        data = self.create_data(x_vector,xname,dacx,y_vector,'none','y_parameter',z_vector,'none','z_parameter')                               # create data file, spyview metafile, copy script
        
        # Define sweep vector for other dac
        m_vector = linspace(mstart,mend,steps+1)
        
        for i in arange(len(x_vector)):
            x = x_vector[i]
            m = m_vector[i]
            ivvi.set(dacm,m)                                                                                                                    # Other dac needs to be set before running take_data
            datavalues = self.take_data(dacx,x)                                                                                                 # Go to next sweep value and take data
            #T_mc = self.read_T()                                                                                                                # Read out mixing chamber temperature
            data.add_data_point(x,0,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])                               # write datapoint into datafile
        
        data.new_block()
        data._write_settings_file()
        data.close_file()
        qt.mend()

    def _3dacs_sweep(self,xname,dacx,xstart,xend,mname,dacm,mstart,mend,nname,dacn,nstart,nend,steps):
        qt.mstart()
        
        # Create sweep vectors
        x_vector = linspace(xstart,xend,steps+1)
        y_vector = [0]
        z_vector = [0]
        
        data = self.create_data(x_vector,xname,dacx,y_vector,'none','y_parameter',z_vector,'none','z_parameter')                               # create data file, spyview metafile, copy script
        
        # Define sweep vector for other dacs
        m_vector = linspace(mstart,mend,steps+1)
        n_vector = linspace(nstart,nend,steps+1)
        
        for i in arange(len(x_vector)):
            x = x_vector[i]
            m = m_vector[i]
            n = n_vector[i]
            ivvi.set(dacm,m)                                                                                                                    # Other dacs needs to be set before running take_data
            ivvi.set(dacn,n)
            datavalues = self.take_data(dacx,x)                                                                                                 # Go to next sweep value and take data
            #T_mc = self.read_T()                                                                                                                # Read out mixing chamber temperature
            data.add_data_point(x,0,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])                               # write datapoint into datafile
        
        data.new_block()
        data._write_settings_file()
        data.close_file()
        qt.mend()
    
		
		
    # 2D scan of one dac vs 2 others; the 2 dacs that are stepped together can have arbitrary start and end values
    def _dac_vs_2dacs(self,xname,dacx,xstart,xend,xsteps,yname,dacy,ystart,yend,mname,dacm,mstart,mend,ymsteps):
        qt.mstart()
        
        # Create sweep vectors
        x_vector = linspace(xstart,xend,xsteps+1)
        y_vector = linspace(ystart,yend,ymsteps+1)
        z_vector = [0]
        
        data = self.create_data(x_vector,xname,dacx,y_vector,yname,dacy,z_vector,'none','z_parameter')                                          # create data file, spyview metafile, copy script
        
        # Define sweep vector for other dac
        m_vector = linspace(mstart,mend,ymsteps+1)
        
        counter = 0
        
        for i in arange(len(y_vector)):
            [starttime, counter] = timetrack.start(counter)
            tstart = timetrack.time()
            y = y_vector[i]
            m = m_vector[i]
            ivvi.set(dacy,y)
            ivvi.set(dacm,m)
            ivvi.set(dacx,x_vector[0])
            #T_mc = self.read_T()                                                                                                                # Read out mixing chamber temperature 
            qt.msleep(1)                                                                                                                   # use explained at the bottom of the script
            
            for x in x_vector:
                datavalues = self.take_data(dacx,x)                                                                                             # Go to next sweep value and take data
                data.add_data_point(x,y,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])                           # write datapoint into datafile
            
            timetrack.remainingtime(starttime,ymsteps+1,counter)                                                                                # Calculate and print remaining scantime
            data.new_block()
            
        data._write_settings_file()                                                                                                             # Overwrite the settings file created at the beginning, this ensures updating the sweep variable with the latest value
        data.close_file()
        qt.mend()
 
 

 # 2D scan of one dac vs 2 others; the 2 dacs that are stepped together can have arbitrary start and end values
    # 2D scan of one dac vs 3 others; the 3 dacs that are stepped together can have arbitrary start and end values
	
    def _dac_vs_3dacs(self,xname,dacx,xstart,xend,xsteps,yname,dacy,ystart,yend,mname,dacm,mstart,mend,nname,dacn,nstart,nend,ymnsteps):
        qt.mstart()
        
        # Create sweep vectors
        x_vector = linspace(xstart,xend,xsteps+1)
        y_vector = linspace(ystart,yend,ymnsteps+1)
        z_vector = [0]
        
        data = self.create_data(x_vector,xname,dacx,y_vector,yname,dacy,z_vector,'none','z_parameter')                                          # create data file, spyview metafile, copy script
        
        # Define sweep vector for other dacs
        m_vector = linspace(mstart,mend,ymnsteps+1)
        n_vector = linspace(nstart,nend,ymnsteps+1)
        
        counter = 0
        
        for i in arange(len(y_vector)):
            [starttime, counter] = timetrack.start(counter)
            tstart = timetrack.time()
            y = y_vector[i]
            m = m_vector[i]
            n = n_vector[i]
            ivvi.set(dacy,y)
            ivvi.set(dacm,m)
            ivvi.set(dacn,n)
            ivvi.set(dacx,x_vector[0])
            #T_mc = self.read_T()                                                                                                                # Read out mixing chamber temperature 
            qt.msleep(2)                                                                                                                   # use explained at the bottom of the script
            
            for x in x_vector:
                datavalues = self.take_data(dacx,x)                                                                                             # Go to next sweep value and take data
                data.add_data_point(x,y,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])                           # write datapoint into datafile
            
            timetrack.remainingtime(starttime,ymnsteps+1,counter)                                                                               # Calculate and print remaining scantime
            data.new_block()
            
        data._write_settings_file()                                                                                                             # Overwrite the settings file created at the beginning, this ensures updating the sweep variable with the latest value
        data.close_file()
        qt.mend()
    
 
    def _5dacs_samesweep(self,xname,dacx,mname,dacm,nname,dacn,pname,dacp,qname,dacq,xstart,xend,steps):
        qt.mstart()
       
        # Create sweep vectors
        x_vector = linspace(xstart,xend,steps+1)
        y_vector = [0]
        z_vector = [0]
       
        data = self.create_data(x_vector,xname,dacx,y_vector,'none','y_parameter',z_vector,'none','z_parameter')                               # create data file, spyview metafile, copy script
       
        # Define sweep vector for other dacs

       
        for i in arange(len(x_vector)):
            x = x_vector[i]
       
            ivvi.set(dacm,x)                                                                                                                    # Other dacs needs to be set before running take_data
            ivvi.set(dacn,x)
            ivvi.set(dacp,x)
            ivvi.set(dacq,x)
           
            datavalues = self.take_data(dacx,x)                                                                                                 # Go to next sweep value and take data
            #T_mc = self.read_T()                                                                                                                # Read out mixing chamber temperature
            data.add_data_point(x,0,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])                               # write datapoint into datafile
       
        data.new_block()
        data._write_settings_file()
        data.close_file()
        qt.mend()
 
 
 
 
    # 2D scan of one dac vs another one
    def _dac_vs_dac(self,xname,dacx,xstart,xend,xsteps,yname,dacy,ystart,yend,ysteps):
        qt.mstart()
        
        # Create sweep vectors
        x_vector = linspace(xstart,xend,xsteps+1)
        y_vector = linspace(ystart,yend,ysteps+1)
        z_vector = [0]
        
        data = self.create_data(x_vector,xname,dacx,y_vector,yname,dacy,z_vector,'none','z_parameter')                                          # create data file, spyview metafile, copy script
        
        counter = 0
        
        for y in y_vector:
            [starttime, counter] = timetrack.start(counter)
            tstart = timetrack.time()
            ivvi.set(dacy,y)
			
            ivvi.set(dacx,x_vector[0])
            qt.msleep(1)                                                                                                                   # use explained at the bottom of the script
            
            for x in x_vector:
                datavalues = self.take_data(dacx,x)                                                                                             # Go to next sweep value and take data
                data.add_data_point(x,y,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])                           # write datapoint into datafile
            
            timetrack.remainingtime(starttime,ysteps+1,counter)                                                                                 # Calculate and print remaining scantime
            data.new_block()
            
        data._write_settings_file()                                                                                                             # Overwrite the settings file created at the beginning, this ensures updating the sweep variable with the latest value
        data.close_file()
        qt.mend()
    
    
    # 2D scan of one dac vs another one
    def _gate_vs_gate(self,xname,dacx,xstart,xend,xsteps,yname,dacy,ystart,yend,ysteps):
        qt.mstart()
        
        # Create sweep vectors
        x_vector = linspace(xstart,xend,xsteps+1)
        y_vector = linspace(ystart,yend,ysteps+1)
        z_vector = [0]
        
        data = self.create_data(x_vector,xname,dacx,y_vector,yname,dacy,z_vector,'none','z_parameter')                                          # create data file, spyview metafile, copy script
        
        counter = 0
        
        for y in y_vector:
            [starttime, counter] = timetrack.start(counter)
            tstart = timetrack.time()
            ivvi.set(dacy,y)
			
            ivvi.set(dacx,x_vector[0])
            qt.msleep(2)                                                                                                                   # use explained at the bottom of the script
            
            for x in x_vector:
                datavalues = self.take_data(dacx,x)                                                                                             # Go to next sweep value and take data
                data.add_data_point(x,y,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])                           # write datapoint into datafile
            
            timetrack.remainingtime(starttime,ysteps+1,counter)                                                                                 # Calculate and print remaining scantime
            data.new_block()
			
            for k in arange(len(x_vector)-1):
                ivvi.set(dacx,x_vector[xsteps-1-k]);                                                                                               
                qt.msleep(0.5)                             
           
          
            
        data._write_settings_file()                                                                                                             # Overwrite the settings file created at the beginning, this ensures updating the sweep variable with the latest value
        data.close_file()
        qt.mend()
    
    


	
    #################### 2D scans ####################
	# 3dac vs magnet
	
    def _3dacs_vs_magnet(self,xname,dacx,xstart,xend,yname,dacy,ystart,yend,zname,dacz,zstart,zend,xyzsteps,Bstart,Bend,Bsteps):
		qt.mstart()
		
		ivvi.set(dacy,ystart)
		ivvi.set(dacx,xstart)
		ivvi.set(dacz,zstart)
		qt.msleep(3)
		
		# Create sweep vectors
		x_vector = linspace(xstart,xend,xyzsteps+1)
		B_vector = linspace(Bstart,Bend,Bsteps+1)
		z_vector = [0]
		
        
		data = self.create_data(x_vector,xname,dacx,B_vector,'B (T)','magnet',z_vector,'none','z_parameter')                                     # create data file, spyview metafile, copy script
		y_vector = linspace(ystart,yend,xyzsteps+1)
		z_vector = linspace(zstart,zend,xyzsteps+1)
		counter = 0
		
		for B in B_vector:
			[starttime, counter] = timetrack.start(counter)
			tstart = timetrack.time();qt.msleep(3)
			
			magnet.set_field(B);  ivvi.set(dacy,y_vector[0]); ivvi.set(dacz,z_vector[0])
			
			ivvi.set(dacx,x_vector[0])
	        
			#T_mc = self.read_T()                                                                                                                # Read out mixing chamber temperature 
			qt.msleep(3)                                                                                                                   # use explained at the bottom of the script
		
			for j in arange(len(x_vector)):
				ivvi.set(dacy,y_vector[j]); ivvi.set(dacz,z_vector[j]); datavalues = self.take_data(dacx,x_vector[j]);  data.add_data_point(x_vector[j],B,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3]);                                                                                                    	
                                     # write datapoint into datafile
            
			timetrack.remainingtime(starttime,Bsteps+1,counter)                                                                                 # Calculate and print remaining scantime
			data.new_block(); 
			
			for k in arange(len(x_vector)):
				ivvi.set(dacy,y_vector[xyzsteps-1-k]); ivvi.set(dacx,x_vector[xyzsteps-1-k]); ivvi.set(dacz,z_vector[xyzsteps-1-k]); qt.msleep(0.2)                                                                                                  	
                             
            
           
			
		data._write_settings_file()                                                                                                             # Overwrite the settings file created at the beginning, this ensures updating the sweep variable with the latest value
		data.close_file()
		qt.mend()
		
	# 3dac vs magnet	
	# 2dac vs magnet
	
    def _2dacs_vs_magnet(self,xname,dacx,xstart,xend,yname,dacy,ystart,yend,xysteps,Bstart,Bend,Bsteps):
		qt.mstart()
		
		ivvi.set(dacy,ystart)
		ivvi.set(dacx,xstart)
		qt.msleep(20)
		
		# Create sweep vectors
		x_vector = linspace(xstart,xend,xysteps+1)
		B_vector = linspace(Bstart,Bend,Bsteps+1)
		z_vector = [0]
		
        
		data = self.create_data(x_vector,xname,dacx,B_vector,'B (T)','magnet',z_vector,'none','z_parameter')                                     # create data file, spyview metafile, copy script
		y_vector = linspace(ystart,yend,xysteps+1)
		counter = 0
		
		for B in B_vector:
			[starttime, counter] = timetrack.start(counter)
			tstart = timetrack.time()
			
			magnet.set_field(B);  ivvi.set(dacy,y_vector[0])
			
			ivvi.set(dacx,x_vector[0])
	        
			T_mc = self.read_T()                                                                                                                # Read out mixing chamber temperature 
			qt.msleep(delay1)                                                                                                                   # use explained at the bottom of the script
		
			for j in arange(len(x_vector)):
				ivvi.set(dacy,y_vector[j]); datavalues = self.take_data(dacx,x_vector[j]);  data.add_data_point(x_vector[j],B,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3],datavalues[4],T_mc);                                                                                                    	
                                     # write datapoint into datafile
            
			timetrack.remainingtime(starttime,Bsteps+1,counter)                                                                                 # Calculate and print remaining scantime
			data.new_block(); 
			
			for k in arange(len(x_vector)):
				ivvi.set(dacy,y_vector[xysteps-1-k]); ivvi.set(dacx,x_vector[xysteps-1-k]);  qt.msleep(0.03)                                                                                                  	
                             
            
           
			
		data._write_settings_file()                                                                                                             # Overwrite the settings file created at the beginning, this ensures updating the sweep variable with the latest value
		data.close_file()
		qt.mend()
    # 2D scan of one dac vs another one
    def _dac_vs_magnet(self,xname,dacx,xstart,xend,xsteps,Bstart,Bend,Bsteps):
        qt.mstart()
        
        # Create sweep vectors
        x_vector = linspace(xstart,xend,xsteps+1)
        B_vector = linspace(Bstart,Bend,Bsteps+1)
        z_vector = [0]
        
        data = self.create_data(x_vector,xname,dacx,B_vector,'B(T)','magnet',z_vector,'none','z_parameter')                                     # create data file, spyview metafile, copy script
        
        counter = 0
        #magnet.set_heater(1)		
        magnet.set_units('T')		

        for B in B_vector:
            [starttime, counter] = timetrack.start(counter)
            tstart = timetrack.time()
            magnet.set_field(B) 			
            #kepco.set_B(B)                                                                                                                      # Specific for kepco as magnet supply
            ivvi.set(dacx,x_vector[0])
            #T_mc = self.read_T()                                                                                                                # Read out mixing chamber temperature 
            qt.msleep(0.01)                                                                                                                   # use explained at the bottom of the script
            
            for x in x_vector:
                datavalues = self.take_data(dacx,x)                                                                                             # Go to next sweep value and take data
                data.add_data_point(x,B,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])                                                                          # write datapoint into datafile
            
            timetrack.remainingtime(starttime,Bsteps+1,counter)                                                                                 # Calculate and print remaining scantime
            data.new_block()
            
        data._write_settings_file()                                                                                                             # Overwrite the settings file created at the beginning, this ensures updating the sweep variable with the latest value
        data.close_file()
        qt.mend()
 

    # dac vs rf power, to do power dependence of Shapiro steps. rf power is stepped with non-uniform steps following P^2 ~ Vrf. Function only works well if you start the sweep at the lowest power.

    def _gate_vs_magnet(self,xname,dacx,xstart,xend,xsteps,Bstart,Bend,Bsteps):
        qt.mstart()
        
        # Create sweep vectors
        x_vector = linspace(xstart,xend,xsteps+1)
        B_vector = linspace(Bstart,Bend,Bsteps+1)
        z_vector = [0]
        
        data = self.create_data(x_vector,xname,dacx,B_vector,'B(T)','magnet',z_vector,'none','z_parameter')                                     # create data file, spyview metafile, copy script
        
        counter = 0
        #magnet.set_heater(1)		
        magnet.set_units('T')		

        for B in B_vector:
            [starttime, counter] = timetrack.start(counter)
            tstart = timetrack.time()
            magnet.set_field(B) 			
            #kepco.set_B(B)                                                                                                                      # Specific for kepco as magnet supply
            ivvi.set(dacx,x_vector[0])
            #T_mc = self.read_T()                                                                                                                # Read out mixing chamber temperature 
            qt.msleep(2)                                                                                                                   # use explained at the bottom of the script
            
            for x in x_vector:
                datavalues = self.take_data(dacx,x)                                                                                             # Go to next sweep value and take data
                data.add_data_point(x,B,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])                                                                          # write datapoint into datafile
            
            timetrack.remainingtime(starttime,Bsteps+1,counter)                                                                                 # Calculate and print remaining scantime
            data.new_block()
            for k in arange(len(x_vector)-1):
                ivvi.set(dacx,x_vector[xsteps-1-k]);                                                                                               
                qt.msleep(1)  

            
        data._write_settings_file()                                                                                                             # Overwrite the settings file created at the beginning, this ensures updating the sweep variable with the latest value
        data.close_file()
        qt.mend()
 

    # dac vs rf power, to do power dependence of Shapiro steps. rf power is stepped with non-uniform steps following P^2 ~ Vrf. Function only works well if you start the sweep at the lowest power.

    
    def _dac_vs_magnetX(self,xname,dacx,xstart,xend,xsteps,Bstart,Bend,Bsteps):
        qt.mstart()
        
        # Create sweep vectors
        x_vector = linspace(xstart,xend,xsteps+1)
        B_vector = linspace(Bstart,Bend,Bsteps+1)
        z_vector = [0]
        
        data = self.create_data(x_vector,xname,dacx,B_vector,'Bx(T)','magnetX',z_vector,'none','z_parameter')                                     # create data file, spyview metafile, copy script
        
        counter = 0
        #magnet.set_heater(1)		
        magnetX.set_units('T')		

        for B in B_vector:
            [starttime, counter] = timetrack.start(counter)
            tstart = timetrack.time()
            magnetX.set_field(B) 			
            #kepco.set_B(B)                                                                                                                      # Specific for kepco as magnet supply
            ivvi.set(dacx,x_vector[0])
            #T_mc = self.read_T()                                                                                                                # Read out mixing chamber temperature 
            qt.msleep(0.01)                                                                                                                   # use explained at the bottom of the script
            
            for x in x_vector:
                datavalues = self.take_data(dacx,x)                                                                                             # Go to next sweep value and take data
                data.add_data_point(x,B,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])                                                                          # write datapoint into datafile
            
            timetrack.remainingtime(starttime,Bsteps+1,counter)                                                                                 # Calculate and print remaining scantime
            data.new_block()
            
        data._write_settings_file()                                                                                                             # Overwrite the settings file created at the beginning, this ensures updating the sweep variable with the latest value
        data.close_file()
        qt.mend()
 

    # dac vs rf power, to do power dependence of Shapiro steps. rf power is stepped with non-uniform steps following P^2 ~ Vrf. Function only works well if you start the sweep at the lowest power.
    

	
   # 2D scan of one dac vs vector magnet
    def _dac_vs_2magnets(self,xname,dacx,xstart,xend,xsteps,Bstart,Bend,Bsteps,sita):
        qt.mstart()
        
        # Create sweep vectors
        x_vector = linspace(xstart,xend,xsteps+1)
        B_vector = linspace(Bstart,Bend,Bsteps+1)
        z_vector = [0]
        
        data = self.create_data(x_vector,xname,dacx,B_vector,'B(T)','Vector_magnet',z_vector,'none','z_parameter')                                     # create data file, spyview metafile, copy script
        
        counter = 0
        #magnet.set_heater(1)		
        magnet.set_units('T')	
        #magnetX.set_heater(1)		
        magnetX.set_units('T')			
        for i in arange(len(B_vector)):
            [starttime, counter] = timetrack.start(counter)
            tstart = timetrack.time()
            B = B_vector[i]				
            Bz = B_vector[i]*cos(sita*pi/180)
            Bx = B_vector[i]*sin(sita*pi/180)
			
            magnet.set_field(Bz) 					
            magnetX.set_field(Bx) 				
            ivvi.set(dacx,x_vector[0])
            qt.msleep(2)                                                                                                                   # use explained at the bottom of the script
            
            for x in x_vector:
                datavalues = self.take_data(dacx,x)                                                                                             # Go to next sweep value and take data
                data.add_data_point(x,B,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])                                                                          # write datapoint into datafile
            
            timetrack.remainingtime(starttime,Bsteps+1,counter)                                                                                 # Calculate and print remaining scantime
            data.new_block()
            
        data._write_settings_file()                                                                                                             # Overwrite the settings file created at the beginning, this ensures updating the sweep variable with the latest value
        data.close_file()
        qt.mend()
    	
	
    def _dac_vs_mwpower(self,xname,dacx,xstart,xend,xsteps,Pstart,Pend,Psteps,freq):
        qt.mstart()
        
        # Create sweep vectors
        x_vector = linspace(xstart,xend,xsteps+1)
        
        # P vector needs to have quadratic shape
        yend = (Pend-Pstart)*(Pend-Pstart)
        y_vector = linspace(0,yend,Psteps+1)
        P_vector = sqrt(y_vector)+Pstart
        
        z_vector = [0]
        
        data = self.create_data(x_vector,xname,dacx,P_vector,'Prf (dBm)','mw',z_vector,'none','z_parameter')                               # create data file, spyview metafile, copy script
        
        counter = 0
        
        # initialization of MW source
        mw.set_power(-145.0)
        mw.set_frequency(freq*1000000000)
        mw.set_status('on')
        
        for P in P_vector:
            [starttime, counter] = timetrack.start(counter)
            tstart = timetrack.time()
            mw.set_power(P)                                                                                                                     # Set mw power
            ivvi.set(dacx,x_vector[0])

            qt.msleep(2)                                                                                                                   # use explained at the bottom of the script
            
            for x in x_vector:
                datavalues = self.take_data(dacx,x)                                                                                             # Go to next sweep value and take data
                data.add_data_point(x,P,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])                           # write datapoint into datafile
            
            timetrack.remainingtime(starttime,Psteps+1,counter)                                                                                 # Calculate and print remaining scantime
            data.new_block()
        
        mw.set_status('off')
        data._write_settings_file()                                                                                                             # Overwrite the settings file created at the beginning, this ensures updating the sweep variable with the latest value
        data.close_file()
        qt.mend()
  
        
    # 2D scan of one dac vs another one
    def _dac_vs_mwfrequency(self,xname,dacx,xstart,xend,xsteps,fstart,fend,fsteps,power):
        qt.mstart()
        
        # Create sweep vectors
        x_vector = linspace(xstart,xend,xsteps+1)
        f_vector = linspace(fstart,fend,fsteps+1)
        z_vector = [0]
        
        data = self.create_data(x_vector,xname,dacx,f_vector,'frequency(GHz)','mw',z_vector,'none','z_parameter')                                     # create data file, spyview metafile, copy script
        
        counter = 0

        mw.set_power(power)
        mw.set_frequency(100e3)
        mw.set_status('on')
        
        for f in f_vector:
            [starttime, counter] = timetrack.start(counter)
            tstart = timetrack.time()
            mw.set_frequency(f*1000000000)			
            ivvi.set(dacx,x_vector[0])
            #T_mc = self.read_T()                                                                                                                # Read out mixing chamber temperature 
            qt.msleep(2)                                                                                                                   # use explained at the bottom of the script
            
            for x in x_vector:
                datavalues = self.take_data(dacx,x)                                                                                             # Go to next sweep value and take data
                data.add_data_point(x,f,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])                                                                          # write datapoint into datafile
            
            timetrack.remainingtime(starttime,fsteps+1,counter)                                                                                 # Calculate and print remaining scantime
            data.new_block()
        mw.set_status('off')       
		
        data._write_settings_file()                                                                                                             # Overwrite the settings file created at the beginning, this ensures updating the sweep variable with the latest value
        data.close_file()
        qt.mend()
    
    
    def _ramp_lockin1_amp(self, amplitude, step):
        x_start = lockin1.get_amplitude()
        x_end = amplitude
        x_vector = linspace(x_start, x_end, step+1)
        
        for x in x_vector:
            lockin1.set_amplitude(x)
            qt.msleep(0.05)
            lockin1.get_amplitude()  
    def _magnet_sweep(self,xname,dacx,x,Bstart,Bend):
        qt.mstart()
        
        # Create sweep vectors
        y_vector = [0]
        B_vector = [Bstart,Bend] #number of steps is undertermined at the beginning.Must added by hand later.
        z_vector = [0]
        magnet.set_field(Bstart)
        B = Bstart
        data = self.create_data(B_vector,'swept B(T)','IPS120-10',y_vector,'none','y_parameter',z_vector,'none','z_parameter')                                     # create data file, spyview metafile, copy script
        print 'Bstart=%r' % Bstart
        counter = 0
        #magnet.set_heater(1)		
        magnet.set_units('T')		
        magnet.set_field_no_wait(Bend)
        # for B in B_vector:
            # [starttime, counter] = timetrack.start(counter)
            # tstart = timetrack.time()
            # magnet.set_field(B) 			
            # kepco.set_B(B)                                                                                                                      # Specific for kepco as magnet supply
            # ivvi.set(dacx,x_vector[0])
            # T_mc = self.read_T()                                                                                                                # Read out mixing chamber temperature 
            # qt.msleep(0.01)                                                                                                                   # use explained at the bottom of the script
            
        # for x in x_vector:
            # datavalues = self.take_data(dacx,x)                                                                                             # Go to next sweep value and take data
            # data.add_data_point(x,B,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])                                                                          # write datapoint into datafile
        while abs(B-Bend)>0.0001 :
            fieldval = magnet.get_field()
            B_vector.append(B)
            datavalues = self.take_data(dacx,x)
            B = float(fieldval)
            print B
            data.add_data_point(x,B,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])            
        
        # timetrack.remainingtime(starttime,Bsteps+1,counter)                                                                                 # Calculate and print remaining scantime
        data.new_block()
            
        data._write_settings_file()                                                                                                             # Overwrite the settings file created at the beginning, this ensures updating the sweep variable with the latest value
        data.close_file()
        qt.mend()   
    
    def _magnet_vs_dac(self,xname,dacx,xstart,xend,xsteps,Bstart,Bend,Bsteps):
        qt.mstart()
        
        # Create sweep vectors
        x_vector = linspace(xstart,xend,xsteps+1)
        B_vector = linspace(Bstart,Bend,Bsteps+1)
        z_vector = [0]
        
        data = self.create_data(x_vector,xname,dacx,B_vector,'B(T)','magnet',z_vector,'none','z_parameter')                                     # create data file, spyview metafile, copy script
        
        counter = 0
        #magnet.set_heater(1)		
        magnet.set_units('T')		

        for x in x_vector:
            [starttime, counter] = timetrack.start(counter)
            tstart = timetrack.time()
            magnet.set_field(B_vector[0]) 			
            #kepco.set_B(B)                                                                                                                      # Specific for kepco as magnet supply
            ivvi.set(dacx,x)
            #T_mc = self.read_T()                                                                                                                # Read out mixing chamber temperature 
            qt.msleep(0.01)                                                                                                                   # use explained at the bottom of the script
            
            for B in B_vector:
                datavalues = self.take_data(dacx,x)  
                magnet.set_field(B)                                                                                                            # Go to next sweep value and take data
                while (abs(magnet.get_field()) - abs(B)) > 0.0005:
                      print(magnet.get_field())
                print(B, "T is reached")
                data.add_data_point(x,B,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])                                                                          # write datapoint into datafile
            
            timetrack.remainingtime(starttime,xsteps+1,counter)                                                                                 # Calculate and print remaining scantime
            data.new_block()
            
        data._write_settings_file()                                                                                                             # Overwrite the settings file created at the beginning, this ensures updating the sweep variable with the latest value
        data.close_file()
        qt.mend()             


    def _magnetX_sweep(self,xname,dacx,x,Bstart,Bend):
        qt.mstart()
        
        # Create sweep vectors
        y_vector = [0]
        B_vector = [Bstart,Bend] #number of steps is undertermined at the beginning.Must added by hand later.
        z_vector = [0]
        magnetX.set_field(Bstart)
        B = Bstart
        data = self.create_data(B_vector,'swept B(T)','IPS120-10',y_vector,'none','y_parameter',z_vector,'none','z_parameter')                                     # create data file, spyview metafile, copy script
        print 'Bstart=%r' % Bstart
        counter = 0
        #magnet.set_heater(1)		
        magnetX.set_units('T')		
        magnetX.set_field_no_wait(Bend)
        # for B in B_vector:
            # [starttime, counter] = timetrack.start(counter)
            # tstart = timetrack.time()
            # magnet.set_field(B) 			
            # kepco.set_B(B)                                                                                                                      # Specific for kepco as magnet supply
            # ivvi.set(dacx,x_vector[0])
            # T_mc = self.read_T()                                                                                                                # Read out mixing chamber temperature 
            # qt.msleep(0.01)                                                                                                                   # use explained at the bottom of the script
            
        # for x in x_vector:
            # datavalues = self.take_data(dacx,x)                                                                                             # Go to next sweep value and take data
            # data.add_data_point(x,B,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])                                                                          # write datapoint into datafile
        while abs(B-Bend)>0.0001 :
            fieldval = magnet.get_field()
            B_vector.append(B)
            datavalues = self.take_data(dacx,x)
            B = float(fieldval)
            print B
            data.add_data_point(x,B,0,datavalues[0],datavalues[1],datavalues[2],datavalues[3])            
        
        # timetrack.remainingtime(starttime,Bsteps+1,counter)                                                                                 # Calculate and print remaining scantime
        data.new_block()
            
        data._write_settings_file()                                                                                                             # Overwrite the settings file created at the beginning, this ensures updating the sweep variable with the latest value
        data.close_file()
        qt.mend() 




   
#################### INITIALIZATION #########################

# DON'T SKIP THIS PART, ITS CRUCIAL FOR PROPER MEASUREMENTS AND DATA PROCESSING!!!

# Gains and ranges
# Please set the gains and ranges before starting measurements. This ensures proper scaling of axis and data in Spyview.
# Make sure that you put the right gain at the right Keithley/Lockin.
GainK1=1e6                      # Gain for Keitley 1
#GainK2=1                     # Gain for Keitley 2
# GainL1=1e7                      # Gain for Lockin 1
#GainL2=1e2                      # Gain for Lockin 2
Vrange=10e-3                     # voltage range in V/V
#Irange=100e-9

#### change the parameters below if necessary ###



Voutput=10.0e-3  # mV
Imeasure=1.0e6    # amplification                 # current range in A/V
RC= 22.5  #KOhm                       #  for amp=1M, R_in=9.17KOhm; for amp=10M, R_in=22.5KOhm; for amp=100M, R_in=182KOhm;;; R_in=11.9KOhm for output*100mV, amp=1M

#lockin setting
Vlockout=500.0e-3
sensitivity= 2000.0e-6

Twait= 0.2
tau = 0.
average = 1

filename='data'

#################### MEASUREMENTS #########################

m = majorana() 
#m._single_dac_sweep('Vg(*30mV)','dac3',200,300,100)
#m._single_dac_sweep('Ibias(*0.1nA)','dac2',-300,300,300)
#magnet.set_field(0.2)
#m._dac_vs_dac('Ibias','dac2',-400,400,200,'Vbg','dac3',-76,300,188)
#m._single_dac_sweep('Vg(*30mV)','dac3',-2,0,20)
#magnet.set_field(0)


m._single_dac_sweep('Vbias(10mV/V)','dac1',400,100,100)

#m._single_dac_sweep('Vgate(30V/V)','dac2',50,45,10)

#m._magnet_sweep('vbias','dac1',100,-0.2,0.2)
#m._magnet_sweep('vbias','dac1',100,0.2,-0.2)

#m._ramp_lockin1_amp(1, 100)
#m._single_dac_sweep('Vbias(10mV/V)','dac1',100,00,100)

m._magnet_sweep('vbias','dac1',100,-0.2,0.2)
m._magnet_sweep('vbias','dac1',100,0.2,-0.2)

m._magnet_sweep('vbias','dac1',100,-0.2,0.2)
m._magnet_sweep('vbias','dac1',100,0.2,-0.2)

m._magnet_sweep('vbias','dac1',100,-0.2,0.2)
m._magnet_sweep('vbias','dac1',100,0.2,-0.2)


#m._magnet_vs_dac('Vbias','dac1',-1000,1000,40,-2,2,100)

#m._dac_vs_dac('Vbias','dac1',1000,-1000,50,'Vbg','dac3',00,300,200)

#m._single_dac_sweep('Vbias(*10uV)','dac1',500,00,100)

# m._single_dac_sweep('Ibias','dac2',100,0,100)

#m._single_dac_sweep('Vgate','dac2',-300,-00,100)

#m._dac_vs_dac('Vbias','dac1',1000,-1000,50,'Vbg','dac3',00,300,200)
#m._single_dac_sweep('Vbias','dac1',-1000,-00,100)
#m._single_dac_sweep('Vgate','dac3',300,00,200)
# #magnet.set_field(0)
# #m._single_dac_sweep('Vgate','dac3',-200,-0,2000)
# #m._single_dac_sweep('Vbias','dac1',000,100,100)

#m._single_dac_sweep('Vbias','dac1',-1000,1000,200)

# #m._dac_vs_magnet('Vbias','dac1',100,100,0,-0.5,0.5,400)

# #m._magnetX_sweep('Vbias','dac1',200,-0.2,0.2)

# #m._single_dac_sweep('Vg','dac3',30,0,300)
# # m._dac_vs_magnet('Ibias','dac2',0,0,0,0,0.05,100)

# #Start at Vgate = 1V (data_72)
# #m._single_dac_sweep('Vgate','dac3',200,150,20)
# #m._single_dac_sweep('Vbias','dac1',100,200,50)

# m._magnet_sweep('Vbias','dac1',200,-0.2,0.2)
# m._magnet_sweep('Vbias','dac1',200,0.2,-0.2)

# m._magnet_sweep('Vbias','dac1',200,-0.2,0.2)
# m._magnet_sweep('Vbias','dac1',200,0.2,-0.2)

# m._magnet_sweep('Vbias','dac1',200,-0.2,0.2)
# m._magnet_sweep('Vbias','dac1',200,0.2,-0.2)

# m._single_dac_sweep('Vbias','dac1',200,000,100)
# m._single_dac_sweep('Vbias','dac3',210,00,100)

# m._dac_vs_dac('Vbias','dac1',1000,-1000,50,'Vbg','dac3',00,200,200)

# m._single_dac_sweep('Vbias','dac1',-1000,000,100)
# m._single_dac_sweep('Vbias','dac3',200,00,100)

# m._dac_vs_magnetX('Vbias','dac1',200,200,0,-0.2,0.2,200)
# m._dac_vs_magnetX('Vbias','dac1',200,200,0,0.2,-0.2,200)

# m._dac_vs_magnetX('Vbias','dac1',200,200,0,-0.2,0.2,200)
# m._dac_vs_magnetX('Vbias','dac1',200,200,0,0.2,-0.2,200)

# m._dac_vs_magnetX('Vbias','dac1',200,200,0,-0.2,0.2,200)
# m._dac_vs_magnetX('Vbias','dac1',200,200,0,0.2,-0.2,200)


#m._dac_vs_magnetX('Vbias','dac1',1000,1000,0,0.2,-0.2,200)


#m._magnet_vs_dac('Vgate','dac8',100,400,30,-0.2,0.2,100)
#m._magnet_vs_dac('Vgate','dac8',100,400,30,0.2,-0.2,100)

#m._dac_vs_magnet('Ibias','dac2',0,0,0,-.5,.5,500)

# magnet.set_field(-0.5)
# m._dac_vs_magnet('Vbias','dac1',10,10,0,-0.2,0.2,200)
# magnet.set_field(0.5)
# m._dac_vs_magnet('Vbias','dac1',10,10,0,0.2,-0.2,200)
# # m._ramp_lockin1_amp(.1,200)
# magnet.set_field(-0.5)
#m._dac_vs_magnet('Ibias','dac2',0,0,0,-.5,.5,500)
# magnet.set_field(0.5)
#m._dac_vs_magnet('Ibias','dac2',0,0,0,.5,-.5,500)

# DC=-10nA, AC=1nA
# m._single_dac_sweep('Ibias','dac2',-10,0,100) #DC 10nA
# m._ramp_lockin1_amp(1,200)    #AC 1nA
# m._magnet_sweep('Ibias','dac2',0,-.5,.5)
# m._magnet_sweep('Ibias','dac2',0,.5,-.5)
# m._magnet_sweep('Ibias','dac2',0,-.5,.5)
# m._magnet_sweep('Ibias','dac2',0,.5,-.5)
# m._single_dac_sweep('Vgate','dac3',0,-100,1000)
# Brange = linspace (-0.5,1,31)
# for B in Brange:
    # magnet.set_field(B)
    # m._single_dac_sweep('Vgate','dac3',-100,100,2000)
    # m._single_dac_sweep('Vgate','dac3',100,-100,2000)
# m._single_dac_sweep('Vgate','dac3',-100,00,1000)    
# magnet.set_field(0)



'''
# DC=100nA, AC=1nA
m._single_dac_sweep('Ibias','dac2',10,100,1000) #DC 10nA
# m._ramp_lockin1_amp(0.1,200)    #AC 1nA

m._dac_vs_magnet('Ibias','dac2',100,100,0,-.5,.5,500)
m._dac_vs_magnet('Ibias','dac2',100,100,0,.5,-.5,500)

m._dac_vs_magnet('Ibias','dac2',100,100,0,-.5,.5,500)
m._dac_vs_magnet('Ibias','dac2',100,100,0,.5,-.5,500)
'''


#m._dac_vs_magnetX('Ibias','dac2',0,0,0,-0.2,0.2,200)
# m._dac_vs_magnetX('Ibias','dac2',0,0,0,0.2,-0.2,200)
# m._dac_vs_magnetX('Ibias','dac2',0,0,0,-0.2,0.2,200)
# m._dac_vs_magnetX('Ibias','dac2',0,0,0,0.2,-0.2,200)
#m._dac_vs_magnetX('Vgate','dac3',-100,100,200,-0.5,0.5,100)

#m._ramp_lockin1_amp(0.01, 100)

#The loop below start with file 909 (the gate scan)
# V_last =  45
# V_start = 100
# V_end = 40
# Steps = 12
# back_gate = linspace(V_start, V_end, Steps+1)
# for Vg in back_gate :
   # m._single_dac_sweep('Vgate','dac2',V_last,Vg,50)
   # #qt.msleep(60)
   # #m._single_dac_sweep('Vgate','dac2',V_last,Vg,100)
   
   # m._magnet_sweep('Vbias','dac1',00,-0.2,0.2)
   # m._magnet_sweep('Vbias','dac1',00,0.2,-0.2)
   # # #qt.msleep(60)
   # m._magnet_sweep('Vbias','dac1',00,-0.2,0.2)
   # m._magnet_sweep('Vbias','dac1',00,0.2,-0.2)
   # # #qt.msleep(60)
   # m._magnet_sweep('Vbias','dac1',00,-0.2,0.2)
   # m._magnet_sweep('Vbias','dac1',00,0.2,-0.2)
   
   # #m._dac_vs_magnetX('Vbias','dac1',1000,1000,0,-2,2,400)
   # #m._dac_vs_magnetX('Vbias','dac1',1000,1000,0,2,-2,400)
    
   # V_last = Vg

# # m._dac_vs_magnet('Vgate','dac2',100,0,100,-0.2,0.2,100)
# # m._dac_vs_magnet('Vgate','dac2',100,0,100,0.2,-0.2,100)
# #m._single_dac_sweep('Vbias(10mV/V)','dac1',100,000,100)

# m._ramp_lockin1_amp(0.0, 100)   
# m._single_dac_sweep('Vgate(30V/V)','dac2',40,0,100) 

# magnet.set_field(0)

   
   
# # # #m._magnet_sweep('Ibias','dac2',0,-.5,.5)
# # # #m._magnet_sweep('Ibias','dac2',0,.5,-.5)

# m._single_dac_sweep('Vbias','dac1',200,00,100)
# m._single_dac_sweep('Vgate','dac3',400,00,100)   
# m._single_dac_sweep('Vbias','dac1',0,1000,100)
# #m._single_dac_sweep('Vgate','dac3',0,200,100)   

# m._dac_vs_dac('Vbias','dac1',1000,-1000,50,'Vbg','dac3',00,300,300)  
# m._single_dac_sweep('Vbias','dac1',-1000,00,100)
# m._single_dac_sweep('Vgate','dac3',300,00,100) 
   
#m._dac_vs_magnetX('Vbias','dac9',0,0,0,0,-0.1,100)   

#m._dac_vs_2magnets('Ibias','dac2',100,-100,25,0,1,20,5)
#m._dac_vs_2magnets('Ibias','dac2',-100,100,25,1,0,20,-5)

#m._dac_vs_magnetX('dac16','dac16',0,5,5,0.5,0,20)
#m._dac_vs_magnet('dac16','dac16',0,5,5,0.5,0,20)

###### Jen's scans 8/23/2017 ##########
#ivvi.set_dac1(0)
#ivvi.set_dac11(-367.6)
#m._dac_vs_dac('Vbias','dac1',-85,115,400,'Vbg','dac11',-367.6,-367.6,1)
#magnet.set_field(0.6)
#magnet.set_field(0.32)
#keithley1.set_averaging(1)
#m._single_dac_sweep('Vbias','dac1',-85,115,1000)
#m._dac_vs_magnet('Vbias','dac1',-85,115,400,0.6,0.2,10)
#ivvi.set_dac1(0)
#keithley1.set_averaging(0)
###### End Jen's scans #####################################



###################1D scan: scan Vbg#####################

#ivvi.set_dac1(500)
#m._single_dac_sweep('Vg(*30mV)','dac3',1500,0,1500)
#m._single_dac_sweep('Vg(*30mV)','dac3',1000,0,300)
#ivvi.set_dac1(0)


###################1D scan: scan Ibias#####################

'''
#ivvi.set_dac3(1300)
m._single_dac_sweep('Ibias(*0.1nA)','dac2',-100,100,200)
m._single_dac_sweep('Ibias(*0.1nA)','dac2',100,-100,200)
'''


###################1D scan: scan Ibias#####################
#m._single_dac_sweep('Vbias','dac1',400,-400,200)
#m._single_dac_sweep('Vbias','dac1',-400,400,200)



###################2D scan:Vbias vs Vbg ###################
#magnet.set_magnetout(0)

#ivvi.set_dac1(-100)
#m._single_dac_sweep('Vbg','dac3',0,-600,100)
#m._dac_vs_dac('Vbias','dac1',-100,100,100,'Vbg','dac3',0,-1000,200)

#ivvi.set_dac3(1500)
#m._dac_vs_magnet('Ibias','dac2',-50,150,150,-50,800,90)
#m._dac_vs_magnet('Ibias','dac2',-50,150,150,800,0,40)


###################2D scan:Ibias vs Vbg ###################

#ivvi.set_dac3(1500)



###################2D scan:Ibias vs B #####################


###################2D scan:Ibias vs vectorB #####################
#_dac_vs_2magnets(self,xname,dacx,xstart,xend,xsteps,Bstart,Bend,Bsteps,sita)



#ivvi.set_dac3(1500)

###################2D scan:Ibias vs mwPower #####################



#Vbg=1500
#m._dac_vs_mwfrequency('Ibias','dac2',-100,500,180,1,4,1000,10)


#m._dac_vs_dac('Ibias','dac2',-30,30,120,'Vbg','dac3',900,815,85)

#ivvi.set_dac3(1500)
#m._dac_vs_2magnets('Vbias(*10uV)','dac1',0,100,100,0,0.01,10,30)



##############

###################2D scan:Ibias vs mwfrequency #####################
#m_dac_vs_mwfrequency(self,xname,dacx,xstart,xend,xsteps,fstart,fend,fsteps,power)

'''
ivvi.set_dac1(1000)
m._5dacs_samesweep('FG1','dac15','FG2','dac16','FG3','dac7','FG4','dac8','BG1','dac11',0,500,500)


ivvi.set_dac1(1000)
m._single_dac_sweep('FG1','dac15',600,-400,400)
m._single_dac_sweep('FG1','dac15',-400,600,400)

m._single_dac_sweep('FG2','dac16',600,-400,400)
m._single_dac_sweep('FG2','dac16',-400,600,400)

m._single_dac_sweep('FG3','dac7',600,-400,400)
m._single_dac_sweep('FG3','dac7',-400,600,400)

m._single_dac_sweep('FG4','dac8',600,-400,400)
m._single_dac_sweep('FG4','dac8',-400,600,400)

m._single_dac_sweep('BG1','dac11',400,-400,400)
m._single_dac_sweep('BG1','dac11',-400,200,200)

m._single_dac_sweep('BG2','dac12',400,-400,400)
m._single_dac_sweep('BG2','dac12',-400,200,200)

m._single_dac_sweep('BG3','dac13',400,-400,400)
m._single_dac_sweep('BG3','dac13',-400,200,200)

ivvi.set_dac1(0)
'''

'''
#Don't modify the program below please, if you need, just copy it!!!
ivvi.set_dac1(200)
m._single_dac_sweep('FG1','dac15',0,400,400)

m._single_dac_sweep('FG2','dac16',0,400,400)

m._single_dac_sweep('FG3','dac7',0,400,400)

m._single_dac_sweep('FG4','dac8',0,400,400)

m._single_dac_sweep('BG1','dac11',0,200,400)

m._single_dac_sweep('BG2','dac12',0,200,400)

m._single_dac_sweep('BG3','dac13',0,200,400)

m._single_dac_sweep('BG4','dac14',0,200,400)



ivvi.set_dac1(0)

m._single_dac_sweep('FG1','dac15',400,0,400)

m._single_dac_sweep('FG2','dac16',400,0,400)

m._single_dac_sweep('FG3','dac7',400,0,400)

m._single_dac_sweep('FG4','dac8',400,0,400)

m._single_dac_sweep('BG1','dac11',400,0,400)

m._single_dac_sweep('BG2','dac12',400,0,400)

m._single_dac_sweep('BG3','dac13',400,0,400)

m._single_dac_sweep('BG4','dac14',400,0,400)

'''


'''
#Don't modify the program below please, if you need, just copy it!!!

ivvi.set_dac1(1000)

m._single_dac_sweep('FG1','dac15',400,-200,400)
m._single_dac_sweep('FG1','dac15',-200,400,400)
 
m._single_dac_sweep('FG2','dac16',400,-200,400)
m._single_dac_sweep('FG2','dac16',-200,400,400)

m._single_dac_sweep('FG3','dac7',400,-200,400)
m._single_dac_sweep('FG3','dac7',-200,400,400)

m._single_dac_sweep('FG4','dac8',400,0,400)
m._single_dac_sweep('FG4','dac8',0,400,400)

m._single_dac_sweep('BG1','dac11',400,-200,400)
m._single_dac_sweep('BG1','dac11',-200,400,400)

m._single_dac_sweep('BG2','dac12',200,0,200)

m._single_dac_sweep('BG3','dac13',200,0,200)




'''




'''
#just for NW-DOWN


ivvi.set_dac1(1000)
m._single_dac_sweep('FG5','dac15',0,400,400)

m._single_dac_sweep('FG6','dac16',0,400,400)

m._single_dac_sweep('FG7','dac7',0,400,400)

m._single_dac_sweep('FG8','dac8',0,400,400)

m._single_dac_sweep('BG3','dac11',0,400,400)

m._single_dac_sweep('BG2','dac12',0,200,0)

m._single_dac_sweep('BG1','dac13',0,200,0)


m._single_dac_sweep('FG5','dac15',400,-300,400)
m._single_dac_sweep('FG5','dac15',-300,400,400)

m._single_dac_sweep('FG6','dac16',400,-300,400)
m._single_dac_sweep('FG6','dac16',-300,400,400)

m._single_dac_sweep('FG7','dac7',400,-300,400)
m._single_dac_sweep('FG7','dac7',-300,400,400)


m._single_dac_sweep('BG3','dac11',400,-300,400)
m._single_dac_sweep('BG3','dac11',-300,400,400)

m._single_dac_sweep('BG2','dac12',200,0,200)
m._single_dac_sweep('BG2','dac12',0,200,200)


m._single_dac_sweep('BG1','dac13',200,0,200)
m._single_dac_sweep('BG1','dac13',0,200,200)


'''



