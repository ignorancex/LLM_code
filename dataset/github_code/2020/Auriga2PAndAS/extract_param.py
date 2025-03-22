#!/usr/bin/python
#########################################################################
## Use to extract a parameter from the raw Auriga simulation           ##
## Oct. 2018 by Guillaume THOMAS guillaume.thomas@nrc-cnrc.gc.ca       ##
## Copyright (c) 2018 National Research Concil. All rights reserved.   ##
#########################################################################

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from astropy.table import Table, hstack
from numpy.linalg import inv


# For using C subroutines
import ctypes as C
double_pointer_pointer=C.POINTER(C.POINTER(C.c_double))
double_pointer = C.POINTER(C.c_double)
long_pointer = C.POINTER(C.c_long) 

import sys
import os


### Save the data ###
def save_data(data,name="data_out.asc"):
    header=' '.join(list(data))
    np.savetxt(name,data,delimiter=" ", header=header)
    print "\nData saved\n"



###################################
### Save the residual particles ###
###################################
def save(part,infile):

	# Get the name
	x=infile.split("_")

	# Save in fits format
	c0 = fits.Column(name='ID', array=part['ID'], format='D')
	c1 = fits.Column(name='met_sim', array=part['met_sim'], format='D')

	t = fits.BinTableHDU.from_columns([c0, c1])

	t.writeto("./results/mock_"+x[1]+"_"+x[2]+"_"+x[3]+"_param.fits",overwrite=True)

	print "Data Saved"


##################################
## Read the section of the file ##
##################################
def read_file(data,param_get,param_name):
	part=pd.DataFrame()
	part['ID']=np.array(data["/Stars/ParticleIDs"][:])
	part[param_name]=np.array(data[param_get][:])
	print "Part Readed"
	return part


###############################
### Read the mock file      ###
###############################
def read_mock(addcat,IDmin,IDmax):
	hdulist = fits.open(addcat)
	data_fits=hdulist[1].data
	part= pd.DataFrame()
	# Load position
	part['ID']=data_fits.field(0).byteswap().newbyteorder()[IDmin:IDmax]
	part['ra']=data_fits.field(1).byteswap().newbyteorder()[IDmin:IDmax]
	part['dec']=data_fits.field(2).byteswap().newbyteorder()[IDmin:IDmax]
	part['xki']=data_fits.field(3).byteswap().newbyteorder()[IDmin:IDmax]
	part['eta']=data_fits.field(4).byteswap().newbyteorder()[IDmin:IDmax]
	part['rhelio']=data_fits.field(5).byteswap().newbyteorder()[IDmin:IDmax]
	part['pmra']=data_fits.field(6).byteswap().newbyteorder()[IDmin:IDmax]
	part['pmdec']=data_fits.field(7).byteswap().newbyteorder()[IDmin:IDmax]
	part['Vrad']=data_fits.field(8).byteswap().newbyteorder()[IDmin:IDmax]
	part['g']=data_fits.field(9).byteswap().newbyteorder()[IDmin:IDmax]
	part['dg']=data_fits.field(10).byteswap().newbyteorder()[IDmin:IDmax]
	part['g0']=data_fits.field(11).byteswap().newbyteorder()[IDmin:IDmax]
	part['i']=data_fits.field(12).byteswap().newbyteorder()[IDmin:IDmax]
	part['di']=data_fits.field(13).byteswap().newbyteorder()[IDmin:IDmax]
	part['i0']=data_fits.field(14).byteswap().newbyteorder()[IDmin:IDmax]
	part['EBV']=data_fits.field(15).byteswap().newbyteorder()[IDmin:IDmax]
	part['nb_field']=data_fits.field(16).byteswap().newbyteorder()[IDmin:IDmax]
	part['x']=data_fits.field(17).byteswap().newbyteorder()[IDmin:IDmax]
	part['y']=data_fits.field(18).byteswap().newbyteorder()[IDmin:IDmax]
	part['z']=data_fits.field(19).byteswap().newbyteorder()[IDmin:IDmax]
	part['Mg']=data_fits.field(20).byteswap().newbyteorder()[IDmin:IDmax]
	part['Mi']=data_fits.field(21).byteswap().newbyteorder()[IDmin:IDmax]
	part['met_sim']=data_fits.field(22).byteswap().newbyteorder()[IDmin:IDmax]
	hdulist.close()
	
	print "\n********************************"
	print "Data loaded: %d" %(len(part))
	
	return part


###############################
### Read the tmp      ###
###############################
def read_tmp(addcat):
	hdulist = fits.open(addcat)
	data_fits=hdulist[1].data
	part= pd.DataFrame()
	# Load position
	part['ID']=data_fits.field(0).byteswap().newbyteorder()
	part['ra']=data_fits.field(1).byteswap().newbyteorder()
	part['dec']=data_fits.field(2).byteswap().newbyteorder()
	part['xki']=data_fits.field(3).byteswap().newbyteorder()
	part['eta']=data_fits.field(4).byteswap().newbyteorder()
	part['rhelio']=data_fits.field(5).byteswap().newbyteorder()
	part['pmra']=data_fits.field(6).byteswap().newbyteorder()
	part['pmdec']=data_fits.field(7).byteswap().newbyteorder()
	part['Vrad']=data_fits.field(8).byteswap().newbyteorder()
	part['g']=data_fits.field(9).byteswap().newbyteorder()
	part['dg']=data_fits.field(10).byteswap().newbyteorder()
	part['g0']=data_fits.field(11).byteswap().newbyteorder()
	part['i']=data_fits.field(12).byteswap().newbyteorder()
	part['di']=data_fits.field(13).byteswap().newbyteorder()
	part['i0']=data_fits.field(14).byteswap().newbyteorder()
	part['EBV']=data_fits.field(15).byteswap().newbyteorder()
	part['nb_field']=data_fits.field(16).byteswap().newbyteorder()
	part['x']=data_fits.field(17).byteswap().newbyteorder()
	part['y']=data_fits.field(18).byteswap().newbyteorder()
	part['z']=data_fits.field(19).byteswap().newbyteorder()
	part['Mg']=data_fits.field(20).byteswap().newbyteorder()
	part['Mi']=data_fits.field(21).byteswap().newbyteorder()
	part['met_sim']=data_fits.field(22).byteswap().newbyteorder()
	part[param_name]=data_fits.field(23).byteswap().newbyteorder()
	hdulist.close()
	
	print "\n********************************"
	print "Data loaded: %d" %(len(part))
	
	return part

###################################
### Save the residual particles ###
###################################
def save(part,num):


	# Save in fits format
	c0 = fits.Column(name='ID', array=part['ID'], format='D')
	c1 = fits.Column(name='RA', array=part['ra'], format='D')
	c2 = fits.Column(name='Dec', array=part['dec'], format='D')
	c3 = fits.Column(name='xki', array=part['xki'], format='D')
	c4 = fits.Column(name='eta', array=part['eta'], format='D')
	c5 = fits.Column(name='rhelio', array=part['rhelio'], format='D')
	c6 = fits.Column(name='pmra', array=part['pmra'], format='D')
	c7 = fits.Column(name='pmdec', array=part['pmdec'], format='D')
	c8 = fits.Column(name='Vrad', array=part['Vrad'], format='D')
	c9 = fits.Column(name='g', array=part['g'], format='D')
	c10 = fits.Column(name='dg', array=part['dg'], format='D')
	c11 = fits.Column(name='g0', array=part['g0'], format='D')
	c12 = fits.Column(name='i', array=part['i'], format='D')
	c13 = fits.Column(name='di', array=part['di'], format='D')
	c14 = fits.Column(name='i0', array=part['i0'], format='D')
	c15 = fits.Column(name='EBV', array=part['EBV'], format='D')
	c16 = fits.Column(name='nb_field', array=part['nb_field'], format='D')
	c17 = fits.Column(name='x', array=part['x'], format='D')
	c18 = fits.Column(name='y', array=part['y'], format='D')
	c19 = fits.Column(name='z', array=part['z'], format='D')
	c20 = fits.Column(name='Mg', array=part['Mg'], format='D')
	c21 = fits.Column(name='Mi', array=part['Mi'], format='D')
	c22 = fits.Column(name='met_sim', array=part['met_sim'], format='D')
	c23 = fits.Column(name=param_name, array=part[param_name], format='D')

	t = fits.BinTableHDU.from_columns([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23])
	if(num>=0):
		t.writeto("tmp%d_extract.fits"%(num),overwrite=True)
	else:
		t.writeto("./results/mockwparam.fits",overwrite=True)
	del(t);del(c0);del(c1);del(c2);del(c3);del(c4);del(c5);del(c6);del(c7);del(c8);
	del(c9);del(c10);del(c11);del(c12);del(c13);del(c14);del(c15);del(c16);del(c17);
	del(c18);del(c19);del(c20);del(c21);del(c22);del(c23);del(part)
	print "Data Saved"







#######################
##       MAIN        ##
#######################
if __name__=='__main__':
    
    num_split=30

	# Read the simulation file and obtain the number of line of it
    if (len(sys.argv[:])<3):
        print "extract_param.py catalogue_path mock_to_add_path"
    else:    
    
        infile=sys.argv[1]
        addcat=sys.argv[2]

        param_get="/Stars/age"#raw_input("Which parameter to get? ")
        param_name="age"#raw_input("What is it name? ")
        
        # Load the Full Auriga catalogue
        data = h5py.File(infile, 'r')
        Auriga=read_file(data,param_get,param_name)
        print len(Auriga)
        Auriga=Auriga.sort_values(by=['ID'])
        Auriga=Auriga.reset_index(drop=True)
        Auriga.drop_duplicates(subset ="ID", keep = 'first', inplace = True)  # Remove duplicate ID
        Auriga=Auriga.sort_values(by=['ID'])
        Auriga=Auriga.reset_index(drop=True)
        print len(Auriga)

        
        # Do the cross match the catalog for each portion of the file (due to memory)
        hdulist = fits.open(addcat)
        data_fits=hdulist[1].data
        hdulist.close()
        len_cat=len(data_fits.field(0).byteswap().newbyteorder())
        len_portion=(int)(len_cat/(num_split*1.0))
        del(data_fits)
        
        for i in range(0,num_split):
            print i
            IDmin=i*len_portion
            IDmax=(i+1)*len_portion
            if(i==num_split):
                IDmax=len_cat+1
            cat=read_mock(addcat,IDmin,IDmax)
            df_left = pd.merge(cat, Auriga, on='ID', how='left')
            print len(df_left),df_left.columns
            save(df_left,i+1)
            del(df_left);del(cat)

            
        del(Auriga)
              
        part=read_tmp("tmp1_extract.fits")
        for i in range(1,num_split):
            print "extract",i
            tmp=read_tmp("tmp%d_extract.fits"%(i+1))
            part=pd.concat([part,tmp])
            del tmp
        
        save(part,-99)
