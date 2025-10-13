#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:36:02 2020

@author:
mik
"""
from io import StringIO
from subprocess import call

import shlex, subprocess

import numpy as np
import matplotlib.pyplot as plt
import tarfile


path_file = "/home/mik/Geant4-10.06p2/g4_work/Pixel/Build/"

frame = 4

n = 2 

filename = "pixel"

# set ion: Atomic Number and Weight (e.g., "38 90" for Sr90, "55 137" for Cs137, "95 241" for Am241 etc.)
setSource ="Sr90"

setThreshold = 0     #eV

setDetec_substra = "true"
setSi_substrat_thick = 225   #um
setDetec_shield = "true"
setDetc_shield_thick = 200  #um

'''
setpixel_x = 1.4     #um
setpixel_y = 1.4     #um
setN_pixel_x = 2592
setN_pixel_y = 1944
'''

setpixel_x = 453.6     #um
setpixel_y = 553.6     #um
setN_pixel_x = 8
setN_pixel_y = 6

setpixel_z = 3     #um
setpixels_gap = 0     #um

Visual_setting = False


for d in range(2):

    setDistance_SD = 2*d   #1mm

    if(setDetec_substra=="true"): filesave="data_"+setSource+"_pixl_thickZ_"+str(setpixel_z)+"um_"+"distSD_"+str(setDistance_SD)+"mm"+"_sustrat"
    else: filesave="data_"+setSource+"_pixl_thickZ_"+str(setpixel_z)+"um_"+"distSD_"+str(setDistance_SD)+"mm"
    call("mkdir "+path_file+filesave, shell=True)    
    
    k = 0
    
    for k in range(int(frame/n)):
        i = 0
        j = 0
        
        o_file3 = open(path_file+setSource+"_SD"+str(setDistance_SD)+"mm_loop_"+str(frame)+".mac", "w")
        
        o_file3.write(
            "# Macro file for the initialization of pixel\n"
            "# in interactive session\n"
            "# Set some default verbose\n\n"
            
            "/control/verbose 2\n"
            "/control/saveHistory\n"
            "/run/verbose 2\n\n"
            
            "/pixel/run/setThreshold " + str(setThreshold)+" eV\n"
            "/pixel/det/setDistance_SD " + str(setDistance_SD)+" mm\n"
            "/pixel/det/setDetec_substra " + setDetec_substra+ "\n"
            "/pixel/det/setSi_substrat_thick " + str(setSi_substrat_thick) +" um\n"
            "/pixel/det/setDetec_shield " + setDetec_shield + "\n"
            "/pixel/det/setDetc_shield_thick " + str(setDetc_shield_thick)+" um\n"
            "/pixel/det/setpixel_x " + str(setpixel_x) + " um\n" #1.4 um
            "/pixel/det/setpixel_y " + str(setpixel_y) + " um\n"#1.4 um
            "/pixel/det/setpixel_z " + str(setpixel_z) + " um\n"#  10-20 µm
            "/pixel/det/setpixels_gap " + str(setpixels_gap) + " um\n"
            "/pixel/det/setN_pixel_x " + str(setN_pixel_x) + "\n"#1296 #2592
            "/pixel/det/setN_pixel_y " + str(setN_pixel_y) + "\n"#972 #1944
            "/pixel/run/setN_pixel_x "+ str(setN_pixel_x) + "\n"
            "/pixel/run/setN_pixel_y "+ str(setN_pixel_y) + "\n"
            "/pixel/det/setSource " + setSource +"\n"
            "/pixel/run/setSource " + setSource +"\n\n"
            
            "#Initialize kernel\n"
            "/run/initialize\n"
        )
        
        if(setSource=="Sr90"):
            o_file3.write(
                "/gps/particle ion\n\n"
                
                "#Sr90\n"
                "/gps/ion 38 90 \n\n"
                
                "# Cylindrical source\n"
                "/gps/pos/type Surface\n"
                "/gps/pos/shape Cylinder\n"
                "/gps/pos/radius 3175  um\n"
                "/gps/pos/halfz  1587.5 um\n\n"
            )
            
        if(setSource=="Cs137"):
            o_file3.write(
                "/gps/particle ion\n\n"
                
                "#Cs137\n"
                "/gps/ion 55 137 \n\n"
                
                "# Cylindrical source\n"
                "/gps/pos/type Surface\n"
                "/gps/pos/shape Cylinder\n"
                "/gps/pos/radius 3175  um\n"
                "/gps/pos/halfz  1587.5 um\n\n"
            )
        
        if(setSource=="Am241"):
            o_file3.write(
                "/gps/particle ion\n\n"
                
                "#Cs137\n"
                "/gps/ion 95 241 \n\n"
                
                "# Cylindrical source\n"
                "/gps/pos/type Surface\n"
                "/gps/pos/shape Cylinder\n"
                "/gps/pos/radius 2 mm\n"
                "/gps/pos/halfz  750 um\n\n"
            ) 
        
        if(setSource=="Fe55"):
            o_file3.write(
                "/gps/particle ion\n\n"
                
                "#Fe55\n"
                "/gps/ion 26 55 \n\n"
                
                "# Cylindrical source\n"
                "/gps/pos/type Surface\n"
                "/gps/pos/shape Cylinder\n"
                "/gps/pos/radius 6 mm\n"
                "/gps/pos/halfz  125 um\n\n"
            )
        
        if(Visual_setting==True):
            o_file3.write(
                "# Visualization setting\n"
                "/control/execute vis.mac\n"
                "/vis/viewer/set/viewpointThetaPhi 90. 180.\n\n"
            )
            
        for i in range(n):
            print("/pixel/run/setFilename "+filename + str(i+n*k) +"_SD"+ str(setDistance_SD)+"mm")
            print("/run/beamOn 13388")
            o_file3.write("/pixel/run/setFilename "+filename + str(i+n*k) +"_SD"+ str(setDistance_SD)+"mm"+"\n")
            if(setSource=="Sr90"):
                o_file3.write("/run/beamOn 13388\n")
            if(setSource=="Cs137"):
                o_file3.write("/run/beamOn 610000\n")
            if(setSource=="Am241"):
                o_file3.write("/run/beamOn 73000\n")
            if(setSource=="Fe55"):
                o_file3.write("/run/beamOn 12000000\n")
        o_file3.close()
        #call("rm "+filecomp, shell=True)
    
        commandlist=  path_file+"pixel "+ setSource+"_SD"+str(setDistance_SD)+"mm_loop_"+str(frame)+".mac"
        #commandlist= "./pixel "+ setSource+"_loop_"+str(frame)+".mac >>/dev/null"
        
        call(commandlist, shell=True)
        
        #args = shlex.split(commandlist)
        #subprocess.call(args)
        
        
        for j in range(n):
            name=path_file+filename + str(j+n*k) +"_SD"+ str(setDistance_SD)+"mm"+"_"+setSource+"_pixel_edep2D.csv"
            print(name)
            #data = np.loadtxt(name, delimiter="\t", skiprows=2, usecols=[0,1,2])
            data = np.loadtxt(name, delimiter="\t", skiprows=1, usecols=[0,1,2])
            #dat = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=[1,2,3])
            
            #height= np.int32(data[0][1])
            #width= np.int32(data[0][0])
            height=setN_pixel_y
            width=setN_pixel_x
            
            if(data.ndim == 1):
                data_2d = np.zeros((height, width))
                data_2d_full = np.zeros((height, width))
                print("Array Zeros")
                
            else:    
                ########################################
                data = np.delete(data, 0, axis=0)
                
                dmax=4.31 #### well capacity
                #dmax=np.amax(data,0)[2]
                #dmin=np.amin(data,0)[2]
                thresh=0.07
                
                dat_b=(data[:,2]-thresh)*255/(dmax-thresh)
                dat_b[dat_b<0]=0
                dat_b[dat_b>255]=255
                
                #height=1944
                #width=2592
    
                data_2d = np.zeros((height, width))
                data_2d_full = np.zeros((height, width))
                data_2d[data[:,1].astype(np.int32),data[:,0].astype(np.int32)]=dat_b.astype(np.int32)
                data_2d_full[data[:,1].astype(np.int32),data[:,0].astype(np.int32)]=data[:,2]
            
            
            np.savez_compressed("Y_CDS_z" +str(int(setDistance_SD/2)).zfill(3)+ "_f"+ str(j+n*k).zfill(3) + ".npz", np.int32(data_2d))
            
            print("Save Compressed npz: "+"Y_CDS_z" +str(setDistance_SD).zfill(3)+ "_f"+ str(j+n*k).zfill(3) + ".npz")
                
            #if(setSource=="Sr90"):
            #    filecomp=filename + str(j+n*k) +"_SD"+ str(setDistance_SD)+"mm"+"_"+setSource+"*.csv "+filename + str(j+n*k) +"_SD"+ str(setDistance_SD)+"mm"+"_"+setSource+"*.root"
            #if(setSource=="Fe55" or setSource=="Am241"  or setSource=="Cs137"): 
            filecomp=filename + str(j+n*k) +"_SD"+ str(setDistance_SD)+"mm"+"_"+setSource+"*.csv "              
                
            namecomp="tar -cjvf "+"\""+"Yz"+str(setDistance_SD).zfill(3)+"mm_f"+str(j+n*k) + "_"+ setSource+".tar.bz2"+"\" "+filecomp
        
            print("Save Compressed "+filecomp+" : "+"Yz0f"+str(j+n*k)+".tar.bz2")
            call(namecomp, shell=True)
                  
            #if(setDetec_substra=="true"): filesave="data_"+setSource+"_pixl_thickZ_"+str(setpixel_z)+"um_"+"distSD_"+str(setDistance_SD)+"mm"+"_sustrat"
            #else: filesave="data_"+setSource+"_pixl_thickZ_"+str(setpixel_z)+"um_"+"distSD_"+str(setDistance_SD)+"mm"
            #if(k==0):call("mkdir "+filesave, shell=True)
                  
            call("mv Y*.npz Y*.tar.bz2 "+path_file+filesave, shell=True)
                         
            print("Delete files: "+filecomp)
            call("rm "+filecomp, shell=True)
            #call("rm *.ascii", shell=True)
            
