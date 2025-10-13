#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Jul 15 11:36:02 2020

@author:
mik
'''
import sys
import os
from io import StringIO
from subprocess import call

import shlex, subprocess

import numpy as np
#import matplotlib.pyplot as plt
#import tarfile


#path_file = '/home/hep_group01/Geant4.10.06.p02/g4_work/pixel0/Build/'
#path_file = '/home/mik/geant4.10.07/g4_work/Pixel_cds_0/Build/'
path_file = '/media/hep_group01/761C6DEA1C6DA5B91/g4_work/Pixel_RGB_xy_emit/Build/'
#path_file = '/home/mik/geant4.10.07/g4_work/Pixel_RGB_xy/Build/'


numberOfThreads = 6

frame = 1

n = 1

filename = 'pixel'

# set ion: Atomic Number and Weight (e.g., '38 90' for Sr90, '55 137' for Cs137, '95 241' for Am241 etc.)
#setSource ='Cs137'

setThreshold = 0     #eV

setDetec_substra = 'false'
setSi_substrat_thick = 3   #um
setDetec_shield = 'false'
setDetc_shield_thick = 200  #um

setDetec_SiO2_layers = 'true'
setDetc_SiO2_layers_thick = 0.225
#set_thick_shiel_down = 500
#set_thick_shiel_up = 2600.
#halfz_sr_cs = 100

#halfz_sr_cs = (3175 - set_thick_shiel_down - set_thick_shiel_up)
#set_thick_shiel_up = (3175 - set_thick_shiel_down -halfz_sr_cs)
#print(set_thick_shiel_down, halfz_sr_cs, set_thick_shiel_up)

setlens_thick = 0.735  # prom 0.5 um
setcolrfltr_thick = 0.9  # 0.79um

setpixel_x = 1.4     #um
setpixel_y = 1.4     #um
setN_pixel_x = 2592
setN_pixel_y = 1944

setpixel_z = 2    #um
setpixels_gap = 0.0    #um

distan_SD = 1

Visual_setting = False

setSaveProcPart = 'false'
#set_thick_shiel_up=0

#ang1 = 0

for set_thick_shiel_down in [
                            # 635.000000000001,  # Cs137
                            # 508.000000000001,  # Cs137
                             254.000000000001, # Sr90
                             #381.000000000001
                             ]:
    #halfz_sr_cs = 100
    for halfz_sr_cs in [ 100,]:
    #for halfz_sr_cs in [0,25,100]:

        set_thick_shiel_up = (3175 - set_thick_shiel_down -halfz_sr_cs)
        print(set_thick_shiel_down, halfz_sr_cs, set_thick_shiel_up)

        for setSource in [
                         'Cs137',
                         'Sr90',
                            ]:
            if(setSource=='Sr90'): n_evt = '1514'
            #if(setSource=='Cs137'): n_evt = '3811'
            if(setSource=='Cs137'): n_evt = '3811'

        #setSource='Sr90'
        #for n_evt in [ '1514', ]:

        #for rot_1 in [ 0.00873, 0.01746, 0.0349, 0.05241, 0.0699, 0.08749  ]:
        #    if( rot_1 == 0.00873): ang1 = 0.5
        #    if( rot_1 == 0.01746): ang1 = 1
        #    if( rot_1 == 0.0349): ang1 = 2
        #    if( rot_1 == 0.05241): ang1 = 3
        #    if( rot_1 == 0.0699): ang1 = 4
        #    if( rot_1 == 0.08749): ang1 = 5
        #    if( rot_1 == 0.105104): ang1 = 6
        #    if( rot_1 == 0.12278): ang1 = 7

            thickback = str(round(set_thick_shiel_down))+'_'+str(halfz_sr_cs)+'um'+'_ev'+str(n_evt) #+'_rot1_'+str(ang1)

            filesave0='data_'+setSource+'_pixl_thickZ_'+str(setpixel_z)+'um'+'_'+str(distan_SD)+'level'
            path1 = path_file+filesave0+'_'+thickback #+ 'r1250'
            path2 = path_file+filesave0+'_edep_pixel'+'_'+thickback# + 'r1250'
            #call('mkdir '+path1, shell=True)
            #call('mkdir '+path2, shell=True)
            try:
                os.makedirs(path1)
            except FileExistsError:
                # directory already exists
                pass
            try:
                os.makedirs(path2)
            except FileExistsError:
                # directory already exists
                pass



            for d in range(distan_SD):

                setDistance_SD = 2*d   #1mm

        #        if(setDetec_substra=='true'): filesave='data_'+setSource+'_pixl_thickZ_'+str(setpixel_z)+'um_'+'distSD_'+str(setDistance_SD)+'mm'+'_sustrat'
        #        else: filesave='data_'+setSource+'_pixl_thickZ_'+str(setpixel_z)+'um_'+'distSD_'+str(setDistance_SD)+'mm'
        #        call('mkdir '+path_file+filesave0+'/'+filesave, shell=True)
                #filesave='data_'+setSource+'_pixl_thickZ_'+str(setpixel_z)+'um_'+'distSD_'+str(setDistance_SD)+'mm'+'_'+str(distan_SD)+'level'+'_rot1_'+str(ang1)+'_csv_tar_bz'
                filesave='data_'+setSource+'_pixl_thickZ_'+str(setpixel_z)+'um_'+'distSD_'+str(setDistance_SD)+'mm'+'_'+str(distan_SD)+'level'+'_csv_tar_bz'
                #call('mkdir '+path1+'/'+filesave, shell=True)
                try:
                    os.makedirs(path1+'/'+filesave)
                except FileExistsError:
                    # directory already exists
                    pass


                k = 0

                for k in range(int(frame/n)):
                #for k in range(250, 500):
                    i = 0
                    j = 0

                    o_file3 = open(path_file+setSource+'_SD'+str(setDistance_SD)+'mm_loop_'+str(frame)+'.mac', 'w')

                    o_file3.write(
                        '# Macro file for the initialization of pixel\n'
                        '# in interactive session\n\n'
                        '# Change the default number of threads (in multi-threaded mode)\n'
                        '/run/numberOfThreads '+str(numberOfThreads)+'\n\n'

                        '# Set some default verbose\n'
                        '/control/verbose 2\n'
                        '/control/saveHistory\n'
                        '/run/verbose 2\n\n'

                        '/pixel/run/setThreshold ' + str(setThreshold) + ' eV\n'
                        '/pixel/det/setDistance_SD ' + str(setDistance_SD) + ' mm\n'
                        '/pixel/det/setDetec_substra ' + setDetec_substra + '\n'
                        '/pixel/det/setSi_substrat_thick ' + str(setSi_substrat_thick) + ' um\n'
                        '/pixel/det/setDetec_shield ' + setDetec_shield + '\n'
                        '/pixel/det/setDetc_shield_thick ' + str(setDetc_shield_thick) + ' um\n'
                        '/pixel/det/setDetec_SiO2_layers ' + setDetec_SiO2_layers + '\n'
                        '/pixel/det/setDetc_SiO2_layers_thick ' + str(setDetc_SiO2_layers_thick) +' um\n'
                        '/pixel/det/setDetc_lens_thick ' + str(setlens_thick) +  ' um\n'
                        '/pixel/det/setDetc_colrfltr_thick ' + str(setcolrfltr_thick) +  ' um\n'
                        '/pixel/det/set_thick_shiel_down ' + str(set_thick_shiel_down) +  ' um\n'
                        '/pixel/det/set_thick_shiel_up ' + str(set_thick_shiel_up) +  ' um\n'


                        '/pixel/det/setpixel_x ' + str(setpixel_x) + ' um\n' #1.4 um
                        '/pixel/det/setpixel_y ' + str(setpixel_y) + ' um\n'#1.4 um
                        '/pixel/det/setpixel_z ' + str(setpixel_z) + ' um\n'#  10-20 µm
                        '/pixel/run/setpixel_x ' + str(setpixel_x) + ' um\n' #1.4 um
                        '/pixel/run/setpixel_y ' + str(setpixel_y) + ' um\n'#1.4 um
                        '/pixel/det/setpixels_gap ' + str(setpixels_gap) + ' um\n'
                        '/pixel/det/setN_pixel_x ' + str(setN_pixel_x) + '\n'#1296 #2592
                        '/pixel/det/setN_pixel_y ' + str(setN_pixel_y) + '\n'#972 #1944
                        '/pixel/run/setN_pixel_x '+ str(setN_pixel_x) + '\n'
                        '/pixel/run/setN_pixel_y '+ str(setN_pixel_y) + '\n'
                        '/pixel/det/setSource ' + setSource +'\n'
                        '/pixel/run/setSource ' + setSource +'\n\n'

                        '/pixel/run/setSaveProcPart '+setSaveProcPart+'\n\n'


                        '#Initialize kernel\n'
                        '/run/initialize\n'
                    )

                    if(setSource=='Sr90'):
                        o_file3.write(
                            '/gps/particle ion\n\n'

                            '#Sr90\n'
                            '/gps/ion 38 90 \n\n'

                            #'/gps/pos/rot1 1 0 '+str(rot_1)+' \n\n'
                            #'/gps/pos/rot2 0 1 '+str(rot_2)+' \n\n'

                            '# Cylindrical source\n'
                            '/gps/pos/type Surface\n'
                            '/gps/pos/shape Cylinder\n'
                            '/gps/pos/radius 1587.5  um\n'
                            #'/gps/pos/radius 3175  um\n'
                            #'/gps/pos/radius 1250  um\n'
                            '/gps/pos/halfz '+str(halfz_sr_cs/2)+' um\n\n'

                            #'# Circle source\n'
                            #'/gps/pos/type Plane\n'
                            #'/gps/pos/shape Circle\n'
                            #'/gps/pos/radius 1500 um\n'

                            #'/gps/pos/centre 0.0 0.0 1587.5 um\n\n'

                            #'/gps/energy 0 keV\n'
                            #'/gps/ang/type iso\n'
                            #'/gps/ang/mintheta   0.0 deg\n'
                            #'/gps/ang/maxtheta  90.0 deg\n'

                        )

                    if(setSource=='Cs137'):
                        o_file3.write(
                            '/gps/particle ion\n\n'

                            #'/gps/pos/rot1 1 0 '+str(rot_1)+' \n\n'
                            #'/gps/pos/rot2 0 1 '+str(rot_2)+' \n\n'

                            '#Cs137\n'
                            '/gps/ion 55 137  \n\n'
                            #'#Ba-137, Excited state'
                            #'/gps/source/add 1.'
                            #'/gps/ion 56 137 0 661.6'

                            '# Cylindrical source\n'
                            '/gps/pos/type Surface\n'
                            '/gps/pos/shape Cylinder\n'
                            '/gps/pos/radius 1587.5  um\n'
                            #'/gps/pos/radius 3175  um\n'
                            #'/gps/pos/radius 2250  um\n'
                            '/gps/pos/halfz '+str(halfz_sr_cs/2)+' um\n\n'

                        )

                    if(setSource=='Am241'):
                        o_file3.write(
                            '/gps/particle ion\n\n'

                            #'/gps/pos/rot1 1 0 '+str(rot_1)+' \n\n'
                            #'/gps/pos/rot2 0 1 '+str(rot_2)+' \n\n'

                            '#Am241\n'
                            '/gps/ion 95 241 \n\n'

                            '# Cylindrical source\n'
                            '/gps/pos/type Surface\n'
                            '/gps/pos/shape Cylinder\n'
                            '/gps/pos/radius 2 mm\n'
                            '/gps/pos/halfz  750 um\n\n'
                        )

                    if(setSource=='Fe55'):
                        o_file3.write(
                            '/gps/particle ion\n\n'

                            #'/gps/pos/rot1 1 0 '+str(rot_1)+' \n\n'
                            #'/gps/pos/rot2 0 1 '+str(rot_2)+' \n\n'

                            '#Fe55\n'
                            '/gps/ion 26 55 \n\n'

                            '# Cylindrical source\n'
                            '/gps/pos/type Surface\n'
                            '/gps/pos/shape Cylinder\n'
                            '/gps/pos/radius 6 mm\n'
                            '/gps/pos/halfz  125 um\n\n'
                        )

                    if(Visual_setting==True):
                        o_file3.write(
                            '# Visualization setting\n'
                            '/control/execute vis.mac\n'
                            '/vis/viewer/set/viewpointThetaPhi 90. 180.\n\n'
                        )

                    for i in range(n):
                        print('/pixel/run/setFilename '+filename + str(i+n*k) +'_SD'+ str(setDistance_SD)+'mm')
                        #print('/run/beamOn 13388')
                        o_file3.write('/pixel/run/setFilename '+filename + str(i+n*k) +'_SD'+ str(setDistance_SD)+'mm'+'\n')
                        if(setSource=='Sr90'):
                            #o_file3.write('/run/beamOn 3347\n')
                            #o_file3.write('/run/beamOn 1598\n')
                            o_file3.write('/run/beamOn '+str(n_evt)+'\n')
                            #o_file3.write('/run/beamOn 4005\n')
                        if(setSource=='Cs137'):
                            #o_file3.write('/run/beamOn 6410000\n')
                            o_file3.write('/run/beamOn '+str(n_evt)+'\n')
                            #o_file3.write('/run/beamOn 1595\n')
                        if(setSource=='Am241'):
                            #o_file3.write('/run/beamOn 73000\n')
                            o_file3.write('/run/beamOn 16238\n')
                        if(setSource=='Fe55'):
                            o_file3.write('/run/beamOn 12000000\n')
                    o_file3.close()
                    #call('rm '+filecomp, shell=True)

                    commandlist =  path_file+'pixel '+ setSource+'_SD'+str(setDistance_SD)+'mm_loop_'+str(frame)+'.mac'
                    #commandlist= './pixel '+ setSource+'_SD'+str(setDistance_SD)+'mm_loop_'+str(frame)+'.mac'
                    print('run')
                    call(commandlist, shell=True)

                    #args = shlex.split(commandlist)
                    #subprocess.call(args)


                    for j in range(n):
                        name=path_file+filename + str(j+n*k) +'_SD'+ str(setDistance_SD)+'mm'+'_'+setSource+'_pixel_edep2D.csv'
                        print(name)
                        #data = np.loadtxt(name, delimiter='\t', skiprows=2, usecols=[0,1,2])
                        data = np.loadtxt(name, delimiter='\t', skiprows=1, usecols=[0,1,2])
                        #dat = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=[1,2,3])

                        #height= np.int32(data[0][1])
                        #width= np.int32(data[0][0])
                        height=setN_pixel_y
                        width=setN_pixel_x

                        if(data.ndim == 1):
                            data_2d = np.zeros((height, width))
                            data_2d_full = np.zeros((height, width))
                            print('Array Zeros')

                        else:
                            ########################################
                            pos_xy=np.array(data[:,:2],dtype=np.int32)
                            dat_eng=np.array(data[:,2],dtype=np.float64)

                            max_x=np.where((pos_xy[:,0]==setN_pixel_x))
                            if(max_x[0].size>0):
                                print(z, f, 'x:  ', max_x)
                                pos_xy[:,0][max_x]=setN_pixel_x-1

                            max_y=np.where((pos_xy[:,1]==setN_pixel_y))
                            if(max_y[0].size>0):
                                print(z, f, 'y:  ', max_y)
                                pos_xy[:,1][max_y]=setN_pixel_y-1

                            data_2d_full = np.zeros((height, width))
                            data_2d = np.zeros((height, width))
                            eng_plus=0

                            for p in range(len(data)):
                                if(data_2d_full[pos_xy[:,1][p].astype(np.int32),pos_xy[:,0][p].astype(np.int32)]==0):
                                    data_2d_full[pos_xy[:,1][p].astype(np.int32),pos_xy[:,0][p].astype(np.int32)]=dat_eng[p]
                                else:
                                    eng_plus=data_2d_full[pos_xy[:,1][p].astype(np.int32),pos_xy[:,0][p].astype(np.int32)]+dat_eng[p]
                                    data_2d_full[pos_xy[:,1][p].astype(np.int32),pos_xy[:,0][p].astype(np.int32)]=eng_plus

                            #dmax=4300 #### well capacity
                            #dmax=np.amax(data,0)[2]
                            #dmin=np.amin(data,0)[2]

                            #thresh=5
                            #pair_eh = 3.6 ## eV

                            #data_2d_full1 = 1000*data_2d_full/pair_eh

                            #data_2d =  np.uint16(np.clip((1000*data_2d_full/pair_eh-thresh)*1023.0/(dmax-thresh), 0, 1023))

                        #np.savez_compressed('RGB_z' +str(int(setDistance_SD/2)).zfill(3)+ '_f'+ str(j+n*k).zfill(3) + '.npz', np.int32(data_2d))
                        #print('Save Compressed npz: '+'RGB_z' +str(setDistance_SD).zfill(3)+ '_f'+ str(j+n*k).zfill(3) + '.npz')
                        #call('mv RGB*.npz '+path2, shell=True)

                        np.savez_compressed(path2+'/'+'E_sim_z' +str(int(setDistance_SD/2)).zfill(3)+ '_f'+ str(j+n*k).zfill(3) + '.npz', np.float64(data_2d_full))
                        print('Save Compressed npz: '+'E_sim_z' +str(setDistance_SD).zfill(3)+ '_f'+ str(j+n*k).zfill(3) + '.npz')
                        #call('mv E_sim_z*.npz '+path2, shell=True)


                       # call('mv *charge_dep*.csv '+path1+'/'+filesave, shell=True)
                        call('cp *'+setSource+'_pixel_edep2D.csv '+path1+'/'+filesave, shell=True)
                        call('cp *'+setSource+'_pixel_edep_zero_emit.csv '+path1+'/'+filesave, shell=True)
                        call('cp *'+setSource+'_pixel_edep_emit.csv '+path1+'/'+filesave, shell=True)
                        call('mv *'+setSource+'_electron_edep*.csv '+path1+'/'+filesave, shell=True)
                        #call('mv *'+setSource+'_gamma_edep*.csv '+path1+'/'+filesave, shell=True)
                        #call('mv *'+setSource+'_electron_dep*.csv '+path1+'/'+filesave, shell=True)
                        #call('mv *'+setSource+'_gamma_dep*.csv '+path1+'/'+filesave, shell=True)
                        call('mv *'+setSource+'_edep_all_electron_arrived_detector*.csv '+path1+'/'+filesave, shell=True)
                        call('mv *'+setSource+'_edep_all_gamma_arrived_detector*.csv '+path1+'/'+filesave, shell=True)
                        call('mv *'+setSource+'_edep_all_particle_arrived_detector*.csv '+path1+'/'+filesave, shell=True)
                        call('mv *'+setSource+'*emit_electron*.csv '+path1+'/'+filesave, shell=True)
                        call('mv *'+setSource+'*emit_gamma*.csv '+path1+'/'+filesave, shell=True)
                        call('mv *'+setSource+'*emit_anti_nu_elec*.csv '+path1+'/'+filesave, shell=True)
                        call('mv *'+setSource+'*emit_positron*.csv '+path1+'/'+filesave, shell=True)


                        '''
                        call('mv *'+setSource+'_electron_edep2D*.csv '+path1+'/'+filesave, shell=True)
                        call('mv *'+setSource+'_edep_all_gamma_arrived_detector*.csv '+path1+'/'+filesave, shell=True)
                        call('mv *'+setSource+'_gamma_edep2D*.csv '+path1+'/'+filesave, shell=True)
                        call('mv *'+setSource+'_edep_all_particle_arrived_detector*.csv '+path1+'/'+filesave, shell=True)
                        call('mv *'+setSource+'*emit_gamma*.csv '+path1+'/'+filesave, shell=True)
                        call('mv *'+setSource+'*emit_electron*.csv '+path1+'/'+filesave, shell=True)
                        '''
                        #call('mv *electron_second_parent_elec*.csv '+path1+'/'+filesave, shell=True)
                        #call('mv *electron_primary*.csv '+path1+'/'+filesave, shell=True)
                        #call('mv *emit_electron*.csv '+path1+'/'+filesave, shell=True)


                        #call('cp *Process_Particles_Name.csv '+path1+'/'+filesave0+'CSV', shell=True)
                        #call('cp *.csv '+path_file+filesave0+'/'+filesave0+thickback+'CSV', shell=True)

                        filecomp=filename + str(j+n*k) +'_SD'+ str(setDistance_SD)+'mm'+'_'+setSource+'*.csv '

                        #namecomp='tar -cjvf '+'\''+'RGBz'+str(setDistance_SD).zfill(3)+'mm_f'+str(j+n*k) + '_'+ setSource+'.tar.bz2'+'\' '+filecomp

                        #print('Save Compressed '+filecomp+' : '+'RGBz0f'+str(j+n*k)+'.tar.bz2')
                        #call(namecomp, shell=True)


                       # call('mv RGB*.npz RGB*.tar.bz2 '+path_file+filesave0+'/'+filesave, shell=True)
                        #call('mv RGB*.tar.bz2 '+path_file+filesave0+'/'+filesave, shell=True)

                        print('Delete files: '+filecomp+',  '+commandlist)
                        call('rm '+filecomp, shell=True)

                        call('rm '+ setSource+'_SD'+str(setDistance_SD)+'mm_loop_'+str(frame)+'.mac', shell=True)
                        #call('rm *.ascii', shell=True)
