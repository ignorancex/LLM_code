#// ============================================================================================= //
#//                                                                                               //
#//       Filename:  LTdwarfIndices_individualBD.py                                               //
#//    Description:  Candidate Variable L and T brown dwarfs determination                        //
#//                  using the Spectral Index method                                              //
#//                                                                                               //
#//        Version:  2.1                                                                          //
#//        Created:  13/06/2023                                                                   //
#//       Compiler:  Python                                                                       //
#//                                                                                               //
#//         Author:  Natalia Oliveros-Gomez                                                       //
#//          Email:  onatalialucia@gmail.com                                                      //
#//        Company:  Grupo Fisica Estelar Universidad de Guanajuato, México                       //
#//                                                                                               //
#// ============================================================================================= //
#
#// ============================================================================================= //
#//    Indications: You only need to modify the brown dwarf spectrum path                         // 
#//                 Indicate if it is L or T type                                                 //
#//                 And if you want to print the index-index graph                                // 
#//                                                                                               //
#//                                                                                               //
#//                                                                                               //
#//   Compile with:  python  LTdwarfIndices                                                       //    
#// ============================================================================================= //


name_data = 'ULASJ141623.txt' #adds the directory and name of the file brown dwarf spectrum normalized and delimiter='\t'
#If your data it's not normalized, activate the line #45, after the lecture of the data
spt = 'L' #add spectral type: T or L

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

import scipy.integrate as integrate

dwarf_data = pd.read_csv(name_data, delimiter='\t', thousands=',', decimal='.', comment='#', names=['lambda', 'flux', 'eflux'])

### If your data it's not normalized, activate the next line:
#dwarf_data['flux'] = dwarf_data['flux']/np.median(dwarf_data['flux'])

# Index definition for division or substract regions in the spectra
def index_div(lamb,flux,num_init,num_final, den_init,den_final):
    
    range_num = np.where(np.logical_and(lamb>=num_init,lamb<=num_final))
    range_den = np.where(np.logical_and(lamb>=den_init,lamb<=den_final))

    num_temp = integrate.simps(flux[range_num[0][0]:range_num[0][len(range_num[0])-1]+1],lamb[range_num[0][0]:range_num[0][len(range_num[0])-1]+1])
    den_temp = integrate.simps(flux[range_den[0][0]:range_den[0][len(range_den[0])-1]+1],lamb[range_den[0][0]:range_den[0][len(range_den[0])-1]+1])
    index_temp = num_temp/den_temp
    return index_temp

def index_res(lamb,flux,num_init,num_final, den_init,den_final):
    
    range_num = np.where(np.logical_and(lamb>=num_init,lamb<=num_final))
    range_den = np.where(np.logical_and(lamb>=den_init,lamb<=den_final))

    num_temp = integrate.simps(flux[range_num[0][0]:range_num[0][len(range_num[0])-1]+1],lamb[range_num[0][0]:range_num[0][len(range_num[0])-1]+1])
    den_temp = integrate.simps(flux[range_den[0][0]:range_den[0][len(range_den[0])-1]+1],lamb[range_den[0][0]:range_den[0][len(range_den[0])-1]+1])
    
    index_temp = den_temp - num_temp
    return index_temp
    
def TdwarfIndices(dwarf_data):
    
    ############################ Spectral indeces template J2228 ###########################
    
    index_temp_J_M2228 = 0.205
    index_temp_H_M2228 = 0.463
    index_temp_HJ_M2228 = 0.4145
    index_temp_J_H_M2228 = 0.0451
    index_temp_Jslope_M2228 = 0.633
    index_temp_Jcurve_M2228 = 0.154
    
    ########################### Calculate the spectral indices ###########################
    
    index_J       = index_div(dwarf_data['lambda'], dwarf_data['flux'],1.15,1.18,1.22,1.25)
    index_H       = index_div(dwarf_data['lambda'], dwarf_data['flux'],1.64,1.67,1.59,1.62)
    index_HJ      = index_div(dwarf_data['lambda'], dwarf_data['flux'],1.51,1.62,1.205,1.315)
    index_J_H     = index_res(dwarf_data['lambda'], dwarf_data['flux'],1.51,1.62,1.205,1.315)
    index_Jslope  = index_div(dwarf_data['lambda'], dwarf_data['flux'],1.30,1.33, 1.27,1.30)
    index_Jcurve  = index_div(dwarf_data['lambda'], dwarf_data['flux'],1.14,1.17, 1.26,1.29)
    
    ########################## define variability areas ###########################
    
    verts00 = [(0.2,0.37), (0.39,0.37), (0.463,0.2010), (0.65,0.05), (0.2,0.05), (0.2,0.37)]
    verts01 = [(0.2,0.55), (0.415,0.55), (0.462,0.4167), (0.4,0.28), (0.2,0.28), (0.2,0.55)]
    verts02 = [(0.038,0.55), (0.065,0.55), (0.065,0.28), (0.057,0.28), (0.038,0.55)]
    verts03 = [(0.067,0.68), (0.067,0.2), (0.0325,0.2), (0.067,0.68)]

    verts10 = [(0.038,0.37), (0.067,0.37), (0.067,0.05), (0.0352,0.2005),(0.038,0.37)]
    verts11 = [(0.43,0.37), (0.63,0.37), (0.65,0.201), (0.63,0.05), (0.43,0.05), (0.43,0.37)]
    verts12 = [(0.43,0.515), (0.646,0.462), (0.75,0.2), (0.43,0.2), (0.43,0.515)]
    verts13 = [(0.43,0.52), (0.63,0.52), (0.65,0.4169), (0.63,0.28), (0.43,0.28), (0.43,0.52)]

    verts20 = [(0.43,0.065), (0.85,0.065), (0.85,0.037), (0.43,0.046), (0.43,0.065)]
    verts21 = [(0.43,0.35), (0.85,-0.07), (0.43,-0.07), (0.43,0.35)]
    verts22 = [(0.2,0.35), (0.68,-0.07), (0.2,-0.07), (0.2,0.35)]
    verts23 = [(0.065,0.34), (0.065,-0.07), (0.025,-0.07),(0.065,0.34)]

    codes_5v = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    codes_4v = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    codes_3v = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    
    path00 = Path(verts00, codes_5v)
    path01 = Path(verts01, codes_5v)
    path02 = Path(verts02, codes_4v)
    path03 = Path(verts03, codes_3v)

    path10 = Path(verts10, codes_4v)
    path11 = Path(verts11, codes_5v)
    path12 = Path(verts12, codes_4v)
    path13 = Path(verts13, codes_5v)

    path20 = Path(verts20, codes_4v)
    path21 = Path(verts21, codes_3v)
    path22 = Path(verts22, codes_3v)
    path23 = Path(verts23, codes_3v)

    patch00 = patches.PathPatch(path00, facecolor='red', lw=1, alpha=0.25)
    patch01 = patches.PathPatch(path01, facecolor='red', lw=1, alpha=0.25)
    patch02 = patches.PathPatch(path02, facecolor='red', lw=1, alpha=0.25)
    patch03 = patches.PathPatch(path03, facecolor='red', lw=1, alpha=0.25)

    patch10 = patches.PathPatch(path10, facecolor='red', lw=1, alpha=0.25)
    patch11 = patches.PathPatch(path11, facecolor='red', lw=1, alpha=0.25)
    patch12 = patches.PathPatch(path12, facecolor='red', lw=1, alpha=0.25)
    patch13 = patches.PathPatch(path13, facecolor='red', lw=1, alpha=0.25)

    patch20 = patches.PathPatch(path20, facecolor='red', lw=1, alpha=0.25)
    patch21 = patches.PathPatch(path21, facecolor='red', lw=1, alpha=0.25)
    patch22 = patches.PathPatch(path22, facecolor='red', lw=1, alpha=0.25)
    patch23 = patches.PathPatch(path23, facecolor='red', lw=1, alpha=0.25)

    ######################### index-index plot ###########################
    
    fig = plt.figure(figsize=(22.0,13.5), constrained_layout = True) 
    plt.rcParams['font.family'] = "DejaVu Sans" 
    plt.rcParams['font.size'] = 14
    plt.rc('xtick', direction='in')
    plt.rc('ytick', direction='in')

    ax1 = fig.add_subplot(341)
    ax1.set_xlabel('H-Index')
    ax1.set_ylabel('J-Index')
    ax1.plot(index_temp_H_M2228, index_temp_J_M2228,marker ='*',markersize=20, color='black')
    ax1.scatter(index_H, index_J,marker ='*' ,s=150, color='blue')

    ax2 = fig.add_subplot(342)
    ax2.set_xlabel('H-Index')
    ax2.set_ylabel('H/J-Index')
    ax2.plot(index_temp_H_M2228, index_temp_HJ_M2228,marker ='*',markersize=20, color='black')
    ax2.scatter(index_H, index_HJ,marker ='*' ,s=150, color='blue')

    ax3 = fig.add_subplot(343)
    ax3.set_xlabel('J-H-Index')
    ax3.set_ylabel('H/J-Index')
    ax3.plot(index_temp_J_H_M2228, index_temp_HJ_M2228,marker ='*',markersize=20, color='black')
    ax3.scatter(index_J, index_HJ,marker ='*' ,s=150, color='blue')

    ax4 = fig.add_subplot(344)
    ax4.set_xlabel('J-H-Index')
    ax4.set_ylabel('H-Index')
    ax4.plot(index_temp_J_H_M2228, index_temp_H_M2228,marker ='*',markersize=20, color='black')
    ax4.scatter(index_J_H, index_H,marker ='*' ,s=150, color='blue')

    ax5 = fig.add_subplot(345)
    ax5.set_xlabel('J-H-Index')
    ax5.set_ylabel('J-Index')
    ax5.plot(index_temp_J_H_M2228, index_temp_J_M2228,marker ='*',markersize=20, color='black')
    ax5.scatter(index_J_H, index_J,marker ='*' ,s=150, color='blue')

    ax6 = fig.add_subplot(346)
    ax6.set_xlabel('Jslope-Index')
    ax6.set_ylabel('J-Index')
    ax6.plot(index_temp_Jslope_M2228, index_temp_J_M2228,marker ='*',markersize=20, color='black')
    ax6.scatter(index_Jslope, index_J,marker ='*' ,s=150, color='blue')

    ax7 = fig.add_subplot(347)
    ax7.set_xlabel('Jslope-Index')
    ax7.set_ylabel('H-Index')
    ax7.plot(index_temp_Jslope_M2228, index_temp_H_M2228,marker ='*',markersize=20, color='black')
    ax7.scatter(index_Jslope, index_H,marker ='*' ,s=150, color='blue')

    ax8 = fig.add_subplot(348)
    ax8.set_xlabel('Jslope-Index')
    ax8.set_ylabel('H/J-Index')
    ax8.plot(index_temp_Jslope_M2228, index_temp_HJ_M2228,marker ='*',markersize=20, color='black')
    ax8.scatter(index_Jslope, index_HJ,marker ='*' ,s=150, color='blue')

    ax9 = fig.add_subplot(349)
    ax9.set_xlabel('Jslope-Index')
    ax9.set_ylabel('J-H-Index')
    ax9.plot(index_temp_Jslope_M2228, index_temp_J_H_M2228,marker ='*',markersize=20, color='black')
    ax9.scatter(index_Jslope, index_J_H,marker ='*' ,s=150, color='blue')

    ax10 = fig.add_subplot(3,4,10)
    ax10.set_xlabel('Jslope-Index')
    ax10.set_ylabel('Jcurve-Index')
    ax10.plot(index_temp_Jslope_M2228, index_temp_Jcurve_M2228,marker ='*',markersize=20, color='black')
    ax10.scatter(index_Jslope, index_Jcurve,marker ='*' ,s=150, color='blue')

    ax11 = fig.add_subplot(3,4,11)
    ax11.set_xlabel('H-Index')
    ax11.set_ylabel('Jcurve-Index')
    ax11.plot(index_temp_H_M2228, index_temp_Jcurve_M2228,marker ='*',markersize=20, color='black')
    ax11.scatter(index_H, index_Jcurve,marker ='*' ,s=150, color='blue')

    ax12 = fig.add_subplot(3,4,12)
    ax12.set_xlabel('J-H-Index')
    ax12.set_ylabel('Jcurve-Index')
    ax12.plot(index_temp_J_H_M2228, index_temp_Jcurve_M2228,marker ='*',markersize=20, color='black', label='Template LP 261-75B')
    ax12.scatter(index_J_H, index_Jcurve,marker ='*' ,s=150, color='blue', label='Your Brown Dwarf')
    ax12.text(0.045, -0.05, 'Variable area', fontsize=12, color='red')
    
    #Define the limits in yx plots
    ax1.set_ylim((index_temp_J_M2228 -0.15,index_temp_J_M2228 + 0.15))
    ax1.set_xlim((index_temp_H_M2228 -0.25,index_temp_H_M2228 + 0.25))
    ax2.set_ylim((index_temp_HJ_M2228 -0.12,index_temp_HJ_M2228 + 0.12))
    ax2.set_xlim((index_temp_H_M2228 -0.25,index_temp_H_M2228 + 0.25))
    ax3.set_ylim((index_temp_HJ_M2228 -0.12,index_temp_HJ_M2228 + 0.12))
    ax3.set_xlim((index_temp_J_H_M2228 -0.02,index_temp_J_H_M2228 + 0.02))
    ax4.set_ylim((index_temp_H_M2228 -0.25,index_temp_H_M2228 + 0.25))
    ax4.set_xlim((index_temp_J_H_M2228 -0.02,index_temp_J_H_M2228 + 0.02))

    ax5.set_ylim((index_temp_J_M2228 -0.15,index_temp_J_M2228 + 0.15))
    ax5.set_xlim((index_temp_J_H_M2228 -0.02,index_temp_J_H_M2228 + 0.02))
    ax6.set_ylim((index_temp_J_M2228 -0.15,index_temp_J_M2228 + 0.15))
    ax6.set_xlim((index_temp_Jslope_M2228 -0.2,index_temp_Jslope_M2228 + 0.2))
    ax7.set_ylim((index_temp_H_M2228 -0.25,index_temp_H_M2228 + 0.25))
    ax7.set_xlim((index_temp_Jslope_M2228 -0.2,index_temp_Jslope_M2228 + 0.2))
    ax8.set_ylim((index_temp_HJ_M2228 -0.1,index_temp_HJ_M2228 + 0.1))
    ax8.set_xlim((index_temp_Jslope_M2228 -0.2,index_temp_Jslope_M2228 + 0.2))

    ax9.set_ylim((index_temp_J_H_M2228 -0.02,index_temp_J_H_M2228 + 0.02))
    ax9.set_xlim((index_temp_Jslope_M2228 -0.2,index_temp_Jslope_M2228 + 0.2))
    ax10.set_ylim((index_temp_Jcurve_M2228 -0.22,index_temp_Jcurve_M2228 + 0.22))
    ax10.set_xlim((index_temp_Jslope_M2228 -0.2,index_temp_Jslope_M2228 + 0.2))
    ax11.set_ylim((index_temp_Jcurve_M2228 -0.22,index_temp_Jcurve_M2228 + 0.22))
    ax11.set_xlim((index_temp_H_M2228 -0.25,index_temp_H_M2228 + 0.25))
    ax12.set_ylim((index_temp_Jcurve_M2228 -0.22,index_temp_Jcurve_M2228 + 0.22))
    ax12.set_xlim((index_temp_J_H_M2228 -0.02,index_temp_J_H_M2228 + 0.02))
    
    #Add variability areas plot
    ax1.add_patch(patch00)
    ax2.add_patch(patch01)
    ax3.add_patch(patch02)
    ax4.add_patch(patch03)
    ax5.add_patch(patch10)
    ax6.add_patch(patch11)
    ax7.add_patch(patch12)
    ax8.add_patch(patch13)
    ax9.add_patch(patch20)
    ax10.add_patch(patch21)
    ax11.add_patch(patch22)
    ax12.add_patch(patch23)
    plt.legend(fontsize=12)
    plt.show()

    
    ######################### Variable or not? #########################
    
    #Define the point indices in each plot
    puntos00_var = [[index_H, index_J],[5,5]]
    puntos01_var = [[index_H, index_HJ],[5,5]]
    puntos02_var = [[index_J_H, index_HJ],[5,5]]
    puntos03_var = [[index_J_H, index_H],[5,5]]
    puntos10_var = [[index_J_H, index_HJ],[5,5]]
    puntos11_var = [[index_Jslope, index_J],[5,5]]
    puntos12_var = [[index_Jslope, index_H],[5,5]]
    puntos13_var = [[index_Jslope, index_HJ],[5,5]]
    puntos20_var = [[index_Jslope, index_J_H],[5,5]]
    puntos21_var = [[index_Jslope, index_Jcurve],[5,5]]
    puntos22_var = [[index_H, index_Jcurve],[5,5]]
    puntos23_var = [[index_J_H, index_Jcurve],[5,5]]
    
    #Determine the point indices inside the variability areas in each plot
    inside00_var = path00.contains_points(puntos00_var)
    inside01_var = path01.contains_points(puntos01_var)
    inside02_var = path02.contains_points(puntos02_var)
    inside03_var = path03.contains_points(puntos03_var)
    inside10_var = path10.contains_points(puntos10_var)
    inside11_var = path11.contains_points(puntos11_var)
    inside12_var = path12.contains_points(puntos12_var)
    inside13_var = path13.contains_points(puntos13_var)
    inside20_var = path20.contains_points(puntos20_var)
    inside21_var = path21.contains_points(puntos21_var)
    inside22_var = path22.contains_points(puntos22_var)
    inside23_var = path23.contains_points(puntos23_var)
    
    
    cant_plots_var=[]
    for i in range(0,2):
        cant_plots_var.append(sum([inside00_var[i],inside01_var[i],inside02_var[i],inside03_var[i], inside10_var[i],
                          inside11_var[i],inside12_var[i],inside13_var[i],inside20_var[i],inside21_var[i],
                          inside22_var[i],inside23_var[i]]))
    #pass threshold? 11
    print('{}{}{}'.format('Your brown dwarf fall in ',cant_plots_var[0],' variable areas of 12'))
    if cant_plots_var[0] > 10:
        a = print('Your brown dwarf is candidate variable')
    else:
        a = print('Your brown dwarf is candidate non-variable')
    return a


def LdwarfIndices(dwarf_data):
    
    ############################ Spectral indeces template LP261-75B ###########################
    
    index_temp_mostH = 1.913080124768479
    index_temp_mostJ = 0.23732289733907183
    index_temp_less = 2.2632193763577337
    index_temp_Jcurve = 1.363828699841246
    index_temp_H2OJ = 0.7366212163364357
    index_temp_CH4J = 0.7414549450272985
    
    ############################# Calculate the spectral indices ###########################
    
    index_mostH  = index_div(dwarf_data['lambda'], dwarf_data['flux'],1.219,1.237,1.376,1.394)
    index_mostJ  = index_div(dwarf_data['lambda'], dwarf_data['flux'],1.37,1.42, 1.57,1.67)
    index_less   = index_div(dwarf_data['lambda'], dwarf_data['flux'],1.670,1.688,1.347,1.365)
    index_Jcurve = index_div(dwarf_data['lambda'], dwarf_data['flux'],1.26,1.29,1.14,1.17)
    index_H2OJ   = index_div(dwarf_data['lambda'], dwarf_data['flux'],1.14,1.165,1.26,1.285)
    index_CH4J   = index_div(dwarf_data['lambda'], dwarf_data['flux'],1.315,1.335,1.26,1.285)
    
    ############################ define variability areas ###########################
    
    verts00 = [(3,5.3), (3,0.78), (0.9,-0.5), (3,5.3)]
    verts01 = [(-0.55,2.262), (0.8,2.262), (0.8,0.75), (-0.55,0.75), (-0.55,2.32)]
    verts02 = [(0,2.72), (1.5,1.75), (1.8,0.75), (0,0.75), (0,2.54)]
    verts03 = [(0.3,3.5), (2.6,0.75), (0.3,0.75), (0.3,3.5)] 
    
    verts10 = [(0.28,2.92), (2.62,-1.8), (0.28,-1.3),(0.28,3.2)]
    verts11 = [(0.31,0.8), (1.37,0.8), (1.37,-0.25),(0.3,-0.25),(0.4,0.8)] 
    verts12 = [(0.3,3.05), (2.38,3.05),(2.38,2.15),(0.3,1.65), (0.3,3.05)] 
    verts13 = [(-0.25,1.905), (1.9,1.905), (1.9,3), (-0.25,3), (-0.25,1.905)]
    
    verts20 = [(-0.5,2.41), (0.8,1.51), (0.8,3), (-0.5,3), (-0.5,2.41)]
    verts21 = [(0.739,0.8), (-0.25,0.8), (-0.25,-0.25), (0.742,-0.25),(0.739,0.8)]
    verts22 = [(0.739,0.8), (1.4,0.8), (1.4,-0.25), (0.742,-0.25),(0.739,0.8)]
    verts23 = [(0.2,2.24), (1.26,2.29), (1.26,0.75),(0.2,0.75),(0.2,2.24)]
    
    verts30 = [(0.18,2.12), (1.26,1.74), (1.26,3), (0.18,3), (0.18,2.12)]
    verts31 = [(-0.18,0.125), (1.8,1.445), (-0.18,1.725), (-0.18,0.125)] 
    verts32 = [(-0.18,1.36), (1.8,1.36), (1.8,0.36), (-0.18,0.36),(-0.18,1.36)] 
    
    codes_5v = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    codes_4v = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    codes_3v = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    path00 = Path(verts00, codes_3v)
    path01 = Path(verts01, codes_4v)
    path02 = Path(verts02, codes_4v)
    path03 = Path(verts03, codes_3v)
    
    path10 = Path(verts10, codes_3v)
    path11 = Path(verts11, codes_4v)
    path12 = Path(verts12, codes_4v)
    path13 = Path(verts13, codes_4v)
    
    path20 = Path(verts20, codes_4v)
    path21 = Path(verts21, codes_4v)
    path22 = Path(verts22, codes_4v)
    path23 = Path(verts23, codes_4v)
    
    path30 = Path(verts30, codes_4v)
    path31 = Path(verts31, codes_3v)
    path32 = Path(verts32, codes_4v)
    
    patch00 = patches.PathPatch(path00, facecolor='red', lw=1, alpha=0.25)
    patch01 = patches.PathPatch(path01, facecolor='red', lw=1, alpha=0.25)
    patch02 = patches.PathPatch(path02, facecolor='red', lw=1, alpha=0.25)
    patch03 = patches.PathPatch(path03, facecolor='red', lw=1, alpha=0.25)
    
    patch10 = patches.PathPatch(path10, facecolor='red', lw=1, alpha=0.25)
    patch11 = patches.PathPatch(path11, facecolor='red', lw=1, alpha=0.25)
    patch12 = patches.PathPatch(path12, facecolor='red', lw=1, alpha=0.25)
    patch13 = patches.PathPatch(path13, facecolor='red', lw=1, alpha=0.25)
    
    patch20 = patches.PathPatch(path20, facecolor='red', lw=1, alpha=0.25)
    patch21 = patches.PathPatch(path21, facecolor='red', lw=1, alpha=0.25)
    patch22 = patches.PathPatch(path22, facecolor='red', lw=1, alpha=0.25)
    patch23 = patches.PathPatch(path23, facecolor='red', lw=1, alpha=0.25)
    
    patch30 = patches.PathPatch(path30, facecolor='red', lw=1, alpha=0.25)
    patch31 = patches.PathPatch(path31, facecolor='red', lw=1, alpha=0.25)
    patch32 = patches.PathPatch(path32, facecolor='red', lw=1, alpha=0.25)
    
    ######################### index-index plot ###########################

    fig = plt.figure(figsize=(26,14), constrained_layout = True) # Cambiar tamaño proporcionalmente, c/u sería 6 x 4.5
    plt.rcParams['font.family'] = "DejaVu Sans" # Tipo de letra general
    plt.rcParams['font.size'] = 28 # Tamaño de letra general
    plt.rc('xtick', direction='in')
    plt.rc('ytick', direction='in')
    
    ax1 = fig.add_subplot(351)
    ax1.set_xlabel('J-index')
    ax1.set_ylabel('max-var')
    ax1.plot(index_temp_mostH,index_temp_less,marker ='*',markersize=20, color='black',zorder=3)
    ax1.plot(index_mostH,index_less,marker ='*',zorder=80, markersize=20)
    ax1.set_ylim((index_temp_less -1.5,index_temp_less + 1.5))
    ax1.set_xlim((index_temp_mostH -1,index_temp_mostH + 1))
    ax1.add_patch(patch00)

    ax2 = fig.add_subplot(352)
    ax2.set_xlabel('H-index')
    ax2.set_ylabel('max-var')
    ax2.plot(index_temp_mostJ,index_temp_less,marker ='*',markersize=20, color='black',zorder=3)
    ax2.plot(index_mostJ,index_less,marker ='*',zorder=80, markersize=20)
    ax2.set_ylim((index_temp_less -1.5,index_temp_less + 1.5))
    ax2.set_xlim((index_temp_mostJ - 0.2,index_temp_mostJ + 0.2))
    ax2.add_patch(patch01)

    ax3 = fig.add_subplot(353)
    ax3.set_xlabel('CH$_4$-J')
    ax3.set_ylabel('max-var')
    ax3.plot(index_temp_CH4J,index_temp_less,marker ='*',markersize=20, color='black',zorder=3)
    ax3.plot(index_CH4J,index_less,marker ='*',zorder=80, markersize=20)
    ax3.set_ylim((index_temp_less -1.5,index_temp_less + 1.5))
    ax3.set_xlim((index_temp_CH4J -0.7,index_temp_CH4J + 0.7))
    ax3.add_patch(patch02)

    ax4 = fig.add_subplot(354)
    ax4.set_xlabel('J-curve')
    ax4.set_ylabel('max-var')
    ax4.plot(index_temp_Jcurve,index_temp_less,marker ='*',markersize=20, color='black',zorder=3)
    ax4.plot(index_Jcurve,index_less,marker ='*',zorder=80, markersize=20)
    ax4.set_xlim((index_temp_Jcurve -1,index_temp_Jcurve + 1))
    ax4.set_ylim((index_temp_less -1.5,index_temp_less + 1.5))
    ax4.add_patch(patch03)

    ax5 = fig.add_subplot(355)
    ax5.set_xlabel('J-curve')
    ax5.set_ylabel('CH$_4$-J')
    ax5.plot(index_temp_Jcurve,index_temp_CH4J,marker ='*',markersize=20, color='black',zorder=3)
    ax5.plot(index_Jcurve,index_CH4J,marker ='*',zorder=80, markersize=20)
    ax5.set_ylim((index_temp_CH4J -0.7,index_temp_CH4J + 0.7))
    ax5.set_xlim((index_temp_Jcurve -1,index_temp_Jcurve + 1))
    ax5.add_patch(patch10)

    ax6 = fig.add_subplot(356)
    ax6.set_xlabel('J-curve')
    ax6.set_ylabel('H-index')
    ax6.plot(index_temp_Jcurve,index_temp_mostJ,marker ='*',markersize=20, color='black',zorder=3)
    ax6.plot(index_Jcurve,index_mostJ,marker ='*',zorder=80, markersize=20)
    ax6.set_xlim((index_temp_Jcurve -1,index_temp_Jcurve + 1))
    ax6.set_ylim((index_temp_mostJ - 0.2,index_temp_mostJ + 0.2))
    ax6.add_patch(patch11)

    ax7 = fig.add_subplot(357)
    ax7.set_xlabel('J-curve')
    ax7.set_ylabel('J-index')
    ax7.plot(index_temp_Jcurve,index_temp_mostH,marker ='*',markersize=20, color='black',zorder=3)
    ax7.plot(index_Jcurve,index_mostH,marker ='*',zorder=80, markersize=20)
    ax7.set_xlim((index_temp_Jcurve -1,index_temp_Jcurve + 1))
    ax7.set_ylim((index_temp_mostH -1,index_temp_mostH + 1))
    ax7.add_patch(patch12)

    ax8 = fig.add_subplot(358)
    ax8.set_xlabel('CH$_4$-J')
    ax8.set_ylabel('J-index')
    ax8.plot(index_temp_CH4J,index_temp_mostH,marker ='*',markersize=20, color='black',zorder=3)
    ax8.plot(index_CH4J,index_mostH,marker ='*',zorder=80, markersize=20)
    ax8.set_xlim((index_temp_CH4J -0.7,index_temp_CH4J + 0.7))
    ax8.set_ylim((index_temp_mostH -1,index_temp_mostH + 1))
    ax8.add_patch(patch13)

    ax9 = fig.add_subplot(359)
    ax9.set_xlabel('H-index')
    ax9.set_ylabel('J-index')
    ax9.plot(index_mostJ,index_mostH,marker ='*',zorder=80, markersize=20)
    ax9.plot(index_temp_mostJ,index_temp_mostH,marker ='*',markersize=20, color='black',zorder=3)
    ax9.set_xlim((index_temp_mostJ - 0.2,index_temp_mostJ + 0.2))
    ax9.set_ylim((index_temp_mostH -1,index_temp_mostH + 1))
    ax9.add_patch(patch20)

    ax10 = fig.add_subplot(3,5,10)
    ax10.set_xlabel('CH$_4$-J')
    ax10.set_ylabel('H-index')
    ax10.plot(index_temp_CH4J,index_temp_mostJ,marker ='*',markersize=20, color='black',zorder=3)
    ax10.plot(index_CH4J,index_mostJ,marker ='*',zorder=80, markersize=20)
    ax10.set_xlim((index_temp_CH4J -0.7,index_temp_CH4J + 0.7))
    ax10.set_ylim((index_temp_mostJ - 0.2,index_temp_mostJ + 0.2))
    ax10.add_patch(patch21)

    ax11 = fig.add_subplot(3,5,11)
    ax11.set_xlabel('H$_2$O-J')
    ax11.set_ylabel('H-index')
    ax11.plot(index_temp_H2OJ,index_temp_mostJ,marker ='*',markersize=20, color='black',zorder=3)
    ax11.plot(index_H2OJ,index_mostJ,marker ='*',zorder=80, markersize=20)
    ax11.set_xlim((index_temp_H2OJ -0.3,index_temp_H2OJ + 0.3))
    ax11.set_ylim((index_temp_mostJ - 0.2,index_temp_mostJ + 0.2))
    ax11.add_patch(patch22)

    ax12 = fig.add_subplot(3,5,12)
    ax12.set_xlabel('H$_2$O-J')
    ax12.set_ylabel('max-var')
    ax12.plot(index_temp_H2OJ,index_temp_less,marker ='*',markersize=20, color='black',zorder=3)
    ax12.plot(index_H2OJ,index_less,marker ='*',zorder=80, markersize=20)
    ax12.set_xlim((index_temp_H2OJ -0.3,index_temp_H2OJ + 0.3))
    ax12.set_ylim((index_temp_less -1.5,index_temp_less + 1.5))
    ax12.add_patch(patch23)

    ax13 = fig.add_subplot(3,5,13)
    ax13.set_xlabel('H$_2$O-J')
    ax13.set_ylabel('J-index')
    ax13.plot(index_temp_H2OJ,index_temp_mostH,marker ='*',markersize=20, color='black',zorder=3)
    ax13.plot(index_H2OJ,index_mostH,marker ='*',zorder=80, markersize=20)
    ax13.set_xlim((index_temp_H2OJ -0.3,index_temp_H2OJ + 0.3))
    ax13.set_ylim((index_temp_mostH -1,index_temp_mostH + 1))
    ax13.add_patch(patch30)

    ax14 = fig.add_subplot(3,5,14)
    ax14.set_xlabel('CH$_4$-J')
    ax14.set_ylabel('H$_2$O-J')
    ax14.plot(index_temp_CH4J,index_temp_H2OJ,marker ='*',markersize=20, color='black',zorder=3)
    ax14.plot(index_CH4J,index_H2OJ,marker ='*',zorder=80, markersize=20)
    ax14.set_xlim((index_temp_CH4J -0.7,index_temp_CH4J + 0.7))
    ax14.set_ylim((index_temp_H2OJ -0.3,index_temp_H2OJ + 0.3))
    ax14.add_patch(patch31)

    ax15 = fig.add_subplot(3,5,15)
    ax15.set_xlabel('H$_2$O-J')
    ax15.set_ylabel('J-curve')
    ax15.plot(index_temp_H2OJ,index_temp_Jcurve,marker ='*',markersize=20, color='black',zorder=3, label='Template 2M2228')
    ax15.plot(index_H2OJ,index_Jcurve,marker ='*',zorder=80, markersize=20, label='Your Brwon Dwarf')
    ax15.set_xlim((index_temp_H2OJ -0.3,index_temp_H2OJ + 0.3))
    ax15.set_ylim((index_temp_Jcurve -1,index_temp_Jcurve + 1))
    ax15.add_patch(patch32)
    ax15.text(0.6,0.8,'Variable area',fontsize=14,color='red')

    plt.legend(fontsize=14)
    plt.plot()
    # Save and display the figure
    if save_index_index_plot == 'Yes':
        plt.savefig('index-index-plot.pdf', bbox_inches='tight')
    if save_index_index_plot == 'No':
        plt.show()
    
    ######################### Variable or not? #########################
    
    #Define the point indices in each plot
    puntos00_var = [[index_mostH, index_less],[5,5]]
    puntos01_var = [[index_mostJ, index_less],[5,5]]
    puntos02_var = [[index_CH4J, index_less],[5,5]]
    puntos03_var = [[index_Jcurve, index_less],[5,5]]
    puntos10_var = [[index_Jcurve, index_CH4J],[5,5]]
    puntos11_var = [[index_Jcurve, index_mostJ],[5,5]]
    puntos12_var = [[index_Jcurve, index_mostH],[5,5]]
    puntos13_var = [[index_CH4J, index_mostH],[5,5]]
    puntos20_var = [[index_mostJ, index_mostH],[5,5]]
    puntos21_var = [[index_CH4J, index_mostJ],[5,5]]
    puntos22_var = [[index_H2OJ, index_mostJ],[5,5]]
    puntos23_var = [[index_H2OJ, index_less],[5,5]]
    puntos30_var = [[index_H2OJ,index_mostH],[5,5]]
    puntos31_var = [[index_CH4J,index_H2OJ],[5,5]]
    puntos32_var = [[index_H2OJ, index_Jcurve],[5,5]]
    
    #Determine the point indices inside the variability areas in each plot
    inside00_var = path00.contains_points(puntos00_var)
    inside01_var = path01.contains_points(puntos01_var)
    inside02_var = path02.contains_points(puntos02_var)
    inside03_var = path03.contains_points(puntos03_var)
    inside10_var = path10.contains_points(puntos10_var)
    inside11_var = path11.contains_points(puntos11_var)
    inside12_var = path12.contains_points(puntos12_var)
    inside13_var = path13.contains_points(puntos13_var)
    inside20_var = path20.contains_points(puntos20_var)
    inside21_var = path21.contains_points(puntos21_var)
    inside22_var = path22.contains_points(puntos22_var)
    inside23_var = path23.contains_points(puntos23_var)
    inside30_var = path30.contains_points(puntos30_var)
    inside31_var = path31.contains_points(puntos31_var)
    inside32_var = path32.contains_points(puntos32_var)
    
    cant_plots_var=[]
    for i in range(0,2):
        cant_plots_var.append(sum([inside00_var[i],inside01_var[i],inside02_var[i],inside03_var[i], inside10_var[i],
                          inside11_var[i],inside12_var[i],inside13_var[i],inside20_var[i],inside21_var[i],
                          inside22_var[i],inside23_var[i],inside30_var[i], inside31_var[i], inside32_var[i]]))
    #pass threshold? 9
    print('{}{}{}'.format('Your brown dwarf fall in ',cant_plots_var[0],' variable areas of 15'))
    if cant_plots_var[0] > 8:
        a = print('Your brown dwarf is candidate variable')
    else:
        a = print('Your brown dwarf is candidate non-variable')
    return a,cant_plots_var[0] 


if spt == 'T':
    TdwarfIndices(dwarf_data)
if spt == 'L':
    LdwarfIndices(dwarf_data)