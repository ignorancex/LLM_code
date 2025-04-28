#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thanks for the help of programming from Gu Hongrui, Huang Qichen, Meng Weiyu, Xiao Kai!!
Author       : Huang,Bowen
Date         : 2023-05-14
Version      : 1.0
E-mail       : 202121160005@mail.bnu.edu.cn
Description  : 

"""

import numpy as np
import pandas as pd
import os

pat = __file__
interC2_T = pd.read_csv(os.path.join(os.path.dirname(pat),'dotruc_interC2_CM.csv'),header=None)
interC3_T = pd.read_csv(os.path.join(os.path.dirname(pat),'dotruc_interC3_CM.csv'),header=None)
interG_T = pd.read_csv(os.path.join(os.path.dirname(pat),'dotruc_interG_CM.csv'),header=None)
interGext_T = pd.read_csv(os.path.join(os.path.dirname(pat),'interGext_CM.csv'),header=None)
interyyC2_T = pd.read_csv(os.path.join(os.path.dirname(pat),'dotruc_interyyC2_CM.csv'),header=None)
interyyC3_T = pd.read_csv(os.path.join(os.path.dirname(pat),'dotruc_interyyC3_CM.csv'),header=None)
interyyG_T = pd.read_csv(os.path.join(os.path.dirname(pat),'dotruc_interyyG_CM.csv'),header=None)
interyyGext_T = pd.read_csv(os.path.join(os.path.dirname(pat),'dotruc_interyyGext_CM.csv'),header=None)

interC2_F = pd.read_csv(os.path.join(os.path.dirname(pat),'untruc_interC2_CM.csv'),header=None)
interC3_F = pd.read_csv(os.path.join(os.path.dirname(pat),'untruc_interC3_CM.csv'),header=None)
interG_F = pd.read_csv(os.path.join(os.path.dirname(pat),'untruc_interG_CM.csv'),header=None)
interGext_F = pd.read_csv(os.path.join(os.path.dirname(pat),'interGext_CM.csv'),header=None)
interyyC2_F = pd.read_csv(os.path.join(os.path.dirname(pat),'untruc_interyyC2_CM.csv'),header=None)
interyyC3_F = pd.read_csv(os.path.join(os.path.dirname(pat),'untruc_interyyC3_CM.csv'),header=None)
interyyG_F = pd.read_csv(os.path.join(os.path.dirname(pat),'untruc_interyyG_CM.csv'),header=None)
interyyGext_F = pd.read_csv(os.path.join(os.path.dirname(pat),'untruc_interyyGext_CM.csv'),header=None)

inter_phot = pd.read_csv(os.path.join(os.path.dirname(pat),'inter_phot.csv'),header=None)
inter_BP = pd.read_csv(os.path.join(os.path.dirname(pat),'inter_BP.csv'),header=None)
inter_RP = pd.read_csv(os.path.join(os.path.dirname(pat),'inter_RP.csv'),header=None)

interC2_T = interC2_T.to_numpy()
interC3_T = interC3_T.to_numpy()
interG_T = interG_T.to_numpy()
interGext_T = interGext_T.to_numpy()
interyyC2_T = interyyC2_T.to_numpy()
interyyC3_T = interyyC3_T.to_numpy()
interyyG_T = interyyG_T.to_numpy()
interyyGext_T = interyyGext_T.to_numpy()

interC2_F = interC2_F.to_numpy()
interC3_F = interC3_F.to_numpy()
interG_F = interG_F.to_numpy()
interGext_F = interGext_F.to_numpy()
interyyC2_F = interyyC2_F.to_numpy()
interyyC3_F = interyyC3_F.to_numpy()
interyyG_F = interyyG_F.to_numpy()
interyyGext_F = interyyGext_F.to_numpy()

inter_phot = inter_phot.to_numpy()
inter_BP = inter_BP.to_numpy()
inter_RP = inter_RP.to_numpy()

un_cauC2 = -1.238505903
un_cauC3 = 1.032292962
do_cauC2 = -1.257463033
do_cauC3 = 1.095023301
cauG_bright = 2.702518
cauG_faint = 17.122173

def correction(flux_origin,G,error=False,Truncation=False,absolute_correction=True):
    if Truncation == True:
        interC2 = interC2_T
        interC3 = interC3_T
        interG = interG_T
        interGext = interGext_T
        interyyC2 = interyyC2_T
        interyyC3 = interyyC3_T
        interyyG = interyyG_T
        interyyGext = interyyGext_T
        cauC2 = do_cauC2
        cauC3 = do_cauC3
    if Truncation == False:
        interC2 = interC2_F
        interC3 = interC3_F
        interG = interG_F
        interGext = interGext_F
        interyyC2 = interyyC2_F
        interyyC3 = interyyC3_F
        interyyG = interyyG_F
        interyyGext = interyyGext_F
        cauC2 = un_cauC2
        cauC3 = un_cauC3
    if not type(error) == bool:
        gaia_wl = np.linspace(3360,3360+20*342,343)
        flux = flux_origin #原流量
        nor = np.mean(flux[62:78]) #归一化位置流量
        nor_error = (sum(error[62:78]**2)**(0.5))/16 #归一化位置误差
        flux_nor = flux/nor # 归一化后流量
        error_nor =  abs(flux_nor)*np.sqrt((error/flux)**2+(nor_error/nor)**2) # 归一化后误差
        error_rel = error_nor/flux_nor # 归一化后相对误差
        del nor_error, error_nor
        # error_rel = error/flux+0.015

        C2_tem1 = np.polyfit(np.ones(22),flux_nor[27:49],0,w=error_rel[27:49]**(-2))
        C2_tem2 = np.polyfit(np.ones(22),flux_nor[49:71],0,w=error_rel[49:71]**(-2))
        C2 = (C2_tem1-C2_tem2)/(np.mean(gaia_wl[27:49])/1000-np.mean(gaia_wl[49:71])/1000)
        C3_tem1 = np.polyfit(np.ones(31),flux_nor[71:102],0,w=error_rel[71:102]**(-2))
        C3_tem2 = np.polyfit(np.ones(31),flux_nor[102:133],0,w=error_rel[102:133]**(-2))
        C3 = (C3_tem1-C3_tem2)/(np.mean(gaia_wl[71:102])/1000-np.mean(gaia_wl[102:133])/1000)
        del flux,error_rel,nor,flux_nor,C2_tem1,C2_tem2,C3_tem1,C3_tem2
        C2 = C2[0]
        C3 = C3[0]
        
    if type(error) == bool:
        gaia_wl = np.linspace(3360,3360+20*342,343)
        flux = flux_origin
        nor = np.mean(flux[62:78])
        flux_nor = flux/nor
        C2_tem1 = np.mean(flux_nor[27:49])
        C2_tem2 = np.mean(flux_nor[49:71])
        C2 = (C2_tem1-C2_tem2)/(np.mean(gaia_wl[27:49])/1000-np.mean(gaia_wl[49:71])/1000)
        C3_tem1 = np.mean(flux_nor[71:102])
        C3_tem2 = np.mean(flux_nor[102:133])
        C3 = (C3_tem1-C3_tem2)/(np.mean(gaia_wl[71:102])/1000-np.mean(gaia_wl[102:133])/1000)
        del flux,nor,flux_nor,C2_tem1,C2_tem2,C3_tem1,C3_tem2
    
    
    if C2<interC2[0,0] or C2>interC2[-1,0] or C3>interC3[-1,0] or C3<interC3[0,0]:
        caution = np.zeros([3],dtype=int)
        # C2C3警告
        if C2<interC2[0,0] or C2>interC2[-1,0]:
            caution[0] = 2
        if C2>=interC2[0,0] and C2<cauC2:
            caution[0] = 1
        if C3>interC3[-1,0] or C3<interC3[0,0]:
            caution[1] = 2
        if C3<=interC3[-1,0] and C3>cauC3:
            caution[1] = 1
        # G警告
        if G<2.703 or G>17.4:
            caution[2] = 3
        if G>=10 and G<=17.122:
            caution[2] = 1
        if G>17.122 and G<=17.4:
            caution[2] = 2
        flux_out = flux_origin
        return flux_out,caution,C2,C3
    
    if C2>interC2[0,0] and C2<interC2[-1,0] and C3<interC3[-1,0] and C3>interC3[0,0]:
        C2_cor = interyyC2[int(np.where(abs(interC2-np.round(C2,3))<0.0001)[0]),:]
        C3_cor = interyyC3[int(np.where(abs(interC3-np.round(C3,3))<0.0001)[0]),:]

        if G<2.703:
            G_cor = interyyG[int(np.where(abs(interG-np.round(2.703,3))<0.0001)[0]),:]
            cor = C2_cor*C3_cor*G_cor
            flux_out = flux_origin*cor
        if G<10 and G>=2.703:
            G_cor = interyyG[int(np.where(abs(interG-np.round(G,3))<0.0001)[0]),:]
            cor = C2_cor*C3_cor*G_cor
            flux_out = flux_origin*cor
        if G>=10 and G<=17.4:
            G_cor = interyyGext[int(np.where(abs(interGext-np.round(G,3))<0.0001)[0]),:]
            if G<=17.122:
                G_inn = np.mean(interyyG[int(np.where(abs(interG-np.round(G,3))<0.0001)[0]),62:78])
                G_cor = G_cor*G_inn
            if G>interG[-1,0]:
                G_inn = np.mean(interyyG[int(np.where(abs(interG-np.round(17.122,3))<0.0001)[0]),62:78])
                G_cor = G_cor*G_inn
            
            if absolute_correction == True:
                G_BP = inter_BP[int(np.where(abs(inter_phot-np.round(G,3))<0.0001)[0]),:]
                G_RP = inter_RP[int(np.where(abs(inter_phot-np.round(G,3))<0.0001)[0]),:]
                G_ext = np.ones(343)
                G_ext[0:105] = G_ext[0:105]*G_BP
                G_ext[105:344] = G_ext[105:344]*G_RP
                G_cor = G_cor*G_ext
                
            cor = C2_cor*C3_cor*G_cor
            flux_out = flux_origin*cor
        if G>17.4:
            G_cor = interyyGext[int(np.where(abs(interGext-np.round(17.4,3))<0.0001)[0]),:]
            G_inn = np.mean(interyyG[int(np.where(abs(interG-np.round(17.122,3))<0.0001)[0]),62:78])
            G_cor = G_cor*G_inn
            
            if absolute_correction == True:
                if G<=17.65:
                    G_BP = inter_BP[int(np.where(abs(inter_phot-np.round(G,3))<0.0001)[0]),:]
                    G_RP = inter_RP[int(np.where(abs(inter_phot-np.round(G,3))<0.0001)[0]),:]
                    G_ext = np.ones(343)
                    G_ext[0:105] = G_ext[0:105]*G_BP
                    G_ext[105:344] = G_ext[105:344]*G_RP
                    G_cor = G_cor*G_ext
                if G>17.65:
                    G_BP = inter_BP[int(np.where(abs(inter_phot-np.round(17.65,3))<0.0001)[0]),:]
                    G_RP = inter_RP[int(np.where(abs(inter_phot-np.round(17.65,3))<0.0001)[0]),:]
                    G_ext = np.ones(343)
                    G_ext[0:105] = G_ext[0:105]*G_BP
                    G_ext[105:344] = G_ext[105:344]*G_RP
                    G_cor = G_cor*G_ext
            
            cor = C2_cor*C3_cor*G_cor
            flux_out = flux_origin*cor
            

                
    
                
        caution = np.zeros([3],dtype=int)
        # C2C3警告
        if C2<interC2[0,0] or C2>interC2[-1,0]:
            caution[0] = 2
        if C2>=interC2[0,0] and C2<cauC2:
            caution[0] = 1
        if C3>interC3[-1,0] or C3<interC3[0,0]:
            caution[1] = 2
        if C3<=interC3[-1,0] and C3>cauC3:
            caution[1] = 1
        # G警告
        if G<2.703 or G>17.4:
            caution[2] = 3
        if G>=10 and G<=17.122:
            caution[2] = 1
        if G>17.122 and G<=17.4:
            caution[2] = 2

        return flux_out,caution,C2,C3
    
def correction_df(dataframe_origin,G,have_error=True,Truncation=False,absolute_correction=True):
    df = pd.DataFrame(data=None,columns=['flux_cor','C2','C3','Caution'])
    dataframe_origin = dataframe_origin.join(df)
    df_len = len(dataframe_origin)
    
    if have_error == True:
        for i in range(0,df_len):
            [flux_new,caution,C2,C3] = correction(dataframe_origin.at[i,'flux'],G[i],\
                                                                  dataframe_origin.at[i,'flux_error'],\
                                                                  Truncation=Truncation,\
                                                                  absolute_correction=absolute_correction)
            dataframe_origin.at[i,'flux_cor'] = flux_new
            dataframe_origin.at[i,'C2'] = C2
            dataframe_origin.at[i,'C3'] = C3
            dataframe_origin.at[i,'Caution'] = caution

    if have_error == False:
        for i in range(0,df_len):
            [flux_new,caution,C2,C3] = correction(dataframe_origin.at[i,'flux'],G[i],error=False,\
                                                                 Truncation=Truncation,\
                                                                 absolute_correction=absolute_correction)
            dataframe_origin.at[i,'flux_cor'] = flux_new
            dataframe_origin.at[i,'C2'] = C2
            dataframe_origin.at[i,'C3'] = C3
            dataframe_origin.at[i,'Caution'] = caution


    return(dataframe_origin)

# def correction_df(dataframe_origin,G,have_error=True,Truncation=False):
#     df = pd.DataFrame(data=None,columns=['flux_cor','C2','C3','Caution'])
#     dataframe_origin = dataframe_origin.join(df)
#     df_len = len(dataframe_origin)
#     if have_error == True:
#         if Truncation == True:
#             for i in range(0,df_len):
#                 [flux_new,caution,C2,C3] = correction(dataframe_origin.at[i,'flux'],G[i],\
#                                                                       dataframe_origin.at[i,'flux_error'],\
#                                                                       Truncation=True)
#                 dataframe_origin.at[i,'flux_cor'] = flux_new
#                 dataframe_origin.at[i,'C2'] = C2
#                 dataframe_origin.at[i,'C3'] = C3
#                 dataframe_origin.at[i,'Caution'] = caution
#         if Truncation == False:
#             for i in range(0,df_len):
#                 [flux_new,caution,C2,C3] = correction(dataframe_origin.at[i,'flux'],G[i],\
#                                                                       dataframe_origin.at[i,'flux_error'],\
#                                                                       Truncation=False)
#                 dataframe_origin.at[i,'flux_cor'] = flux_new
#                 dataframe_origin.at[i,'C2'] = C2
#                 dataframe_origin.at[i,'C3'] = C3
#                 dataframe_origin.at[i,'Caution'] = caution
#     if have_error == False:
#         if Truncation == True:
#                 for i in range(0,df_len):
#                     [flux_new,caution,C2,C3] = correction(dataframe_origin.at[i,'flux'],G[i],error=False,\
#                                                                       Truncation=True)
#                     dataframe_origin.at[i,'flux_cor'] = flux_new
#                     dataframe_origin.at[i,'C2'] = C2
#                     dataframe_origin.at[i,'C3'] = C3
#                     dataframe_origin.at[i,'Caution'] = caution
                    
#         if Truncation == False:
#             for i in range(0,df_len):
#                 [flux_new,caution,C2,C3] = correction(dataframe_origin.at[i,'flux'],G[i],error=False,\
#                                                                       Truncation=False)
#                 dataframe_origin.at[i,'flux_cor'] = flux_new
#                 dataframe_origin.at[i,'C2'] = C2
#                 dataframe_origin.at[i,'C3'] = C3
#                 dataframe_origin.at[i,'Caution'] = caution

#     return(dataframe_origin)


