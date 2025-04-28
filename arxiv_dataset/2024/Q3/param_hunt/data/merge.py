# merges vertically
# only really necessary for training files!
import numpy as np
import pandas as pd

# are we using the short setup or the SHiP geometry?
iSHIP = 1

n_obs = 1 # 1 for bkg only, 1 for posterior training file, more for signal
n_subs = 10 # how many subfiles to merge
i_subs = 70 # number of starting file 
m_min, m_max = 0.1, 4.5
tau_min, tau_max = 0.01, 100 # short
z_min, z_max, z_cal, l_x, l_y = 10, 35, 35, 1.25, 1.25 # short 

if iSHIP:
	tau_min, tau_max = 0.05, 500 # SHiP
	z_min, z_max, z_cal, l_x, l_y = 32, 82, 93, 2, 3 # SHiP 




df_arr = []
for subs in range(i_subs,i_subs+n_subs):
    setfile = "event_set"+str(subs)+"_"+str(n_obs)+"_m_"+str(m_min)+"_"+str(m_max)+"_tm_"+str(tau_min)+"_"+str(tau_max)+"_c_"+str(z_min)+"_"+str(z_max)+"_"+str(l_x)+"_"+str(l_y)+".csv"
    df_arr.append(pd.read_csv(setfile))
    
dftry=pd.concat([df_arr[i] for i in range(n_subs)])
# change name accordingly, especially the counter
#dftry.to_csv("event_bkg_"+str(1)+"_m_"+str(m_min)+"_"+str(m_max)+"_tm_"+str(tau_min)+"_"+str(tau_max)+"_c_"+str(z_min)+"_"+str(z_max)+"_"+str(l_x)+"_"+str(l_y)+".csv", index=False)
#dftry.to_csv("event_train_11_m_"+str(m_min)+"_"+str(m_max)+"_tm_"+str(tau_min)+"_"+str(tau_max)+"_c_"+str(z_min)+"_"+str(z_max)+"_"+str(l_x)+"_"+str(l_y)+".csv", index=False)
dftry.to_csv("event_train_post_m_"+str(m_min)+"_"+str(m_max)+"_tm_"+str(tau_min)+"_"+str(tau_max)+"_c_"+str(z_min)+"_"+str(z_max)+"_"+str(l_x)+"_"+str(l_y)+".csv", index=False)
# for signal rename by hand malp to malp_0 and ctau_alp to ctau_alp_0
