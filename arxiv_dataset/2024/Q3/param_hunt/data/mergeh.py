# merge horizontally
# only needed for background events (both train and obs)
import numpy as np
import pandas as pd

# are we using the short setup or the SHiP geometry?
iSHIP = 1
train = 1 # are we creating obs files or training file combining two background training files

i_subs = 5 # first subfile to join
n_subs = 13 # how many bkg events in sample
if train:
	i_subs = 0
	n_subs = 2
	
m_min, m_max = 0.1, 4.5
tau_min, tau_max = 0.01, 100 # short
z_min, z_max, z_cal, l_x, l_y = 10, 35, 35, 1.25, 1.25 # short 

if iSHIP:
	tau_min, tau_max = 0.05, 500 # SHiP
	z_min, z_max, z_cal, l_x, l_y = 32, 82, 93, 2, 3 # SHiP 


df_arr = []
for isub, subs in enumerate(range(i_subs,i_subs+n_subs)):
    setfile = "event_set"+str(subs)+"_"+str(1)+"_m_"+str(m_min)+"_"+str(m_max)+"_tm_"+str(tau_min)+"_"+str(tau_max)+"_c_"+str(z_min)+"_"+str(z_max)+"_"+str(l_x)+"_"+str(l_y)+".csv"
    if train:
		# bkg training files have a specific name
	    setfile = "event_bkg_"+str(subs)+"_m_"+str(m_min)+"_"+str(m_max)+"_tm_"+str(tau_min)+"_"+str(tau_max)+"_c_"+str(z_min)+"_"+str(z_max)+"_"+str(l_x)+"_"+str(l_y)+".csv"
    df = pd.read_csv(setfile)
    df = df.rename(columns={"m_alp": "m_alp_"+str(isub), "ctau_alp": "ctau_alp_"+str(isub),"E1_0": "E1_"+str(isub),"px1_0": "px1_"+str(isub),"py1_0":"py1_"+str(isub),"pz1_0":"pz1_"+str(isub),"E2_0":"E2_"+str(isub),"px2_0":"px2_"+str(isub),"py2_0":"py2_"+str(isub),"pz2_0":"pz2_"+str(isub),"chx1_0":"chx1_"+str(isub), "chy1_0":"chy1_"+str(isub),"chz1_0":"chz1_"+str(isub),"chx2_0":"chx2_"+str(isub),"chy2_0":"chy2_"+str(isub),"chz2_0":"chz2_"+str(isub),"Vx_0":"Vx_"+str(isub),"Vy_0":"Vy_"+str(isub),"Vz_0":"Vz_"+str(isub)})
    df_arr.append(df)
    
dftry=pd.concat([df_arr[i] for i in range(n_subs)], axis=1)
if train ==0:
	dftry.to_csv("event_bkg_"+str(n_subs)+"_m_"+str(m_min)+"_"+str(m_max)+"_tm_"+str(tau_min)+"_"+str(tau_max)+"_c_"+str(z_min)+"_"+str(z_max)+"_"+str(l_x)+"_"+str(l_y)+".csv", index=False)
elif train:
	dftry.to_csv("event_train_00_m_"+str(m_min)+"_"+str(m_max)+"_tm_"+str(tau_min)+"_"+str(tau_max)+"_c_"+str(z_min)+"_"+str(z_max)+"_"+str(l_x)+"_"+str(l_y)+".csv", index=False)

