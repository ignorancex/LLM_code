import numpy as np
import pandas as pd

# can comment out next three if not interested in parallelization
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

# CHOOSE GEOMETRY HERE
# z_min, z_max, z_cal, l_x, l_y = 10, 35, 35, 1.25, 1.25 # short experiment
z_min, z_max, z_cal, l_x, l_y= 32, 82, 93, 2, 3 #l_x, l_y are half lengths of the calorimeter

dr_gg, E_min = 0.1, 1 # in meters and GeV
m_min, m_max = 0.1, 4.5
tau_min, tau_max = 0.05, 500


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nevents', type=int, help="total generated events", default=100)
parser.add_argument('--nobs', type=int, help="number observed events", default=1)
parser.add_argument('--mfixed', type=float, help="fixed mass", default=0)
parser.add_argument('--taufixed', type=float, help="fixed lifetime", default=0)
parser.add_argument('--set_num', type=int, help="subset number", default=0)
args = parser.parse_args()

nev  = args.nevents
n_obs = args.nobs
fixed_mass= args.mfixed
fixed_tau= args.taufixed
set_num = args.set_num

if fixed_mass:
    m_min, m_max = fixed_mass, fixed_mass
    
if fixed_tau:
    tau_min, tau_max = fixed_tau, fixed_tau



par = 1      # parallelization
tau_mass = 1 # tau actually indicates the tau/mass parameter

from ALP_decay import b_meson_decay, alp_decay, random_event, generate_events

if par:
    def processInput(i):
        np.random.seed() #necessary?
        return generate_events(1, n_obs, m_min, m_max, tau_min, tau_max, z_min=z_min, z_max=z_max, z_cal=z_cal, equal_weights = True, l_x = l_x, l_y=l_y, dr_gg = dr_gg, E_min = E_min, tau_mass = tau_mass)[0]
    events=Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in range(nev))
    
else:
    events = generate_events(nev, n_obs, m_min, m_max, tau_min, tau_max, z_min=z_min, z_max=z_max, z_cal=z_cal, equal_weights = True, l_x = l_x, l_y=l_y, dr_gg = dr_gg, E_min = E_min, tau_mass = tau_mass)


df = pd.DataFrame.from_records(events)
df_out = pd.concat([df['m_alp'], df['ctau_alp']], axis=1)


for iOBS in range(n_obs):
    df_g1 = pd.DataFrame(df["gamma_1_4momentum_"+str(iOBS)].to_list(), columns=[s + '_' +str(iOBS) for s in ['E1','px1', 'py1', 'pz1']])
    df_g2 = pd.DataFrame(df["gamma_2_4momentum_"+str(iOBS)].to_list(), columns=[s + '_' +str(iOBS) for s in ['E2','px2', 'py2', 'pz2']])
    df_ch1 = pd.DataFrame(df["gamma_1_calo_hit_"+str(iOBS)].to_list(), columns=[s + '_' +str(iOBS) for s in ['chx1', 'chy1', 'chz1']])
    df_ch2 = pd.DataFrame(df["gamma_2_calo_hit_"+str(iOBS)].to_list(), columns=[s + '_' +str(iOBS) for s in ['chx2', 'chy2', 'chz2']])
    df_V = pd.DataFrame(df["decay_position_"+str(iOBS)].to_list(), columns=[s + '_' +str(iOBS) for s in ['Vx', 'Vy', 'Vz']])


    df_out = pd.concat([df_out, df_g1, df_g2, df_ch1, df_ch2, df_V], axis=1) # same name for the variables, but different entries    

if tau_mass:    
    df_out.to_csv("/x/morandini/data/event_set"+str(set_num)+"_"+str(n_obs)+"_m_"+str(m_min)+"_"+str(m_max)+"_tm_"+str(tau_min)+"_"+str(tau_max)+"_c_"+str(z_min)+"_"+str(z_max)+"_"+str(l_x)+"_"+str(l_y)+".csv", index= False)
else:
    df_out.to_csv("/x/morandini/data/event_set"+str(set_num)+"_"+str(n_obs)+"_m_"+str(m_min)+"_"+str(m_max)+"_t_"+str(tau_min)+"_"+str(tau_max)+"_c_"+str(z_min)+"_"+str(z_max)+"_"+str(l_x)+"_"+str(l_y)+".csv", index= False)

