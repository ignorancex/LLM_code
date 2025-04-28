import numpy as np
import pandas as pd
import gudhi as gd
from angio import angio_abm

#initialize simulation

#### chemoattractant distribution
# linear : profile linearly increases with x
# tumor  : profile max as the point (1,0.5) imitating a tumor center at that location
C_dist = 'linear'

#max time units a simulation can run
final_time = 20.0

#chemotaxis rate
chemo_rate = 'const'

psi = 0.1

paramsfile = snakemake.input[0]

params_i = int(snakemake.params['index'])

params = pd.read_csv(paramsfile,header=None)

hapt,chi = params.iloc[params_i,:]
    
#initialize ABM simulation
A = angio_abm(C_dist,rho = hapt,t_final = final_time,chi = chi,chemo_rate = chemo_rate,psi = psi)

#initialize chemoattractant and TAF grids, sprout locations
A.IC_generate()
A.sprout_initialize()

#Record biological data (sprouts, tips, branches)
A.record_bio_data()

#Run the ABM until either one of the sprouts reaches x=0.95
#or when time exceeds max_sim_time
j = 0
max_x = 0
while max_x < 0.95:
    #move sprouts
    A.move_sprouts()
    #update TAF , chemoattractant
    A.update_grids()
    #perform branching
    A.branch()

    #Save bio info
    A.record_bio_data()

    #max x value reached by sprout tips
    max_x = np.max(A.X[A.N==1])

    j+=1
    if A.dt*j > final_time:
        #stop simulation if time exceeds max_sim_time
        break

outfile = snakemake.output[0]
#save extended persistence data
A.plane_sweeping_TDA(outfile)

#save simulation image
bio_data = A.get_bio_data()
imagefile = snakemake.output[1]
np.save(imagefile,bio_data['N'])
