# This code generates multiple configurations for generating VCell simulation data

import os
import random
num_itrs = 100
mode = 'rule_egfr' #'rule_ran'
reaction_rate_config_id = 0


# rule based EGFR tutorial

# specify reaction rates
# I tried, only Kf works, DirectHalf modifies the forward reaction rate, InverseHalf modifies the reverse reaction rate

Kf_R_Grb2_interaction_DirectHalf_0_default =  0.001
Kf_R_Grb2_interaction_InverseHalf_0_default =  0.05
Kf_R_ShcP_interaction_DirectHalf_0_default = 4.5e-4
Kf_R_ShcP_interaction_InverseHalf_0_default = 0.3
Kf_R_ShcU_interaction_DirectHalf_0_default = 0.045
Kf_R_ShcU_interaction_InverseHalf_0_default = 0.6
Kf_ShcDephosp_DirectHalf_0_default = 0.005
Kf_ShcDephosp_InverseHalf_0_default = 0
Kf_Shc_phosph_DirectHalf_0_default = 3.0
Kf_Shc_phosph_InverseHalf_0_default = 0
Kf_Y1_dephosph_DirectHalf_0_default = 4.5
Kf_Y1_dephosph_InverseHalf_0_default = 0
Kf_Y1_phosph_DirectHalf_0_default = 0.5
Kf_Y1_phosph_InverseHalf_0_default = 0
Kf_Y2_dephosph_DirectHalf_0_default = 4.5
Kf_Y2_dephosph_InverseHalf_0_default = 0
Kf_Y2_phosph_DirectHalf_0_default = 0.5
Kf_Y2_phosph_InverseHalf_0_default = 0
Kf_dimeriz_DirectHalf_0_default = 0.001
Kf_dimeriz_InverseHalf_0_default = 0.1
Kf_ligand_bind_DirectHalf_0_default = 0.003
Kf_ligand_bind_InverseHalf_0_default = 0.06

random_range = 0.5
random_range_substance = 0.9

Kf_R_Grb2_interaction_DirectHalf_0_rand =  random.uniform(Kf_R_Grb2_interaction_DirectHalf_0_default*(1-random_range),Kf_R_Grb2_interaction_DirectHalf_0_default*(1+random_range))
Kf_R_Grb2_interaction_InverseHalf_0_rand =  random.uniform(Kf_R_Grb2_interaction_InverseHalf_0_default*(1-random_range),Kf_R_Grb2_interaction_InverseHalf_0_default*(1+random_range))
Kf_R_ShcP_interaction_DirectHalf_0_rand = random.uniform(Kf_R_ShcP_interaction_DirectHalf_0_default*(1-random_range),Kf_R_ShcP_interaction_DirectHalf_0_default*(1+random_range))
Kf_R_ShcP_interaction_InverseHalf_0_rand = random.uniform(Kf_R_ShcP_interaction_InverseHalf_0_default*(1-random_range),Kf_R_ShcP_interaction_InverseHalf_0_default*(1+random_range))
Kf_R_ShcU_interaction_DirectHalf_0_rand = random.uniform(Kf_R_ShcU_interaction_DirectHalf_0_default*(1-random_range),Kf_R_ShcU_interaction_DirectHalf_0_default*(1+random_range))
Kf_R_ShcU_interaction_InverseHalf_0_rand = random.uniform(Kf_R_ShcU_interaction_InverseHalf_0_default*(1-random_range),Kf_R_ShcU_interaction_InverseHalf_0_default*(1+random_range))
Kf_ShcDephosp_DirectHalf_0_rand = random.uniform(Kf_ShcDephosp_DirectHalf_0_default*(1-random_range),Kf_ShcDephosp_DirectHalf_0_default*(1+random_range))
Kf_ShcDephosp_InverseHalf_0_rand = random.uniform(Kf_ShcDephosp_InverseHalf_0_default*(1-random_range),Kf_ShcDephosp_InverseHalf_0_default*(1+random_range))
Kf_Shc_phosph_DirectHalf_0_rand =random.uniform(Kf_Shc_phosph_DirectHalf_0_default*(1-random_range),Kf_Shc_phosph_DirectHalf_0_default*(1+random_range))
Kf_Shc_phosph_InverseHalf_0_rand = random.uniform(Kf_Shc_phosph_InverseHalf_0_default*(1-random_range),Kf_Shc_phosph_InverseHalf_0_default*(1+random_range))
Kf_Y1_dephosph_DirectHalf_0_rand = random.uniform(Kf_Y1_dephosph_DirectHalf_0_default*(1-random_range),Kf_Y1_dephosph_DirectHalf_0_default*(1+random_range))
Kf_Y1_dephosph_InverseHalf_0_rand =random.uniform(Kf_Y1_dephosph_InverseHalf_0_default*(1-random_range),Kf_Y1_dephosph_InverseHalf_0_default*(1+random_range))
Kf_Y1_phosph_DirectHalf_0_rand = random.uniform(Kf_Y1_phosph_DirectHalf_0_default*(1-random_range),Kf_Y1_phosph_DirectHalf_0_default*(1+random_range))
Kf_Y1_phosph_InverseHalf_0_rand = random.uniform(Kf_Y1_phosph_InverseHalf_0_default*(1-random_range),Kf_Y1_phosph_InverseHalf_0_default*(1+random_range))
Kf_Y2_dephosph_DirectHalf_0_rand = random.uniform(Kf_Y2_dephosph_DirectHalf_0_default*(1-random_range),Kf_Y2_dephosph_DirectHalf_0_default*(1+random_range))
Kf_Y2_dephosph_InverseHalf_0_rand = random.uniform(Kf_Y2_dephosph_InverseHalf_0_default*(1-random_range),Kf_Y2_dephosph_InverseHalf_0_default*(1+random_range))
Kf_Y2_phosph_DirectHalf_0_rand =random.uniform(Kf_Y2_phosph_DirectHalf_0_default*(1-random_range),Kf_Y2_phosph_DirectHalf_0_default*(1+random_range))
Kf_Y2_phosph_InverseHalf_0_rand = random.uniform(Kf_Y2_phosph_InverseHalf_0_default*(1-random_range),Kf_Y2_phosph_InverseHalf_0_default*(1+random_range))
Kf_dimeriz_DirectHalf_0_rand = random.uniform(Kf_dimeriz_DirectHalf_0_default*(1-random_range),Kf_dimeriz_DirectHalf_0_default*(1+random_range))
Kf_dimeriz_InverseHalf_0_rand = random.uniform(Kf_dimeriz_InverseHalf_0_default*(1-random_range),Kf_dimeriz_InverseHalf_0_default*(1+random_range))
Kf_ligand_bind_DirectHalf_0_rand = random.uniform(Kf_ligand_bind_DirectHalf_0_default*(1-random_range),Kf_ligand_bind_DirectHalf_0_default*(1+random_range))
Kf_ligand_bind_InverseHalf_0_rand = random.uniform(Kf_ligand_bind_InverseHalf_0_default*(1-random_range),Kf_ligand_bind_InverseHalf_0_default*(1+random_range))



# specify substance initial solution

Grb2_init_nM_default = 100
L_init_nM_default = 680
R_init_nM_default = 58
ShcP_init_nM_default = 0
ShcU_init_nM_default = 150

Grb2_init_nM_range=[max(Grb2_init_nM_default*(1-random_range_substance),0),Grb2_init_nM_default*(1+random_range_substance)]
L_init_nM_range=[max(L_init_nM_default*(1-random_range_substance),0),L_init_nM_default*(1+random_range_substance)]
R_init_nM_range=[max(R_init_nM_default*(1-random_range_substance),0),R_init_nM_default*(1+random_range_substance)]
ShcP_init_nM_range=[max(ShcP_init_nM_default*(1-random_range_substance),0),ShcP_init_nM_default*(1+random_range_substance)]
ShcU_init_nM_range =[max(ShcU_init_nM_default*(1-random_range_substance),0),ShcU_init_nM_default*(1+random_range_substance)]
#config_path_egfr = f'/store/MS-GODE/data/VCell_config_rule_egfr_size{num_itrs}_Grb2_{Kf_R_Grb2_interaction_DirectHalf_0_rand:.4f}_{Kf_R_Grb2_interaction_InverseHalf_0_rand:.3f}_ShcP_{Kf_R_ShcP_interaction_DirectHalf_0_rand:.5f}_{Kf_R_ShcP_interaction_InverseHalf_0_rand:.2f}_ShcU_{Kf_R_ShcU_interaction_DirectHalf_0_rand:.3f}_{Kf_R_ShcU_interaction_InverseHalf_0_rand:.2f}_ShcDe_{Kf_ShcDephosp_DirectHalf_0_rand:.4f}_{Kf_ShcDephosp_InverseHalf_0_rand:.1f}_Shcpho_{Kf_Shc_phosph_DirectHalf_0_rand:.1f}_{Kf_Shc_phosph_InverseHalf_0_rand:.1f}_Y1d_{Kf_Y1_dephosph_DirectHalf_0_rand:.1f}_{Kf_Y1_dephosph_InverseHalf_0_rand:.1f}_Y1p_{Kf_Y1_phosph_DirectHalf_0_rand:.1f}_{Kf_Y1_phosph_InverseHalf_0_rand:.1f}_Y2d_{Kf_Y2_dephosph_DirectHalf_0_rand:.1f}_{Kf_Y2_dephosph_InverseHalf_0_rand:.1f}_Y2p_{Kf_Y2_phosph_DirectHalf_0_rand:.1f}_{Kf_Y2_phosph_InverseHalf_0_rand:.1f}_dim_{Kf_dimeriz_DirectHalf_0_rand:.4f}_{Kf_dimeriz_InverseHalf_0_rand:.2f}_lig_{Kf_ligand_bind_DirectHalf_0_rand:.4f}_{Kf_ligand_bind_InverseHalf_0_rand:.3f}_{reaction_rate_config_id}.dat'
config_path_egfr = f'/store/MS-GODE/data/VCell_config_rule_egfr_size{num_itrs}_{reaction_rate_config_id}.dat'


# rule based Ran transport tutorial
# I tried, only Kf works, DirectHalf modifies the forward reaction rate, InverseHalf modifies the reverse reaction rate
Kf_C_p1_DirectHalf_0_range = [5,15] #[5,15] # default 10.0
Kf_C_p1_InverseHalf_0_range = [0.5,1.5] # default 1.0
Kf_C_p2_DirectHalf_0_range = [5,15]
Kf_C_p2_InverseHalf_0_range = [0.5,1.5]
Kf_C_p3_DirectHalf_0_range = [5,15]
Kf_C_p3_InverseHalf_0_range = [0.5,1.5]
Kf_Ran_C_bind_cyt_DirectHalf_0_range = [0.5,1.5]
Kf_Ran_C_bind_cyt_InverseHalf_0_range = [50,150]
Kf_Ran_C_bind_nuc_DirectHalf_0_range = [0.5,1.5]
Kf_Ran_C_bind_nuc_InverseHalf_0_range = [50,150]
Kf_Ran_RCC1_bind_DirectHalf_0_range = [0.5,1.5]
Kf_Ran_RCC1_bind_InverseHalf_0_range = [50,150]
#Kf_C_p1_DirectHalf_0_range = [5,15]
#Kf_C_p1_InverseHalf_0_range = [0.5,4.5]
#Kr_Transport_DirectHalf_0=55, Kf_C_p1_InverseHalf_0=12
Ran_C_nuc_init_um_range=[0,10e-4]
RCC1_init_um_range=[0,10e-4]
Kf_C_p1 = random.uniform(Kf_C_p1_DirectHalf_0_range[0],Kf_C_p1_DirectHalf_0_range[1])
Kr_C_p1 = random.uniform(Kf_C_p1_InverseHalf_0_range[0],Kf_C_p1_InverseHalf_0_range[1])
Kf_C_p2 = random.uniform(Kf_C_p2_DirectHalf_0_range[0],Kf_C_p2_DirectHalf_0_range[1])
Kr_C_p2 = random.uniform(Kf_C_p2_InverseHalf_0_range[0],Kf_C_p2_InverseHalf_0_range[1])
Kf_C_p3 = random.uniform(Kf_C_p3_DirectHalf_0_range[0],Kf_C_p3_DirectHalf_0_range[1])
Kr_C_p3 = random.uniform(Kf_C_p3_InverseHalf_0_range[0],Kf_C_p3_InverseHalf_0_range[1])
Kf_Ran_C_bind_cyt = random.uniform(Kf_Ran_C_bind_cyt_DirectHalf_0_range[0],Kf_Ran_C_bind_cyt_DirectHalf_0_range[1])
Kr_Ran_C_bind_cyt = random.uniform(Kf_Ran_C_bind_cyt_InverseHalf_0_range[0],Kf_Ran_C_bind_cyt_InverseHalf_0_range[1])
Kf_Ran_C_bind_nuc = random.uniform(Kf_Ran_C_bind_nuc_DirectHalf_0_range[0],Kf_Ran_C_bind_nuc_DirectHalf_0_range[1])
Kr_Ran_C_bind_nuc = random.uniform(Kf_Ran_C_bind_nuc_InverseHalf_0_range[0],Kf_Ran_C_bind_nuc_InverseHalf_0_range[1])
Kf_Ran_RCC1_bind = random.uniform(Kf_Ran_RCC1_bind_DirectHalf_0_range[0],Kf_Ran_RCC1_bind_DirectHalf_0_range[1])
Kr_Ran_RCC1_bind = random.uniform(Kf_Ran_RCC1_bind_InverseHalf_0_range[0],Kf_Ran_RCC1_bind_InverseHalf_0_range[1])

config_path_ran = f'/store/MS-GODE/data/VCell_config_rule_ran_size{num_itrs}_C_p1_{Kf_C_p1:.2f}_{Kr_C_p1:.2f}_C_p2_{Kf_C_p2:.2f}_{Kr_C_p2:.2f}_C_p3_{Kf_C_p3:.2f}_{Kr_C_p3:.2f}_cyt_{Kf_Ran_C_bind_cyt:.2f}_{Kr_Ran_C_bind_cyt:.2f}_nuc_{Kf_Ran_C_bind_nuc:.2f}_{Kr_Ran_C_bind_nuc:.2f}_{reaction_rate_config_id}.dat'

config_paths = {'rule_egfr':config_path_egfr, 'rule_ran':config_path_ran}

config_path = config_paths[mode]

while os.path.exists(config_path):
    print(f'This file exists: {config_path}')
    reaction_rate_config_id_old = reaction_rate_config_id
    reaction_rate_config_id+=1
    config_path = config_path.replace(f'_{reaction_rate_config_id_old}.dat',f'_{reaction_rate_config_id}.dat')

config_str = ''

for ite in range(num_itrs):
    if mode is 'rule_ran':
        new_config = f'Ran_C_nuc_init={random.uniform(Ran_C_nuc_init_um_range[0],Ran_C_nuc_init_um_range[1])},RCC1_init={random.uniform(RCC1_init_um_range[0],RCC1_init_um_range[1])},Kf_C_p1_DirectHalf_0={Kf_C_p1},Kf_C_p1_InverseHalf_0={Kr_C_p1},Kf_C_p2_DirectHalf_0={Kf_C_p2},Kf_C_p2_InverseHalf_0={Kr_C_p2},Kf_C_p3_DirectHalf_0={Kf_C_p3},Kf_C_p3_InverseHalf_0={Kr_C_p3},Kf_Ran_C_bind_cyt_DirectHalf_0={Kf_Ran_C_bind_cyt},Kf_Ran_C_bind_cyt_InverseHalf_0={Kr_Ran_C_bind_cyt},Kf_Ran_C_bind_nuc_DirectHalf_0={Kf_Ran_C_bind_nuc},Kf_Ran_C_bind_nuc_InverseHalf_0={Kr_Ran_C_bind_nuc}\n'
    elif mode is 'rule_egfr':
        new_config = f'Grb2_init_nM={random.uniform(Grb2_init_nM_range[0], Grb2_init_nM_range[1])},L_init_nM={random.uniform(L_init_nM_range[0], L_init_nM_range[1])},R_init_nM={random.uniform(R_init_nM_range[0], R_init_nM_range[1])},ShcP_init_nM={random.uniform(ShcP_init_nM_range[0], ShcP_init_nM_range[1])},ShcU_init_nM={random.uniform(ShcU_init_nM_range[0], ShcU_init_nM_range[1])}\n'
    config_str = config_str + new_config

with open(f'{config_path}','w') as f:
    f.write(config_str)