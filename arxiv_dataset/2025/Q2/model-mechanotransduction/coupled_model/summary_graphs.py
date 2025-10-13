import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pyvista as pv

# define colours
myblue = (0,119/235,187/235)
myred=(187/235,85/235,102/235)
myyellow=(221/235,170/235,51/235)
mygrey=(187/235,187/235,187/235)
mygreen="#66BB55"
mymagenta="#7733DD"
myblack="#48494B"

# define text size
axes_labels = 25
axes_ticks = 20
legend_labels = 20
rcParams['font.size'] = 20

# define parameters
dt = 0.05
T = 10
k6 = 1
Es = np.array([0.001, 0.01, 0.1, 0.3, 1, 5.7, 10, 50, 100, 7000000])
n = len(Es)
order = np.array([0, 2, 4, 6, 1, 5, 3, 7])

# loop for different boundary conditions for linear elasticity (partfixed = no deformation in the z-direction at the bottom of the cell)
for bcs in ['_partfixed','_neumann']:
    
    # for loop for different stimuli (2D and 3D)
    for stim in ['_2D','']:
        
        # create Figures 5, 10, 15, and 20
        fig, axs = plt.subplots(4,4, sharex=True, sharey='row', figsize=(24,16), layout='constrained')
        k = 0
        
        # for loop for different cell shapes (1 = radially symmetric, 0 = lamellidpodium)
        for meshnr in [1,0]:

            # create empty arrays to save variables
            Ec = np.zeros((8,n))
            Ec_min = np.zeros((8,n))
            Ec_max = np.zeros((8,n))
            u_div_l2 = np.zeros((8,n))
            u_div_min = np.zeros((8,n))
            u_div_max = np.zeros((8,n))
            ca = np.zeros((8,n))
            ca_min = np.zeros((8,n))
            ca_max = np.zeros((8,n))
            pa = np.zeros((8,n))
            pa_min = np.zeros((8,n))
            pa_max = np.zeros((8,n))

            j = 0

            # for loop for different couplings (1-> Ec=0.6, C1=0; 2-> Ec=0.6, C1=1; 3-> Ec=f(phi), C1=0; 4-> Ec=f(phi), C1=1)
            for coupling in [1,2,3,4]:
                i = 0

                # for loop for different substrate stiffnesses E
                for bigE in Es:
                    #print('STATE:', bcs, stim, meshnr, coupling, bigE)

                    # read from files saved from simulations and find mean of phi_a, rho_a, div(u) and Ec/f(phi) to plot
                    file_ca = pv.read("../results/simulations/coupled"+str(coupling)+str(stim)+"_ca"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_"+str(bigE)+"E000199.vtu")
                    cell_ca = file_ca.point_data_to_cell_data()
                    ca_values = cell_ca.get_array(cell_ca.array_names[0])

                    sized = file_ca.compute_cell_sizes(length=False, area=False)
                    cell_volumes = np.abs(sized.cell_data["Volume"])
                    tot_vol = np.sum(cell_volumes)
                    ca[order[j],i] = np.sum(ca_values*cell_volumes) / tot_vol
                    #print('mean ca', np.sum(ca_values*cell_volumes) / tot_vol)
                    
                    file_pa = pv.read("../results/simulations/coupled"+str(coupling)+str(stim)+"_p"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_"+str(bigE)+"E000199.vtu")
                    cell_pa = file_pa.point_data_to_cell_data()
                    pa_values = cell_pa.get_array(cell_pa.array_names[0])

                    sized = file_pa.compute_cell_sizes(length=False, volume=False)
                    cell_areas = np.abs(sized.cell_data["Area"])
                    tot_vol = np.sum(cell_areas)
                    pa[order[j],i] = (np.sum(pa_values*cell_areas) / tot_vol)*10**(-5)
                    #print('mean pa', np.sum(pa_values*cell_areas) / tot_vol)
                    
                    file_u = pv.read("../results/simulations/coupled"+str(coupling)+str(stim)+"_u"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_"+str(bigE)+"E000199.vtu")
                    diverg = file_u.compute_derivative(gradient=False, divergence=True)
                    cell_divu = diverg.point_data_to_cell_data()
                    divu_values = cell_divu.get_array('divergence')

                    sized = file_u.compute_cell_sizes(length=False, area=False)
                    cell_volumes = np.abs(sized.cell_data["Volume"])
                    tot_vol = np.sum(cell_volumes)
                    u_div_l2[order[j],i] = np.sum(divu_values*cell_volumes) / tot_vol
                    #print('mean div(u)', np.sum(divu_values*cell_volumes) / tot_vol)
                    
                    file_ec = pv.read("../results/simulations/coupled"+str(coupling)+str(stim)+"_ec"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_"+str(bigE)+"E000199.vtu")
                    cell_ec = file_ec.point_data_to_cell_data()
                    ec_values = cell_ec.get_array(cell_ec.array_names[0])

                    sized = file_ec.compute_cell_sizes(length=False, area=False)
                    cell_volumes = np.abs(sized.cell_data["Volume"])
                    tot_vol = np.sum(cell_volumes)
                    Ec[order[j],i] = np.sum(ec_values*cell_volumes) / tot_vol
                    #print('mean ec', np.sum(ec_values*cell_volumes) / tot_vol)
                    
                    if bcs == '_neumann':
                        variables = np.load('../results/temp/tempstats'+str(coupling)+str(stim)+'-oth-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+'_'+str(bigE)+'E.npy')
                        tempstats = np.load('../results/temp/tempstats'+str(coupling)+str(stim)+'-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+'_'+str(bigE)+'E.npy')
                        tempstats_mm = np.load('../results/temp/tempstats'+str(coupling)+str(stim)+'-minmax-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+'_'+str(bigE)+'E.npy')
                    else:
                        variables = np.load('../results/temp/tempstats'+str(coupling)+str(bcs)+str(stim)+'-oth-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+'_'+str(bigE)+'E.npy')
                        tempstats = np.load('../results/temp/tempstats'+str(coupling)+str(bcs)+str(stim)+'-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+'_'+str(bigE)+'E.npy')
                        tempstats_mm = np.load('../results/temp/tempstats'+str(coupling)+str(bcs)+str(stim)+'-minmax-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+'_'+str(bigE)+'E.npy')

                    Ec_min[order[j],i] = variables[3,-1]
                    Ec_max[order[j],i] = variables[4,-1]

                    u_div_min[order[j],i] = variables[7,-1]
                    u_div_max[order[j],i] = variables[8,-1]

                    ca_min[order[j],i] = tempstats_mm[0,-1]
                    ca_max[order[j],i] = tempstats_mm[1,-1]

                    pa_min[order[j],i] = tempstats_mm[4,-1]*10**(-5)
                    pa_max[order[j],i] = tempstats_mm[5,-1]*10**(-5)

                    i += 1
                
                j += 1

            # for loop to add extra values of C1=0.5, 2
            for C1 in [0.5,2.0]:
                for coupling in [2,4]:
                    i = 0
                    for bigE in Es:
                        #print('STATE:', bcs, stim, meshnr, coupling, bigE)

                        # read from files saved from simulations and find mean of phi_a, rho_a, div(u) and Ec/f(phi) to plot
                        file_ca = pv.read("../results/simulations/coupled"+str(coupling)+str(stim)+"_ca"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+'_C1='+str(C1)+"_"+str(bigE)+"E000199.vtu")
                        cell_ca = file_ca.point_data_to_cell_data()
                        ca_values = cell_ca.get_array(cell_ca.array_names[0])

                        sized = file_ca.compute_cell_sizes(length=False, area=False)
                        cell_volumes = np.abs(sized.cell_data["Volume"])
                        tot_vol = np.sum(cell_volumes)
                        ca[order[j],i] = np.sum(ca_values*cell_volumes) / tot_vol
                        #print('mean ca', np.sum(ca_values*cell_volumes) / tot_vol)
                        
                        file_pa = pv.read("../results/simulations/coupled"+str(coupling)+str(stim)+"_p"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+'_C1='+str(C1)+"_"+str(bigE)+"E000199.vtu")
                        cell_pa = file_pa.point_data_to_cell_data()
                        pa_values = cell_pa.get_array(cell_pa.array_names[0])

                        sized = file_pa.compute_cell_sizes(length=False, volume=False)
                        cell_areas = np.abs(sized.cell_data["Area"])
                        tot_vol = np.sum(cell_areas)
                        pa[order[j],i] = (np.sum(pa_values*cell_areas) / tot_vol)*10**(-5)
                        #print('mean pa', np.sum(pa_values*cell_areas) / tot_vol)
                        
                        file_u = pv.read("../results/simulations/coupled"+str(coupling)+str(stim)+"_u"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+'_C1='+str(C1)+"_"+str(bigE)+"E000199.vtu")
                        diverg = file_u.compute_derivative(gradient=False, divergence=True)
                        cell_divu = diverg.point_data_to_cell_data()
                        divu_values = cell_divu.get_array('divergence')

                        sized = file_u.compute_cell_sizes(length=False, area=False)
                        cell_volumes = np.abs(sized.cell_data["Volume"])
                        tot_vol = np.sum(cell_volumes)
                        u_div_l2[order[j],i] = np.sum(divu_values*cell_volumes) / tot_vol
                        #print('mean div(u)', np.sum(divu_values*cell_volumes) / tot_vol)
                        
                        file_ec = pv.read("../results/simulations/coupled"+str(coupling)+str(stim)+"_ec"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+'_C1='+str(C1)+"_"+str(bigE)+"E000199.vtu")
                        cell_ec = file_ec.point_data_to_cell_data()
                        ec_values = cell_ec.get_array(cell_ec.array_names[0])

                        sized = file_ec.compute_cell_sizes(length=False, area=False)
                        cell_volumes = np.abs(sized.cell_data["Volume"])
                        tot_vol = np.sum(cell_volumes)
                        Ec[order[j],i] = np.sum(ec_values*cell_volumes) / tot_vol
                        #print('mean ec', np.sum(ec_values*cell_volumes) / tot_vol)
                        
                        if bcs == '_neumann':
                            variables = np.load('../results/temp/tempstats'+str(coupling)+str(stim)+'-oth-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+'_C1='+str(C1)+'_'+str(bigE)+'E.npy')
                            tempstats = np.load('../results/temp/tempstats'+str(coupling)+str(stim)+'-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+'_C1='+str(C1)+'_'+str(bigE)+'E.npy')
                            tempstats_mm = np.load('../results/temp/tempstats'+str(coupling)+str(stim)+'-minmax-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+'_C1='+str(C1)+'_'+str(bigE)+'E.npy')
                        else:
                            variables = np.load('../results/temp/tempstats'+str(coupling)+str(bcs)+str(stim)+'-oth-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+'_C1='+str(C1)+'_'+str(bigE)+'E.npy')
                            tempstats = np.load('../results/temp/tempstats'+str(coupling)+str(bcs)+str(stim)+'-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+'_C1='+str(C1)+'_'+str(bigE)+'E.npy')
                            tempstats_mm = np.load('../results/temp/tempstats'+str(coupling)+str(bcs)+str(stim)+'-minmax-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+'_C1='+str(C1)+'_'+str(bigE)+'E.npy')

                        Ec_min[order[j],i] = variables[3,-1]
                        Ec_max[order[j],i] = variables[4,-1]

                        u_div_min[order[j],i] = variables[7,-1]
                        u_div_max[order[j],i] = variables[8,-1]

                        ca_min[order[j],i] = tempstats_mm[0,-1]
                        ca_max[order[j],i] = tempstats_mm[1,-1]

                        pa_min[order[j],i] = tempstats_mm[4,-1]*10**(-5)
                        pa_max[order[j],i] = tempstats_mm[5,-1]*10**(-5)

                        i += 1
                    
                    j += 1

            #creat error bars from mean and min/max values
            Ec_err = np.array([Ec - Ec_min, Ec_max - Ec])    
            u_div_err = np.array([u_div_l2 - u_div_min, u_div_max - u_div_l2])
            ca_err = np.array([ca - ca_min, ca_max - ca])
            pa_err = np.array([pa - pa_min, pa_max - pa])

            # plot everything
            axs[0,k].set_xscale('log')
            axs[0,k].set_ylim(0, 2.5)
            axs[0,k].set_xlabel('$E$ (kPa)', fontsize=axes_labels)
            axs[0,k].set_ylabel('$f(\phi_a)$ (kPa)', fontsize=axes_labels)
            axs[0,k].grid(alpha=0.5, which='both', linestyle=':')
            axs[0,k].errorbar(Es, Ec[0,:], yerr=Ec_err[:,0,:], marker='o', markerfacecolor=myblue, markeredgecolor=myblue, linestyle='-', color=myblue, capsize=10, label='$C_1=0$')
            axs[0,k].errorbar(Es, Ec[1,:], yerr=Ec_err[:,0,:], marker='o', markerfacecolor=myred, markeredgecolor=myred, linestyle='-', color=myred, capsize=10, label='$C_1=0.05$')
            axs[0,k].errorbar(Es, Ec[2,:], yerr=Ec_err[:,0,:], marker='o', markerfacecolor=mygreen, markeredgecolor=mygreen, linestyle='-', color=mygreen, capsize=10, label='$C_1=0.1$')
            axs[0,k].errorbar(Es, Ec[3,:], yerr=Ec_err[:,0,:], marker='o', markerfacecolor=myyellow, markeredgecolor=myyellow, linestyle='-', color=myyellow, capsize=10, label='$C_1=0.2$')

            axs[0,k+1].set_xscale('log')
            axs[0,k+1].set_ylim(0, 2.5)
            axs[0,k+1].set_xlabel('$E$ (kPa)', fontsize=axes_labels)
            axs[0,k+1].set_ylabel('$f(\phi_a)$ (kPa)', fontsize=axes_labels)
            axs[0,k+1].grid(alpha=0.5, which='both', linestyle=':')
            axs[0,k+1].errorbar(Es, Ec[4,:], yerr=Ec_err[:,0,:], marker='o', markerfacecolor=myblue, markeredgecolor=myblue, linestyle='-', color=myblue, capsize=10, label='$C_1=0$')
            axs[0,k+1].errorbar(Es, Ec[5,:], yerr=Ec_err[:,0,:], marker='o', markerfacecolor=myred, markeredgecolor=myred, linestyle='-', color=myred, capsize=10, label='$C_1=0.05$')
            axs[0,k+1].errorbar(Es, Ec[6,:], yerr=Ec_err[:,0,:], marker='o', markerfacecolor=mygreen, markeredgecolor=mygreen, linestyle='-', color=mygreen, capsize=10, label='$C_1=0.1$')
            axs[0,k+1].errorbar(Es, Ec[7,:], yerr=Ec_err[:,0,:], marker='o', markerfacecolor=myyellow, markeredgecolor=myyellow, linestyle='-', color=myyellow, capsize=10, label='$C_1=0.2$')

            axs[1,k].set_xscale('log')
            axs[1,k].set_ylim(-0.2,2.3)
            axs[1,k].set_xlabel('$E$ (kPa)', fontsize=axes_labels)
            axs[1,k].set_ylabel('div($u$)', fontsize=axes_labels)
            axs[1,k].grid(alpha=0.5, which='both', linestyle=':')
            axs[1,k].errorbar(Es, u_div_l2[0,:], yerr=u_div_err[:,0,:], marker='o', markerfacecolor=myblue, markeredgecolor=myblue, linestyle='-', color=myblue, capsize=10, label='$C_1=0$')
            axs[1,k].errorbar(Es, u_div_l2[1,:], yerr=u_div_err[:,0,:], marker='o', markerfacecolor=myred, markeredgecolor=myred, linestyle='-', color=myred, capsize=10, label='$C_1=0.05$')
            axs[1,k].errorbar(Es, u_div_l2[2,:], yerr=u_div_err[:,0,:], marker='o', markerfacecolor=mygreen, markeredgecolor=mygreen, linestyle='-', color=mygreen, capsize=10, label='$C_1=0.1$')
            axs[1,k].errorbar(Es, u_div_l2[3,:], yerr=u_div_err[:,0,:], marker='o', markerfacecolor=myyellow, markeredgecolor=myyellow, linestyle='-', color=myyellow, capsize=10, label='$C_1=0.2$')

            axs[1,k+1].set_xscale('log')
            axs[1,k+1].set_ylim(-0.2,2.3)
            axs[1,k+1].set_xlabel('$E$ (kPa)', fontsize=axes_labels)
            axs[1,k+1].set_ylabel('div($u$)', fontsize=axes_labels)
            axs[1,k+1].grid(alpha=0.5, which='both', linestyle=':')
            axs[1,k+1].errorbar(Es, u_div_l2[4,:], yerr=u_div_err[:,0,:], marker='o', markerfacecolor=myblue, markeredgecolor=myblue, linestyle='-', color=myblue, capsize=10, label='$C_1=0$')
            axs[1,k+1].errorbar(Es, u_div_l2[5,:], yerr=u_div_err[:,0,:], marker='o', markerfacecolor=myred, markeredgecolor=myred, linestyle='-', color=myred, capsize=10, label='$C_1=0.05$')
            axs[1,k+1].errorbar(Es, u_div_l2[6,:], yerr=u_div_err[:,0,:], marker='o', markerfacecolor=mygreen, markeredgecolor=mygreen, linestyle='-', color=mygreen, capsize=10, label='$C_1=0.1$')
            axs[1,k+1].errorbar(Es, u_div_l2[7,:], yerr=u_div_err[:,0,:], marker='o', markerfacecolor=myyellow, markeredgecolor=myyellow, linestyle='-', color=myyellow, capsize=10, label='$C_1=0.2$')

            axs[2,k].set_xscale('log')
            axs[2,k].set_ylim(0.2,1.2)
            axs[2,k].set_xlabel('$E$ (kPa)', fontsize=axes_labels)
            axs[2,k].set_ylabel('$\phi_a$ ($\mu$mol/dm$^3$)', fontsize=axes_labels)
            axs[2,k].grid(alpha=0.5, which='both', linestyle=':')
            axs[2,k].errorbar(Es, ca[0,:], yerr=ca_err[:,0,:], marker='o', markerfacecolor=myblue, markeredgecolor=myblue, linestyle='-', color=myblue, capsize=10, label='$C_1=0$')
            axs[2,k].errorbar(Es, ca[1,:], yerr=ca_err[:,0,:], marker='o', markerfacecolor=myred, markeredgecolor=myred, linestyle='-', color=myred, capsize=10, label='$C_1=0.05$')
            axs[2,k].errorbar(Es, ca[2,:], yerr=ca_err[:,0,:], marker='o', markerfacecolor=mygreen, markeredgecolor=mygreen, linestyle='-', color=mygreen, capsize=10, label='$C_1=0.1$')
            axs[2,k].errorbar(Es, ca[3,:], yerr=ca_err[:,0,:], marker='o', markerfacecolor=myyellow, markeredgecolor=myyellow, linestyle='-', color=myyellow, capsize=10, label='$C_1=0.2$')
            
            axs[2,k+1].set_xscale('log')
            axs[2,k+1].set_ylim(0.2,1.2)
            axs[2,k+1].set_xlabel('$E$ (kPa)', fontsize=axes_labels)
            axs[2,k+1].set_ylabel('$\phi_a$ ($\mu$mol/dm$^3$)', fontsize=axes_labels)
            axs[2,k+1].grid(alpha=0.5, which='both', linestyle=':')
            axs[2,k+1].errorbar(Es, ca[4,:], yerr=ca_err[:,0,:], marker='o', markerfacecolor=myblue, markeredgecolor=myblue, linestyle='-', color=myblue, capsize=10, label='$C_1=0$')
            axs[2,k+1].errorbar(Es, ca[5,:], yerr=ca_err[:,0,:], marker='o', markerfacecolor=myred, markeredgecolor=myred, linestyle='-', color=myred, capsize=10, label='$C_1=0.05$')
            axs[2,k+1].errorbar(Es, ca[6,:], yerr=ca_err[:,0,:], marker='o', markerfacecolor=mygreen, markeredgecolor=mygreen, linestyle='-', color=mygreen, capsize=10, label='$C_1=0.1$')
            axs[2,k+1].errorbar(Es, ca[7,:], yerr=ca_err[:,0,:], marker='o', markerfacecolor=myyellow, markeredgecolor=myyellow, linestyle='-', color=myyellow, capsize=10, label='$C_1=0.2$')
       
            axs[3,k].set_xscale('log')
            axs[3,k].set_ylim(0, 1*10**(-5))
            axs[3,k].set_xlabel('$E$ (kPa)', fontsize=axes_labels)
            axs[3,k].set_ylabel(r'$\rho_a$ ($\mu$mol/dm$^2$)', fontsize=axes_labels)
            axs[3,k].grid(alpha=0.5, which='both', linestyle=':')
            axs[3,k].errorbar(Es, pa[0,:], yerr=pa_err[:,0,:], marker='o', markerfacecolor=myblue, markeredgecolor=myblue, linestyle='-', color=myblue, capsize=10, label='$C_1=0$')
            axs[3,k].errorbar(Es, pa[1,:], yerr=pa_err[:,0,:], marker='o', markerfacecolor=myred, markeredgecolor=myred, linestyle='-', color=myred, capsize=10, label='$C_1=0.05$')
            axs[3,k].errorbar(Es, pa[2,:], yerr=pa_err[:,0,:], marker='o', markerfacecolor=mygreen, markeredgecolor=mygreen, linestyle='-', color=mygreen, capsize=10, label='$C_1=0.1$')
            axs[3,k].errorbar(Es, pa[3,:], yerr=pa_err[:,0,:], marker='o', markerfacecolor=myyellow, markeredgecolor=myyellow, linestyle='-', color=myyellow, capsize=10, label='$C_1=0.2$')
          
            axs[3,k+1].set_xscale('log')
            axs[3,k+1].set_ylim(0, 1*10**(-5))
            axs[3,k+1].set_xlabel('$E$ (kPa)', fontsize=axes_labels)
            axs[3,k+1].set_ylabel(r'$\rho_a$ ($\mu$mol/dm$^2$)', fontsize=axes_labels)
            axs[3,k+1].grid(alpha=0.5, which='both', linestyle=':')
            axs[3,k+1].errorbar(Es, pa[4,:], yerr=pa_err[:,0,:], marker='o', markerfacecolor=myblue, markeredgecolor=myblue, linestyle='-', color=myblue, capsize=10, label='$C_1=0$')
            axs[3,k+1].errorbar(Es, pa[5,:], yerr=pa_err[:,0,:], marker='o', markerfacecolor=myred, markeredgecolor=myred, linestyle='-', color=myred, capsize=10, label='$C_1=0.05$')
            axs[3,k+1].errorbar(Es, pa[6,:], yerr=pa_err[:,0,:], marker='o', markerfacecolor=mygreen, markeredgecolor=mygreen, linestyle='-', color=mygreen, capsize=10, label='$C_1=0.1$')
            axs[3,k+1].errorbar(Es, pa[7,:], yerr=pa_err[:,0,:], marker='o', markerfacecolor=myyellow, markeredgecolor=myyellow, linestyle='-', color=myyellow, capsize=10, label='$C_1=0.2$')

            k = 2

        for ax in fig.get_axes():
            ax.label_outer()
        
        #create legend with preferred layout
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, -0.05), loc="lower center", ncol = len(ax.lines))
        
        plt.savefig('../results/figures/AllGraphsMean'+str(bcs)+str(stim)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+'.png',bbox_inches='tight')
        #plt.show()
        plt.close()
