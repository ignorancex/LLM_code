import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import numpy as np
import pyvista as pv

# define colors
myblue = (0,119/235,187/235)
myred=(187/235,85/235,102/235)
myyellow=(221/235,170/235,51/235)
mygrey=(187/235,187/235,187/235)
mygreen="#66BB55"
mymagenta="#7733DD"
myblack="#48494B"

# define text sizes
axes_labels = 25
axes_ticks = 20
legend_labels = 20
rcParams['font.size'] = 20

# define parameters
dt = 0.05
T = 10
meshnr = 1
Es = np.array([0.001, 0.01, 0.1, 0.3, 1.0, 5.7, 10.0, 50.0, 100.0, 7000000.0])
n = len(Es)

# for loop for different diffusion coefficients
for D1 in [40, 100]:
        # define empty numpy arrays to store variables
        ca = np.zeros(n)
        ca_min = np.zeros(n)
        ca_max = np.zeros(n)
        pa = np.zeros(n)
        pa_min = np.zeros(n)
        pa_max = np.zeros(n)
        ca_2D = np.zeros(n)
        ca_min_2D = np.zeros(n)
        ca_max_2D = np.zeros(n)
        pa_2D = np.zeros(n)
        pa_min_2D = np.zeros(n)
        pa_max_2D = np.zeros(n)
        pa_2xD = np.zeros(n)
        pa_min_2xD = np.zeros(n)
        pa_max_2xD = np.zeros(n)

        # initialize counter
        i = 0

        # for loop for different values of substrate stiffness E
        for E in Es:
            # find mean of phi_a (ca) and rho_a (ca) from simulations results for both 3D, 2D and 2xD stimulus
            file_ca = pv.read("../results/simulations/rhomodel3D_ca_reduced_dt="+str(dt)+"_T="+str(T)+"_D="+str(D1)+"_"+str(E)+"E000199.vtu")
            cell_ca = file_ca.point_data_to_cell_data()
            ca_values = cell_ca.get_array(cell_ca.array_names[0])

            sized = file_ca.compute_cell_sizes(length=False, area=False)
            cell_volumes = np.abs(sized.cell_data["Volume"])
            tot_vol = np.sum(cell_volumes)
            ca[i] = np.sum(ca_values*cell_volumes) / tot_vol
            #print('mean ca', np.sum(ca_values*cell_volumes) / tot_vol)

            file_ca_2D = pv.read("../results/simulations/rhomodel2D_ca_reduced_dt="+str(dt)+"_T="+str(T)+"_D="+str(D1)+"_"+str(E)+"E000199.vtu")
            cell_ca_2D = file_ca_2D.point_data_to_cell_data()
            ca_2D_values = cell_ca_2D.get_array(cell_ca_2D.array_names[0])

            sized = file_ca_2D.compute_cell_sizes(length=False, area=False)
            cell_volumes = np.abs(sized.cell_data["Volume"])
            tot_vol = np.sum(cell_volumes)
            ca_2D[i] = np.sum(ca_2D_values*cell_volumes) / tot_vol
            #print('mean ca', np.sum(ca_values*cell_volumes) / tot_vol)
            
            file_pa = pv.read("../results/simulations/rhomodel3D_p_reduced_dt="+str(dt)+"_T="+str(T)+"_D="+str(D1)+"_"+str(E)+"E000199.vtu")
            cell_pa = file_pa.point_data_to_cell_data()
            pa_values = cell_pa.get_array(cell_pa.array_names[0])

            sized = file_pa.compute_cell_sizes(length=False, volume=False)
            cell_areas = np.abs(sized.cell_data["Area"])
            tot_vol = np.sum(cell_areas)
            pa[i] = (np.sum(pa_values*cell_areas) / tot_vol)*10**(-5)
            #print('mean pa', np.sum(pa_values*cell_areas) / tot_vol)

            file_pa_2D = pv.read("../results/simulations/rhomodel2D_p_reduced_dt="+str(dt)+"_T="+str(T)+"_D="+str(D1)+"_"+str(E)+"E000199.vtu")
            cell_pa_2D = file_pa_2D.point_data_to_cell_data()
            pa_2D_values = cell_pa_2D.get_array(cell_pa_2D.array_names[0])

            sized = file_pa_2D.compute_cell_sizes(length=False, volume=False)
            cell_areas = np.abs(sized.cell_data["Area"])
            tot_vol = np.sum(cell_areas)
            pa_2D[i] = (np.sum(pa_2D_values*cell_areas) / tot_vol)*10**(-15)
            #print('mean pa', np.sum(pa_values*cell_areas) / tot_vol)
            
            file_pa_2xD = pv.read("../results/simulations/rhomodel2xD_p_reduced_dt="+str(dt)+"_T="+str(T)+"_D="+str(D1)+"_"+str(E)+"E000199.vtu")
            cell_pa_2xD = file_pa_2xD.point_data_to_cell_data()
            pa_2xD_values = cell_pa_2xD.get_array(cell_pa_2xD.array_names[0])

            sized = file_pa_2xD.compute_cell_sizes(length=False, volume=False)
            cell_areas = np.abs(sized.cell_data["Area"])
            tot_vol = np.sum(cell_areas)
            pa_2xD[i] = (np.sum(pa_2xD_values*cell_areas) / tot_vol)*10**(-5)
            #print('mean pa', np.sum(pa_values*cell_areas) / tot_vol)

            # get min and max values of phi_a (ca) and rho_a (ca) from saved arrays druing simulations for both 3D, 2D and 2xD stimulus
            tempstats = np.load('../results/temp/tempstats0_3D-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_D='+str(D1)+'_'+str(E)+'E.npy')
            tempstats_mm = np.load('../results/temp/tempstats0_3D-minmax-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_D='+str(D1)+'_'+str(E)+'E.npy')
            tempstats_2D = np.load('../results/temp/tempstats0_2D-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_D='+str(D1)+'_'+str(E)+'E.npy')
            tempstats_mm_2D = np.load('../results/temp/tempstats0_2D-minmax-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_D='+str(D1)+'_'+str(E)+'E.npy')
            tempstats_2xD = np.load('../results/temp/tempstats0_2xD-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_D='+str(D1)+'_'+str(E)+'E.npy')
            tempstats_mm_2xD = np.load('../results/temp/tempstats0_2xD-minmax-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_D='+str(D1)+'_'+str(E)+'E.npy')

            ca_min[i] = tempstats_mm[0,-1]
            ca_max[i] = tempstats_mm[1,-1]
            ca_min_2D[i] = tempstats_mm_2D[0,-1]
            ca_max_2D[i] = tempstats_mm_2D[1,-1]

            pa_min[i] = tempstats_mm[4,-1]*10**(-5)
            pa_max[i] = tempstats_mm[5,-1]*10**(-5)
            pa_min_2D[i] = tempstats_mm_2D[4,-1]*10**(-5)
            pa_max_2D[i] = tempstats_mm_2D[5,-1]*10**(-5)
            pa_min_2xD[i] = tempstats_mm_2xD[4,-1]*10**(-5)
            pa_max_2xD[i] = tempstats_mm_2xD[5,-1]*10**(-5)
            
            i += 1

        # create error bars from min and max value compared to the mean
        ca_err = np.array([ca - ca_min, ca_max - ca])
        pa_err = np.array([pa - pa_min, pa_max - pa])
        ca_err_2D = np.array([ca_2D - ca_min_2D, ca_max_2D - ca_2D])
        pa_err_2D = np.array([pa_2D - pa_min_2D, pa_max_2D - pa_2D])
        pa_err_2xD = np.array([pa_2xD - pa_min_2xD, pa_max_2xD - pa_2xD])

        # plot left graph (phi_a vs E) in Figure 22 and 24
        fig = plt.figure(figsize=(12,8))
        plt.xscale('log')
        plt.xlabel('E (kPa)', fontsize=axes_labels)
        plt.ylabel('$\phi_a$ ($\mu$mol/dm$^3$)', fontsize=axes_labels)
        plt.ylim(0.2,1.2)
        plt.xticks(fontsize=axes_ticks)
        plt.yticks(fontsize=axes_ticks)
        plt.grid(alpha=0.3, which='both', linestyle=':')
        plt.errorbar(Es, ca, yerr=ca_err, marker='o', markerfacecolor=myblue, markeredgecolor=myblue, linestyle='-', color=myblue, capsize=10, label='3D stimulus')
        plt.errorbar(Es, ca_2D, yerr=ca_err_2D, marker='o', markerfacecolor=myred, markeredgecolor=myred, linestyle='-', color=myred, capsize=10, label='2xD/2D stimulus')
        plt.legend(loc="upper left", fontsize = 'large')
        plt.savefig('../results/figures/PhiTotvsEs0_allD-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_D='+str(D1)+'.png')
        #plt.show()
        plt.close()

        # plot right graph (rho_a vs E) in Figure 22 and 24
        fig = plt.figure(figsize=(12,8))
        plt.xscale('log')
        plt.xlabel('E (kPa)', fontsize=axes_labels)
        plt.ylabel(r'$\rho_a$ ($\mu$mol/dm$^2$)', fontsize=axes_labels)
        plt.ylim(0, 0.9*10**(-5))
        plt.xticks(fontsize=axes_ticks)
        plt.yticks(fontsize=axes_ticks)
        plt.grid(alpha=0.3, which='both', linestyle=':')
        plt.errorbar(Es, pa, yerr=pa_err, marker='o', markerfacecolor=myblue, markeredgecolor=myblue, linestyle='-', color=myblue, capsize=10, label='3D stimulus')
        plt.errorbar(Es, pa_2xD, yerr=pa_err_2xD, marker='o', markerfacecolor=myred, markeredgecolor=myred, linestyle='-', color=myred, capsize=10, label='2xD stimulus')
        plt.errorbar(Es, pa_2D, yerr=pa_err_2D, marker='o', markerfacecolor=mygreen, markeredgecolor=mygreen, linestyle='-', color=mygreen, capsize=10, label='2D stimulus')
        plt.legend(loc="upper left", fontsize = 'large')
        plt.savefig('../results/figures/RhoAvsEs0_allD-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_D='+str(D1)+'.png')
        #plt.show()
        plt.close()
        
