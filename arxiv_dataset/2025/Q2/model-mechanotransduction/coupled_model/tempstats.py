import matplotlib.pyplot as plt
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

# defines text size
axes_labels = 25
axes_ticks = 20
legend_labels = 20
rcParams['font.size'] = 20

#define parameters
dt = 0.05
T = 10
k6 = 1
Es = np.array([0.1, 5.7])
time_steps = np.arange(200)
time = np.arange(0,T*10,dt*10)
time_string = [str(x).zfill(3) for x in time_steps]
n = len(time)
order = np.array([0, 2, 4, 6, 1, 5, 3, 7])
# following choices gives Figure 25, but other choices can be made by creating a for loop
bcs = '_neumann' #['_partfixed','_neumann']
stim = '' #['_2D','']
meshnr = 1 #[0,1]

# create Figure 25 from data saved during simulations
fig, axs = plt.subplots(4,4, sharex=True, sharey='row', figsize=(24,16), layout='constrained')
k = 0

# loop over different substrate stiffnesses E
for bigE in Es:
    # create empty numpy arrays for the values to be plotted
    Ec = np.zeros((8,n))
    u_div_l2 = np.zeros((8,n))
    ca = np.zeros((8,n))
    pa = np.zeros((8,n))

    j = 0

    # loop over different couplings
    for coupling in [1,2,3,4]:
        i = 0
        # loop over different time steps
        for t_str in time_string:
            #print('STATE:', bcs, stim, meshnr, coupling, bigE)

            # find the mean of certain values from simulation results
            file_ca = pv.read("../results/simulations/coupled"+str(coupling)+str(stim)+"_ca"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_"+str(bigE)+"E000"+t_str+".vtu")
            cell_ca = file_ca.point_data_to_cell_data()
            ca_values = cell_ca.get_array(cell_ca.array_names[0])

            sized = file_ca.compute_cell_sizes(length=False, area=False)
            cell_volumes = np.abs(sized.cell_data["Volume"])
            tot_vol = np.sum(cell_volumes)
            ca[order[j],i] = np.sum(ca_values*cell_volumes) / tot_vol
            #print('mean ca', np.sum(ca_values*cell_volumes) / tot_vol)
            
            file_pa = pv.read("../results/simulations/coupled"+str(coupling)+str(stim)+"_p"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_"+str(bigE)+"E000"+t_str+".vtu")
            cell_pa = file_pa.point_data_to_cell_data()
            pa_values = cell_pa.get_array(cell_pa.array_names[0])

            sized = file_pa.compute_cell_sizes(length=False, volume=False)
            cell_areas = np.abs(sized.cell_data["Area"])
            tot_vol = np.sum(cell_areas)
            pa[order[j],i] = (np.sum(pa_values*cell_areas) / tot_vol)*10**(-5)
            #print('mean pa', np.sum(pa_values*cell_areas) / tot_vol)
            
            file_u = pv.read("../results/simulations/coupled"+str(coupling)+str(stim)+"_u"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_"+str(bigE)+"E000"+t_str+".vtu")
            diverg = file_u.compute_derivative(gradient=False, divergence=True)
            cell_divu = diverg.point_data_to_cell_data()
            divu_values = cell_divu.get_array('divergence')

            sized = file_u.compute_cell_sizes(length=False, area=False)
            cell_volumes = np.abs(sized.cell_data["Volume"])
            tot_vol = np.sum(cell_volumes)
            u_div_l2[order[j],i] = np.sum(divu_values*cell_volumes) / tot_vol
            #print('mean div(u)', np.sum(divu_values*cell_volumes) / tot_vol)
            
            file_ec = pv.read("../results/simulations/coupled"+str(coupling)+str(stim)+"_ec"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_"+str(bigE)+"E000"+t_str+".vtu")
            cell_ec = file_ec.point_data_to_cell_data()
            ec_values = cell_ec.get_array(cell_ec.array_names[0])

            sized = file_ec.compute_cell_sizes(length=False, area=False)
            cell_volumes = np.abs(sized.cell_data["Volume"])
            tot_vol = np.sum(cell_volumes)
            Ec[order[j],i] = np.sum(ec_values*cell_volumes) / tot_vol
            #print('mean ec', np.sum(ec_values*cell_volumes) / tot_vol)
            
            i += 1
        
        j += 1

    for C1 in [0.5,2.0]:
        for coupling in [2,4]:
            i = 0
            for t_str in time_string:
                #print('STATE:', bcs, stim, meshnr, coupling, bigE)

                # find the mean of certain values from simulation results
                file_ca = pv.read("../results/simulations/coupled"+str(coupling)+str(stim)+"_ca"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+'_C1='+str(C1)+"_"+str(bigE)+"E000"+t_str+".vtu")
                cell_ca = file_ca.point_data_to_cell_data()
                ca_values = cell_ca.get_array(cell_ca.array_names[0])

                sized = file_ca.compute_cell_sizes(length=False, area=False)
                cell_volumes = np.abs(sized.cell_data["Volume"])
                tot_vol = np.sum(cell_volumes)
                ca[order[j],i] = np.sum(ca_values*cell_volumes) / tot_vol
                #print('mean ca', np.sum(ca_values*cell_volumes) / tot_vol)
                
                file_pa = pv.read("../results/simulations/coupled"+str(coupling)+str(stim)+"_p"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+'_C1='+str(C1)+"_"+str(bigE)+"E000"+t_str+".vtu")
                cell_pa = file_pa.point_data_to_cell_data()
                pa_values = cell_pa.get_array(cell_pa.array_names[0])

                sized = file_pa.compute_cell_sizes(length=False, volume=False)
                cell_areas = np.abs(sized.cell_data["Area"])
                tot_vol = np.sum(cell_areas)
                pa[order[j],i] = (np.sum(pa_values*cell_areas) / tot_vol)*10**(-5)
                #print('mean pa', np.sum(pa_values*cell_areas) / tot_vol)
                
                file_u = pv.read("../results/simulations/coupled"+str(coupling)+str(stim)+"_u"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+'_C1='+str(C1)+"_"+str(bigE)+"E000"+t_str+".vtu")
                diverg = file_u.compute_derivative(gradient=False, divergence=True)
                cell_divu = diverg.point_data_to_cell_data()
                divu_values = cell_divu.get_array('divergence')

                sized = file_u.compute_cell_sizes(length=False, area=False)
                cell_volumes = np.abs(sized.cell_data["Volume"])
                tot_vol = np.sum(cell_volumes)
                u_div_l2[order[j],i] = np.sum(divu_values*cell_volumes) / tot_vol
                #print('mean div(u)', np.sum(divu_values*cell_volumes) / tot_vol)
                
                file_ec = pv.read("../results/simulations/coupled"+str(coupling)+str(stim)+"_ec"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+'_C1='+str(C1)+"_"+str(bigE)+"E000"+t_str+".vtu")
                cell_ec = file_ec.point_data_to_cell_data()
                ec_values = cell_ec.get_array(cell_ec.array_names[0])

                sized = file_ec.compute_cell_sizes(length=False, area=False)
                cell_volumes = np.abs(sized.cell_data["Volume"])
                tot_vol = np.sum(cell_volumes)
                Ec[order[j],i] = np.sum(ec_values*cell_volumes) / tot_vol
                #print('mean ec', np.sum(ec_values*cell_volumes) / tot_vol)

                i += 1
            
            j += 1

    # plot everything
    axs[0,k].set_ylim(0, 2)
    axs[0,k].set_xlabel('time (s)', fontsize=axes_labels)
    axs[0,k].set_ylabel('$f(\phi_a)$ (kPa)', fontsize=axes_labels)
    axs[0,k].grid(alpha=0.5, which='both', linestyle=':')
    axs[0,k].plot(time, Ec[0,:], linestyle='-', color=myblue, label='$C_1=0$')
    axs[0,k].plot(time, Ec[1,:], linestyle='-', color=myred, label='$C_1=0.05$')
    axs[0,k].plot(time, Ec[2,:], linestyle='-', color=mygreen, label='$C_1=0.1$')
    axs[0,k].plot(time, Ec[3,:], linestyle='-', color=myyellow, label='$C_1=0.2$')

    axs[0,k+1].set_ylim(0, 2)
    axs[0,k+1].set_xlabel('time (s)', fontsize=axes_labels)
    axs[0,k+1].set_ylabel('$f(\phi_a)$ (kPa)', fontsize=axes_labels)
    axs[0,k+1].grid(alpha=0.5, which='both', linestyle=':')
    axs[0,k+1].plot(time, Ec[4,:], linestyle='-', color=myblue, label='$C_1=0$')
    axs[0,k+1].plot(time, Ec[5,:], linestyle='-', color=myred, label='$C_1=0.05$')
    axs[0,k+1].plot(time, Ec[6,:], linestyle='-', color=mygreen, label='$C_1=0.1$')
    axs[0,k+1].plot(time, Ec[7,:], linestyle='-', color=myyellow, label='$C_1=0.2$')

    axs[1,k].set_ylim(-0.2,2)
    axs[1,k].set_xlabel('time (s)', fontsize=axes_labels)
    axs[1,k].set_ylabel('div($u$)', fontsize=axes_labels)
    axs[1,k].grid(alpha=0.5, which='both', linestyle=':')
    axs[1,k].plot(time[1:], u_div_l2[0,1:], linestyle='-', color=myblue, label='$C_1=0$')
    axs[1,k].plot(time[1:], u_div_l2[1,1:], linestyle='-', color=myred, label='$C_1=0.05$')
    axs[1,k].plot(time[1:], u_div_l2[2,1:], linestyle='-', color=mygreen, label='$C_1=0.1$')
    axs[1,k].plot(time[1:], u_div_l2[3,1:], linestyle='-', color=myyellow, label='$C_1=0.2$')

    axs[1,k+1].set_ylim(-0.2,2)
    axs[1,k+1].set_xlabel('time (s)', fontsize=axes_labels)
    axs[1,k+1].set_ylabel('div($u$)', fontsize=axes_labels)
    axs[1,k+1].grid(alpha=0.5, which='both', linestyle=':')
    axs[1,k+1].plot(time[1:], u_div_l2[4,1:], linestyle='-', color=myblue, label='$C_1=0$')
    axs[1,k+1].plot(time[1:], u_div_l2[5,1:], linestyle='-', color=myred, label='$C_1=0.05$')
    axs[1,k+1].plot(time[1:], u_div_l2[6,1:], linestyle='-', color=mygreen, label='$C_1=0.1$')
    axs[1,k+1].plot(time[1:], u_div_l2[7,1:], linestyle='-', color=myyellow, label='$C_1=0.2$')

    axs[2,k].set_ylim(0.2,1)
    axs[2,k].set_xlabel('time (s)', fontsize=axes_labels)
    axs[2,k].set_ylabel('$\phi_a$ ($\mu$mol/dm$^3$)', fontsize=axes_labels)
    axs[2,k].grid(alpha=0.5, which='both', linestyle=':')
    axs[2,k].plot(time, ca[0,:], linestyle='-', color=myblue, label='$C_1=0$')
    axs[2,k].plot(time, ca[1,:], linestyle='-', color=myred, label='$C_1=0.05$')
    axs[2,k].plot(time, ca[2,:], linestyle='-', color=mygreen, label='$C_1=0.1$')
    axs[2,k].plot(time, ca[3,:], linestyle='-', color=myyellow, label='$C_1=0.2$')
    
    axs[2,k+1].set_ylim(0.2,1)
    axs[2,k+1].set_xlabel('time (s)', fontsize=axes_labels)
    axs[2,k+1].set_ylabel('$\phi_a$ ($\mu$mol/dm$^3$)', fontsize=axes_labels)
    axs[2,k+1].grid(alpha=0.5, which='both', linestyle=':')
    axs[2,k+1].plot(time, ca[4,:], linestyle='-', color=myblue, label='$C_1=0$')
    axs[2,k+1].plot(time, ca[5,:], linestyle='-', color=myred, label='$C_1=0.05$')
    axs[2,k+1].plot(time, ca[6,:], linestyle='-', color=mygreen, label='$C_1=0.1$')
    axs[2,k+1].plot(time, ca[7,:], linestyle='-', color=myyellow, label='$C_1=0.2$')

    axs[3,k].set_ylim(0, 0.8*10**(-5))
    axs[3,k].set_xlabel('time (s)', fontsize=axes_labels)
    axs[3,k].set_ylabel(r'$\rho_a$ ($\mu$mol/dm$^2$)', fontsize=axes_labels)
    axs[3,k].grid(alpha=0.5, which='both', linestyle=':')
    axs[3,k].plot(time, pa[0,:], linestyle='-', color=myblue, label='$C_1=0$')
    axs[3,k].plot(time, pa[1,:], linestyle='-', color=myred, label='$C_1=0.05$')
    axs[3,k].plot(time, pa[2,:], linestyle='-', color=mygreen, label='$C_1=0.1$')
    axs[3,k].plot(time, pa[3,:], linestyle='-', color=myyellow, label='$C_1=0.2$')
    
    axs[3,k+1].set_ylim(0, 0.8*10**(-5))
    axs[3,k+1].set_xlabel('time (s)', fontsize=axes_labels)
    axs[3,k+1].set_ylabel(r'$\rho_a$ ($\mu$mol/dm$^2$)', fontsize=axes_labels)
    axs[3,k+1].grid(alpha=0.5, which='both', linestyle=':')
    axs[3,k+1].plot(time, pa[4,:], linestyle='-', color=myblue, label='$C_1=0$')
    axs[3,k+1].plot(time, pa[5,:], linestyle='-', color=myred, label='$C_1=0.05$')
    axs[3,k+1].plot(time, pa[6,:], linestyle='-', color=mygreen, label='$C_1=0.1$')
    axs[3,k+1].plot(time, pa[7,:], linestyle='-', color=myyellow, label='$C_1=0.2$')

    k = 2

for ax in fig.get_axes():
    ax.label_outer()

# create appropriate legend with nice layout
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, -0.05), loc="lower center", ncol = len(ax.lines))

plt.savefig('../results/figures/TimeGraphsMean'+str(bcs)+str(stim)+"_m"+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+'.png',bbox_inches='tight')
#plt.show()
plt.close()
