import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pyvista as pv

myblue = (0,119/235,187/235)
myred=(187/235,85/235,102/235)
myyellow=(221/235,170/235,51/235)
mygrey=(187/235,187/235,187/235)
mygreen="#66BB55"
mymagenta="#7733DD"
myblack="#48494B"

axes_labels = 25
axes_ticks = 20
legend_labels = 20
rcParams['font.size'] = 20

def compute_ca_from_file(file):
    file_ca = pv.read(file)
    cell_ca = file_ca.point_data_to_cell_data()
    ca_values = cell_ca.get_array(cell_ca.array_names[0])

    sized = file_ca.compute_cell_sizes(length=False, area=False)
    cell_volumes = np.abs(sized.cell_data["Volume"])
    tot_vol = np.sum(cell_volumes)
    return np.sum(ca_values*cell_volumes) / tot_vol

def compute_pa_from_file(file):
    file_pa = pv.read(file)
    cell_pa = file_pa.point_data_to_cell_data()
    pa_values = cell_pa.get_array(cell_pa.array_names[0])

    sized = file_pa.compute_cell_sizes(length=False, volume=False)
    cell_areas = np.abs(sized.cell_data["Area"])
    tot_vol = np.sum(cell_areas)
    return (np.sum(pa_values*cell_areas) / tot_vol)

def compute_u_from_file(file):
    file_u = pv.read(file)
    diverg = file_u.compute_derivative(gradient=False, divergence=True)
    cell_divu = diverg.point_data_to_cell_data()
    divu_values = cell_divu.get_array('divergence')

    sized = file_u.compute_cell_sizes(length=False, area=False)
    cell_volumes = np.abs(sized.cell_data["Volume"])
    tot_vol = np.sum(cell_volumes)
    return np.sum(divu_values*cell_volumes) / tot_vol

def compute_ec_from_file(file):
    file_ec = pv.read(file)
    cell_ec = file_ec.point_data_to_cell_data()
    ec_values = cell_ec.get_array(cell_ec.array_names[0])

    sized = file_ec.compute_cell_sizes(length=False, area=False)
    cell_volumes = np.abs(sized.cell_data["Volume"])
    tot_vol = np.sum(cell_volumes)
    return np.sum(ec_values*cell_volumes) / tot_vol

k6 = 1
coupling = 4
meshnr = 1
dt = 0.05
T = 10
stim = ''
bcs = '_neumann'

fig, axs = plt.subplots(4,3, sharex=True, sharey='row', figsize=(16,20), layout='constrained')
k = 0

for bigE in [0.1, 5.7, 7000000.0]:

    Ec = np.zeros((6,4))
    u_div_l2 = np.zeros((6,4))
    ca = np.zeros((6,4))
    pa = np.zeros((6,4))        


    ca_or = compute_ca_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_ca"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_"+str(bigE)+"E000199.vtu")
    pa_or = compute_pa_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_p"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_"+str(bigE)+"E000199.vtu")
    u_div_l2_or = compute_u_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_u"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_"+str(bigE)+"E000199.vtu")
    Ec_or = compute_ec_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_ec"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_"+str(bigE)+"E000199.vtu")

    C1 = 1
    k6 = 1
    nu_ = 0.3
    k7 = 0.2
    p = 2.6
    k8 = 10**(1/2.6)

    i = 0
    for C1 in [0.8*C1, 0.9*C1, 1.1*C1, 1.2*C1]:  
        ca[0,i] = compute_ca_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_ca"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")
        pa[0,i] = compute_pa_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_p"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")
        u_div_l2[0,i] = compute_u_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_u"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")
        Ec[0,i] = compute_ec_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_ec"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")

        i += 1

    C1 = 1
    i = 0
    for k6 in [0.8*k6, 0.9*k6, 1.1*k6, 1.2*k6]: 
        ca[1,i] = compute_ca_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_ca"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")
        pa[1,i] = compute_pa_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_p"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")
        u_div_l2[1,i] = compute_u_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_u"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")
        Ec[1,i] = compute_ec_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_ec"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")

        i += 1

    k6 = 1
    i = 0
    for k7 in [0.8*k7, 0.9*k7, 1.1*k7, 1.2*k7]: 
        ca[2,i] = compute_ca_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_ca"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")
        pa[2,i] = compute_pa_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_p"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")
        u_div_l2[2,i] = compute_u_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_u"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")
        Ec[2,i] = compute_ec_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_ec"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")

        i += 1

    k7 = 0.2
    i = 0
    for k8 in [0.8*k8, 0.9*k8, 1.1*k8, 1.2*k8]: 
        ca[3,i] = compute_ca_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_ca"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")
        pa[3,i] = compute_pa_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_p"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")
        u_div_l2[3,i] = compute_u_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_u"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")
        Ec[3,i] = compute_ec_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_ec"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")

        i += 1

    k8 = 10**(1/2.6)
    i = 0
    for p in [0.8*p, 0.9*p, 1.1*p, 1.2*p]: 
        ca[4,i] = compute_ca_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_ca"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")
        pa[4,i] = compute_pa_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_p"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")
        u_div_l2[4,i] = compute_u_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_u"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")
        Ec[4,i] = compute_ec_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_ec"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")

        i += 1

    p = 2.6
    i = 0
    for nu_ in [0.8*nu_, 0.9*nu_, 1.1*nu_, 1.2*nu_]: 
        ca[5,i] = compute_ca_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_ca"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")
        pa[5,i] = compute_pa_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_p"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")
        u_div_l2[5,i] = compute_u_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_u"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")
        Ec[5,i] = compute_ec_from_file("../results/simulations/coupled"+str(coupling)+str(stim)+"_ec"+str(bcs)+"_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E000199.vtu")

        i += 1



    parameters = ['$C_1$', '$k_6$', '$k_7$', '$k_8$', 'p', r'$\nu_c$']
    x = np.arange(len(parameters))  # the label locations
    width = 0.2  # the width of the bars
    labels = ['-20%', '-10%', '+10%', '+20%']
    colors = [myyellow, mygreen, myred, myblue]

    for multiplier in range(4):
        offset = width * multiplier
        rects = axs[0,k].bar(x + offset - 0.5*width, (ca[:,multiplier]-ca_or)/ca_or*100, width, color=colors[multiplier], label=labels[multiplier])

    axs[0,k].axhline(0, color=myblack)
    axs[0,k].grid(alpha=0.5, which='both', linestyle=':')
    axs[0,k].set_ylim(-15, 15)
    axs[0,k].set_xlim(-1.5*width, 5+3.5*width)
    axs[0,k].set_ylabel(r'$\phi_a^{diff}$ (%)', fontsize=axes_labels)
    for i in x[:-1]:
        axs[0,k].axvline(i+offset+0.5*width, ls='--', color=mygrey)

    for multiplier in range(4):
        offset = width * multiplier
        rects = axs[1,k].bar(x + offset - 0.5*width, (pa[:,multiplier]-pa_or)/pa_or*100, width, color=colors[multiplier], label=labels[multiplier])

    axs[1,k].axhline(0, color=myblack)
    axs[1,k].grid(alpha=0.5, which='both', linestyle=':')
    axs[1,k].set_ylim(-15, 15)
    axs[0,k].set_xlim(-1.5*width, 5+3.5*width)
    axs[1,k].set_ylabel(r'$\rho_a^{diff}$ (%)', fontsize=axes_labels)
    for i in x[:-1]:
        axs[1,k].axvline(i+offset+0.5*width, ls='--', color=mygrey)
    
    for multiplier in range(4):
        offset = width * multiplier
        rects = axs[2,k].bar(x + offset - 0.5*width, (u_div_l2[:,multiplier]-u_div_l2_or)/u_div_l2_or*100, width, color=colors[multiplier], label=labels[multiplier])

    axs[2,k].axhline(0, color=myblack)
    axs[2,k].grid(alpha=0.5, which='both', linestyle=':')
    axs[2,k].set_ylim(-40, 70)
    axs[0,k].set_xlim(-1.5*width, 5+3.5*width)
    axs[2,k].set_ylabel(r'div$(u)^{diff}$ (%)', fontsize=axes_labels)
    for i in x[:-1]:
        axs[2,k].axvline(i+offset+0.5*width, ls='--', color=mygrey)
    
    for multiplier in range(4):
        offset = width * multiplier
        rects = axs[3,k].bar(x + offset - 0.5*width, (Ec[:,multiplier]-Ec_or)/Ec_or*100, width, color=colors[multiplier], label=labels[multiplier])

    axs[3,k].axhline(0, color=myblack)
    axs[3,k].grid(alpha=0.5, which='both', linestyle=':')
    axs[3,k].set_ylim(-40, 70)
    axs[0,k].set_xlim(-1.5*width, 5+3.5*width)
    axs[3,k].set_ylabel(r'$E_c^{diff}$ (%)', fontsize=axes_labels)
    for i in x[:-1]:
        axs[3,k].axvline(i+offset+0.5*width, ls='--', color=mygrey)
    
    k += 1


for ax in fig.get_axes():
    ax.set_xticks(x + width, parameters)
    #ax.set_ylim(0, 2.5)
    ax.label_outer()

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, -0.05), loc="lower center", ncol = 4)

plt.savefig('../results/figures/param_sens_anal_vdiff.png',bbox_inches='tight')
#plt.show()
plt.close()
