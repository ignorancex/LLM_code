#%pip install seaborn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

NocoCtr1 = np.array([0.063, 0.065, 0.066, 0.064, 0.032, 0.03, 0.172, 0.119, 0.013, 0.15, 
                     0.053, 0.046, 0.428, 0.25, 0.095, 0.102, 0.011]); 

NocoCtr2 = np.array([0.044, 0.044, 0.14,
        0.192, 0.278, 0.368, 0.366, 0.363, 0.326, 0.217, 0.273, 0.377, 0.395,
        0.155, 0.201, 0.14, 0.052, 0.094, 0.027, 0.193, 0.194, 0.193, 0.228,
        0.12, 0.112, 0.238, 0.25, 0.326, 0.348, 0.061, 0.196, 0.104, 0.109]);

DMSOMF1 = np.array([0.011, 0.006, 0.18, 0.29, 0.268, 0.067, 0.19, 0.077, 0.116, 0.123,
        0.029, 0.029, 0.01, 0.193, 0.279, 0.046, 0.172, 0.09, 0.237, 0.073,
        0.01, 0.01, 0.058, 0.056, 0.158, 0.042]);

DMSOMF2 = np.array([0.407, 0.295, 0.316, 0.307, 0.067, 0.127, 0.041, 0.255, 0.219, 
                    0.392, 0.622, 0.166, 0.198, 0.297, 0.179, 0.125, 0.032, 0.157, 0.191, 
                    0.044, 0.153, 0.274, 0.036,0.024, 0.185, 0.042, 0.368, 0.13, 0.215, 
                    0.177, 0.178, 0.136]);

NocoMF1 = np.array([0.014, 0.074, 0.14, 0.009, 0.067, 0.068, 0.234, 0.019, 0.062, 0.046,
        0.107, 0.382, 0.054, 0.084, 0.026, 0.061, 0.238, 0.048, 0.036, 0.077,
        0.027, 0.087, 0.031, 0.052, 0.002, 0.015, 0.03, 0.032, 0.091, 0.014,
        0.054, 0.164, 0.005, 0.091, 0.001, 0.203, 0.05, 0.104, 0.501, 0.422,
        0.075, 0.528, 0.46, 0.042]);

NocoMF2 = np.array([0.059, 0.28, 0.076, 0.164, 0.074, 0.177,
        0.145, 0.131, 0.078, 0.11, 0.176, 0.123, 0.21, 0.212, 0.202, 0.086,
        0.191, 0.199, 0.184, 0.114, 0.061, 0.105, 0.141, 0.211, 0.118, 0.223,
        0.057, 0.117, 0.291]);

DMSOCtr1 = np.array([0.104, 0.011, 0.254, 0.003, 0.148, 0.11, 0.283, 0.06, 0.061, 0.002,
        0.003, 0.079, 0.445, 0.495, 0.024, 0.003, 0.001, 0.001, 0.27]);

DMSOCtr2 = np.array([0.017,
        0.011, 0.016, 0.028, 0.134, 0.185, 0.314, 0.284, 0.231, 0.08, 0.084,
        0.322, 0.103, 0.359, 0.304, 0.109, 0.194, 0.061, 0.136, 0.128, 0.318, 0.148,
        0.046, 0.15, 0.188, 0.161, 0.246, 0.206, 0.254]);

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3.54,3.54), dpi=600) #E.g., Science Mag uses a width of 3.54in (variable height); 600 dpi typical for pub; 

# #"If our main text uses a font size of 12pt, we can make the labels slightly larger than the text, and the ticks slightly smaller:"
SMALL_SIZE = 10
MEDIUM_SIZE = 12 #size of main text
BIGGER_SIZE = 15

plt.rc('font',  size     =SMALL_SIZE)     # controls default text sizes
plt.rc('axes',  titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes',  labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('legend',fontsize =MEDIUM_SIZE)    # legend fontsize
plt.rc('figure',titlesize=BIGGER_SIZE)    # fontsize of the figure title

conditions = ["DMSO-geofield", "Noco-geofield", "DMSO-MF", "Noco-MF"]
miny = 0.0
maxy = 0.65

axs.spines[['right', 'top']].set_visible(False)
#axs.set_ylabel('F-actin alignment level',rotation=0, labelpad=20)
axs.yaxis.set_label_coords(0, 1)
axs.set_xlim(0, len(conditions))  # set x axis limits to appropriate values
axs.set_ylim(miny, maxy)          # set y axis limits to appropriate values
axs.set_title('F-actin alignment level')

# setting ticks for y-axis 
axs.set_yticks([0, 0.25, 0.5]) 
# setting label for y tick 
axs.set_yticklabels(['0','25%', '50%']) 

# setting ticks for x-axis 
axs.set_xticks([-0.25, 0.75, 1.75, 2.75]) 
# setting label for y tick 
axs.set_xticklabels(["DMSO+geofield \n (N=48)", "Noco+geofield \n (N=50)", "DMSO+MF \n (N=58)", "Noco+MF \n (N=73)"],rotation=45) 

##################################
###USUAL VIOLIN

DMSOCtr = np.concatenate((DMSOCtr1,DMSOCtr2), axis = 0)
NocoCtr = np.concatenate((NocoCtr1,NocoCtr2), axis = 0)
DMSOMF = np.concatenate((DMSOMF1,DMSOMF2), axis = 0)
NocoMF = np.concatenate((NocoMF1,NocoMF2), axis = 0)

# plot violin plot
parts = sns.violinplot([DMSOCtr,NocoCtr,DMSOMF,NocoMF],color="grey",inner='stick',linewidth=1, fill=True, ax = axs) 
plt.setp(parts.collections, alpha=0.2)

axs.tick_params(bottom=False)

axs.plot(0, np.mean(DMSOCtr), marker="o", markersize=5, markeredgecolor="k", markerfacecolor="k",zorder=6)
axs.plot(1, np.mean(NocoCtr), marker="o", markersize=5, markeredgecolor="k", markerfacecolor="k",zorder=5)
axs.plot(2, np.mean(DMSOMF), marker="o", markersize=5, markeredgecolor="k", markerfacecolor="k",zorder=4)
axs.plot(3, np.mean(NocoMF), marker="o", markersize=5, markeredgecolor="k", markerfacecolor="k",zorder=3)

axs.plot(0, np.median(DMSOCtr), marker="x", markersize=6, markeredgecolor="k", markerfacecolor="k",zorder=6)
axs.plot(1, np.median(NocoCtr), marker="x", markersize=6, markeredgecolor="k", markerfacecolor="k",zorder=5)
axs.plot(2, np.median(DMSOMF), marker="x", markersize=6, markeredgecolor="k", markerfacecolor="k",zorder=4)
axs.plot(3, np.median(NocoMF), marker="x", markersize=6, markeredgecolor="k", markerfacecolor="k",zorder=3)

axs.vlines(0, np.mean(DMSOCtr)-np.std(DMSOCtr), np.mean(DMSOCtr)+np.std(DMSOCtr), color='grey', lw=2,zorder=5)
axs.vlines(1, np.mean(NocoCtr)-np.std(NocoCtr), np.mean(NocoCtr)+np.std(NocoCtr), color='grey', lw=2,zorder=4)
axs.vlines(2, np.mean(DMSOMF)-np.std(DMSOMF), np.mean(DMSOMF)+np.std(DMSOMF), color='grey', lw=2,zorder=3)
axs.vlines(3, np.mean(NocoMF)-np.std(NocoMF), np.mean(NocoMF)+np.std(NocoMF), color='grey', lw=2,zorder=2)

plt.subplots_adjust(bottom=0, wspace=1.75)
fig.tight_layout()
plt.show()
plt.savefig('figure4.png', bbox_inches='tight')