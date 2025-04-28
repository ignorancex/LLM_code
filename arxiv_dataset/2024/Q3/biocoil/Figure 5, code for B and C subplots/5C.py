import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import uncertainties as unc  
import uncertainties.unumpy as unp ### <---- this is to do "easy" error propagation
                                   ### unp.nominal_values(a) / unp.std_devs(a)

Exp1Ctr = np.array([1573.812,1586.912,1612.492]);
Exp1MF = np.array([1738.972,1737.882,1764.792]);
uExp1Ctr = unp.uarray([np.mean(Exp1Ctr)],[np.std(Exp1Ctr)]);
uExp1MF = unp.uarray([np.mean(Exp1MF)],[np.std(Exp1MF)]);
# uExp1Ctr = array([1591.0720000000001+/-16.06268553719045], dtype=object)
# uExp1MF = array([1747.2153333333335+/-12.436543821424864], dtype=object)
# uExp1MF/uExp1Ctr = array([1.0981371888471+/-0.013564736031925937], dtype=object)

Exp2Ctr = np.array([1168.595,1561.195,1614.095]);
Exp2MF = np.array([1253.695,1898.695,1910.495]);
uExp2Ctr = unp.uarray([np.mean(Exp2Ctr)],[np.std(Exp2Ctr)]);
uExp2MF = unp.uarray([np.mean(Exp2MF)],[np.std(Exp2MF)]);
# uExp2Ctr = array([1447.9616666666668+/-198.71907026978786], dtype=object)
# uExp2MF = array([1687.6283333333333+/-306.8750161801308], dtype=object)
# uExp2MF/uExp2Ctr = array([1.1655200356363025+/-0.2655238864811309], dtype=object)

Exp3Ctr = np.array([1072.544,1025.044,1030.444,1061.044,975.644,1069.944,1135.344,1171.844,1098.544,1107.444,1161.244,1208.744]);
Exp3MF = np.array([1194.492,1146.292,1116.992,1091.292,1141.992,1077.792,1003.392,1051.492,1226.692,1178.992,1124.292,1179.092]);
uExp3Ctr = unp.uarray([np.mean(Exp3Ctr)],[np.std(Exp3Ctr)]);
uExp3MF = unp.uarray([np.mean(Exp3MF)],[np.std(Exp3MF)]);
# uExp3Ctr = array([1093.1523333333334+/-64.92130439146217], dtype=object)
# uExp3MF = array([1127.7336666666667+/-61.431038006495996], dtype=object)
# uExp3MF/uExp3Ctr = array([1.0316345053464644+/-0.08313700461958033], dtype=object)

Exp4Ctr = np.array([1601.7415,1493.7125,1449.6175]);
Exp4MF = np.array([1607.019,1579.005, 1501.479]);
uExp4Ctr = unp.uarray([np.mean(Exp4Ctr)],[np.std(Exp4Ctr)]);
uExp4MF = unp.uarray([np.mean(Exp4MF)],[np.std(Exp4MF)]);
# uExp4Ctr = array([1515.0238333333334+/-63.906481363178045], dtype=object)
# uExp4MF = array([1562.5010000000002+/-44.63899201370927], dtype=object)
# uExp4MF/uExp4Ctr = array([1.0313375708170929+/-0.052542485462205246], dtype=object)

# (uExp1MF/uExp1Ctr + uExp2MF/uExp2Ctr + uExp3MF/uExp3Ctr + uExp4MF/uExp4Ctr)/4.0 = array([1.08165732516174+/-0.0708693386942312], dtype=object)

# not normalized
# scipy.stats.mannwhitneyu([1591.0720000000001,1447.9616666666668,1093.1523333333334,1515.0238333333334],[1747.2153333333335,1687.6283333333333,1127.7336666666667,1562.5010000000002])
# MannwhitneyuResult(statistic=4.0, pvalue=0.34285714285714286)

# normalized to Control
# scipy.stats.mannwhitneyu([1,1,1,1],[1.0981371888471,1.1655200356363025,1.0316345053464644,1.0313375708170929])
# MannwhitneyuResult(statistic=0.0, pvalue=0.021070570134378658)

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

miny = 0.89
maxy = 1.5

axs.spines[['right', 'top']].set_visible(False)
axs.set_ylabel('MF/geofield \n fluo counts \n (norm.)',rotation=0) #, labelpad=10)
axs.set_xlim(0.38, 0.62)  # set x axis limits to appropriate values
axs.set_ylim(miny,maxy)          # set y axis limits to appropriate values
axs.set_title('F-actin polymerization')

# setting ticks for y-axis 
axs.set_yticks([1.0, 1.2]) 
# setting label for y tick 
axs.set_yticklabels(['100%', '120%']) 

# setting ticks for x-axis 
axs.set_xticks([0.4,0.45,0.5,0.55,0.6]) 
# setting label for y tick 
axs.set_xticklabels(["N=3", "N=3", "N=3", "N=12", "Average"],rotation=45) 

xpoints = np.array([1, 8])
ypoints = np.array([3, 10])

plt.errorbar(np.array([0.4,0.45,0.5,0.55,0.6]),np.array([unp.nominal_values(uExp1MF/uExp1Ctr)[0],unp.nominal_values(uExp2MF/uExp2Ctr)[0],unp.nominal_values(uExp4MF/uExp4Ctr)[0],unp.nominal_values(uExp3MF/uExp3Ctr)[0],unp.nominal_values((uExp1MF/uExp1Ctr + uExp2MF/uExp2Ctr + uExp3MF/uExp3Ctr + uExp4MF/uExp4Ctr)/4.0)[0]]),yerr =np.array([unp.std_devs(uExp1MF/uExp1Ctr)[0],unp.std_devs(uExp2MF/uExp2Ctr)[0],unp.std_devs(uExp4MF/uExp4Ctr)[0],unp.std_devs(uExp3MF/uExp3Ctr)[0],unp.std_devs((uExp1MF/uExp1Ctr + uExp2MF/uExp2Ctr + uExp3MF/uExp3Ctr + uExp4MF/uExp4Ctr)/4.0)[0]]),fmt='o', markersize=5, color = 'k')
axs.tick_params(bottom=False)
axs.yaxis.set_label_coords(-0.05,0.65)
plt.axhline(y=1, color='k', linestyle=':')
axs.spines['left'].set_bounds(0.89, 1.25)

plt.subplots_adjust(bottom=0, wspace=1.75)

fig.tight_layout()
plt.show()
#plt.savefig('figure4.png', bbox_inches='tight')
