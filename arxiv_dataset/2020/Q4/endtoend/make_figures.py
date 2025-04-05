import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import os.path
from os import path
import sys
plt.rcParams['text.usetex'] = True #Let TeX do the typsetting



# # # # # # # FILE PATHS # # # # # # #

include_climate = int(sys.argv[1]) # analyze analysis with or without climate concern level feature      

if include_climate:
    datapath_estimation = "output/with_climate/estimation/"
    datapath_algorithm = "output/with_climate/algorithm/"
    datapath_endtoend = "output/with_climate/endtoend/"
else:
    datapath_estimation = "output/without_climate/estimation/"
    datapath_algorithm = "output/without_climate/algorithm/"
    datapath_endtoend = "output/without_climate/endtoend/"
    
# check which folders exist
estimation_exists = path.exists(datapath_estimation)
algorithm_exists = path.exists(datapath_algorithm)
endtoend_exists = path.exists(datapath_endtoend)

if estimation_exists==False:
    print("estimation data not found - no figures can be generated")
    quit()
    
# # # # # # READ DATA # # # # # # # #
    
# ESTIMATION DATA
pool_df = pd.read_csv(datapath_estimation+"pool_qis.csv") # get POOL DATA including qi estimates, pi estimates
pool_df['pi'] = pickle.load( open(datapath_estimation+"pool_pis.pickle", "rb" ) )['pool_pis']

ESS_df = pd.read_csv(datapath_estimation+"background_qis.csv") # get ESS DATA including qi estimates
ESS_df['wqi'] = ESS_df['qi']*ESS_df['weight'] # compute weighted qis

beta0 = pickle.load( open( datapath_estimation+"betas.pickle", "rb" ) )['baseline'] # get beta estimates
betas_temp = pickle.load( open(datapath_estimation+"betas.pickle", "rb" ) )['value_multipliers']

twocorr_df = pd.read_csv(datapath_estimation+'pairwise_correlations.csv') 

# ALGORITHM DATA
if algorithm_exists:
    alg_output = pickle.load( open(datapath_algorithm+"histogram_data.pickle", "rb" ) )['fv_representation_dist']

# END TO END DATA
if endtoend_exists:
    n = 100000
    if include_climate:
        rvals = [600000]
    else:
        rvals = [10000,11000,12000,13000,14000,15000,60000]

    e2e_df = pd.read_csv(datapath_endtoend+'end_to_end_r'+str(rvals[0])+'_n'+str(n)+'.csv')
    e2e_df = e2e_df.rename(columns={'normalized good' : 'normgood_r'+str(rvals[0])})
    e2e_df = e2e_df.rename(columns={'normalized any' : 'normany_r'+str(rvals[0])})
    e2e_df = e2e_df.rename(columns={'normalized satisfying 1&2' : 'norm12_r'+str(rvals[0])})
    for i in range(1,len(rvals)):
        temp_df = pd.read_csv(datapath_endtoend+'end_to_end_r'+str(rvals[i])+'_n'+str(n)+'.csv')
        temp_df = temp_df.rename(columns={'normalized good' : 'normgood_r'+str(rvals[i])})
        temp_df = temp_df.rename(columns={'normalized any' : 'normany_r'+str(rvals[i])})
        temp_df = temp_df.rename(columns={'normalized satisfying 1&2' : 'norm12_r'+str(rvals[i])})
        e2e_df = e2e_df.merge(temp_df[['Unnamed: 0','norm12_r'+str(rvals[i]),'normgood_r'+str(rvals[i]),'normany_r'+str(rvals[i])]], on='Unnamed: 0')


# # # # # # CONSTRUCT DATA # # # # # # # #

# set parameters
k = 110 
pool_n = pool_df.count()[0]
ESS_n = ESS_df['weight'].sum()

features = list(pool_df) 
features.remove('pi')
features.remove('qi')
features.remove('Unnamed: 0')
featurevalues = {}
for f in features:
    featurevalues[f] = list(pool_df[f].unique())

# manually re-order
featurevalues['age_bucket'] = sorted(featurevalues['age_bucket'])
featurevalues['edu_bucket'] = ['No Qualifications/Level 1','Level 2/Level 3/Apprenticeship/Other','Level 4 and above']
featurevalues['geo_bucket'] = ['North East','North West','Yorkshire and The Humber',
                                'East Midlands','West Midlands', 'East of England','London','South East',
                                'South West','Wales','Scotland','Northern Ireland']
if include_climate==True:
    featurevalues['climate_concern_level'] = ['Not at all concerned', 'Not very concerned', 'Fairly concerned','Very concerned','Other']


# pool and background counts by feature value (only require estimation data)
pool_n_fv = {}
ESS_n_fv = {}
betas = {}
names = {}
for f in features:
    pool_n_fv[f] = {}
    ESS_n_fv[f] = {}
    betas[f] = {}
    names[f]={}
    for v in featurevalues[f]:
        pool_n_fv[f][v] = pool_df[pool_df[f]==v][f].count()
        ESS_n_fv[f][v] = ESS_df[ESS_df[f]==v]['weight'].sum() 
        betas[f][v] = betas_temp[(f,v)]  
        # set names
        if v =='No Qualifications/Level 1':
            names[f][v] = 'Level 1 and below'
        elif v == 'Level 2/Level 3/Apprenticeship/Other':
            names[f][v] = 'Level 2-3 or other'
        elif v == 'Yorkshire and The Humber':
            names[f][v] = 'Yorkshire/Humber'
        else:   
            names[f][v] = v



# calculate stats on algorithm output
if algorithm_exists:
    stats = {}
    stats['mean'] = {}
    stats['max'] = {}
    stats['min'] = {}
    stats['lowci'] = {}  # 95 perc CI
    stats['upci'] = {}   # 95 perc CI
    stats['lowci_n'] = {} # 90 perc CI
    stats['upci_n'] = {}  # 90 perc CI
    stats['popfrac'] = {}
    stats['exprep'] = {}
    stats['maxdist_nfci'] = {}
    stats['maxdist'] = {}


    for f in features:
        stats['mean'][f] = {}
        stats['max'][f] = {}
        stats['min'][f] = {}
        stats['lowci'][f] = {}
        stats['upci'][f] = {}
        stats['lowci_n'][f] = {}
        stats['upci_n'][f] = {}
        stats['popfrac'][f] = {}
        stats['exprep'][f] = {}
        stats['maxdist_nfci'][f] = {}
        stats['maxdist'][f] = {}
             
        # 
        for v in featurevalues[f]:
            dist = alg_output[(f,v)]
                    
            # min / max
            stats['max'][f][v] = max(list(dist.keys()))
            stats['min'][f][v] = min(list(dist.keys()))
            
            # mean
            mean = 0
            for l in dist.keys():
                mean = mean+l*dist[l]
            stats['mean'][f][v] = mean
            
            # 95% confidence intervals      
            c = 0
            for l in sorted(list(dist.keys())):
                c = c+dist[l]
                if c >= 0.025:
                    stats['lowci'][f][v] = l
                    break           
            
            u = 0
            for l in reversed(sorted(list(dist.keys()))):
                u = u+dist[l]
                if u >= 0.025:
                    stats['upci'][f][v] = l
                    break
            
            # 90% confidence intervals
            c = 0
            for l in sorted(list(dist.keys())):
                c = c+dist[l]
                if c >= 0.05:
                    stats['lowci_n'][f][v] = l
                    break           
            
            u = 0
            for l in reversed(sorted(list(dist.keys()))):
                u = u+dist[l]
                if u >= 0.05:
                    stats['upci_n'][f][v] = l
                    break
            
            # population fraction
            stats['popfrac'][f][v] = ESS_n_fv[f][v]/ESS_n*110
            stats['exprep'][f][v] = pool_df[pool_df[f]==v]['pi'].sum()
            
            # maximum distance
            stats['maxdist'][f][v] = max(abs(stats['popfrac'][f][v]-stats['min'][f][v]),abs(stats['popfrac'][f][v]-stats['max'][f][v]))
            
            # maximum distance with 95% confidence
            if stats['min'][f][v] <= stats['popfrac'][f][v] <= stats['max'][f][v]:
                stats['maxdist_nfci'][f][v] = max(abs(stats['popfrac'][f][v] - stats['lowci'][f][v]),abs(stats['popfrac'][f][v] - stats['upci'][f][v]))
            else:
                stats['maxdist_nfci'][f][v] = max(abs(stats['popfrac'][f][v] - stats['lowci_n'][f][v]),abs(stats['popfrac'][f][v] - stats['upci_n'][f][v]))
    

# # # # # # # PLOTS # # # # # # # 
# (generated in the order in which they appear in the paper / appendix)
def save(strFile):
    if os.path.isfile(strFile):
        os.remove(strFile)   # Opt.: os.system("rm "+strFile)
    plt.savefig(strFile,bbox_inches='tight')

# PLOT: end-to-end guarantee
if endtoend_exists & include_climate==False:
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,0.8))

    norm = 'normgood'

    Rs = ['60000']

    colors = ['b','g','r','c','y','m']
    c = 0
    for r in Rs:
        color = colors[c]
        ax.scatter(e2e_df['qi'].values,e2e_df[norm+'_r'+str(r)].values,alpha=0.8,color=color,marker='+')
        c = c+1

    ax.set_yticks([.9,1,1.1])
    ax.set_yticklabels(['$.9\,k/n$','$k/n$','$1.1\,k/n$'])
    ax.text(-.012,1.02,'$\\mathit{end}$-$\\mathit{to}$-$\\mathit{end}$',fontsize=12)
    ax.text(-.012,0.955,'$\\mathit{probability}$',fontsize=12)
    ax.text(0.044,.8,'$q_i$',fontsize=12)
    ax.set_xlim([0,0.08])
    ax.set_ylim([0.9,1.1])


    if include_climate:
        save('figures/end_to_end_withclimate.pdf')
        plt.close()
    else:
        save('figures/end_to_end.pdf')
        plt.close()



# PLOT: realized representation
if algorithm_exists:
    def box(x,y,length):
        return patches.Rectangle((x,y),length,0.8,linewidth=0.8,edgecolor='k',facecolor='k')

    def rect2(x,y):
        return patches.Rectangle((x,y),1,0.8,linewidth=0.8,edgecolor='w',facecolor='k')

    # draws tiny distribution bars
    def minidist(x,y,width):
        return patches.Rectangle((x,y),1,width,linewidth=0,facecolor='k')

    custom_orange = (251/255,140/255,0)
    plotstyle = 'rectangles'  # can also do mini distributions ('minidist')

    # set plot parameters
    if include_climate:
        plotrows = 18
        features1 = {'gender':[1,2],
                     'age_bucket':[4,5,6,7],
                     'edu_bucket':[9,10,11],
                     'climate_concern_level':[13,14,15,16,17]
                    }
        features2 = {'geo_bucket':[1,2,3,4,5,6,7,8,9,10,11,12],
                     'urban_rural':[14,15],
                     'ethnicity_bucket':[17,18]}
    else:
        plotrows = 15
        features1 = {'gender':[1,2],
                     'age_bucket':[4,5,6,7],
                     'edu_bucket':[9,10,11],
                     'urban_rural':[13,14]
                    }
        features2 = {'geo_bucket':[1,2,3,4,5,6,7,8,9,10,11,12],
                     'ethnicity_bucket':[14,15]}


    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,5.5))
    fig.tight_layout(pad=8)
    width = 0.5    

    for plot in [1,2]:
        yticks = []
        yticklabels = []
        
        if plot==1:
            ax = ax1
            featureset = features1
        else:
            ax = ax2
            featureset = features2

        for f in featureset.keys():
            v_ind = plotrows-featureset[f][0]
            yticks.extend(featureset[f])

            for v in featurevalues[f]:
                yticklabels.append(names[f][v]) # ticks

                if plotstyle == 'rectangles': # put in bars
                    ax.add_patch(box(stats['min'][f][v],v_ind+0.1,stats['max'][f][v]-stats['min'][f][v])) # main box
                    ax.add_patch(rect2(stats['exprep'][f][v]-1.3,v_ind+0.1)) # target box
                    ax.vlines(stats['popfrac'][f][v]-0.5,ymin = v_ind, ymax = v_ind+1, color=custom_orange)

                v_ind = v_ind-1
                
        # axes settings
        ax.set_yticks(ticks=[plotrows-(x-0.5) for x in reversed(yticks)])
        ax.set_yticklabels(labels=reversed(yticklabels))
        ax.set_xticks(ticks=np.arange(0,120,10))
        ax.set_xlabel('Number of seats (out of 110)')
        ax.set_xlim([0,110])
        ax.set_ylim([-0.2,plotrows+1])
                                        
        # custom stuff per subplot
        if plot==1:
            if include_climate:
                # shading
                ax.axhspan(10.6, 15.6, facecolor='0.2', alpha=0.1)
                ax.axhspan(-0.2,6.6, facecolor='0.2', alpha=0.1)
                # headers
                ax.text(-2.5,18.1,'\\textbf{Gender}',fontweight='bold',horizontalalignment='right')
                ax.text(-2.5,15.1,'\\textbf{Age}',fontweight='bold',horizontalalignment='right')
                ax.text(-2.5,10.1,'\\textbf{Education}',fontweight='bold',horizontalalignment='right')
                ax.text(-2.5,6.1,'\\textbf{Climate Concern}',fontweight='bold',horizontalalignment='right')  
            else:
                # shading
                ax.axhspan(7.7, 12.7, facecolor='0.2', alpha=0.1)
                ax.axhspan(-0.2,3.7, facecolor='0.2', alpha=0.1)
                # headers
                ax.text(-2.5,15.1,'\\textbf{Gender}',fontweight='bold',horizontalalignment='right')
                ax.text(-2.5,12.1,'\\textbf{Age}',fontweight='bold',horizontalalignment='right')
                ax.text(-2.5,7.1,'\\textbf{Education}',fontweight='bold',horizontalalignment='right')
                ax.text(-2.5,3.1,'\\textbf{Urban/Rural}',fontweight='bold',horizontalalignment='right')  

        else: 
            if include_climate:
                # shading
                ax.axhspan(2.6, 5.6, facecolor='0.2', alpha=0.1)
                # headers
                ax.text(-2.5,18.1,'\\textbf{Region}',fontweight='bold',horizontalalignment='right')  
                ax.text(-2.5,5.1,'\\textbf{Urban/Rural}',fontweight='bold',horizontalalignment='right')
                ax.text(-2.5,2.1,'\\textbf{Ethnicity}',fontweight='bold',horizontalalignment='right')
            else:
                # shading
                ax.axhspan(-0.2, 2.7, facecolor='0.2', alpha=0.1)
                # headers
                ax.text(-2.5,2.1,'\\textbf{Ethnicity}',fontweight='bold',horizontalalignment='right')  
                ax.text(-2.5,15.1,'\\textbf{Region}',fontweight='bold',horizontalalignment='right')

    # save figure
    if include_climate:
        save('figures/realized_representation_withclimate.pdf')
        plt.close()
    else:
        save('figures/realized_representation.pdf')
        plt.close()

# LEGEND for above plot
    def box(x,y,length,width):
        return patches.Rectangle((x,y),length,width,linewidth=1,edgecolor='k',facecolor='k')

    def rect2(x,y,length,width):
        return patches.Rectangle((x,y),length,width,linewidth=1,edgecolor='w',facecolor='k')

    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,0.5))

    for s in ["right","left","top","bottom"]:
        side = ax.spines[s]
        side.set_visible(False)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    ax.set_xlim([0,1000])
    ax.set_ylim([0,1])

    texty = 0.35

    x = 2
    ax.text(x,texty,'\\textbf{Legend:}',fontsize=14,fontweight='bold')


    # # ideal representation
    x = x + 140
    ymin = 0.25
    ymax = 0.75
    ax.vlines(x,ymin =ymin, ymax = ymax, color=custom_orange,linewidth=5)
    ax.text(x+15,texty,'proportional no. seats',fontsize=13.5)

    # #  expected representation
    x = x+280
    y = 0.25
    length = 10
    height = 0.5
    ax.add_patch(box(x,y,length,height)) 

    x = x+2
    y = 0.29
    length = 6
    height = 0.42
    ax.add_patch(rect2(x,y,length,height))
    ax.text(x + length + 15,texty,'expectated no. seats',fontsize=13.5)

    # #  realized representation
    x = x + 280
    y = 0.25
    length = 35
    height = 0.5
    ax.add_patch(box(x,y,length,height)) 
    ax.text(x + length+15,texty,'range in no. seats over all panels in distribution',fontsize=13.5)


    save('figures/realized_representation_legend.pdf')
    plt.close()







# PLOT: Pool / background data composition
ordered_features = {'gender':[0,1],
                    'age_bucket':[3,4,5,6],
                    'edu_bucket':[8,9,10],
                    'geo_bucket':[12,13,14,15,16,17,18,19,20,21,22,23],
                    'urban_rural':[25,26],
                    'ethnicity_bucket':[28,29],
                    'climate_concern_level':[31,32,33,34,35]
                    }

if include_climate:
    FV = 30

else:
    FV = 25
    x = ordered_features.pop('climate_concern_level')
    
# construct data for plot
ticklabels = np.empty(FV,dtype = object)
ticks = []
pool = np.zeros(FV)
ESS = np.zeros(FV)

i = 0
for f in ordered_features.keys():
    ticks.extend(ordered_features[f])
    for v in featurevalues[f]:
        ticklabels[i] = names[f][v]
        pool[i] = pool_n_fv[f][v]/pool_df['qi'].count()
        ESS[i] = ESS_df[ESS_df[f]==v]['weight'].sum()/ESS_df['weight'].sum()
        i += 1


fig, ax = plt.subplots(figsize=(12,5))
width = 0.33        # the width of the bars
p1 = ax.bar(ticks, pool, width, bottom=0)
p2 = ax.bar(np.array(ticks) + width, ESS, width, bottom=0)

ax.set_xticks(np.array(ticks) + width / 2)
ax.set_xticklabels(ticklabels)
plt.xticks(rotation=90)
plt.ylabel('density')
ax.legend((p1[0], p2[0]), ('Pool', 'Background'),loc='upper left')
ax.autoscale_view()

v = -0.02

if include_climate:
    plt.text(-1,v,'\\textbf{Gender}' ,rotation=90, verticalalignment='top')  
    plt.text(2,v,'\\textbf{Age}' ,rotation=90, verticalalignment='top')  
    plt.text(7,v,'\\textbf{Education}' ,rotation=90,verticalalignment='top')  
    plt.text(11,v,'\\textbf{Region}', rotation=90,verticalalignment='top')  
    plt.text(24,v,'\\textbf{Urban/Rural}', rotation=90,verticalalignment='top')  
    plt.text(27,v,'\\textbf{Ethnicity}', rotation=90,verticalalignment='top')
    plt.text(30,v,'\\textbf{Climate Concern}', rotation=90,verticalalignment='top')

    save('figures/data_composition_withclimate.pdf')
    plt.close()
    
else:
    plt.text(-1,v,'\\textbf{Gender}' ,rotation=90, verticalalignment='top')  
    plt.text(2,v,'\\textbf{Age}' ,rotation=90, verticalalignment='top')  
    plt.text(7,v,'\\textbf{Education}' ,rotation=90,verticalalignment='top')  
    plt.text(11,v,'\\textbf{Region}', rotation=90,verticalalignment='top')  
    plt.text(24,v,'\\textbf{Urban/Rural}', rotation=90,verticalalignment='top')  
    plt.text(27,v,'\\textbf{Ethnicity}', rotation=90,verticalalignment='top')
    
    save('figures/data_composition.pdf')
    plt.close()







# PLOT: betas
def box(x,y,length):
    return patches.Rectangle((x,y),length,0.4,linewidth=0.8,edgecolor='k',facecolor='k')


# set plot parameters
if include_climate:
    plotrows = 18
    features1 = {'gender':[1,2],
                 'age_bucket':[4,5,6,7],
                 'edu_bucket':[9,10,11],
                 'climate_concern_level':[13,14,15,16,17]
                }
    features2 = {'geo_bucket':[1,2,3,4,5,6,7,8,9,10,11,12],
                 'urban_rural':[14,15],
                 'ethnicity_bucket':[17,18]}
else:
    plotrows = 15
    features1 = {'gender':[1,2],
                 'age_bucket':[4,5,6,7],
                 'edu_bucket':[9,10,11],
                 'urban_rural':[13,14]
                }
    features2 = {'geo_bucket':[1,2,3,4,5,6,7,8,9,10,11,12],
                 'ethnicity_bucket':[14,15]}


fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,5.5))
fig.tight_layout(pad=8)
width = 0.5    

for plot in [1,2]:
    yticks = []
    yticklabels = []
    
    if plot==1:
        ax = ax1
        featureset = features1
    else:
        ax = ax2
        featureset = features2

    for f in featureset.keys():
        v_ind = plotrows-featureset[f][0]
        yticks.extend(featureset[f])

        for v in featurevalues[f]:
            yticklabels.append(names[f][v]) # ticks
            ax.add_patch(box(0,v_ind+0.1,betas[f][v]*100)) # main box
            ax.text(betas[f][v]*100+1,v_ind,str(int(round(betas[f][v]*100,0))))
            
            v_ind = v_ind-1
            
    # axes
    ax.set_yticks(ticks=[plotrows-(x-0.5) for x in reversed(yticks)])
    ax.set_yticklabels(labels=reversed(yticklabels))
    ax.set_xticks(ticks=np.arange(0,110,10))
    ax.set_xlabel('$\\beta $')
    ax.set_xlim([0,109])
    ax.set_ylim([-0.2,plotrows+1])
                                    
    # custom stuff per subplot
    if plot==1:
        if include_climate:
            # shading
            ax.axhspan(10.6, 15.6, facecolor='0.2', alpha=0.1)
            ax.axhspan(-0.2,6.6, facecolor='0.2', alpha=0.1)
            # headers
            ax.text(-2.5,18.1,'\\textbf{Gender}',fontweight='bold',horizontalalignment='right')
            ax.text(-2.5,15.1,'\\textbf{Age}',fontweight='bold',horizontalalignment='right')
            ax.text(-2.5,10.1,'\\textbf{Education}',fontweight='bold',horizontalalignment='right')
            ax.text(-2.5,6.1,'\\textbf{Climate Concern}',fontweight='bold',horizontalalignment='right')  
        else:
            # shading
            ax.axhspan(7.7, 12.7, facecolor='0.2', alpha=0.1)
            ax.axhspan(-0.2,3.7, facecolor='0.2', alpha=0.1)
            # headers
            ax.text(-2.5,15.1,'\\textbf{Gender}',fontweight='bold',horizontalalignment='right')
            ax.text(-2.5,12.1,'\\textbf{Age}',fontweight='bold',horizontalalignment='right')
            ax.text(-2.5,7.1,'\\textbf{Education}',fontweight='bold',horizontalalignment='right')
            ax.text(-2.5,3.1,'\\textbf{Urban/Rural}',fontweight='bold',horizontalalignment='right')  

    else: 
        if include_climate:
            # shading
            ax.axhspan(2.6, 5.6, facecolor='0.2', alpha=0.1)
            # headers
            ax.text(-2.5,18.1,'\\textbf{Region}',fontweight='bold',horizontalalignment='right')  
            ax.text(-2.5,5.1,'\\textbf{Urban/Rural}',fontweight='bold',horizontalalignment='right')
            ax.text(-2.5,2.1,'\\textbf{Ethnicity}',fontweight='bold',horizontalalignment='right')
        else:
            # shading
            ax.axhspan(-0.2, 2.7, facecolor='0.2', alpha=0.1)
            # headers
            ax.text(-2.5,2.1,'\\textbf{Ethnicity}',fontweight='bold',horizontalalignment='right')  
            ax.text(-2.5,15.1,'\\textbf{Region}',fontweight='bold',horizontalalignment='right')

if include_climate:
    save('figures/betas_withclimate.pdf')
    plt.close()
else:
    save('figures/betas.pdf')
    plt.close()









# PLOT: qi distribution
if include_climate:
    bins = np.arange(0,30,1)
else:
    bins = np.arange(0,10,1)
    
# construct data
counts_pool, bins, bars = plt.hist(pool_df['qi'].values*100, bins = bins, label=['Pool', 'Background'])
plt.close()
density_pool = counts_pool/pool_df['qi'].count()

density_background = np.zeros(len(density_pool))
for i in range(len(bins)-1):
    density_background[i] = ESS_df[(bins[i]/100 <= ESS_df['qi']) & (ESS_df['qi'] < bins[i+1]/100)]['weight'].sum()/ESS_df['weight'].sum()
                                                                                                                        
bins=bins[:-1]
width = 0.4

plt.bar(np.array(bins+0.3),density_pool,width=width)
plt.bar(np.array(bins)+0.3+width,density_background,width=width)

plt.legend(['Pool','Background Sample'])
plt.ylabel("density")
plt.xlabel("$q_i$ (\%)")
plt.axvline(x=2.9,color='k')
plt.ylim([0,0.4])
plt.text(3.1,.36,'$\overline{q}=2.9\%$')

if include_climate:
    plt.xticks(np.arange(0,16,1))
    plt.xlim([0,15])
    save('figures/qis_withclimate.pdf')
    plt.close()
else:
    plt.xlim([1,8])
    save('figures/qis.pdf')
    plt.close()


# count people who were dropped due to privacy issues
bins = np.arange(0,40,1)

counts_pool, bins, bars = plt.hist(pool_df['qi'].values*100, bins = bins, label=['Pool', 'Background'])
plt.close()
density_pool = counts_pool/pool_df['qi'].count()
density_background = np.zeros(len(density_pool))

for i in range(len(bins)-1):
    density_background[i] = ESS_df[(bins[i]/100 <= ESS_df['qi']) & (ESS_df['qi'] < bins[i+1]/100)]['weight'].sum()/ESS_df['weight'].sum()
                                                   
pcount = 0
bcount = 0
for i in range(len(density_pool)):
    if (0 < density_pool[i]*1715 < 7) or (0 < density_background[i]*1915 < 7):
        pcount+=density_pool[i]
        bcount+=density_background[i]
print("percent people in pool data not shown in qi distribution figure: " + str(round(pcount*100,2))+"%")
print("percent people in background data not shown in qi distribution figure: " + str(round(bcount*100,2))+"%")
print()








# PLOT: qi Calibration Test
if include_climate:
    bins = np.arange(0,21,1)
else:
    bins = np.arange(0,10,1)

counts_pool, bins, bars = plt.hist(pool_df['qi'].values*100, bins = bins, label=['Pool', 'Background'])
plt.close()
density_pool = counts_pool/pool_df['qi'].count()

density_background = np.zeros(len(density_pool))
for i in range(len(bins)-1):
    density_background[i] = ESS_df[(bins[i]/100 <= ESS_df['qi']) & (ESS_df['qi'] < bins[i+1]/100)]['wqi'].sum()/ESS_df['wqi'].sum()

# make plot
width = 0.4
bins = bins[:-1]

plt.bar(np.array(bins)+0.3,density_pool,width=width)
plt.bar(np.array(bins)+0.3+width,density_background,width=width)
plt.ylim([0,0.3])
plt.legend(['Realized Pool','Hypothetical Pool'])
plt.ylabel("density")
plt.xlabel("$q_i$ (\%)")
plt.axvline(x=2.9,color='k')

if include_climate:
    plt.xlim([1,20])
    plt.ylim([0,0.15])
    plt.text(3.1,.14,'$\overline{q}=2.9\%$')
    save('figures/qis_calibration_withclimate.pdf')
    plt.close()
else:
    plt.xlim([1,8])
    plt.ylim([0,0.3])
    plt.text(3.1,.28,'$\overline{q}=2.9\%$')
    save('figures/qis_calibration.pdf')
    plt.close()




# count people in bins that were dropped due to privacy issues
bins = np.arange(0,40,1)

counts_pool, bins, bars = plt.hist(pool_df['qi'].values*100, bins = bins, label=['Pool', 'Background'])
plt.close()
density_pool = counts_pool/pool_df['qi'].count()
density_background = np.zeros(len(density_pool))

for i in range(len(bins)-1):
    density_background[i] = ESS_df[(bins[i]/100 <= ESS_df['qi']) & (ESS_df['qi'] < bins[i+1]/100)]['wqi'].sum()/ESS_df['wqi'].sum()
          
pcount = 0
bcount = 0
for i in range(len(density_pool)):
    if (0 < density_pool[i]*1715 < 7) or (0 < density_background[i]*1915 < 7):
        pcount+=density_pool[i]
        bcount+=density_background[i]
print("percent people in pool data not shown in qi calibration figure: " + str(round(pcount*100,2))+"%")
print("percent people in background data not shown in qi calibration figure: " + str(round(bcount*100,2))+"%")









# PLOT: real vs. hypothetical pool
ordered_features = {'gender':[0,1],
                    'age_bucket':[3,4,5,6],
                    'edu_bucket':[8,9,10],
                    'geo_bucket':[12,13,14,15,16,17,18,19,20,21,22,23],
                    'urban_rural':[25,26],
                    'ethnicity_bucket':[28,29],
                    'climate_concern_level':[31,32,33,34,35]
                    }

if include_climate:
    FV = 30

else:
    FV = 25
    x = ordered_features.pop('climate_concern_level')
    
# construct data for plot
ticklabels = np.empty(FV,dtype = object)
ticks = []
pool = np.zeros(FV)
ESS = np.zeros(FV)

i = 0
for f in ordered_features.keys():
    ticks.extend(ordered_features[f])
    for v in featurevalues[f]:
        ticklabels[i] = names[f][v]
        pool[i] = pool_n_fv[f][v]
        ESS[i] = 60000*ESS_df[ESS_df[f]==v]['wqi'].sum()/ESS_df['qi'].count()
        i += 1

fig, ax = plt.subplots(figsize=(12,5))
width = 0.33        # the width of the bars
p1 = ax.bar(ticks, pool, width, bottom=0)
p2 = ax.bar(np.array(ticks) + width, ESS, width, bottom=0)

#ax.set_title('Pool vs. Expected Pool Drawn from Background Data')
ax.set_xticks(np.array(ticks) + width / 2)
ax.set_xticklabels(ticklabels)
plt.xticks(rotation=90)
plt.ylabel('frequency')
ax.legend((p1[0], p2[0]), ('Realized Pool', 'Hypothetical Pool'),loc='upper left')
ax.autoscale_view()

v = -40

if include_climate:
    plt.text(-1,v,'\\textbf{Gender}' ,rotation=90, verticalalignment='top')  
    plt.text(2,v,'\\textbf{Age}' ,rotation=90, verticalalignment='top')  
    plt.text(7,v,'\\textbf{Education}' ,rotation=90,verticalalignment='top')  
    plt.text(11,v,'\\textbf{Region}', rotation=90,verticalalignment='top')  
    plt.text(24,v,'\\textbf{Urban/Rural}', rotation=90,verticalalignment='top')  
    plt.text(27,v,'\\textbf{Ethnicity}', rotation=90,verticalalignment='top')
    plt.text(30,v,'\\textbf{Climate Concern}', rotation=90,verticalalignment='top')

    save('figures/realvhypothetical_withclimate.pdf')
    plt.close()
    
else:
    plt.text(-1,v,'\\textbf{Gender}' ,rotation=90, verticalalignment='top')  
    plt.text(2,v,'\\textbf{Age}' ,rotation=90, verticalalignment='top')  
    plt.text(7,v,'\\textbf{Education}' ,rotation=90,verticalalignment='top')  
    plt.text(11,v,'\\textbf{Region}', rotation=90,verticalalignment='top')  
    plt.text(24,v,'\\textbf{Urban/Rural}', rotation=90,verticalalignment='top')  
    plt.text(27,v,'\\textbf{Ethnicity}', rotation=90,verticalalignment='top')
    
    save('figures/realvhypothetical.pdf')
    plt.close()







# PLOT: second-order correlations
plt.scatter(twocorr_df['density in pool'].values,twocorr_df['qᵢ-weighted density in background'],linewidths=1,alpha=0.5)
x = np.linspace(0, 0.7, 100)
plt.plot(x,x,color='k')
#plt.xlim([0,0.7])
#plt.ylim([0,0.7])
plt.xlabel('fraction of realized pool')
plt.ylabel('fraction of hypothetical pool')

if include_climate:
    save('figures/twocorrelations_withclimate.pdf')
    plt.close()
else:
    save('figures/twocorrelations.pdf')
    plt.close()









# PLOT: end-to-end probabilities - extended results
if endtoend_exists:
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,3))

    norm = 'norm12'

    if include_climate:
        Rs = ['600000']
    else:
        Rs = ['10000','11000','12000','15000','60000']

    colors = ['r','g','c',custom_orange,'b','y','m']
    c = 0
    for r in Rs:
        color = colors[c]
        ax.scatter(e2e_df['qi'].values,e2e_df[norm+'_r'+str(r)].values,alpha=0.8,color=color,marker='+')
        c = c+1

    ax.set_yticks([0,.9,1,1.1])
    ax.set_yticklabels(['0','$.9\,k/n$','$k/n$','$1.1\,k/n$'])
    ax.text(-.012,0.5,'$\\mathit{end}$-$\\mathit{to}$-$\\mathit{end}$',fontsize=12)
    ax.text(-.012,0.4,'$\\mathit{probability}$',fontsize=12)
    ax.text(0.049,-0.25,'$q_i$',fontsize=12)
    ax.set_xlim([0,0.1])
    ax.set_ylim([-0.1,1.1])
    ax.legend(['$r = $ '+str(x) for x in Rs])
    #ax.legend(['$r = 10\,000$','$r = 15\,000$','$r = 60\,000$'])

    if include_climate:
        save('figures/end_to_end_largerange_withclimate.pdf')
        plt.close()
    else:
        save('figures/end_to_end_largerange.pdf')
        plt.close()

