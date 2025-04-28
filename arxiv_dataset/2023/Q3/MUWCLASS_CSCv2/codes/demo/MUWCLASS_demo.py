#!/usr/bin/env python
# coding: utf-8
# %%

# # MUWCLASS Pipeline on CSCv2 Demonstration
# ## see our paper at https://arxiv.org/abs/2206.13656
# 
# ### Hui Yang1, Jeremy Hare2, Oleg Kargaltsev1, Igor Volkov1
# ### 1 The George Washington University 2 NASA GSFC
# 
# ### Contact huiyang@gwu.edu if you have any questions
# 
# This notebook presents a demonstration of classifying Chandra Source Catalog v2 (CSCv2) using the MUltiWavelength Machine Learning CLASSification Pipeline with CSCv2 and multiwavelength data
# 
# This notebook is available at https://github.com/huiyang-astro/MUWCLASS_CSCv2
# 
# 
# * This notebook was run in CIAO 4.14 with Python 3.9 
# * run the follow code to create a new conda environment ciao-4.14-muwclass; if you already have ciao-4.14 installed with Python 3.9, you can use your own conda environment with additional Python packages installed from below
# * conda create -n ciao-4.14-muwclass -c https://cxc.cfa.harvard.edu/conda/ciao -c conda-forge ciao sherpa ds9 ciao-contrib caldb_main marx python=3.9
# 
# 
# * run 'bash install-packages.sh' under ciao-4.14-muwclass environment to install all required packages 
# 
# * then, make sure to enable widgetsnbextension and ipyaladin, run
# * jupyter nbextension enable --py widgetsnbextension
# * jupyter nbextension enable --py --sys-prefix ipyaladin on your terminal
# 
# * You might also need to manually register the existing ds9 with the xapns name server by selecting the ds9 File->XPA->Connect menu option so your ds9 will be fully accessible to pyds9.
# 

# ## Related Posters:
# 
# Machine Learning Classification of Variable Galactic X-ray Sources from Chandra Source Catalog
# * https://submissions.mirasmart.com/HEAD19/itinerary/PresentationDetail.aspx?evdid=77
# 
# X-ray Source Populations in NGC 3532 and Other Intermediate Age Star Clusters
# * https://submissions.mirasmart.com/HEAD19/itinerary/PresentationDetail.aspx?evdid=113

# %%


import numpy as np
import pandas as pd

import astropy
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
import astroquery
from astropy.table import QTable
import pyvo as vo
import pyds9 as ds9

import time
from pathlib import Path
import multiprocessing as mp

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, display
import seaborn as sns
import ipyaladin.aladin_widget as ipyal
import holoviews as hv
from bokeh.io import show
from bokeh.layouts import layout
import sklearn 

import sys  
sys.path.insert(0, '../')

from test_library     import  plot_confusion_matrix, plot_Feature_Importance_withSTD, class_train_model_and_classify
from muwclass_library import class_prepare, class_train_and_classify, class_save_res, col_rename, confident_flag, confident_sigma, find_confident, plot_classifier_matrix_withSTD, prepare_cols,class_prepare_oversample
from other_libaray import prepare_field, dict_update, combine_class_result, plot_bbsed, plot_class_matrix, prepare_evts_plot_xray_class, interactive_Ximg_class, CMweight_probability
from prepare_library import CSCview_conesearch


plt.rcParams.update({'font.size': 25})
params = {'legend.fontsize': 'large',
          #'figure.figsize': (15, 5),
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
plt.rcParams.update(params)


get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# %%


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))


# %%


print("Versions:")
print(f"  NumPy       - {np.__version__}")
print(f"  Matplotlib  - {matplotlib.__version__}")
print(f"  AstroPy     - {astropy.__version__}")
print(f"  PyDS9       - {vo.__version__}")
print(f"  Astroquery  - {astroquery.__version__}")
print(f"  Seaborn     - {sns.__version__}")
print(f"  Holoviews     - {hv.__version__}")
print(f"  sklearn       - {sklearn.__version__}")


# ### workflow chart and the training dataset source breakdown of the MUWCLASS pipeline

# %%


display(Image(filename='../../files/figs/flowchart_TD_breakdown.png',width = 600))


# %%





# #### Define a circular region of interest with detections available from CSCv2 (ACIS chips only)
# 
# * field_name
# * RA, DEC, radius: RA (right ascension) and Dec (declination) coordinates and radius (in arcmin) of the field

# %%


field_name = 'NGC3532'

RA = 166.28875

DEC = -58.85

radius = 12 # in arcmin 



# Some other examples of regions of interest 

#field_name = 'Trumpler18'
#RA, DEC,radius = 167.9712649, -60.6184559, 12


# creating the directories for saving the data and results 

data_dir = f'./data/{field_name}' # data directory to save the file

dir_out = f'./{field_name}_results' # classification results directory
dir_plot = dir_out+'/plot' # plots directory
Path(dir_out).mkdir(parents=True, exist_ok=True)
Path(dir_plot).mkdir(parents=True, exist_ok=True)
Path(data_dir).mkdir(parents=True, exist_ok=True)


# #### Cone search -- new version using CSCview_conesearch function
# 
# * search_engine: The database can be accessed using either curl or wget tool. You may run into some bugs when downloading data using curl, then wget is suggested to use. 
# * csc_version: curl can access to different Chandra Source Catalog version, including 2.0 and current version (which includes more data and keeps updating on a daily basis). wget can only access to 2.0 version so the csc_version parameter does not apply to wget 
# 

# %%


search_engine = 'wget'#'wget'#'curl'

# installing wget is required to run with the 'wget' option

csc_version = '2.0'#'2.0' #'current' # '2.0'

df_res = CSCview_conesearch(field_name, RA, DEC, radius,'./data/query',engine=search_engine,csc_version=csc_version,data_dir=data_dir)

#df_res['ra'] = Angle(df_res['ra'], 'hourangle').degree
#df_res['dec'] = Angle(df_res['dec'], 'deg').degree
df_res['usrid'] = np.nan
for i, name in enumerate(df_res['name'].unique()):
    df_res.loc[df_res['name']==name, 'usrid'] = i+1
    
#df_res.to_csv(f'{data_dir}/{field_name}_conesearch.csv',index=False)
# you can filter on the properties of X-ray sources, i.e., X-ray significance if you uncomment the following line

# df_res = df_res[df_res.significance>=3.].reset_index()


# %%


df_res


# #### Cone search -- old version since it can not access to CSC Current Database
# 
# * query with the PyVo package 
# * only significant X-ray sources (significance>=3) are selected for classification, which can be changed to your own purposes

# %%


'''
cone = vo.dal.SCSService('http://cda.cfa.harvard.edu/csc2scs/coneSearch')
field = SkyCoord(RA, DEC, unit="deg") 
maxrad = radius * u.arcmin
results = cone.search(pos=field, radius=maxrad, verbosity=2)
df_res = results.to_table().to_pandas()
df_sig = df_res[df_res.significance>=3.].reset_index()
print(len(df_sig),'sources left with S/N >=3 from',len(df_res), 'sources')
'''


# %%





# * prepare_field function extracts per-observation CSC data, calculates average X-ray fluxes and hardness ratio, X-ray variability from multiple X-ray detections, and applies cross-matching with multiwavelength catalogs (GAIA, 2MASS, WISE)
# 

# %%


df_MW = prepare_field(df_res, data_dir, './data/query', field_name, name_col='name',search_mode='cone_search',engine=search_engine,csc_version=csc_version,create_perobs=False,convert_hms_to_deg=True, gaia_precomputed=True)


# #### read training data , field data and absroption file that is needed to calculate absorption correction 

# %%


TD = pd.read_csv('../../files/CSC_TD_MW_remove.csv')
TD['Class'] = TD['Class'].replace({'NS_BIN':'LMXB'})
TD = prepare_cols(TD, cp_thres=0, TD=True, NS_MWdrop=False, STAR_classremove=['HM-STAR','LM-STAR','YSO']) # some filtering 

field = pd.read_csv(f'{data_dir}/{field_name}_MW_remove.csv')
print(len(field[field.name.isin(TD.name)]), 'field sources are in the TD:')
print(TD.loc[TD.name.isin(field.name), ['name','Class']])
TD.loc[TD.name.isin(field.name), :].to_csv(f'{data_dir}/TD_{field_name}_MW_remove.csv',index=False)
field = field[~field.name.isin(TD.name)].reset_index(drop=True)
field.to_csv(f'{data_dir}/{field_name}_MW_remove.csv',index=False)

field = prepare_cols(field, cp_thres=0, vphas=False,gaiadata=False) # apply with some filterings

# tbabs from xpsec Wilms, Allen & McCray (2000)
tbabs_ene, tbabs_cross = np.loadtxt('../../files/tbabs.data', skiprows=0)
tbabs_cross *= 100


# #### MUWCLASS pipeline settings

# %%


Xcat = 'CSC' # CSCv2 catalog is used 
Uncer_flag = True # applying Monte-Carlo samplings to account the measurement uncertainties 
num_sample = 100 # number of MC samplings, we use 1000 for our classification results in https://arxiv.org/abs/2206.13656, 100 is runnable with a few minutes on a laptop
class_labels = ['AGN','CV','HM-STAR','LM-STAR','HMXB','LMXB','NS','YSO'] # there are 8 classes of X-ray sources in our current pipeline 


# #### Preprocessing, including Monte-Carlo sampling of feature uncertainties, applying field-specific reddening on AGNs from the TD, standardization, oversampling to mitigate imbalance in TD (see Section 3 in https://arxiv.org/abs/2206.13656)

# %%


from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SMOTEN, KMeansSMOTE

oversample_algorithms = {
    'SMOTE': SMOTE( k_neighbors=4, n_jobs=-1), 
    'ADASYN':ADASYN(n_neighbors=4, n_jobs=-1), 
    'BorderlineSMOTE':BorderlineSMOTE(k_neighbors=4,n_jobs=-1), 
    #'KMeansSMOTE':KMeansSMOTE(cluster_balance_threshold=0.01,k_neighbors=4, n_jobs=-1) 
}

oversample_algo = 'SMOTE' # you can add other oversampling algorithms into the list above and change the algorithm that you want to use here


# %%


# this step takes a few minutes, and it increases linearly as you increase the num_sample (number of MC samplings)

num_cores = mp.cpu_count()
pool = mp.Pool(int(num_cores))

get_ipython().run_line_magic('time', "arr = [pool.apply(class_prepare_oversample, args=(TD, field, True, RA, DEC, Xcat, 'nodist', Uncer_flag, i, i, tbabs_ene, tbabs_cross,oversample_algorithms[oversample_algo])) for i in range(num_sample)]")


# #### Our pipeline uses Random Forest (RF) algorithm as the default model. 
# #### In this demo, we offer the flexibilities that users can choose their preferred model available from scikit-learn package. 
# 
# https://scikit-learn.org/stable/supervised_learning.html
# 

# %%


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier

model_dict = {
    'RF': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='sqrt', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=True, random_state=None, verbose=0, warm_start=False),
    'SVC':SVC(gamma='auto',probability=True), 
    'KNeighbor': KNeighborsClassifier(n_neighbors=10),
    'bagging': BaggingClassifier(), 
    'extratrees': ExtraTreesClassifier(n_estimators=300), 
    'GB': GradientBoostingClassifier()
}

model = 'RF' # you can add models into the model list above and change the model that you want to use here


# %%





# #### running the classification (training, applying model to field data and saving classification results )

# %%


get_ipython().run_cell_magic('time', '', "# this step takes a few minutes\n\narr2 = list(zip(arr, [model_dict[model]]*len(arr)))\n\nif __name__ == '__main__':\n    \n    with mp.Pool() as pool:\n        res = pool.map(class_train_model_and_classify, arr2) \n#print(res)\nclass_save_res(res, dir_out)\n")


# #### combining the classification results (MC samplings provide the ability to calculate uncertainties on classification probability) 

# %%


TD_evaluation = pd.read_csv('../../files/LOO_classes.csv')
field_mw_class = combine_class_result(field_name, data_dir, dir_out, class_labels, TD_evaluation,radius)#,weight_CM=True)

field_XCLASS = field_mw_class.rename(columns={'significance':'Signif.'})




# #### select confident classified sources based on Confidence Threshold parameter (see Eq. (7) in https://arxiv.org/abs/2206.13656) for visualization 

# %%


CT_cut = 2 # which can be changed


# #### combining the classification results (MC samplings provide the ability to calculate uncertainties on classification probability) & saving the file for XCLASS plotting

# %%


field_XCLASS.loc[field_XCLASS.CT>=CT_cut,['name','ra','dec','Class','F_b','F_h','F_m','F_s','P_inter','P_intra','Signif.','G','BP','RP','J','H','K','W1','W2','W3']].to_csv(f'{dir_out}/{field_name}_XCLASS.csv')


# ## Visualization of classification results 

# ### MUWCLASS pipeline performance evaluation 

# %%


# confusion matrix
recall_cm = plot_confusion_matrix(pd.read_csv('../../files/LOO_classes.csv'),classes=class_labels,title='Norm. Confusion Matrix (%)',width=450,height=400)
show(recall_cm)

# feature importance
feaimps = pd.read_csv('../../files/feature_importance.csv')
fea_imp = plot_Feature_Importance_withSTD(feaimps.iloc[0], feaimps.iloc[1],feaimps.columns,400,800)


# ### XCLASS interactive plotting website of TD overplotted with field data 
# 
# https://home.gwu.edu/~kargaltsev/XCLASS/
# 
# * click on Choose File and upload {dir_out}/{field_name}_XCLASS.csv to overplot the field data
# 

# #### Visualizing the classification matrix
# The classification matrix shows the probability array (the probability to be classified to 8 classes of X-ray emitters from TD) in each row for each source in the field 

# %%


# the sources to plot the classification matrix can be defined by users

# this example plots the top 2 least confidently classified sources and top 3 most confidently classified sources
df_class_matrix = field_mw_class[field_mw_class.CT>0].sort_values(by=['CT'])
df_class_matrix = pd.concat([df_class_matrix.head(2), df_class_matrix.tail(3)])

plot_class_matrix(field_name, df_class_matrix, dir_plot, class_labels)


# #### The histograms of all and confident classifications 

# %%


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
field_mw_class['Class'].value_counts().plot(kind='bar', ax=ax1, title='All Classification')
field_mw_class[field_mw_class.CT>=CT_cut]['Class'].value_counts().plot(kind='bar', ax=ax2,title='Confident Classification')


# #### multi-band spectrum of TD overplotted with field data (in cyan)
# * set plot_classes the classes of sources that you want to plot the broadband SEDs
# * The fluxes are normalized so that the medium band fluxes are all the same
# * The color indicates the density of the sources from the TD In the multi-band spectrum

# %%


TD.columns


# %%


TD_mw = TD.rename(columns={'Fcsc_s':'F_s','Fcsc_m':'F_m','Fcsc_h':'F_h','flux_aper90_ave_b':'F_b',\
                            'Gmag':'G','BPmag':'BP','RPmag':'RP','Jmag':'J','Hmag':'H','Kmag':'K','W1mag':'W1','W2mag':'W2','W3mag':'W3',\
                            'var_intra_prob':'P_intra', 'var_inter_prob':'P_inter'})


# %%


plot_classes = ['YSO','LMXB','LM-STAR','NS','CV','AGN']
plotlist = [plot_bbsed(TD_mw, field_mw_class, dir_plot, plot_class=c, save_class=[], confidence=True, TD_name_col=False) for c in plot_classes]

layout = hv.Layout(plotlist).cols(3)

layout


# #### prepare the event file and plot X-ray images with classification results overplotted (these images are saved in plot directory)
# 
# ##### a ds9 window will pop up for data processing which can be ignored 

# %%


# this takes a few minutes up to an hour to run depending on how many ObsIDs are used for the field

evt2_data,  fn_evt2 = prepare_evts_plot_xray_class(field_name, RA, DEC, radius, data_dir, dir_out)


# #### plotting multi-band field images with ipyaladin 

# %%


aladin = ipyal.Aladin(target=str(RA)+' '+str(DEC), fov=radius/20)


dtb = QTable.from_pandas(field_mw_class[field_mw_class.conf_flag>0])

aladin


# %%


aladin.add_table(dtb)


# #### Interactive plotting of X-ray image overplotted with classification results 

# %%


interactive_Ximg_class(field_name, evt2_data, fn_evt2, dir_out, RA, DEC, radius,include_TD=True)

#interactive_Ximg_class(src, evt2_data, fn_evt2, src_dir,RA=df_single.loc[0, 'RA_ICRS'],DEC=df_single.loc[0, 'DE_ICRS'], radius=df_single.loc[0, 'PU_TeV'],include_TD=True)
        


# %%





# ## Q&A
# * What kind of user interface would you like for the MUWCLASS pipeline? 
# * Do you have any comments or suggestions that can help us improve the MUWCLASS pipeline?

# %%




