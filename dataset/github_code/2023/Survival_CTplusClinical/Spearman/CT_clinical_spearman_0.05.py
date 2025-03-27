# Import Libraries

import time
import os
import shutil
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import KFold
import random
from torch.utils.data import Subset

from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import ROCAUCMetric
from monai.utils import first, set_determinism
from monai.transforms import (
    Compose,
    Activations,
    AsChannelFirstd,
    EnsureChannelFirstd,
    AddChanneld,
    AsDiscrete,
    Spacingd,
    LoadImaged,
    RandFlipd,
    RandRotated,
    RandZoomd,
    ScaleIntensityRanged,
    Resized,
    Orientationd,
    ToTensord,
    RandAffined,
    RandGaussianNoised
)

from monai.data import Dataset, DataLoader
from monai.utils import set_determinism


from efficientnet_pytorch_3d import EfficientNet3D

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

import torch # For building the networks 
import torchtuples as tt # Some useful functions
import torch.nn as nn
import torch.nn.functional as F

from pycox.models import LogisticHazard, CoxPH, MTLR
from pycox.evaluation import EvalSurv
from pycox.models.loss import NLLLogistiHazardLoss

from sklearn.preprocessing import StandardScaler

from lifelines import KaplanMeierFitter


# Two important lines for reproducibility of results

# In[297]:


np.random.seed(121274)
_ = torch.manual_seed(121274)


#Get the images,tumors based on the order in ISUP folder

#total files before augmentation

# In[298]:


data_dir = '/home/mary/Documents/kidney_ds/ISUP_C'
class_names = sorted([x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))])
num_class = len(class_names)
image_files = [[os.path.join(data_dir, class_name,'image', x) 
                for x in os.listdir(os.path.join(data_dir, class_name, 'image'))] 
               for class_name in class_names]

tumor_files = [[os.path.join(data_dir, class_name,'label', x) 
                for x in os.listdir(os.path.join(data_dir, class_name, 'label'))] 
               for class_name in class_names]

image_file_list = []
tumor_file_list = []
image_label_list = []

for i, class_name in enumerate(class_names):
    
    image_file_list.extend(sorted(image_files[i]))
    tumor_file_list.extend(sorted(tumor_files[i]))
    image_label_list.extend([i] * len(image_files[i]))

    
num_total = len(image_label_list)

print('Total image count:', num_total)
print('Total label count:', len(tumor_file_list))
print("Label names:", class_names)
print("Label counts:", [len(image_files[i]) for i in range(num_class)])
print("Percent of every class:", [int(((len(image_files[i])/num_total)*100)) for i in range(num_class)])


# See the order of patients and classes

# In[299]:


order_of_cases = []
classes = []
for i in image_file_list:
    order_of_cases.append(int(i[58:63]))
    classes.append(int(i[43:44]))


# Making Custom dataset

# In[300]:


class KDataset(Dataset):

    def __init__(self, image_files, tumor_files, labels):
        self.image_files = image_files
        self.tumor_files = tumor_files
        self.labels = labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        im =  self.image_files[index]['image']
        lb = self.tumor_files[index]['label']
        imlb = torch.cat((im, lb),0)        
        return imlb, self.labels[index]


# Monai transformers

# In[301]:


train_transforms = Compose([
    LoadImaged(keys=['image']),
    AddChanneld(keys=['image']),
    Spacingd(keys=['image'], pixdim=(1.5, 1.5, 2)),
    Orientationd(keys=['image'], axcodes="RAS"),
    ScaleIntensityRanged(keys='image', a_min=-200, a_max=500, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=['image'], spatial_size=[128, 128, 128]),
    ToTensord(keys=['image'])
])
     
val_transforms = Compose([
    LoadImaged(keys=['image']),
    AddChanneld(keys=['image']),
    Orientationd(keys=['image'],axcodes="RAS"),
    ScaleIntensityRanged(keys='image', a_min=-200, a_max=500, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=['image'],spatial_size=[128, 128, 128]),
    ToTensord(keys=['image'])
])

lb_transforms = Compose([
    LoadImaged(keys=['label']),
    AddChanneld(keys=['label']),
    Orientationd(keys=['label'],axcodes="RAS"),
    Resized(keys=['label'],spatial_size=[128, 128, 128]),
    ToTensord(keys=['label'])
])


# all files without augmentation

# In[302]:


all_files = [{"image": image_name} for image_name in image_file_list]
all_files_l = [{"label": label_name} for label_name in tumor_file_list]
all_im = Dataset(data=all_files, transform=train_transforms)
all_lb = Dataset(data=all_files_l, transform=lb_transforms)


# In[303]:


all_ds = KDataset(all_im, all_lb, image_label_list)
all_loader = DataLoader(all_ds, batch_size = 1, shuffle = False)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extract survival times, events and clinical columns of patients

# In[304]:


Survival_df = pd.read_csv ('/home/mary/Documents/kidney_ds/survival/merge_clinicals/kits_numberversion.csv')

cl1 = Survival_df['comorbidities/diabetes_mellitus_with_end_organ_damage'].tolist()
cl2 = Survival_df['comorbidities/metastatic_solid_tumor'].tolist()
cl3 = Survival_df['comorbidities/myocardial_infarction'].tolist()
cl4 = Survival_df['chewing_tobacco_use'].tolist()
cl5 = Survival_df['comorbidities/uncomplicated_diabetes_mellitus'].tolist()
cl6 = Survival_df['intraoperative_complications/injury_to_surrounding_organ'].tolist()
cl7 = Survival_df['comorbidities/copd'].tolist()
cl8 = Survival_df['comorbidities/mild_liver_disease'].tolist()
cl9 = Survival_df['comorbidities/hemiplegia_from_stroke'].tolist()
cl10 = Survival_df['comorbidities/malignant_lymphoma'].tolist()
cl11 = Survival_df['pathologic_size'].tolist()
cl12 = Survival_df['comorbidities/congestive_heart_failure'].tolist()
cl13 = Survival_df['radiographic_size'].tolist()

# In[305]:


def surv_times(Survival_df):

    n_get_target = lambda Survival_df: (Survival_df['case_id'], Survival_df['vital_days_after_surgery'].values.astype(int),Survival_df['event'].values.astype(int))
    raw_target = n_get_target(Survival_df)
    return raw_target

Survival_df = pd.read_csv ('/home/mary/Downloads/kits.csv')
raw_target = surv_times(Survival_df)

numbers_p = []
case_ids_label = []
for i,path in enumerate(image_file_list):
    pathn = path[58:63]
    numbers_p.append(pathn)

case_numbers = []
for i,strnumber in enumerate(numbers_p):
    strnumber = int(strnumber)
    case_numbers.append(strnumber)


# We have to find survival time and events in the order of files we have based on ISUP classes
# In[306]:


raw_tar_case = np.array([raw_target[0][i] for i in case_numbers])
raw_tar_st = np.array([raw_target[1][i] for i in case_numbers])
raw_tar_ev = np.array([raw_target[2][i] for i in case_numbers])
cl1d = np.array([cl1[i] for i in case_numbers])
cl2d = np.array([cl2[i] for i in case_numbers])
cl3d = np.array([cl3[i] for i in case_numbers])
cl4d = np.array([cl4[i] for i in case_numbers])
cl5d = np.array([cl5[i] for i in case_numbers])
cl6d = np.array([cl6[i] for i in case_numbers])
cl7d = np.array([cl7[i] for i in case_numbers])
cl8d = np.array([cl8[i] for i in case_numbers])
cl9d = np.array([cl9[i] for i in case_numbers])
cl10d = np.array([cl10[i] for i in case_numbers])
cl11d = np.array([cl11[i] for i in case_numbers])
cl12d = np.array([cl12[i] for i in case_numbers])
cl13d = np.array([cl13[i] for i in case_numbers])

dataf = list(zip(raw_tar_case, raw_tar_st, raw_tar_ev, cl1d, cl2d, cl3d, cl4d, cl5d, cl6d, cl7d, cl8d, cl9d, cl10d, cl11d, cl12d, cl13d))
pp = pd.DataFrame(data = dataf, columns = ['case_id','survival','event', 'cl1d', 'cl2d', 'cl3d', 'cl4d', 'cl5d', 'cl6d', 'cl7d', 'cl8d', 'cl9d', 'cl10d', 'cl11d', 'cl12d', 'cl13d'])
pp.to_csv('kits_label_244_spear0.05.csv',index=False)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Split dataset to train,validation and test

# In[307]:


fold2_indexes = pd.read_csv ('/home/mary/Documents/kidney_ds/survival/fold2_indexes.csv')


# In[308]:


train_index_fold2 = fold2_indexes['train'].tolist()

val_index_fold2 = fold2_indexes['validation'].tolist()
val_index_fold2 = [x for x in val_index_fold2 if np.isnan(x) == False]
val_index_fold2 = [int(x) for x in val_index_fold2]

test_index_fold2 = fold2_indexes['test'].tolist()
test_index_fold2 = [x for x in test_index_fold2 if np.isnan(x) == False]
test_index_fold2 = [int(x) for x in test_index_fold2]


# merging the vlaidation index with train index

# In[309]:


train_val_index_fold2 = sorted(train_index_fold2 + val_index_fold2)


# Survival Analysis

# In[310]:


featuredf = pd.read_csv ('/home/mary/Documents/kidney_ds/survival/244_features_noneaugmented.csv')

df_train = featuredf.loc[train_val_index_fold2]
df_test = featuredf.loc[test_index_fold2]

scaler = StandardScaler()

imfeat_train = scaler.fit_transform(df_train).astype('float32')
imfeat_test = scaler.transform(df_test).astype('float32')


# Extract clinical data from the ordered files we saved before 

# In[311]:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
label_d = pd.read_csv ('kits_label_244_spear0.05.csv')

df_train_target = label_d.loc[train_val_index_fold2]
df_test_target = label_d.loc[test_index_fold2]


#We will standardize the 2 numerical covariates, and leave the binary covariates as is**

# In[312]:


# df_train_target[['age','tumorsize']] = scaler.fit_transform(df_train_target[['age','tumorsize']]).astype('float32')
# df_test_target[['age','tumorsize']] = scaler.transform(df_test_target[['age','tumorsize']]).astype('float32')


# In[313]:


col1_train = (df_train_target['cl1d'].astype('float32')).to_numpy()
col2_train = (df_train_target['cl2d'].astype('float32')).to_numpy()
col3_train = (df_train_target['cl3d'].astype('float32')).to_numpy()
col4_train = (df_train_target['cl4d'].astype('float32')).to_numpy()
col5_train = (df_train_target['cl5d'].astype('float32')).to_numpy()
col6_train = (df_train_target['cl6d'].astype('float32')).to_numpy()
col7_train = (df_train_target['cl7d'].astype('float32')).to_numpy()
col8_train = (df_train_target['cl8d'].astype('float32')).to_numpy()
col9_train = (df_train_target['cl9d'].astype('float32')).to_numpy()
col10_train = (df_train_target['cl10d'].astype('float32')).to_numpy()
col11_train = (df_train_target['cl11d'].astype('float32')).to_numpy()
col12_train = (df_train_target['cl12d'].astype('float32')).to_numpy()
col13_train = (df_train_target['cl13d'].astype('float32')).to_numpy()


col1_test = (df_test_target['cl1d'].astype('float32')).to_numpy()
col2_test = (df_test_target['cl2d'].astype('float32')).to_numpy()
col3_test = (df_test_target['cl3d'].astype('float32')).to_numpy()
col4_test = (df_test_target['cl4d'].astype('float32')).to_numpy()
col5_test = (df_test_target['cl5d'].astype('float32')).to_numpy()
col6_test = (df_test_target['cl6d'].astype('float32')).to_numpy()
col7_test = (df_test_target['cl7d'].astype('float32')).to_numpy()
col8_test = (df_test_target['cl8d'].astype('float32')).to_numpy()
col9_test = (df_test_target['cl9d'].astype('float32')).to_numpy()
col10_test = (df_test_target['cl10d'].astype('float32')).to_numpy()
col11_test = (df_test_target['cl11d'].astype('float32')).to_numpy()
col12_test = (df_test_target['cl12d'].astype('float32')).to_numpy()
col13_test = (df_test_target['cl13d'].astype('float32')).to_numpy()

# In[314]:


col1_train = col1_train.reshape(len(col1_train),1)
col2_train = col2_train.reshape(len(col2_train),1)
col3_train = col3_train.reshape(len(col3_train),1)
col4_train = col4_train.reshape(len(col4_train),1)
col5_train = col5_train.reshape(len(col5_train),1)
col6_train = col6_train.reshape(len(col6_train),1)
col7_train = col7_train.reshape(len(col7_train),1)
col8_train = col8_train.reshape(len(col8_train),1)
col9_train = col9_train.reshape(len(col9_train),1)
col10_train = col10_train.reshape(len(col10_train),1)
col11_train = col11_train.reshape(len(col11_train),1)
col12_train = col12_train.reshape(len(col12_train),1)
col13_train = col13_train.reshape(len(col13_train),1)

col1_test = col1_test.reshape(len(col1_test),1)
col2_test = col2_test.reshape(len(col2_test),1)
col3_test = col3_test.reshape(len(col3_test),1)
col4_test = col4_test.reshape(len(col4_test),1)
col5_test = col5_test.reshape(len(col5_test),1)
col6_test = col6_test.reshape(len(col6_test),1)
col7_test = col7_test.reshape(len(col7_test),1)
col8_test = col8_test.reshape(len(col8_test),1)
col9_test = col9_test.reshape(len(col9_test),1)
col10_test = col10_test.reshape(len(col10_test),1)
col11_test = col11_test.reshape(len(col11_test),1)
col12_test = col12_test.reshape(len(col12_test),1)
col13_test = col13_test.reshape(len(col13_test),1)

# Extract the targets from kits_label_augmented.csv

# Also Make a train dataset that is combination of image feature and clinical dataset**

# In[315]:


get_target = lambda label_d: (label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
n_get_target = lambda label_d: (label_d['case_id'], label_d['survival'].values.astype(int), label_d['event'].values.astype(int))

num_durations = 15
labtrans = LogisticHazard.label_transform(num_durations)

target_train = labtrans.fit_transform(*get_target(df_train_target))
target_test = labtrans.transform(*get_target(df_test_target))

train = tt.tuplefy((imfeat_train, col1_train, col2_train, col3_train, col4_train, col5_train,
                    col6_train, col7_train, col8_train, col9_train, 
                    col10_train, col11_train, col12_train, col13_train), target_train)
test = tt.tuplefy((imfeat_test, col1_test, col2_test, col3_test, col4_test,
                   col5_test, col6_test, col7_test, col8_test,
                   col9_test, col10_test, col11_test, col12_test, col13_test), target_test)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Extract the died people in train and test dataset

# In[316]:


caseid_test, durations_test, events_test = n_get_target(df_test_target)
caseid_train, durations_train, events_train = n_get_target(df_train_target)


# In[317]:


died_person_train = [[],[],[]]
alive_person_train = [[],[],[]]
for i,item in enumerate(events_train):
    if item == 1:
        died_person_train[0].append(i)
        died_person_train[1].append(durations_train[i])
        died_person_train[2].append(list(caseid_train)[i])
    else :
        alive_person_train[0].append(i)
        alive_person_train[1].append(durations_train[i])
        alive_person_train[2].append(list(caseid_train)[i])

died_person_test = [[],[],[]]
alive_person_test = [[],[],[]]
for i,item in enumerate(events_test):
    if item == 1:
        died_person_test[0].append(i)
        died_person_test[1].append(durations_test[i])
        died_person_test[2].append(list(caseid_test)[i])
    else :
        alive_person_test[0].append(i)
        alive_person_test[1].append(durations_test[i])
        alive_person_test[2].append(list(caseid_test)[i])


#Define a net for Logistic hazard that the clinical values would be merged in the middle of network**

# In[318]:

# In[319]:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class survnet_2(torch.nn.Module):
    
    def __init__(self, in_features, out_features):
        
       
        super(survnet_2, self).__init__()

        self.fc1 = torch.nn.Linear(in_features, 1000)
        self.fc2 = torch.nn.Linear(1000+13, 500)
        self.fc3 = torch.nn.Linear(500, 100)
        self.fc4 = torch.nn.Linear(100, out_features)
        
        self.relu = torch.nn.ReLU()
        self.BN1 = torch.nn.BatchNorm1d(1000)
        self.BN2 = torch.nn.BatchNorm1d(500)
        self.BN3 = torch.nn.BatchNorm1d(100)

        self.dropout = torch.nn.Dropout(0.1)
        
    
    def forward(self, imfeat_train, col1_train, col2_train, col3_train, 
                col4_train, col5_train, col6_train, col7_train, col8_train,
                col9_train, col10_train, col11_train, col12_train, col13_train):
        
        x1 = self.fc1(imfeat_train)
        
        x2 = self.relu(x1)
        
        x3 = self.BN1(x2)
        
        x4 = self.dropout(x3)
        
        merged_data = torch.cat((x4, col1_train, col2_train, col3_train, col4_train,
                                 col5_train, col6_train, col7_train, col8_train,
                                 col9_train, col10_train, col11_train, col12_train, col13_train), dim=1)
        
        x5 = self.fc2(merged_data)
        
        x6 = self.relu(x5)
        
        x7 = self.BN2(x6)
        
        x8 = self.dropout(x7)
        
        x9 = self.fc3(x8)
        
        x10 = self.relu(x9)
        
        x11 = self.BN3(x10)
        
        x12 = self.dropout(x11)
        
        x13 = self.fc4(x12)
        
        return x13

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# In[917]:


in_features = imfeat_train.shape[1]
out_features = labtrans.out_features
net = survnet_2(in_features, out_features)
model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
# model_MTLR = MTLR(net, tt.optim.Adam(0.01), duration_index=labtrans_MTLR.cuts)


#Training

# In[903]:

# batch_size = 100
# epochs = 100
# callbacks = [tt.cb.EarlyStopping(patience=20)]
# # log = model.fit(*train, batch_size, epochs, None , True)
# log = model.fit(*train, batch_size, epochs, callbacks, True, val_data = test)


# In[911]:


# model.save_model_weights('merged_spearman0.1_79CIuuuu.pt')


# In[918]:


model.load_model_weights('merged_spearman0.05_83CI.pt')


# In[919]:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_train = (imfeat_train, col1_train, col2_train, col3_train, col4_train,
           col5_train, col6_train, col7_train, col8_train,
           col9_train, col10_train, col11_train, col12_train, col13_train)
x_test = (imfeat_test, col1_test, col2_test, col3_test, col4_test,
          col5_test, col6_test, col7_test, col8_test,
          col9_test, col10_test, col11_test, col12_test, col13_test)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
surv_train_disc = model.predict_surv_df(x_train)
surv_test_disc = model.predict_surv_df(x_test)



# In[920]:


surv_test_cont = model.interpolate(40).predict_surv_df(x_test)
surv_train_cont = model.interpolate(40).predict_surv_df(x_train)


#Indexes for evaluating survival model

#  C-index
# In[921]:


ev = EvalSurv(surv_test_cont, durations_test, events_test, censor_surv='km')
ctd = ev.concordance_td('antolini')
print(f'C_td Score for test: {ctd}')

ev_train = EvalSurv(surv_train_cont, durations_train, events_train, censor_surv='km')
ctd_train = ev_train.concordance_td('antolini')
print(f'C_td Score for train: {ctd_train}')


#AUC
# In[922]:


import sksurv
from sksurv.metrics import cumulative_dynamic_auc
start_t = min(durations_test)
end_t = max(durations_test)
times = np.arange(start_t, end_t, ((end_t-start_t)/(model.predict(x_test).squeeze().shape[1])))

# start by defining a function to transform 1D arrays to the 
# structured arrays needed by sksurv
def transform_to_struct_array(times, events):
    return sksurv.util.Surv.from_arrays(events, times)

# then call the AUC metric from sksurv
AUC_metric = cumulative_dynamic_auc(
    transform_to_struct_array(durations_train, events_train), 
    transform_to_struct_array(durations_test, events_test),
    model.predict(x_test).squeeze(),
    times)

print(f'AUC metric is:{AUC_metric[1]}')

#Integerated Brier Score
# In[908]:


time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)

IBS = ev.integrated_brier_score(time_grid)
IBS_train = ev_train.integrated_brier_score(time_grid)

print(f'IBS Score for test: {IBS}')
print(f'IBS_train Score for train: {IBS_train}')


#Violin Plot

# In[923]:


data_1 = np.asarray(surv_test_disc.iloc[-1, died_person_test[0][:]])
data_2 = np.asarray(surv_test_disc.iloc[-1, alive_person_test[0][:]])
data_3 = np.asarray(surv_train_disc.iloc[-1, died_person_train[0][:]])
data_4 = np.asarray(surv_train_disc.iloc[-1, alive_person_train[0][:]])

data = list([data_1, data_2, data_3, data_4])
fig, ax = plt.subplots()
ax.violinplot(data, showmeans=True, showmedians=False)
ax.set_title('violin graph')
ax.set_xlabel('Different Kinds of Dataset')
ax.set_ylabel('Survival Probabilities')
xticklabels = ['Dead_Test', 'Censored_Test', 'Dead_Train', 'Censored_Train']
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(xticklabels)
ax.yaxis.grid(True)
red = '#B60D0D'
blue = '#23BDC4'
green = '#23C47B'
khaki = '#D2A03D'
violet = '#B86FCD'

violin_parts = ax.violinplot(data, showmeans=True, showmedians=False)

i=0
for vp in violin_parts['bodies']:
    if i==0:
        vp.set_facecolor(blue)
        vp.set_edgecolor(red)
        vp.set_linewidth(1)
        vp.set_alpha(0.5)
    
    if i ==1:
        vp.set_facecolor(green)
        vp.set_edgecolor(red)
        vp.set_linewidth(1)
        vp.set_alpha(0.5)
        
    if i ==2:
        vp.set_facecolor(violet)
        vp.set_edgecolor(red)
        vp.set_linewidth(1)
        vp.set_alpha(0.5)
    
    if i ==3:
        vp.set_facecolor(khaki)
        vp.set_edgecolor(red)
        vp.set_linewidth(1)
        vp.set_alpha(0.5)
    i+=1    
plt.savefig('Violin_plot.eps', format='eps')


#Curves for dead people in test dataset

# In[910]:


surv_test_cont.iloc[:, died_person_test[0][:]].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')


# In[244]:

#Curves for censored people in test dataset

# In[455]:


# surv_test_cont.iloc[:, alive_person_test[0][:]].plot(drawstyle='steps-post')
# plt.ylabel('S(t | x)')
# _ = plt.xlabel('Time')





