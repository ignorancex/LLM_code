import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


clinical_data = pd.read_csv("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/ClinicalData/Result_fin.csv",encoding='unicode_escape') 
MRI_location = "/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/_raw_bids_hno/"

labelled_data = pd.read_csv("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/results_JAMA_aug2023/plots/labelled_dataset.csv") 

unlabelled_data_relaxed = pd.read_csv("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/results_JAMA_aug2023/plots/JAMA_aug23_50_percent_threshold.csv")
unlabelled_data_conserv = pd.read_csv("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/results_JAMA_aug2023/plots/JAMA_aug23_conservative.csv")
#include column diseased_label with value 0 
unlabelled_data_relaxed['diseased_label'] = 0
unlabelled_data_conserv['diseased_label'] = 0

sinus_data_relaxed = pd.concat([labelled_data,unlabelled_data_relaxed])
sinus_data_conserv = pd.concat([labelled_data,unlabelled_data_conserv])

# get all the folders in the MRI_location
folders = os.listdir(MRI_location)  
#remove sub- from name of folders
folders = [folder.replace("sub-","") for folder in folders] 


#Get all data in Pseu 
pseu_data = clinical_data['Pseu'] 
#convert to list of strings
pseu_data = pseu_data.astype(str).tolist()


#Find intersection of folders and pseu_data
intersection = list(set(folders) & set(pseu_data)) 


# Select only the rows in clinical_data that have a corresponding folder in intersection 
clinical_data = clinical_data[clinical_data['Pseu'].isin(intersection)] 

#save the filtered clinical_data 

#clinical_data.to_csv("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/ClinicalData/Result_fin_filtered.csv",index=False)






# Convert Leuk, CRP, hsCRP, and HCH_SVBMI0001 to float 
#print("Leuk unique values: ",clinical_data['Leuk'].unique())
#print("CRP unique values: ",clinical_data['CRP'].unique())
#print("hsCRP unique values: ",clinical_data['hsCRP'].unique())
#print("HCH_SVBMI0001 unique values: ",clinical_data['HCH_SVBMI0001'].unique())
#print("klin_pneumo_006 unique values: ",clinical_data['klin_pneumo_006'].unique())
#print("klin_pneumo_008 unique values: ",clinical_data['klin_pneumo_008'].unique())
#print("klin_allerg_030 unique values: ",clinical_data['klin_allerg_030'].unique())
print("ZA unique values: ",clinical_data['ZA'].unique())


#Replace all NP in Leuk with NaN
clinical_data['Leuk'] = clinical_data['Leuk'].replace('NP', np.nan)
#Replace all NP in CRP with NaN
clinical_data['CRP'] = clinical_data['CRP'].replace('NP', np.nan)
#Replace all NP in hsCRP with NaN
clinical_data['hsCRP'] = clinical_data['hsCRP'].replace('NP', np.nan)
#Replace all NP in HCH_SVBMI0001 with NaN
clinical_data['HCH_SVBMI0001'] = clinical_data['HCH_SVBMI0001'].replace('NP', np.nan)
#Replace all NP in klin_pneumo_006 with NaN
clinical_data['klin_pneumo_006'] = clinical_data['klin_pneumo_006'].replace('NP', np.nan)

#Replace all <5 in CRP with 5
clinical_data['CRP'] = clinical_data['CRP'].replace('<5', 5)
#Replace all <0.02 in hsCRP with 0.02
clinical_data['hsCRP'] = clinical_data['hsCRP'].replace('<0.02', 0.02)

#Convert Leuk, CRP, hsCRP, and HCH_SVBMI0001 to float


clinical_data['Leuk'] = clinical_data['Leuk'].astype(float)
clinical_data['CRP'] = clinical_data['CRP'].astype(float)
clinical_data['hsCRP'] = clinical_data['hsCRP'].astype(float)
clinical_data['HCH_SVBMI0001'] = clinical_data['HCH_SVBMI0001'].astype(float)




# keep only Leuk rows that are not NaN  
leuk_data = clinical_data[clinical_data['Leuk'].notna()] 
# keep only CRP rows that are not NaN
CRP_data = clinical_data[clinical_data['CRP'].notna()]
# keep only hsCRP rows that are not NaN
hsCRP_data = clinical_data[clinical_data['hsCRP'].notna()]
# keep only HCH_SVBMI0001 rows that are not NaN
HCH_SVBMI0001_data = clinical_data[clinical_data['HCH_SVBMI0001'].notna()]

# HCH_ANAM0023 
HCH_ANAM0023_data = clinical_data[clinical_data['HCH_ANAM0023'].notna()]
HCH_ANAM0023_data['HCH_ANAM0023'] = HCH_ANAM0023_data['HCH_ANAM0023'].astype(str)
#replace any value that begins with 0 with 0 and any value that begins with 1 with 1 , 2 with 2 and 3 with 3, 9999 with 9999
HCH_ANAM0023_data['HCH_ANAM0023'] = HCH_ANAM0023_data['HCH_ANAM0023'].replace(to_replace ='^0.*', value = 0, regex = True)
HCH_ANAM0023_data['HCH_ANAM0023'] = HCH_ANAM0023_data['HCH_ANAM0023'].replace(to_replace ='^1.*', value = 1, regex = True)
HCH_ANAM0023_data['HCH_ANAM0023'] = HCH_ANAM0023_data['HCH_ANAM0023'].replace(to_replace ='^2.*', value = 2, regex = True)
HCH_ANAM0023_data['HCH_ANAM0023'] = HCH_ANAM0023_data['HCH_ANAM0023'].replace(to_replace ='^3.*', value = 3, regex = True) 
HCH_ANAM0023_data['HCH_ANAM0023'] = HCH_ANAM0023_data['HCH_ANAM0023'].replace(to_replace ='^9999.*', value = 9999, regex = True) 
#remove all rows with 9999
HCH_ANAM0023_data = HCH_ANAM0023_data[HCH_ANAM0023_data['HCH_ANAM0023'] != 9999] 

#convert to int
HCH_ANAM0023_data['HCH_ANAM0023'] = HCH_ANAM0023_data['HCH_ANAM0023'].astype(int) 
# HCH_HAUT0149 
HCH_HAUT0149_data = clinical_data[clinical_data['HCH_HAUT0149'].notna()]
HCH_HAUT0149_data['HCH_HAUT0149'] = HCH_HAUT0149_data['HCH_HAUT0149'].astype(str)
#replace any value that begins with 0 with 0 and any value that begins with 1 with 1 
HCH_HAUT0149_data['HCH_HAUT0149'] = HCH_HAUT0149_data['HCH_HAUT0149'].replace(to_replace ='^0.*', value = 0, regex = True) 
HCH_HAUT0149_data['HCH_HAUT0149'] = HCH_HAUT0149_data['HCH_HAUT0149'].replace(to_replace ='^1.*', value = 1, regex = True)
#convert to int
HCH_HAUT0149_data['HCH_HAUT0149'] = HCH_HAUT0149_data['HCH_HAUT0149'].astype(int)


# keep only klin_hno_006_data rows that are not NaN and convert to int  Only keep rows that have value 0 or 1 in klin_pneumo_006 # No and Yes
klin_hno_006_data = clinical_data[clinical_data['klin_hno_006'].notna()]
klin_hno_006_data['klin_hno_006_data'] = klin_hno_006_data['klin_hno_006'].astype(int) 
klin_hno_006_data = klin_hno_006_data[klin_hno_006_data['klin_hno_006'].isin([0,1])] 


# keep only klin_pneumo_007 rows that are not NaN and convert to int  Only keep rows that have value 0 or 1 in klin_pneumo_006 # No and Yes
klin_pneumo_006_data = clinical_data[clinical_data['klin_pneumo_006'].notna()]
klin_pneumo_006_data['klin_pneumo_006'] = klin_pneumo_006_data['klin_pneumo_006'].astype(int) 
klin_pneumo_006_data = klin_pneumo_006_data[klin_pneumo_006_data['klin_pneumo_006'].isin([0,1])] 


# keep only klin_pneumo_008 rows that are not NaN and convert to int Only keep rows that have value 0 or 1 in klin_pneumo_008 # No and Yes
klin_pneumo_007_data = clinical_data[clinical_data['klin_pneumo_007'].notna()]
klin_pneumo_007_data['klin_pneumo_007'] = klin_pneumo_007_data['klin_pneumo_007'].astype(int)
klin_pneumo_007_data = klin_pneumo_007_data[klin_pneumo_007_data['klin_pneumo_007'].isin([0,1])]

# keep only klin_pneumo_008 rows that are not NaN and convert to int Only keep rows that have value 0 or 1 in klin_pneumo_008 # No and Yes
klin_pneumo_008_data = clinical_data[clinical_data['klin_pneumo_008'].notna()]
klin_pneumo_008_data['klin_pneumo_008'] = klin_pneumo_008_data['klin_pneumo_008'].astype(int)
klin_pneumo_008_data = klin_pneumo_008_data[klin_pneumo_008_data['klin_pneumo_008'].isin([0,1])]

# keep only klin_allerg_029 rows that are not NaN and convert to int Only keep rows that have value 0 or 1 in klin_allerg_029 # No and Yes
klin_allerg_029_data = clinical_data[clinical_data['klin_allerg_029'].notna()]
klin_allerg_029_data['klin_allerg_029'] = klin_allerg_029_data['klin_allerg_029'].astype(int)
klin_allerg_029_data = klin_allerg_029_data[klin_allerg_029_data['klin_allerg_029'].isin([0,1])]

# keep only klin_allerg_030 rows that are not NaN and convert to int Only keep rows 

klin_allerg_030_data = clinical_data[clinical_data['klin_allerg_030'].notna()]
klin_allerg_030_data['klin_allerg_030'] = klin_allerg_030_data['klin_allerg_030'].astype(int)
#remove 9999 from klin_allerg_030
klin_allerg_030_data = klin_allerg_030_data[klin_allerg_030_data['klin_allerg_030'] != 9999] 
klin_allerg_030_data = klin_allerg_030_data[klin_allerg_030_data['klin_allerg_030'] != 8888] 

# keep only klin_allerg_032 rows that are not NaN and convert to int Only keep rows that have value 0 or 1 in klin_allerg_032 # No and Yes
klin_allerg_032_data = clinical_data[clinical_data['klin_allerg_032'].notna()]
klin_allerg_032_data['klin_allerg_032'] = klin_allerg_032_data['klin_allerg_032'].astype(int)
#remove 9999 from klin_allerg_030
klin_allerg_032_data = klin_allerg_032_data[klin_allerg_032_data['klin_allerg_032'] != 9999] 
klin_allerg_032_data = klin_allerg_032_data[klin_allerg_032_data['klin_allerg_032'] != 8888] 
klin_allerg_032_data = klin_allerg_032_data[klin_allerg_032_data['klin_allerg_032'] != -99] 




# keep only HCH_SVSEX0001 rows that are not NaN  and convert to int
HCH_SVSEX0001_data = clinical_data[clinical_data['HCH_SVSEX0001'].notna()]
HCH_SVSEX0001_data['HCH_SVSEX0001'] = HCH_SVSEX0001_data['HCH_SVSEX0001'].astype(int)

# keep only HCH_SVAGE0001 rows that are not NaN  and convert to int
HCH_SVAGE0001_data = clinical_data[clinical_data['HCH_SVAGE0001'].notna()]
HCH_SVAGE0001_data['HCH_SVAGE0001'] = HCH_SVAGE0001_data['HCH_SVAGE0001'].astype(int)


# Find total men using HCH_SVSEX0001_data
HCH_SVSEX0001_data_men = HCH_SVSEX0001_data[HCH_SVSEX0001_data['HCH_SVSEX0001'] == 0]
# Find total women using HCH_SVSEX0001_data
HCH_SVSEX0001_data_women = HCH_SVSEX0001_data[HCH_SVSEX0001_data['HCH_SVSEX0001'] == 1]

print('Total Men in number and percentage', len(HCH_SVSEX0001_data_men), HCH_SVSEX0001_data_men['HCH_SVSEX0001'].count()/len(HCH_SVSEX0001_data)) 
print('Total Women in number and percentage',  len(HCH_SVSEX0001_data_women), HCH_SVSEX0001_data_women['HCH_SVSEX0001'].count()/len(HCH_SVSEX0001_data)) 


# Find Mean and Standard deviation of age using HCH_SVAGE0001_data
print('Mean age', HCH_SVAGE0001_data['HCH_SVAGE0001'].mean())
print('Standard deviation of age', HCH_SVAGE0001_data['HCH_SVAGE0001'].std()) 







 



