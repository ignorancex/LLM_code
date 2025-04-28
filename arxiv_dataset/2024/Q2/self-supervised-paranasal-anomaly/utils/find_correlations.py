import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


clinical_data = pd.read_csv("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/ClinicalData/Result_fin.csv",encoding='unicode_escape') 
MRI_location = "/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/_raw_bids_hno/"

labelled_data = pd.read_csv("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/results_JAMA_aug2023/plots_oct2023/labelled_dataset.csv") 

unlabelled_data_relaxed = pd.read_csv("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/results_JAMA_aug2023/plots_oct2023/JAMA_aug23_50_percent_threshold.csv")
unlabelled_data_conserv = pd.read_csv("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/results_JAMA_aug2023/plots_oct2023/JAMA_aug23_conservative.csv")
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
HCH_ANAM0023_data = HCH_ANAM0023_data[HCH_ANAM0023_data['HCH_ANAM0023'] != 2] 
#make all 2 values as 3 
#HCH_ANAM0023_data['HCH_ANAM0023'] = HCH_ANAM0023_data['HCH_ANAM0023'].replace(to_replace =2, value = 3, regex = True)
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

# keep only HCH_SVISCED_02 rows that are not NaN  and convert to int
HCH_SVISCED_02_data = clinical_data[clinical_data['HCH_SVISCED_02'].notna()]
HCH_SVISCED_02_data['HCH_SVISCED_02'] = HCH_SVISCED_02_data['HCH_SVISCED_02'].astype(int) 

# keep only HCH_SVBMI0001 rows that are not NaN  and convert to float
HCH_SVBMI0001_data = clinical_data[clinical_data['HCH_SVBMI0001'].notna()]
HCH_SVBMI0001_data['HCH_SVBMI0001'] = HCH_SVBMI0001_data['HCH_SVBMI0001'].astype(float)

# keep only ZA rows that are not NaN  and convert to float
ZA_data = clinical_data[clinical_data['ZA'].notna()]
# remove 'np' from ZA 
ZA_data = ZA_data[ZA_data['ZA'] != 'np'] 
ZA_data['ZA'] = ZA_data['ZA'].astype(float)
print("ZA_data unique values: ",ZA_data['ZA'].unique())

print("HCH_ANAM0023 data shape: ",HCH_ANAM0023_data.shape,clinical_data.shape) # Do you smoke  1 - Yes, 2 - Not smoked for 6 months , 3 - never smoked
print("HCH_HAUT0149_data data shape: ",HCH_HAUT0149_data.shape,clinical_data.shape) # Neurodermitis 0 - No, 1 - Yes 
print("klin_hno_006_data data shape: ",klin_hno_006_data.shape,clinical_data.shape) # Were you operated for HNO? 0 - No, 1 - Yes
print("klin_pneumo_006 data shape: ",klin_pneumo_006_data.shape,clinical_data.shape)
print("klin_pneumo_007 data shape: ",klin_pneumo_007_data.shape,clinical_data.shape)
print("klin_pneumo_008 data shape: ",klin_pneumo_008_data.shape,clinical_data.shape) #Bronchitis or COPD

print("klin_allerg_029 data shape: ",klin_allerg_029_data.shape,clinical_data.shape) #Allergy 1 question
print("klin_allerg_030 data shape: ",klin_allerg_030_data.shape,clinical_data.shape) #Allergy 2 question
print("klin_allerg_032 data shape: ",klin_allerg_032_data.shape,clinical_data.shape) #Allergy 3 question

print("Leuk data shape: ",leuk_data.shape,clinical_data.shape)
print("CRP data shape: ",CRP_data.shape,clinical_data.shape)
print("hsCRP data shape: ",hsCRP_data.shape,clinical_data.shape)


print("HCH_SVSEX0001 data shape: ",HCH_SVSEX0001_data.shape,clinical_data.shape) #Sex
print("HCH_SVAGE0001 data shape: ",HCH_SVAGE0001_data.shape,clinical_data.shape)  # Age
print("HCH_SVISCED_02 data shape: ",HCH_SVISCED_02_data.shape,clinical_data.shape) #Education 
print("HCH_SVBMI0001 data shape: ",HCH_SVBMI0001_data.shape,clinical_data.shape) #BMI
print("ZA data shape: ",ZA_data.shape,clinical_data.shape) #Alcohol consumption per day


def find_correlation(clinical_data, sinus_data,variable_name=None,map_variable_value=None, plot_type= [], test_type = [], result_file_name = None , plot_title = None):

    if variable_name is None:
        # assert error 
        assert variable_name is not None, "variable_name is None"


    if map_variable_value is not None:

        #change variable value based on map_variable_value dictionary 
        for key in map_variable_value.keys():
            clinical_data[variable_name] = clinical_data[variable_name].replace(key,map_variable_value[key]) 

    
    # remove sub- from patient_id in sinus_data
    sinus_data['patient_id'] = sinus_data['patient_id'].str.replace('sub-','') 
    # print length of unique patient_id in sinus_data
    print('Unique Patients',len(set(sinus_data['patient_id'])))
    
    # create a dataframe with patient_id having mean >0.5 and smax = smax_r from sinus data
    sinus_data_smaxr = sinus_data[(sinus_data['mean'] > 0.5) & (sinus_data['smax'] == 'smax_r')] 
    sinus_data_smaxl = sinus_data[(sinus_data['mean'] > 0.5) & (sinus_data['smax'] == 'smax_l')] 

    # choose intersection of patient_id from sinus_data_smaxr and sinus_data_smaxl based on set intersection patient_id
    patient_id_smaxr = set(sinus_data_smaxr['patient_id']) 
    patient_id_smaxl = set(sinus_data_smaxl['patient_id'])
    anomaly_patient_id_smaxr_smaxl = patient_id_smaxr.intersection(patient_id_smaxl)

    # choose patient_id only present in sinus_data_smaxr and not in sinus_data_smaxl
    anomaly_patient_id_smaxr_only = patient_id_smaxr - anomaly_patient_id_smaxr_smaxl 

    # choose patient_id only present in sinus_data_smaxl and not in sinus_data_smaxr
    anomaly_patient_id_smaxl_only = patient_id_smaxl - anomaly_patient_id_smaxr_smaxl

    # choose patient_id with no anomaly
    normal_patient_id = set(sinus_data['patient_id']) - anomaly_patient_id_smaxr_only - anomaly_patient_id_smaxl_only - anomaly_patient_id_smaxr_smaxl

    # choose patient_id with anomaly
    anomaly_patient_id = anomaly_patient_id_smaxr_only | anomaly_patient_id_smaxl_only | anomaly_patient_id_smaxr_smaxl 

    
    print("anomaly_patient_id_smaxr_smaxl: ",len(set(anomaly_patient_id_smaxr_smaxl)))
    print("anomaly_patient_id_smaxr_only: ",len(set(anomaly_patient_id_smaxr_only)))
    print("anomaly_patient_id_smaxl_only: ",len(set(anomaly_patient_id_smaxl_only)))
    print("normal_patient_id: ",len(set(normal_patient_id)))
    print("anomaly_patient_id: ",len(set(anomaly_patient_id)))

    control_group = clinical_data[clinical_data['Pseu'].isin(normal_patient_id)] 
    treatment_group = clinical_data[clinical_data['Pseu'].isin(anomaly_patient_id)]

    control_group_variable = control_group[variable_name]
    treatment_group_variable = treatment_group[variable_name] 

    #create dataframe with value and category 
    control_group_variable_df = pd.DataFrame({'value':control_group_variable,'category':['Control']*len(control_group_variable)})
    treatment_group_variable_df = pd.DataFrame({'value':treatment_group_variable,'category':['Treatment']*len(treatment_group_variable)}) 

    #concatenate control_group_variable_df and treatment_group_variable_df
    df = pd.concat([control_group_variable_df,treatment_group_variable_df],axis=0) 

    # calculate total number of patients in control group and percentage of patients in control group
    total_patients_control_group = len(control_group_variable)
    percentage_patients_control_group = len(control_group_variable)/len(clinical_data) * 100

    # calculate total number of patients in treatment group and percentage of patients in treatment group
    total_patients_treatment_group = len(treatment_group_variable)
    percentage_patients_treatment_group = len(treatment_group_variable)/len(clinical_data) * 100 

    #calculate percentage with respect to total number of control and treatment group patients
    percentage_patients_control_group_total = len(control_group_variable)/(len(control_group_variable)+len(treatment_group_variable)) * 100 
    percentage_patients_treatment_group_total = len(treatment_group_variable)/(len(control_group_variable)+len(treatment_group_variable)) * 100
    control_group_variable_unique_values = None 
    treatment_group_variable_unique_values = None
    #check if control_group_variable is numeric
    if control_group_variable.dtype == 'int64' or control_group_variable.dtype == 'float64':
    # calculate 95% confidence interval for control group value 
    
        control_group_variable_mean = control_group_variable.mean()
        control_group_variable_std = control_group_variable.std()
        control_group_variable_95_confidence_interval = [control_group_variable_mean - 1.96*control_group_variable_std,control_group_variable_mean + 1.96*control_group_variable_std] 
        control_group_variable_95_confidence_interval = ','.join([str(round(x,2)) for x in control_group_variable_95_confidence_interval])
    else:
        control_group_variable_95_confidence_interval = 'NA' 
        control_group_variable_mean = 'NA'
        control_group_variable_std = 'NA'

        #count of unique values in control_group_variable 
        control_group_variable_unique_values = control_group_variable.value_counts().to_dict() 

        # percentage count of unique values in control_group_variable with respect to total number in control group
        control_group_variable_unique_values_percentage = {key:round(value/total_patients_control_group*100,2) for key,value in control_group_variable_unique_values.items()}

        

    #check if treatment_group_variable is numeric
    if treatment_group_variable.dtype == 'int64' or treatment_group_variable.dtype == 'float64':
    # calculate 95% confidence interval for treatment group value
        treatment_group_variable_mean = treatment_group_variable.mean()
        treatment_group_variable_std = treatment_group_variable.std()
        treatment_group_variable_95_confidence_interval = [treatment_group_variable_mean - 1.96*treatment_group_variable_std,treatment_group_variable_mean + 1.96*treatment_group_variable_std]
        treatment_group_variable_95_confidence_interval = ','.join([str(round(x,2)) for x in treatment_group_variable_95_confidence_interval])
        #make it string separated by comma
    else:
        treatment_group_variable_95_confidence_interval = 'NA'
        treatment_group_variable_mean = 'NA'
        treatment_group_variable_std = 'NA'

        #count of unique values in treatment_group_variable
        treatment_group_variable_unique_values = treatment_group_variable.value_counts().to_dict()

        # percentage count of unique values in treatment_group_variable with respect to total number in treatment group
        treatment_group_variable_unique_values_percentage = {key:round(value/total_patients_treatment_group*100,2) for key,value in treatment_group_variable_unique_values.items()}
        


    #save all this information in a text file
    result_file_name = variable_name if result_file_name is None else result_file_name

    # save as a text file 

    with open(f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/results_JAMA_aug2023/statistical_tests_oct2023/{result_file_name}_meta.txt", "w") as text_file:

        text_file.write(f"Total number of patients in control group: {total_patients_control_group}\n")
        text_file.write(f"Percentage of patients in control group: {percentage_patients_control_group}\n")
        text_file.write(f"Total number of patients in treatment group: {total_patients_treatment_group}\n")
        text_file.write(f"Percentage of patients in treatment group: {percentage_patients_treatment_group}\n")
        text_file.write(f"Percentage of patients in control group with respect to total number of control + treatment : {percentage_patients_control_group_total}\n")
        text_file.write(f"Percentage of patients in treatment group with respect to total number of  control + treatment : {percentage_patients_treatment_group_total}\n")
        text_file.write(f"control group value mean: {control_group_variable_mean}\n")
        text_file.write(f"treatment group value mean: {treatment_group_variable_mean}\n")
        text_file.write(f"control group value std: {control_group_variable_std}\n")
        text_file.write(f"treatment group value std: {treatment_group_variable_std}\n")
        text_file.write(f"95% confidence interval for control group value: {control_group_variable_95_confidence_interval}\n")
        text_file.write(f"95% confidence interval for treatment group value: {treatment_group_variable_95_confidence_interval}\n")

        if control_group_variable_unique_values is not None:
            text_file.write(f"control group value unique values: {control_group_variable_unique_values}\n")

            text_file.write(f"control group value unique values percentage: {control_group_variable_unique_values_percentage}\n")

        if treatment_group_variable_unique_values is not None:
            text_file.write(f"treatment group value unique values: {treatment_group_variable_unique_values}\n")

            text_file.write(f"treatment group value unique values percentage: {treatment_group_variable_unique_values_percentage}\n")

        

    #create a dataframe with all the information    
    if 'piechart' in plot_type:
        # increase figure size
        
        #draw pie chart for control group with number of patients in each category, and percentage of patients in each category and draw pie chart for treatment group with number of patients in each category, and percentage of patients in each category side by side
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        title = 'Pie chart for control group and treatment group' if plot_title is None else plot_title
        fig.suptitle(title)

        # check if yes and no in control_group_variable and treatment_group_variable 
        if 'Female' in control_group_variable.value_counts().index and 'Male' in control_group_variable.value_counts().index:
            colors = {'Female':'tab:blue', 'Male':'tab:orange'}
            ax1.pie(control_group_variable.value_counts(),colors=[colors[v] for v in control_group_variable.value_counts().keys()], labels=control_group_variable.value_counts().index,autopct='%1.1f%%')
            #add set title for ax1 with line break
            ax1.set_title('No incidental finding in \n LMS or/and RMS')

            #print([colors[v] for v in control_group_variable.value_counts().keys()],control_group_variable.value_counts().keys())
            
            ax2.pie(treatment_group_variable.value_counts(),colors=[colors[v] for v in control_group_variable.value_counts().keys()],labels=control_group_variable.value_counts().index,autopct='%1.1f%%')
            ax2.set_title('Incidental finding in \n LMS or/and RMS')

            #print([colors[v] for v in treatment_group_variable.value_counts().keys()],control_group_variable.value_counts().keys())

        
        else:
            
        

            ax1.pie(control_group_variable.value_counts(), labels=control_group_variable.value_counts().index,autopct='%1.1f%%')
            #add set title for ax1 with line break
            ax1.set_title('No incidental finding in \n LMS or/and RMS')
            
            ax2.pie(treatment_group_variable.value_counts(),labels=treatment_group_variable.value_counts().index,autopct='%1.1f%%')
            ax2.set_title('Incidental finding in \n LMS or/and RMS')
            # Mention patient count in each category in control group and treatment group 

        ax1.text(-0.5, -1.5, 'Patient count: '+str(len(control_group_variable)), style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        ax2.text(-0.5, -1.5, 'Patient count: '+str(len(treatment_group_variable)), style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        

        #save figure with 500 dpi
        plt.savefig( f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/results_JAMA_aug2023/statistical_tests_oct2023/plots_oct2023/{result_file_name}_pie.pdf",bbox_inches='tight',dpi=500) 
        plt.close()
        plt.clf()
    if 'boxplot' in plot_type:

        #draw boxplot seaborn
        title = 'Boxplot for control group and treatment group' if plot_title is None else plot_title
        ax = sns.boxplot(x="category", y="value", data=df)
        ax.set_title(title)
        #x axis label
        ax.set_xlabel('Category')
        #y axis label
        if variable_name == 'Leuk':
            ax.set_ylabel('Leukocytes/μl')
        elif variable_name == 'CRP':
            ax.set_ylabel('CRP (mg/L)')
        elif variable_name == 'ZA':
            ax.set_ylabel('Alcohol Comsumption (g/day)')
        else:
            ax.set_ylabel(variable_name)
        #save figure with 500 dpi
        plt.savefig( f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/results_JAMA_aug2023/statistical_tests_oct2023/plots_oct2023/{result_file_name}_boxplot.pdf",bbox_inches='tight',dpi=500)
        plt.close()
        plt.clf()

        # no outlier boxplot

        ax = sns.boxplot(x="category", y="value", data=df,showfliers=False) 
        ax.set_title(title)
        #x axis label
        ax.set_xlabel('Category')
        #y axis label
        if variable_name == 'Leuk':
            ax.set_ylabel('Leukocytes/μl')
        elif variable_name == 'CRP':
            ax.set_ylabel('CRP (mg/L)')
        elif variable_name == 'HCH_SVAGE0001':
            ax.set_ylabel('Age (years)')
        elif variable_name == 'ZA':
            ax.set_ylabel('Alcohol Comsumption (g/day)')

        else:
            ax.set_ylabel(variable_name)
        #save figure with 500 dpi
        plt.savefig( f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/results_JAMA_aug2023/statistical_tests_oct2023/plots_oct2023/{result_file_name}_boxplot_no_outliers.pdf",bbox_inches='tight',dpi=500)
        plt.close()
        plt.clf()

    if 'chi_square' in test_type:
        from scipy.stats import chi2_contingency
        print(pd.crosstab(df['category'],df['value']))
        #chi square test
        chi2, p, dof, ex = chi2_contingency(pd.crosstab(df['category'],df['value'])) 

        print(f"Chi-squared value: {chi2}")
        print(f"P-value: {p}")
        print(f"Degrees of freedom: {dof}")
        print("Expected values: \n", ex)

        # save chi2, p, dof, ex in a dataframe and save it in a csv file
        chi2_p_dof_ex_df = pd.DataFrame({'chi2':[chi2],'p':[p],'dof':[dof],'ex':[ex]}) 
        chi2_p_dof_ex_df.to_csv(f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/results_JAMA_aug2023/statistical_tests_oct2023/files/{result_file_name}_chi.csv",index=False)
        #plt.show()

    if 'pointbiserialr' in test_type:

        from scipy.stats import pointbiserialr 
        #replace category value of control and treatment with 0 and 1 respectively
        df['category'] = df['category'].replace(['Control','Treatment'],[0,1])
        

        correlation_coefficient, p = pointbiserialr(df['category'],df['value']) 
        

        print(f"Correlation coefficient: {correlation_coefficient}")
        print(f"P-value: {p}")

        # save correlation_coefficient, p in a dataframe and save it in a csv file
        correlation_coefficient_p_df = pd.DataFrame({'correlation_coefficient':[correlation_coefficient],'p':[p]}) 
        correlation_coefficient_p_df.to_csv(f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/results_JAMA_aug2023/statistical_tests_oct2023/files/{result_file_name}_pointbiserial.csv",index=False)
    
    

    # find patient_ids only present in sinus_data_smaxr and not in sinus_data_smaxl


    
find_correlation(HCH_ANAM0023_data, sinus_data_relaxed,variable_name='HCH_ANAM0023',
                 map_variable_value={1:'Yes',2:'Not smoked for 6 months',3: 'Never smoked'},
                 plot_type=['piechart'],test_type=['chi_square'],
                 result_file_name='HCH_ANAM0023_yes_no_conserv', plot_title='Do you smoke?') 


"""
find_correlation(HCH_HAUT0149_data, sinus_data_relaxed,variable_name='HCH_HAUT0149',
                 map_variable_value={0:'No',1:'Yes'},
                 plot_type=['piechart'],test_type=['chi_square'],
                 result_file_name='HCH_HAUT0149', plot_title='Do you have neurodermitis?')

find_correlation(klin_hno_006_data, sinus_data_relaxed,variable_name='klin_hno_006',
                    map_variable_value={0:'No',1:'Yes'},
                    plot_type=['piechart'],test_type=['chi_square'],
                    result_file_name='klin_hno_006', plot_title='Have you ever had an operation in the ear, nose and throat area?')

find_correlation(klin_pneumo_006_data, sinus_data_relaxed,variable_name='klin_pneumo_006',
                    map_variable_value={0:'No',1:'Yes'},
                    plot_type=['piechart'],test_type=['chi_square'],
                    result_file_name='klin_pneumo_006', plot_title='Have you had a cough and sputum that lasted longer than 3 months for 2 consecutive years?')

find_correlation(klin_pneumo_007_data, sinus_data_relaxed,variable_name='klin_pneumo_007',
                    map_variable_value={0:'No',1:'Yes'},
                    plot_type=['piechart'],test_type=['chi_square'],
                    result_file_name='klin_pneumo_007', plot_title='Have you ever been diagnosed with bronchial asthma by a doctor?')

find_correlation(klin_pneumo_008_data, sinus_data_relaxed,variable_name='klin_pneumo_008',
                    map_variable_value={0:'No',1:'Yes'},
                    plot_type=['piechart'],test_type=['chi_square'],
                    result_file_name='klin_pneumo_008', plot_title='Have you ever been diagnosed with chronic bronchitis or COPD by a doctor?')


find_correlation(klin_allerg_029_data, sinus_data_relaxed,variable_name='klin_allerg_029',
                    map_variable_value={0:'No',1:'Yes'},
                    plot_type=['piechart'],test_type=['chi_square'],
                    result_file_name='klin_allerg_029', plot_title='Have you ever been diagnosed with an allergy by a doctor?')


find_correlation(leuk_data, sinus_data_relaxed,variable_name='Leuk',
                    plot_type=['boxplot'],test_type=['pointbiserialr'],
                    result_file_name='leuk', plot_title='Leukocyte count per microliter')

find_correlation(CRP_data, sinus_data_relaxed,variable_name='CRP',
                    plot_type=['boxplot'],test_type=['pointbiserialr'],
                    result_file_name='CRP', plot_title='C-reactive protein in mg/l')

find_correlation(hsCRP_data, sinus_data_relaxed,variable_name='hsCRP',
                    plot_type=['boxplot'],test_type=['pointbiserialr'],
                    result_file_name='hsCRP', plot_title='High-sensitivity C-reactive protein in mg/l')

find_correlation(HCH_SVAGE0001_data, sinus_data_relaxed,variable_name='HCH_SVAGE0001',
                    plot_type=['boxplot'],test_type=['pointbiserialr'],
                    result_file_name='HCH_SVAGE0001', plot_title='Is there relationship between age and incidental findings?')



find_correlation(HCH_SVSEX0001_data, sinus_data_relaxed,variable_name='HCH_SVSEX0001',
                 map_variable_value={0:'Male',1:'Female'},
                    plot_type=['piechart'],test_type=['chi_square'],
                    result_file_name='HCH_SVSEX0001', plot_title='Is there relationship between sex and incidental findings?')
                 

find_correlation(HCH_SVBMI0001_data, sinus_data_relaxed,variable_name='HCH_SVBMI0001',
                    plot_type=['boxplot'],test_type=['pointbiserialr'],
                    result_file_name='HCH_SVBMI0001', plot_title='Body mass index')


find_correlation(ZA_data, sinus_data_relaxed,variable_name='ZA',
                    plot_type=['boxplot'],test_type=['pointbiserialr'],
                    result_file_name='ZA', plot_title='Is there relationship between alcohol consumption and incidental findings?')

                 





find_correlation(HCH_ANAM0023_data, sinus_data_conserv,variable_name='HCH_ANAM0023',
                 map_variable_value={1:'Yes',2:'Not smoked for 6 months',3: 'Never smoked'},
                 plot_type=['piechart'],test_type=['chi_square'],
                 result_file_name='HCH_ANAM0023_conserv', plot_title='Do you smoke?') 



find_correlation(HCH_HAUT0149_data, sinus_data_conserv,variable_name='HCH_HAUT0149',
                 map_variable_value={0:'No',1:'Yes'},
                 plot_type=['piechart'],test_type=['chi_square'],
                 result_file_name='HCH_HAUT0149_conserv', plot_title='Do you have neurodermitis?')

find_correlation(klin_hno_006_data, sinus_data_conserv,variable_name='klin_hno_006',
                    map_variable_value={0:'No',1:'Yes'},
                    plot_type=['piechart'],test_type=['chi_square'],
                    result_file_name='klin_hno_006_conserv', plot_title='Have you ever had an operation in the ear, nose and throat area?')

find_correlation(klin_pneumo_006_data, sinus_data_conserv,variable_name='klin_pneumo_006',
                    map_variable_value={0:'No',1:'Yes'},
                    plot_type=['piechart'],test_type=['chi_square'],
                    result_file_name='klin_pneumo_006_conserv', plot_title='Have you had a cough and sputum that lasted longer than 3 months for 2 consecutive years?')

find_correlation(klin_pneumo_007_data, sinus_data_conserv,variable_name='klin_pneumo_007',
                    map_variable_value={0:'No',1:'Yes'},
                    plot_type=['piechart'],test_type=['chi_square'],
                    result_file_name='klin_pneumo_007_conserv', plot_title='Have you ever been diagnosed with bronchial asthma by a doctor?')

find_correlation(klin_pneumo_008_data, sinus_data_conserv,variable_name='klin_pneumo_008',
                    map_variable_value={0:'No',1:'Yes'},
                    plot_type=['piechart'],test_type=['chi_square'],
                    result_file_name='klin_pneumo_008_conserv', plot_title='Have you ever been diagnosed with chronic bronchitis or COPD by a doctor?')


find_correlation(klin_allerg_029_data, sinus_data_conserv,variable_name='klin_allerg_029',
                    map_variable_value={0:'No',1:'Yes'},
                    plot_type=['piechart'],test_type=['chi_square'],
                    result_file_name='klin_allerg_029_conserv', plot_title='Have you ever been diagnosed with an allergy by a doctor?')


find_correlation(leuk_data, sinus_data_conserv,variable_name='Leuk',
                    plot_type=['boxplot'],test_type=['pointbiserialr'],
                    result_file_name='leuk_conserv', plot_title='Leukocyte count per microliter')

find_correlation(CRP_data, sinus_data_conserv,variable_name='CRP',
                    plot_type=['boxplot'],test_type=['pointbiserialr'],
                    result_file_name='CRP_conserv', plot_title='C-reactive protein in mg/l')

find_correlation(hsCRP_data, sinus_data_conserv,variable_name='hsCRP',
                    plot_type=['boxplot'],test_type=['pointbiserialr'],
                    result_file_name='hsCRP_conserv', plot_title='High-sensitivity C-reactive protein in mg/l')

find_correlation(HCH_SVAGE0001_data, sinus_data_conserv,variable_name='HCH_SVAGE0001',
                    plot_type=['boxplot'],test_type=['pointbiserialr'],
                    result_file_name='HCH_SVAGE0001_conserv', plot_title='Is there relationship between age and incidental findings?')



find_correlation(HCH_SVSEX0001_data, sinus_data_conserv,variable_name='HCH_SVSEX0001',
                 map_variable_value={0:'Male',1:'Female'},
                    plot_type=['piechart'],test_type=['chi_square'],
                    result_file_name='HCH_SVSEX0001_conserv', plot_title='Is there relationship between sex and incidental findings?')
                 

find_correlation(HCH_SVBMI0001_data, sinus_data_conserv,variable_name='HCH_SVBMI0001',
                    plot_type=['boxplot'],test_type=['pointbiserialr'],
                    result_file_name='HCH_SVBMI0001_conserv', plot_title='Body mass index')


find_correlation(ZA_data, sinus_data_conserv,variable_name='ZA',
                    plot_type=['boxplot'],test_type=['pointbiserialr'],
                    result_file_name='ZA_conserv', plot_title='Is there relationship between alcohol consumption and incidental findings?')

                 




print("HCH_ANAM0023 data shape: ",HCH_ANAM0023_data.shape,clinical_data.shape) # Do you smoke  1 - Yes, 2 - Not smoked for 6 months , 3 - never smoked
print("HCH_HAUT0149_data data shape: ",HCH_HAUT0149_data.shape,clinical_data.shape) # Neurodermitis 0 - No, 1 - Yes 
print("klin_hno_006_data data shape: ",klin_hno_006_data.shape,clinical_data.shape) # Were you operated for HNO? 0 - No, 1 - Yes
print("klin_pneumo_006 data shape: ",klin_pneumo_006_data.shape,clinical_data.shape)
print("klin_pneumo_007 data shape: ",klin_pneumo_007_data.shape,clinical_data.shape)
print("klin_pneumo_008 data shape: ",klin_pneumo_008_data.shape,clinical_data.shape) #Bronchitis or COPD

print("klin_allerg_029 data shape: ",klin_allerg_029_data.shape,clinical_data.shape) #Allergy 1 question
print("klin_allerg_030 data shape: ",klin_allerg_030_data.shape,clinical_data.shape) #Allergy 2 question
print("klin_allerg_032 data shape: ",klin_allerg_032_data.shape,clinical_data.shape) #Allergy 3 question

print("Leuk data shape: ",leuk_data.shape,clinical_data.shape)
print("CRP data shape: ",CRP_data.shape,clinical_data.shape)
print("hsCRP data shape: ",hsCRP_data.shape,clinical_data.shape)


print("HCH_SVSEX0001 data shape: ",HCH_SVSEX0001_data.shape,clinical_data.shape) #Sex
print("HCH_SVAGE0001 data shape: ",HCH_SVAGE0001_data.shape,clinical_data.shape)  # Age
print("HCH_SVISCED_02 data shape: ",HCH_SVISCED_02_data.shape,clinical_data.shape) #Education 
print("HCH_SVBMI0001 data shape: ",HCH_SVBMI0001_data.shape,clinical_data.shape) #BMI
print("ZA data shape: ",ZA_data.shape,clinical_data.shape) #Alcohol consumption per day

"""



