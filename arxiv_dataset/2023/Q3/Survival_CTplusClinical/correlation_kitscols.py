#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:53:29 2023

@author: mary
"""

from pathlib import Path
import pandas as pd

clinical_dir = Path('/home/mary/Downloads/kits.csv')

#delete the rows that tumor isup grade has none value for them

total_df = pd.read_csv(clinical_dir)
mask = pd.isnull(total_df['tumor_isup_grade'])
rows_with_nan = total_df.index[mask]

# new_df = total_df.dropna(subset=["tumor_isup_grade"])                 


#finding columns that for 244 patients they have none values so we wont use
#these columns
listofnans = []
listofnans_cols = []

for i in range(total_df.shape[1]):
    listofnans.append(total_df.iloc[:,i].isnull().values.any())

i = 0    
for item in listofnans:
    if item == True:
       listofnans_cols.append(total_df.columns[i]) 

    i+=1


#delete the columns that can not be used as inputs to the model    
cols_should_delete =['case_id','pathology_n_stage','pathology_m_stage','vital_status','voxel_spacing/x_spacing','voxel_spacing/y_spacing','voxel_spacing/z_spacing' ]    
new_df_withoutnan = total_df.drop([item for item in listofnans_cols], axis=1)
new_df_withoutnan = new_df_withoutnan.drop([item for item in cols_should_delete], axis=1)
new_df_withoutnan.to_csv('kits_clinical.csv')

#change categorical columns to numbers and True False columns to 0 and 1

#hospitalization row 18 is not number, we change it to zero
df_without_nan_dir = Path('/home/mary/Documents/kidney_ds/survival/merge_clinicals/kits_clinical.csv')
df_without_nan = pd.read_csv(df_without_nan_dir)
df_without_nan.loc[16,'hospitalization'] = 0

#leave these columns
#Age_at_nephrectomy, Body_mass_index, Hospitalization, Radiographic_size, Pathologic_size
#change the other 33 columns

for i in range(df_without_nan.shape[0]):
    if i not in rows_with_nan:
        
        if df_without_nan.loc[i,'gender'] == 'male':
            df_without_nan.loc[i,'gender']= 0
        elif df_without_nan.loc[i,'gender'] == 'female':
            df_without_nan.loc[i,'gender']= 1
    
        if df_without_nan.loc[i,'comorbidities/myocardial_infarction'] == True:
           df_without_nan.loc[i,'comorbidities/myocardial_infarction'] = 0
        elif df_without_nan.loc[i,'comorbidities/myocardial_infarction'] == False:
            df_without_nan.loc[i,'comorbidities/myocardial_infarction'] = 1
                              
        if df_without_nan.loc[i,'comorbidities/congestive_heart_failure'] == True:
            df_without_nan.loc[i,'comorbidities/congestive_heart_failure'] = 0
        elif df_without_nan.loc[i,'comorbidities/congestive_heart_failure'] == False:
             df_without_nan.loc[i,'comorbidities/congestive_heart_failure'] = 1
    
        if df_without_nan.loc[i,'comorbidities/peripheral_vascular_disease'] == True:
           df_without_nan.loc[i,'comorbidities/peripheral_vascular_disease'] = 0
        elif df_without_nan.loc[i,'comorbidities/peripheral_vascular_disease'] == False:
            df_without_nan.loc[i,'comorbidities/peripheral_vascular_disease'] = 1
    
        if df_without_nan.loc[i,'comorbidities/cerebrovascular_disease'] == True:
           df_without_nan.loc[i,'comorbidities/cerebrovascular_disease'] = 0
        elif df_without_nan.loc[i,'comorbidities/cerebrovascular_disease'] == False:
            df_without_nan.loc[i,'comorbidities/cerebrovascular_disease'] = 1                     
    
        if df_without_nan.loc[i,'comorbidities/dementia'] == True:
           df_without_nan.loc[i,'comorbidities/dementia'] = 0
        elif df_without_nan.loc[i,'comorbidities/dementia'] == False:
            df_without_nan.loc[i,'comorbidities/dementia'] = 1 
                    
        if df_without_nan.loc[i,'comorbidities/copd'] == True:
           df_without_nan.loc[i,'comorbidities/copd'] = 0
        elif df_without_nan.loc[i,'comorbidities/copd'] == False:
            df_without_nan.loc[i,'comorbidities/copd'] = 1 
            
        if df_without_nan.loc[i,'comorbidities/connective_tissue_disease'] == True:
           df_without_nan.loc[i,'comorbidities/connective_tissue_disease'] = 0
        elif df_without_nan.loc[i,'comorbidities/connective_tissue_disease'] == False:
            df_without_nan.loc[i,'comorbidities/connective_tissue_disease'] = 1 
            
        if df_without_nan.loc[i,'comorbidities/peptic_ulcer_disease'] == True:
           df_without_nan.loc[i,'comorbidities/peptic_ulcer_disease'] = 0
        elif df_without_nan.loc[i,'comorbidities/peptic_ulcer_disease'] == False:
            df_without_nan.loc[i,'comorbidities/peptic_ulcer_disease'] = 1 
            
        if df_without_nan.loc[i,'comorbidities/uncomplicated_diabetes_mellitus'] == True:
           df_without_nan.loc[i,'comorbidities/uncomplicated_diabetes_mellitus'] = 0
        elif df_without_nan.loc[i,'comorbidities/uncomplicated_diabetes_mellitus'] == False:
            df_without_nan.loc[i,'comorbidities/uncomplicated_diabetes_mellitus'] = 1 
            
        if df_without_nan.loc[i,'comorbidities/diabetes_mellitus_with_end_organ_damage'] == True:
           df_without_nan.loc[i,'comorbidities/diabetes_mellitus_with_end_organ_damage'] = 0
        elif df_without_nan.loc[i,'comorbidities/diabetes_mellitus_with_end_organ_damage'] == False:
            df_without_nan.loc[i,'comorbidities/diabetes_mellitus_with_end_organ_damage'] = 1 
            
        if df_without_nan.loc[i,'comorbidities/chronic_kidney_disease'] == True:
           df_without_nan.loc[i,'comorbidities/chronic_kidney_disease'] = 0
        elif df_without_nan.loc[i,'comorbidities/chronic_kidney_disease'] == False:
            df_without_nan.loc[i,'comorbidities/chronic_kidney_disease'] = 1 
            
        if df_without_nan.loc[i,'comorbidities/hemiplegia_from_stroke'] == True:
           df_without_nan.loc[i,'comorbidities/hemiplegia_from_stroke'] = 0
        elif df_without_nan.loc[i,'comorbidities/hemiplegia_from_stroke'] == False:
            df_without_nan.loc[i,'comorbidities/hemiplegia_from_stroke'] = 1 
    
        if df_without_nan.loc[i,'comorbidities/leukemia'] == True:
           df_without_nan.loc[i,'comorbidities/leukemia'] = 0
        elif df_without_nan.loc[i,'comorbidities/leukemia'] == False:
            df_without_nan.loc[i,'comorbidities/leukemia'] = 1 
        
        if df_without_nan.loc[i,'comorbidities/malignant_lymphoma'] == True:
           df_without_nan.loc[i,'comorbidities/malignant_lymphoma'] = 0
        elif df_without_nan.loc[i,'comorbidities/malignant_lymphoma'] == False:
            df_without_nan.loc[i,'comorbidities/malignant_lymphoma'] = 1 
        
        if df_without_nan.loc[i,'comorbidities/localized_solid_tumor'] == True:
           df_without_nan.loc[i,'comorbidities/localized_solid_tumor'] = 0
        elif df_without_nan.loc[i,'comorbidities/localized_solid_tumor'] == False:
            df_without_nan.loc[i,'comorbidities/localized_solid_tumor'] = 1 
        
        if df_without_nan.loc[i,'comorbidities/metastatic_solid_tumor'] == True:
           df_without_nan.loc[i,'comorbidities/metastatic_solid_tumor'] = 0
        elif df_without_nan.loc[i,'comorbidities/metastatic_solid_tumor'] == False:
            df_without_nan.loc[i,'comorbidities/metastatic_solid_tumor'] = 1 
        
        if df_without_nan.loc[i,'comorbidities/mild_liver_disease'] == True:
           df_without_nan.loc[i,'comorbidities/mild_liver_disease'] = 0
        elif df_without_nan.loc[i,'comorbidities/mild_liver_disease'] == False:
            df_without_nan.loc[i,'comorbidities/mild_liver_disease'] = 1 
            
        if df_without_nan.loc[i,'comorbidities/moderate_to_severe_liver_disease'] == True:
           df_without_nan.loc[i,'comorbidities/moderate_to_severe_liver_disease'] = 0
        elif df_without_nan.loc[i,'comorbidities/moderate_to_severe_liver_disease'] == False:
            df_without_nan.loc[i,'comorbidities/moderate_to_severe_liver_disease'] = 1 
        
        if df_without_nan.loc[i,'comorbidities/aids'] == True:
           df_without_nan.loc[i,'comorbidities/aids'] = 0
        elif df_without_nan.loc[i,'comorbidities/aids'] == False:
            df_without_nan.loc[i,'comorbidities/aids'] = 1 
        
        if df_without_nan.loc[i,'smoking_history'] == 'current_smoker':
           df_without_nan.loc[i,'smoking_history'] = 0
        elif df_without_nan.loc[i,'smoking_history'] == 'previous_smoker':
            df_without_nan.loc[i,'smoking_history'] = 1
        elif df_without_nan.loc[i,'smoking_history'] == 'never_smoked':
            df_without_nan.loc[i,'smoking_history'] = 2
        
        if df_without_nan.loc[i,'chewing_tobacco_use'] == 'quit_in_last_3mo':
           df_without_nan.loc[i,'chewing_tobacco_use'] = 0
        elif df_without_nan.loc[i,'chewing_tobacco_use'] == 'never_or_not_in_last_3mo':
            df_without_nan.loc[i,'chewing_tobacco_use'] = 1 
        
        if df_without_nan.loc[i,'intraoperative_complications/blood_transfusion'] == True:
           df_without_nan.loc[i,'intraoperative_complications/blood_transfusion'] = 0
        elif df_without_nan.loc[i,'intraoperative_complications/blood_transfusion'] == False:
            df_without_nan.loc[i,'intraoperative_complications/blood_transfusion'] = 1 
    
        if df_without_nan.loc[i,'intraoperative_complications/injury_to_surrounding_organ'] == True:
           df_without_nan.loc[i,'intraoperative_complications/injury_to_surrounding_organ'] = 0
        elif df_without_nan.loc[i,'intraoperative_complications/injury_to_surrounding_organ'] == False:
            df_without_nan.loc[i,'intraoperative_complications/injury_to_surrounding_organ'] = 1 
            
        if df_without_nan.loc[i,'intraoperative_complications/cardiac_event'] == True:
           df_without_nan.loc[i,'intraoperative_complications/cardiac_event'] = 0
        elif df_without_nan.loc[i,'intraoperative_complications/cardiac_event'] == False:
            df_without_nan.loc[i,'intraoperative_complications/cardiac_event'] = 1 
        
        if df_without_nan.loc[i,'malignant'] == True:
           df_without_nan.loc[i,'malignant'] = 0
        elif df_without_nan.loc[i,'malignant'] == False:
            df_without_nan.loc[i,'malignant'] = 1 
            
        if df_without_nan.loc[i,'pathology_t_stage'] == '4':
           df_without_nan.loc[i,'pathology_t_stage'] = 0
        elif df_without_nan.loc[i,'pathology_t_stage'] == '3':
            df_without_nan.loc[i,'pathology_t_stage'] = 1 
        elif df_without_nan.loc[i,'pathology_t_stage'] == '2b':
            df_without_nan.loc[i,'pathology_t_stage'] = 2
        elif df_without_nan.loc[i,'pathology_t_stage'] == '2a':
            df_without_nan.loc[i,'pathology_t_stage'] = 3
        elif df_without_nan.loc[i,'pathology_t_stage'] == '1b':
            df_without_nan.loc[i,'pathology_t_stage'] = 4
        elif df_without_nan.loc[i,'pathology_t_stage'] == '1a':
            df_without_nan.loc[i,'pathology_t_stage'] = 5 
        
            
        if df_without_nan.loc[i,'tumor_histologic_subtype'] == 'chromophobe':
           df_without_nan.loc[i,'tumor_histologic_subtype'] = 0
        elif df_without_nan.loc[i,'tumor_histologic_subtype'] == 'clear_cell_rcc':
            df_without_nan.loc[i,'tumor_histologic_subtype'] = 1 
        elif df_without_nan.loc[i,'tumor_histologic_subtype'] == 'papillary':
            df_without_nan.loc[i,'tumor_histologic_subtype'] = 2 
        elif df_without_nan.loc[i,'tumor_histologic_subtype'] == 'rcc_unclassified':
            df_without_nan.loc[i,'tumor_histologic_subtype'] = 3 
        elif df_without_nan.loc[i,'tumor_histologic_subtype'] == 'clear_cell_papillary_rcc':
            df_without_nan.loc[i,'tumor_histologic_subtype'] = 4 
        elif df_without_nan.loc[i,'tumor_histologic_subtype'] == 'collecting_duct_undefined':
            df_without_nan.loc[i,'tumor_histologic_subtype'] = 5 
        elif df_without_nan.loc[i,'tumor_histologic_subtype'] == 'multilocular_cystic_rcc':
            df_without_nan.loc[i,'tumor_histologic_subtype'] = 6 
        elif df_without_nan.loc[i,'tumor_histologic_subtype'] == 'urothelial':
            df_without_nan.loc[i,'tumor_histologic_subtype'] = 7 
        elif df_without_nan.loc[i,'tumor_histologic_subtype'] == 'other':
            df_without_nan.loc[i,'tumor_histologic_subtype'] = 8 
    
        
        if df_without_nan.loc[i,'surgery_type'] == 'open':
           df_without_nan.loc[i,'surgery_type'] = 0
        elif df_without_nan.loc[i,'surgery_type'] == 'robotic':
            df_without_nan.loc[i,'surgery_type'] = 1 
        elif df_without_nan.loc[i,'surgery_type'] == 'laparoscopic':
            df_without_nan.loc[i,'surgery_type'] = 2 
        
        if df_without_nan.loc[i,'surgical_procedure'] == 'radical_nephrectomy':
           df_without_nan.loc[i,'surgical_procedure'] = 0
        elif df_without_nan.loc[i,'surgical_procedure'] == 'partial_nephrectomy':
            df_without_nan.loc[i,'surgical_procedure'] = 1 
        
        if df_without_nan.loc[i,'surgical_approach'] == 'Transperitoneal':
           df_without_nan.loc[i,'surgical_approach'] = 0
        elif df_without_nan.loc[i,'surgical_approach'] == 'Retroperitoneal':
            df_without_nan.loc[i,'surgical_approach'] = 1 
        elif df_without_nan.loc[i,'surgical_approach'] == 'Trans_to_Retro':
            df_without_nan.loc[i,'surgical_approach'] = 2 
            
            
        if df_without_nan.loc[i,'cytoreductive'] == True:
           df_without_nan.loc[i,'cytoreductive'] = 0
        elif df_without_nan.loc[i,'cytoreductive'] == False:
            df_without_nan.loc[i,'cytoreductive'] = 1 
            
        if df_without_nan.loc[i,'positive_resection_margins'] == True:
           df_without_nan.loc[i,'positive_resection_margins'] = 0
        elif df_without_nan.loc[i,'positive_resection_margins'] == False:
            df_without_nan.loc[i,'positive_resection_margins'] = 1 

for i in range(df_without_nan.shape[0]):
    # if i not in rows_with_nan:
        
    if df_without_nan.loc[i,'gender'] == 'male':
        df_without_nan.loc[i,'gender']= 0
    elif df_without_nan.loc[i,'gender'] == 'female':
        df_without_nan.loc[i,'gender']= 1

    if df_without_nan.loc[i,'comorbidities/myocardial_infarction'] == True:
       df_without_nan.loc[i,'comorbidities/myocardial_infarction'] = 0
    elif df_without_nan.loc[i,'comorbidities/myocardial_infarction'] == False:
        df_without_nan.loc[i,'comorbidities/myocardial_infarction'] = 1
                          
    if df_without_nan.loc[i,'comorbidities/congestive_heart_failure'] == True:
        df_without_nan.loc[i,'comorbidities/congestive_heart_failure'] = 0
    elif df_without_nan.loc[i,'comorbidities/congestive_heart_failure'] == False:
         df_without_nan.loc[i,'comorbidities/congestive_heart_failure'] = 1

    if df_without_nan.loc[i,'comorbidities/peripheral_vascular_disease'] == True:
       df_without_nan.loc[i,'comorbidities/peripheral_vascular_disease'] = 0
    elif df_without_nan.loc[i,'comorbidities/peripheral_vascular_disease'] == False:
        df_without_nan.loc[i,'comorbidities/peripheral_vascular_disease'] = 1

    if df_without_nan.loc[i,'comorbidities/cerebrovascular_disease'] == True:
       df_without_nan.loc[i,'comorbidities/cerebrovascular_disease'] = 0
    elif df_without_nan.loc[i,'comorbidities/cerebrovascular_disease'] == False:
        df_without_nan.loc[i,'comorbidities/cerebrovascular_disease'] = 1                     

    if df_without_nan.loc[i,'comorbidities/dementia'] == True:
       df_without_nan.loc[i,'comorbidities/dementia'] = 0
    elif df_without_nan.loc[i,'comorbidities/dementia'] == False:
        df_without_nan.loc[i,'comorbidities/dementia'] = 1 
                
    if df_without_nan.loc[i,'comorbidities/copd'] == True:
       df_without_nan.loc[i,'comorbidities/copd'] = 0
    elif df_without_nan.loc[i,'comorbidities/copd'] == False:
        df_without_nan.loc[i,'comorbidities/copd'] = 1 
        
    if df_without_nan.loc[i,'comorbidities/connective_tissue_disease'] == True:
       df_without_nan.loc[i,'comorbidities/connective_tissue_disease'] = 0
    elif df_without_nan.loc[i,'comorbidities/connective_tissue_disease'] == False:
        df_without_nan.loc[i,'comorbidities/connective_tissue_disease'] = 1 
        
    if df_without_nan.loc[i,'comorbidities/peptic_ulcer_disease'] == True:
       df_without_nan.loc[i,'comorbidities/peptic_ulcer_disease'] = 0
    elif df_without_nan.loc[i,'comorbidities/peptic_ulcer_disease'] == False:
        df_without_nan.loc[i,'comorbidities/peptic_ulcer_disease'] = 1 
        
    if df_without_nan.loc[i,'comorbidities/uncomplicated_diabetes_mellitus'] == True:
       df_without_nan.loc[i,'comorbidities/uncomplicated_diabetes_mellitus'] = 0
    elif df_without_nan.loc[i,'comorbidities/uncomplicated_diabetes_mellitus'] == False:
        df_without_nan.loc[i,'comorbidities/uncomplicated_diabetes_mellitus'] = 1 
        
    if df_without_nan.loc[i,'comorbidities/diabetes_mellitus_with_end_organ_damage'] == True:
       df_without_nan.loc[i,'comorbidities/diabetes_mellitus_with_end_organ_damage'] = 0
    elif df_without_nan.loc[i,'comorbidities/diabetes_mellitus_with_end_organ_damage'] == False:
        df_without_nan.loc[i,'comorbidities/diabetes_mellitus_with_end_organ_damage'] = 1 
        
    if df_without_nan.loc[i,'comorbidities/chronic_kidney_disease'] == True:
       df_without_nan.loc[i,'comorbidities/chronic_kidney_disease'] = 0
    elif df_without_nan.loc[i,'comorbidities/chronic_kidney_disease'] == False:
        df_without_nan.loc[i,'comorbidities/chronic_kidney_disease'] = 1 
        
    if df_without_nan.loc[i,'comorbidities/hemiplegia_from_stroke'] == True:
       df_without_nan.loc[i,'comorbidities/hemiplegia_from_stroke'] = 0
    elif df_without_nan.loc[i,'comorbidities/hemiplegia_from_stroke'] == False:
        df_without_nan.loc[i,'comorbidities/hemiplegia_from_stroke'] = 1 

    if df_without_nan.loc[i,'comorbidities/leukemia'] == True:
       df_without_nan.loc[i,'comorbidities/leukemia'] = 0
    elif df_without_nan.loc[i,'comorbidities/leukemia'] == False:
        df_without_nan.loc[i,'comorbidities/leukemia'] = 1 
    
    if df_without_nan.loc[i,'comorbidities/malignant_lymphoma'] == True:
       df_without_nan.loc[i,'comorbidities/malignant_lymphoma'] = 0
    elif df_without_nan.loc[i,'comorbidities/malignant_lymphoma'] == False:
        df_without_nan.loc[i,'comorbidities/malignant_lymphoma'] = 1 
    
    if df_without_nan.loc[i,'comorbidities/localized_solid_tumor'] == True:
       df_without_nan.loc[i,'comorbidities/localized_solid_tumor'] = 0
    elif df_without_nan.loc[i,'comorbidities/localized_solid_tumor'] == False:
        df_without_nan.loc[i,'comorbidities/localized_solid_tumor'] = 1 
    
    if df_without_nan.loc[i,'comorbidities/metastatic_solid_tumor'] == True:
       df_without_nan.loc[i,'comorbidities/metastatic_solid_tumor'] = 0
    elif df_without_nan.loc[i,'comorbidities/metastatic_solid_tumor'] == False:
        df_without_nan.loc[i,'comorbidities/metastatic_solid_tumor'] = 1 
    
    if df_without_nan.loc[i,'comorbidities/mild_liver_disease'] == True:
       df_without_nan.loc[i,'comorbidities/mild_liver_disease'] = 0
    elif df_without_nan.loc[i,'comorbidities/mild_liver_disease'] == False:
        df_without_nan.loc[i,'comorbidities/mild_liver_disease'] = 1 
        
    if df_without_nan.loc[i,'comorbidities/moderate_to_severe_liver_disease'] == True:
       df_without_nan.loc[i,'comorbidities/moderate_to_severe_liver_disease'] = 0
    elif df_without_nan.loc[i,'comorbidities/moderate_to_severe_liver_disease'] == False:
        df_without_nan.loc[i,'comorbidities/moderate_to_severe_liver_disease'] = 1 
    
    if df_without_nan.loc[i,'comorbidities/aids'] == True:
       df_without_nan.loc[i,'comorbidities/aids'] = 0
    elif df_without_nan.loc[i,'comorbidities/aids'] == False:
        df_without_nan.loc[i,'comorbidities/aids'] = 1 
    
    if df_without_nan.loc[i,'smoking_history'] == 'current_smoker':
       df_without_nan.loc[i,'smoking_history'] = 0
    elif df_without_nan.loc[i,'smoking_history'] == 'previous_smoker':
        df_without_nan.loc[i,'smoking_history'] = 1
    elif df_without_nan.loc[i,'smoking_history'] == 'never_smoked':
        df_without_nan.loc[i,'smoking_history'] = 2
    
    if df_without_nan.loc[i,'chewing_tobacco_use'] == 'quit_in_last_3mo':
       df_without_nan.loc[i,'chewing_tobacco_use'] = 0
    elif df_without_nan.loc[i,'chewing_tobacco_use'] == 'never_or_not_in_last_3mo':
        df_without_nan.loc[i,'chewing_tobacco_use'] = 1 
    
    if df_without_nan.loc[i,'intraoperative_complications/blood_transfusion'] == True:
       df_without_nan.loc[i,'intraoperative_complications/blood_transfusion'] = 0
    elif df_without_nan.loc[i,'intraoperative_complications/blood_transfusion'] == False:
        df_without_nan.loc[i,'intraoperative_complications/blood_transfusion'] = 1 

    if df_without_nan.loc[i,'intraoperative_complications/injury_to_surrounding_organ'] == True:
       df_without_nan.loc[i,'intraoperative_complications/injury_to_surrounding_organ'] = 0
    elif df_without_nan.loc[i,'intraoperative_complications/injury_to_surrounding_organ'] == False:
        df_without_nan.loc[i,'intraoperative_complications/injury_to_surrounding_organ'] = 1 
        
    if df_without_nan.loc[i,'intraoperative_complications/cardiac_event'] == True:
       df_without_nan.loc[i,'intraoperative_complications/cardiac_event'] = 0
    elif df_without_nan.loc[i,'intraoperative_complications/cardiac_event'] == False:
        df_without_nan.loc[i,'intraoperative_complications/cardiac_event'] = 1 
    
    if df_without_nan.loc[i,'malignant'] == True:
       df_without_nan.loc[i,'malignant'] = 0
    elif df_without_nan.loc[i,'malignant'] == False:
        df_without_nan.loc[i,'malignant'] = 1 
        
    if df_without_nan.loc[i,'pathology_t_stage'] == '4':
       df_without_nan.loc[i,'pathology_t_stage'] = 0
    elif df_without_nan.loc[i,'pathology_t_stage'] == '3':
        df_without_nan.loc[i,'pathology_t_stage'] = 1 
    elif df_without_nan.loc[i,'pathology_t_stage'] == '2b':
        df_without_nan.loc[i,'pathology_t_stage'] = 2
    elif df_without_nan.loc[i,'pathology_t_stage'] == '2a':
        df_without_nan.loc[i,'pathology_t_stage'] = 3
    elif df_without_nan.loc[i,'pathology_t_stage'] == '1b':
        df_without_nan.loc[i,'pathology_t_stage'] = 4
    elif df_without_nan.loc[i,'pathology_t_stage'] == '1a':
        df_without_nan.loc[i,'pathology_t_stage'] = 5 
    
        
    if df_without_nan.loc[i,'tumor_histologic_subtype'] == 'chromophobe':
       df_without_nan.loc[i,'tumor_histologic_subtype'] = 0
    elif df_without_nan.loc[i,'tumor_histologic_subtype'] == 'clear_cell_rcc':
        df_without_nan.loc[i,'tumor_histologic_subtype'] = 1 
    elif df_without_nan.loc[i,'tumor_histologic_subtype'] == 'papillary':
        df_without_nan.loc[i,'tumor_histologic_subtype'] = 2 
    elif df_without_nan.loc[i,'tumor_histologic_subtype'] == 'rcc_unclassified':
        df_without_nan.loc[i,'tumor_histologic_subtype'] = 3 
    elif df_without_nan.loc[i,'tumor_histologic_subtype'] == 'clear_cell_papillary_rcc':
        df_without_nan.loc[i,'tumor_histologic_subtype'] = 4 
    elif df_without_nan.loc[i,'tumor_histologic_subtype'] == 'collecting_duct_undefined':
        df_without_nan.loc[i,'tumor_histologic_subtype'] = 5 
    elif df_without_nan.loc[i,'tumor_histologic_subtype'] == 'multilocular_cystic_rcc':
        df_without_nan.loc[i,'tumor_histologic_subtype'] = 6 
    elif df_without_nan.loc[i,'tumor_histologic_subtype'] == 'urothelial':
        df_without_nan.loc[i,'tumor_histologic_subtype'] = 7 
    elif df_without_nan.loc[i,'tumor_histologic_subtype'] == 'other':
        df_without_nan.loc[i,'tumor_histologic_subtype'] = 8 

    
    if df_without_nan.loc[i,'surgery_type'] == 'open':
       df_without_nan.loc[i,'surgery_type'] = 0
    elif df_without_nan.loc[i,'surgery_type'] == 'robotic':
        df_without_nan.loc[i,'surgery_type'] = 1 
    elif df_without_nan.loc[i,'surgery_type'] == 'laparoscopic':
        df_without_nan.loc[i,'surgery_type'] = 2 
    
    if df_without_nan.loc[i,'surgical_procedure'] == 'radical_nephrectomy':
       df_without_nan.loc[i,'surgical_procedure'] = 0
    elif df_without_nan.loc[i,'surgical_procedure'] == 'partial_nephrectomy':
        df_without_nan.loc[i,'surgical_procedure'] = 1 
    
    if df_without_nan.loc[i,'surgical_approach'] == 'Transperitoneal':
       df_without_nan.loc[i,'surgical_approach'] = 0
    elif df_without_nan.loc[i,'surgical_approach'] == 'Retroperitoneal':
        df_without_nan.loc[i,'surgical_approach'] = 1 
    elif df_without_nan.loc[i,'surgical_approach'] == 'Trans_to_Retro':
        df_without_nan.loc[i,'surgical_approach'] = 2 
        
        
    if df_without_nan.loc[i,'cytoreductive'] == True:
       df_without_nan.loc[i,'cytoreductive'] = 0
    elif df_without_nan.loc[i,'cytoreductive'] == False:
        df_without_nan.loc[i,'cytoreductive'] = 1 
        
    if df_without_nan.loc[i,'positive_resection_margins'] == True:
       df_without_nan.loc[i,'positive_resection_margins'] = 0
    elif df_without_nan.loc[i,'positive_resection_margins'] == False:
        df_without_nan.loc[i,'positive_resection_margins'] = 1 


df_without_nan.to_csv('kits_numberversion_244.csv')

#calculating Pearson correlation
kitsnumberversion_dir = Path('/home/mary/Documents/kidney_ds/survival/merge_clinicals/kits_numberversion_244.csv')
df_kitsnumberv = pd.read_csv(kitsnumberversion_dir)

correlation_matrix = df_kitsnumberv.corr(method='spearman')
cor_survival = correlation_matrix['vital_days_after_surgery']


#finding importance of variables with random forest
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

X = df_kitsnumberv.drop('vital_days_after_surgery', axis=1)
y = df_kitsnumberv['vital_days_after_surgery']

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print(feature_importances.sort_values(ascending=False))