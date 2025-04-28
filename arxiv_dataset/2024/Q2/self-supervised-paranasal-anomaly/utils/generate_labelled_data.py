import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

csv = pd.read_csv("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset/JAMA-labelled-1000/splits/crop_size_65_std_factor_1/fold_1/train_val_test.csv")

#sort by folder
csv = csv.sort_values(by=['folder']) 

#get values in folder column
folders = csv['folder'].values
patient_ids = []
smax_labels = []
confident_labels = []
disease_types = []

for folder in folders: 

    #split folder name into parts
    parts = folder.split("/")
    patient_id = parts[-2]
    disease_type = parts[-1]

    patient_ids.append(patient_id) 

    if disease_type == "normal-l":
        smax_labels.append('smax-l')
        confident_labels.append(0)
        disease_types.append(0) 

    elif disease_type == "normal-r":
        smax_labels.append('smax-r')
        confident_labels.append(0)
        disease_types.append(0)

    elif disease_type == "mucosal_thickening-r":

        smax_labels.append('smax-r')
        confident_labels.append(1)
        disease_types.append(1)

    elif disease_type == "mucosal_thickening-l":

        smax_labels.append('smax-l')
        confident_labels.append(1)
        disease_types.append(1)

    elif disease_type == "polyps-r":

        smax_labels.append('smax-r')
        confident_labels.append(1)
        disease_types.append(2)

    elif disease_type == "polyps-l":

        smax_labels.append('smax-l')
        confident_labels.append(1)
        disease_types.append(2)

    elif disease_type == "cysts-r":

        smax_labels.append('smax-r')
        confident_labels.append(1)
        disease_types.append(3)

    elif disease_type == "cysts-l":

        smax_labels.append('smax-l')
        confident_labels.append(1)
        disease_types.append(3)

    elif disease_type == "fully_occupied-l":

        smax_labels.append('smax-l')
        confident_labels.append(1)
        disease_types.append(4)

    elif disease_type == "fully_occupied-r":

        smax_labels.append('smax-r')
        confident_labels.append(1)  
        disease_types.append(4)


#create new dataframe with new columns
new_df = pd.DataFrame({'patient_id': patient_ids, 'smax': smax_labels, 'mean': confident_labels, 'stdev': [0]*len(confident_labels)   ,'disease_label': disease_types})
#save to csv
new_df.to_csv("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/results_JAMA_aug2023/plots/labelled_dataset.csv", index=False)




    



    