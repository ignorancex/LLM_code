import pandas as pd
import numpy as np
import os
import sys
import tqdm
import random
import time
import ast

sys.path.append(os.getcwd())
from sklearn.model_selection import train_test_split
from local_data.constants import *

# % ----------------------
# PRETRAINING DATASETS


def prepare_padchest_partition_pretrain():

    # Frontal/All options
    only_frontal = True

    # Target categories
    categories = ['copd signs', 'adenopathy', 'air trapping', 'alveolar pattern', 'aortic atheromatosis',
                  'aortic elongation', 'apical pleural thickening', 'atelectasis', 'bronchiectasis',
                  'bronchovascular markings', 'bullas', 'calcified adenopathy', 'calcified densities',
                  'calcified granuloma', 'calcified pleural thickening', 'callus rib fracture', 'cardiomegaly',
                  'cavitation', 'chronic changes', 'consolidation', 'costophrenic angle blunting',
                  'descendent aortic elongation', 'diaphragmatic eventration', 'emphysema', 'fibrotic band',
                  'flattened diaphragm', 'fracture', 'goiter', 'granuloma', 'gynecomastia', 'heart insufficiency',
                  'heart valve', 'hemidiaphragm elevation', 'hiatal hernia', 'hilar congestion', 'hilar enlargement',
                  'hyperinflated lung', 'hypoexpansion', 'increased density', 'infiltrates', 'interstitial pattern',
                  'kyphosis', 'laminar atelectasis', 'lobar atelectasis', 'lung opacities', 'mammary prosthesis',
                  'mastectomy', 'metal', 'minor fissure thickening', 'nipple shadow', 'no finding', 'nodule',
                  'osteopenia', 'osteosynthesis material', 'pacemaker', 'pleural effusion', 'pleural thickening',
                  'pneumonia','pneumothorax', 'pseudonodule', 'pulmonary edema', 'pulmonary fibrosis', 'pulmonary mass',
                  'reticular interstitial pattern', 'rib fracture', 'sclerotic bone lesion', 'scoliosis', 'sternotomy',
                  'suboptimal study', 'superior mediastinal enlargement', 'support devices', 'supra aortic elongation',
                  'suture material', 'tracheal shift', 'tuberculosis', 'tuberculosis sequelae', 'volume loss'
                  'vascular hilar enlargement', 'vertebral anterior compression', 'vertebral degenerative changes']

    # Paths
    dataset_path = "PadChest/"
    images_path = "images/"

    # Read dataframe
    dataframe = pd.read_csv(PATH_DATASETS + dataset_path + "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")
    dataframe = dataframe[[not pd.isna(iLabel) for iLabel in dataframe["Labels"]]]

    # Adequate labels
    labels = dataframe["Labels"].to_list()
    labels = [ast.literal_eval(iLabel) for iLabel in labels]

    for i in range(5):
        for iCase in range(len(labels)):
            for iFinding in range(len(labels[iCase])):
                if labels[iCase][iFinding] != "":
                    labels[iCase][iFinding] = labels[iCase][iFinding].lower()
                    if labels[iCase][iFinding][0] == " ":
                        labels[iCase][iFinding] = labels[iCase][iFinding][1:]
                    if labels[iCase][iFinding][-1] == " ":
                        labels[iCase][iFinding] = labels[iCase][iFinding][:-1]
                    if labels[iCase][iFinding] == "empyema":
                        labels[iCase][iFinding] = "emphysema"
                    if "atelectasis" in labels[iCase][iFinding] and "atelectasis" not in labels[iCase]:
                        labels[iCase].append("atelectasis")
                    if "pneumonia" in labels[iCase][iFinding] and "pneumonia" not in labels[iCase]:
                        labels[iCase].append("pneumonia")
                    if "pleural effusion" in labels[iCase][iFinding] and "pleural effusion" not in labels[iCase]:
                        labels[iCase].append("pleural effusion")
                    if "pleural thickening" in labels[iCase][iFinding] and "pleural thickening" not in labels[iCase]:
                        labels[iCase].append("pleural thickening")
                    if "tuberculosis" in labels[iCase][iFinding] and "tuberculosis" not in labels[iCase]:
                        labels[iCase].append("tuberculosis")
                    if "fracture" in labels[iCase][iFinding] and "fracture" not in labels[iCase]:
                        labels[iCase].append("fracture")
                    if "miliary opacities" in labels[iCase][iFinding] and "lung opacities" not in labels[iCase]:
                        labels[iCase].append("lung opacities")
                    if "reticular interstitial pattern" in labels[iCase][iFinding] and "lung opacities" not in labels[iCase]:
                        labels[iCase].append("lung opacities")
                    if "reticulonodular interstitial pattern" in labels[iCase][iFinding] and "lung opacities" not in labels[iCase]:
                        labels[iCase].append("lung opacities")
                    if "interstitial pattern" in labels[iCase][iFinding] and "lung opacities" not in labels[iCase]:
                        labels[iCase].append("lung opacities")
                    if "pulmonary mass" in labels[iCase][iFinding] and "lung opacities" not in labels[iCase]:
                        labels[iCase].append("lung opacities")
                    if "nodule" in labels[iCase][iFinding] and "nodule" not in labels[iCase]:
                        labels[iCase].append("nodule")
                    if "central venous catheter" in labels[iCase][iFinding]:
                        labels[iCase][iFinding] = "central venous catheter"
                    if "heart valve" in labels[iCase][iFinding]:
                        labels[iCase][iFinding] = "heart valve"
                    if "normal" in labels[iCase][iFinding]:
                        labels[iCase][iFinding] = "no finding"
                    if "central venous catheter" in labels[iCase][iFinding]:
                        labels[iCase][iFinding] = "central venous catheter"
                    if "device" in labels[iCase][iFinding]:
                        labels[iCase][iFinding] = "support devices"
                    if " tube" in labels[iCase][iFinding]:
                        labels[iCase][iFinding] = "support devices"
                    if "catheter" in labels[iCase][iFinding]:
                        labels[iCase][iFinding] = "support devices"

            if "unchanged" in labels[iCase]:
                labels[iCase].remove("unchanged")
            if "exclude" in labels[iCase]:
                labels[iCase].remove("exclude")

    # Re-couple labels
    dataframe["Labels"] = labels

    if only_frontal:
        dataframe = dataframe[(dataframe["Projection"] == "AP") | (dataframe["Projection"] == "PA")]

    data = []
    for i in range(len(dataframe)):
        t1 = time.time()

        relative_path = dataframe["ImageID"].values[i].replace(".png", ".jpeg")

        if os.path.isfile(PATH_DATASETS + dataset_path + images_path + relative_path):
            data.append({"image": dataset_path + images_path + relative_path,
                         "prompts": [],
                         "prompts_categories": [],
                         "study_categories": [iCategory for iCategory in categories if iCategory in dataframe["Labels"].values[i]]})

        t2 = time.time()
        print(str(i) + "/" + str(len(dataframe)) + " -- " + str(t2 - t1), end="\r")

    # Create dataframe
    df = pd.DataFrame(data)

    # Save dataframe
    if only_frontal:
        df.to_csv("./local_data/partitions/pretraining/PadChest-train-frontal.csv")
    else:
        df.to_csv("./local_data/partitions/pretraining/PadChest-train.csv")


def prepare_mimic_partition_pretrain():

    # Frontal/All options
    only_frontal = True

    # Target categories
    categories = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity',
                  'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                  'Pleural Other', 'Fracture', 'Support Devices']

    table_labels_sentences = pd.read_csv(PATH_DATASETS + "MIMIC-CXR-2/2.0.0/" + "labels_individual_sentences.csv")
    table_test = pd.read_csv("./local_data/partitions/transferability/mimic_5x200.csv")
    table_partitions = pd.read_csv(PATH_DATASETS + "MIMIC-CXR-2/2.0.0/" + "mimic-cxr-2.0.0-split.csv")
    table_metadata = pd.read_csv(PATH_DATASETS + "MIMIC-CXR-2/2.0.0/" + "mimic-cxr-2.0.0-metadata.csv")

    # Quit subjects used for testing samples
    test_subjects = [id.split("/")[4][1:] for id in table_test["Path"].to_list()]
    table_partitions = table_partitions[np.logical_not(np.isin(table_partitions["subject_id"], test_subjects))]

    # Select only chest frontal views
    if only_frontal:
        table_metadata = table_metadata[(table_metadata["ViewPosition"] == "PA") | (table_metadata["ViewPosition"] == "AP")]

    # Align partition and metadata tables
    table_partitions = table_partitions[np.in1d(list(table_partitions["dicom_id"]), list(table_metadata["dicom_id"]))]
    table_partitions = table_partitions.reset_index()

    images = table_partitions["dicom_id"]
    studies = table_partitions["study_id"]
    subjects = table_partitions["subject_id"]
    data = []
    for i in range(len(table_partitions)):
        t1 = time.time()

        image_id = images[i]
        study_id = studies[i]
        subject_id = subjects[i]

        folder = str(subject_id)[:2]

        relative_path = "files/p" + folder + "/p" + str(subject_id) + "/s" + str(study_id) + "/" + image_id + ".jpg"
        if os.path.isfile(PATH_DATASETS + "MIMIC-CXR-2/2.0.0/" + relative_path):

            subTable_prompts = table_labels_sentences.query('patient == "' + "s" + str(study_id) + '"')
            if len(subTable_prompts) > 0:
                prompts_categories = [[categories[iCategory] for iCategory in range(len(categories)) if subTable_prompts[categories].values[iPrompt,iCategory] == 1] for iPrompt in range(subTable_prompts.shape[0])]
                study_categories = np.unique(np.concatenate(prompts_categories)).tolist()
                if len(study_categories) > 1 and 'No Finding' in study_categories:
                    study_categories.remove('No Finding')

                data.append({"image": "MIMIC-CXR-2/2.0.0/" + relative_path,
                             "prompts": list(subTable_prompts["sentences"]),
                             "prompts_categories": prompts_categories,
                             "study_categories": study_categories})

                t2 = time.time()
                print(str(i) + "/" + str(len(table_partitions)) + " -- " + str(t2-t1), end="\n")

    # Create dataframe
    df = pd.DataFrame(data)

    # Save dataframe
    if only_frontal:
        df.to_csv("./local_data/partitions/pretraining/MIMIC-CXR-2-train-frontal.csv")
    else:
        df.to_csv("./local_data/partitions/pretraining/MIMIC-CXR-2-train.csv")


def prepare_chexpert_partition_pretrain():

    # Frontal/All options
    only_frontal = True

    # Target categories
    categories = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity',
                  'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                  'Pleural Other', 'Fracture', 'Support Devices']

    # Train partition
    table_train = pd.read_csv(PATH_DATASETS + "CheXpert/" + "train_visualCheXbert.csv")
    # Test partition
    table_test = pd.read_csv("./local_data/partitions/transferability/chexpert_5x200.csv")

    # Remove testing subjects from train subset
    subjects = [id.split("/")[2] for id in table_train["Path"].to_list()]
    test_subjects = [id.split("/")[2] for id in table_test["Path"].to_list()]
    table_train = table_train[np.logical_not(np.in1d(subjects, test_subjects))]
    table_train = table_train.reset_index()

    # Select only chest PA radiographs
    if only_frontal:
        table_train = table_train[(table_train["Frontal/Lateral"] == "Frontal")]
        table_train = table_train.reset_index()

    data = []
    for i in range(len(table_train)):
        t1 = time.time()

        relative_path = table_train["Path"][i]

        if os.path.isfile(PATH_DATASETS + "CheXpert/CheXpert-v1.0/" + relative_path):
            data.append({"image": "CheXpert/CheXpert-v1.0/" + relative_path,
                         "prompts": [],
                         "prompts_categories": [],
                         "study_categories": [iCategory for iCategory in categories if table_train[iCategory][i]]})

        t2 = time.time()
        print(str(i) + "/" + str(len(table_train)) + " -- " + str(t2 - t1), end="\r")

    # Create dataframe
    df = pd.DataFrame(data)

    # Save dataframe
    if only_frontal:
        df.to_csv("./local_data/partitions/pretraining/CheXpert-train-frontal.csv")
    else:
        df.to_csv("./local_data/partitions/pretraining/CheXpert-train.csv")

# % ----------------------
# TRANSFERABILITY DATASETS


def prepare_nih_lt_partition():

    # Set paths and dataframe folders
    path_dataset = "NIH/"
    train_paths = {"path": "images/", "dataframe": "LongTailCXR/nih-cxr-lt_single-label_train.csv"}
    val_paths = {"path": "images/", "dataframe": "LongTailCXR/nih-cxr-lt_single-label_balanced-test.csv"}
    test_paths = {"path": "images/", "dataframe": "LongTailCXR/nih-cxr-lt_single-label_balanced-val.csv"}

    # Retrieve and join dataframes
    df_train = pd.read_table(PATH_DATASETS + path_dataset + train_paths["dataframe"], delimiter=",")
    df_val = pd.read_table(PATH_DATASETS + path_dataset + val_paths["dataframe"], delimiter=",")
    df_test = pd.read_table(PATH_DATASETS + path_dataset + test_paths["dataframe"], delimiter=",")

    # Prepare training data
    df = pd.concat([df_train, df_val])
    # Rename columns
    df = df.rename(columns={'id': 'Path', 'Pleural_Thickening': 'Pleural Thickening'})
    # Images Path
    df["Path"] = ["images/" + iFile for iFile in df["Path"].to_list()]
    df.to_csv("./local_data/partitions/transferability/nih_train.csv")

    # Prepare training data
    df = df_test
    # Rename columns
    df = df.rename(columns={'id': 'Path', 'Pleural_Thickening': 'Pleural Thickening'})
    # Prepare test data
    df["Path"] = ["images/" + iFile for iFile in df["Path"].to_list()]
    df.to_csv("./local_data/partitions/transferability/nih_test.csv")


def prepare_rsna_pneumonia_partition():
    n = 6000

    # Read original partition dataframe
    df = pd.read_csv(PATH_DATASETS + "RSNA_PNEUMONIA/" + "stage_2_detailed_class_info.csv")

    # create labels
    """
    "Lung Opacity (bounding box) - a finding on chest radiograph that in a patient with cough and fever has a high 
    likelihood of being pneumonia"
    From the challenge organizers: "In the cases labeled Not Normal/No Lung Opacity, no lung opacity refers to no 
    opacity suspicious for pneumonia. Other non-pneumonia opacities may be present. Also, some of the not normal cases
    have subtle abnormalities which require a trained eye to discern. (Which, for now, keeps radiologists around.)"
    """

    df["Normal"] = df["class"].apply(lambda x: 1 if x == "Normal" else 0)
    df["Pneumonia"] = df["class"].apply(lambda x: 1 if x == "Lung Opacity" else 0)
    df = df[(df["Normal"].values + df["Pneumonia"].values) == 1]

    # no encoded pixels mean healthy
    df["Path"] = df["patientId"].apply(lambda x: "stage_2_train_images/" + (x + ".dcm"))

    df = df[["Path", "Normal", "Pneumonia"]]

    idx_pneumonia = list(np.squeeze(np.argwhere(df["Pneumonia"].values == 1)))[:n]
    idx_normal = list(np.squeeze(np.argwhere(df["Pneumonia"].values == 0)))

    # Resample balanced dataset
    random.seed(42)
    idx_normal = random.sample(idx_normal, len(idx_pneumonia))
    df = df[df.index.isin(idx_normal + idx_pneumonia)]

    # split data
    train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=0)
    train_df = train_df.reset_index()
    test_val_df = test_val_df.reset_index()

    train_df.to_csv("./local_data/partitions/transferability/RSNA_train.csv")
    test_val_df.to_csv("./local_data/partitions/transferability/RSNA_test.csv")


def prepare_rsna_pneumonia_partition_gloria():
    n = 6000

    # Read original partition dataframe
    df = pd.read_csv(PATH_DATASETS + "RSNA_PNEUMONIA/" + "stage_2_train_labels.csv")

    # create bounding boxes
    def create_bbox(row):
        if row["Target"] == 0:
            return 0
        else:
            x1 = row["x"]
            y1 = row["y"]
            x2 = x1 + row["width"]
            y2 = y1 + row["height"]
            return [x1, y1, x2, y2]

    df["bbox"] = df.apply(lambda x: create_bbox(x), axis=1)

    # aggregate multiple boxes
    df = df[["patientId", "bbox"]]
    df = df.groupby("patientId").agg(list)
    df = df.reset_index()
    df["bbox"] = df["bbox"].apply(lambda x: None if x == [0] else x)

    # create labels
    df["Normal"] = df["bbox"].apply(lambda x: 1 if x == None else 0)
    df["Pneumonia"] = df["bbox"].apply(lambda x: 0 if x == None else 1)

    # no encoded pixels mean healthy
    df["Path"] = df["patientId"].apply(lambda x: "stage_2_train_images/" + (x + ".dcm"))

    df = df[["Path", "Normal", "Pneumonia", "bbox"]]

    idx_pneumonia = list(np.squeeze(np.argwhere(df["Pneumonia"].values == 1)))[:n]
    idx_normal = list(np.squeeze(np.argwhere(df["Pneumonia"].values == 0)))

    # Resample balanced dataset
    random.seed(42)
    idx_normal = random.sample(idx_normal, len(idx_pneumonia))
    df = df[df.index.isin(idx_normal + idx_pneumonia)]

    # split data
    train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=0)

    train_df.to_csv("./local_data/partitions/transferability/RSNA_gloria_train.csv")
    test_val_df.to_csv("./local_data/partitions/transferability/RSNA_gloria_test.csv")


def prepare_covid_partition():

    path_dataset = "COVID-19_Radiography_Dataset/"

    covid_path, n_covid_test, n_covid_train = "COVID/images/", 1000, 300
    normal_path, n_normal_test, n_normal_train = "Normal/images/", 1000, 300
    pneumonia_path, n_pneumonia_test, n_pneumonia_train = "ViralPneumonia/images/", 1000, 300
    opacities_path, opacities_test, n_opacities_train = "Lung_Opacity/images/", 1000, 300

    files_covid = os.listdir(PATH_DATASETS + path_dataset + covid_path)
    files_normal = os.listdir(PATH_DATASETS + path_dataset + normal_path)
    files_pneumonia = os.listdir(PATH_DATASETS + path_dataset + pneumonia_path)
    files_opacities = os.listdir(PATH_DATASETS + path_dataset + opacities_path)

    files_covid = [covid_path + iFile for iFile in files_covid]
    files_normal = [normal_path + iFile for iFile in files_normal]
    files_pneumonia = [pneumonia_path + iFile for iFile in files_pneumonia]
    files_opacities = [opacities_path + iFile for iFile in files_opacities]

    # Test partition
    files_covid_test = files_covid[:n_covid_test]
    files_normal_test = files_normal[:n_normal_test]
    files_pneumonia_test = files_pneumonia[:n_pneumonia_test]
    files_opacities_test = files_opacities[:n_opacities_train]

    labels_covid = [1 for i in files_covid_test] + [0 for i in files_normal_test] + [0 for i in files_pneumonia_test] + [0 for i in files_opacities_test]
    labels_no_covid = [0 for i in files_covid_test] + [1 for i in files_normal_test] + [0 for i in files_pneumonia_test] + [0 for i in files_opacities_test]
    labels_pneumonia = [0 for i in files_covid_test] + [0 for i in files_normal_test] + [1 for i in files_pneumonia_test] + [0 for i in files_opacities_test]
    labels_opacities = [0 for i in files_covid_test] + [0 for i in files_normal_test] + [0 for i in files_pneumonia_test] + [1 for i in files_opacities_test]

    # Create table
    df = pd.DataFrame(list(zip(files_covid_test + files_normal_test + files_pneumonia_test + files_opacities_test,
                               labels_covid, labels_no_covid, labels_pneumonia, labels_opacities)),
                      columns=['Path', 'COVID', 'Normal', 'Pneumonia', 'Lung Opacity'])
    df.to_csv("./local_data/partitions/transferability/covid_test.csv")

    # Train partition
    files_covid_train = files_covid[n_covid_test:n_covid_test+n_covid_train]
    files_normal_train = files_normal[n_normal_test:n_normal_test+n_normal_train]
    files_pneumonia_train = files_pneumonia[n_pneumonia_test:n_pneumonia_test + n_pneumonia_train]
    files_opacities_train = files_opacities[n_opacities_train:n_opacities_train + n_opacities_train]

    labels_covid = [1 for i in files_covid_train] + [0 for i in files_normal_train] + [0 for i in files_pneumonia_train] + [0 for i in files_opacities_train]
    labels_no_covid = [0 for i in files_covid_train] + [1 for i in files_normal_train] + [0 for i in files_pneumonia_train] + [0 for i in files_opacities_train]
    labels_pneumonia = [0 for i in files_covid_train] + [0 for i in files_normal_train] + [1 for i in files_pneumonia_train] + [0 for i in files_opacities_train]
    labels_opacities = [0 for i in files_covid_train] + [0 for i in files_normal_train] + [0 for i in files_pneumonia_train] + [1 for i in files_opacities_train]

    # Create table
    df = pd.DataFrame(list(zip(files_covid_train + files_normal_train + files_pneumonia_train + files_opacities_train,
                               labels_covid, labels_no_covid, labels_pneumonia, labels_opacities)),
                      columns=['Path', 'COVID', 'Normal', 'Pneumonia', 'Lung Opacity'])
    df.to_csv("./local_data/partitions/transferability/covid_train.csv")


def prepare_siim_pneumothorax_partition():

    n_train = 2400
    n_test = 600

    # Read original training dataset
    df = pd.read_csv(PATH_DATASETS + "SIIM_Pneumothorax/" + "train-rle.csv")

    # get image paths
    img_paths = {}
    for subdir, dirs, files in tqdm.tqdm(os.walk(PATH_DATASETS + "SIIM_Pneumothorax/" + "dicom-images-train")):
        for f in files:
            if "dcm" in f:
                # remove dcm
                file_id = f[:-4]
                img_paths[file_id] = os.path.join(subdir.replace(PATH_DATASETS + "SIIM_Pneumothorax/", ""), f)

    # no encoded pixels mean healthy
    df["Pneumothorax"] = df.apply(
        lambda x: 0.0 if x[" EncodedPixels"] == " -1" else 1.0, axis=1
    )
    df["Normal"] = df.apply(
        lambda x: 1.0 if x[" EncodedPixels"] == " -1" else 0.0, axis=1
    )
    df["Path"] = df["ImageId"].apply(lambda x: img_paths[x])
    df = df[["Path", "Normal", "Pneumothorax"]]

    idx_train = list(np.argwhere(df["Pneumothorax"].values == 1)[:n_train].flatten()) +\
          list(np.argwhere(df["Normal"].values == 1)[:n_train].flatten())

    idx_test = list(np.argwhere(df["Pneumothorax"].values == 1)[n_train:n_train+n_test].flatten()) +\
          list(np.argwhere(df["Normal"].values == 1)[n_train:n_train+n_test].flatten())

    train_df = df.iloc[idx_train].reset_index()
    test_df = df.iloc[idx_test].reset_index()

    print(f"Number of train samples: {len(train_df)}")
    print(train_df["Pneumothorax"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["Pneumothorax"].value_counts())

    train_df.to_csv("./local_data/partitions/transferability/SIIM_train.csv")
    test_df.to_csv("./local_data/partitions/transferability/SIIM_test.csv")


def prepare_vindr_partition():

    # Define number of samples to retrieve
    n_train = 350
    n_test = 60

    # Define class categories
    categories_vindr = ['No finding', 'Bronchitis', 'Brocho-pneumonia', 'Bronchiolitis', 'Situs inversus', 'Pneumonia',
                        'Pleuro-pneumonia', 'Diagphramatic hernia', 'Tuberculosis', 'Congenital emphysema', 'CPAM',
                        'Hyaline membrane disease', 'Mediastinal tumor', 'Lung tumor']
    # Select categories with large number of samples
    categories_vindr_recurrent = ["No finding", "Pneumonia", "Bronchitis", "Brocho-pneumonia", "Bronchiolitis"]

    # Set paths and dataframe folders
    path_dataset = "VinDr-PCXR/"
    test_paths = {"path": "test/", "dataframe": "image_labels_test.csv"}
    train_paths = {"path": "train/", "dataframe": "image_labels_train.csv"}

    # Load datadrames
    df_train = pd.read_csv(PATH_DATASETS + path_dataset + train_paths["dataframe"])
    df_test = pd.read_csv(PATH_DATASETS + path_dataset + test_paths["dataframe"])

    # Create Path column
    df_train["Path"] = [train_paths["path"] + i + ".dicom" for i in df_train["image_id"].to_list()]
    df_test["Path"] = [test_paths["path"] + i + ".dicom" for i in df_test["image_id"].to_list()]

    # Filter target categories
    df_train = df_train[["Path"] + categories_vindr_recurrent]
    df_test = df_test[["Path"] + categories_vindr_recurrent]

    # Train partition
    df = df_train
    # Filter samples with more than one annotated category
    df = df[df[categories_vindr_recurrent].values.sum(-1) == 1]
    # Select n samples per category
    indexes_all = []
    for iCat in categories_vindr_recurrent:
        # Select category index
        idx_category = (np.argwhere(df[iCat].values)[0:n_train]).flatten()
        indexes_all.extend(idx_category)
    df = df.iloc[indexes_all, :]
    df = df.reset_index()
    # Rename column
    df = df.rename(columns={'No finding': 'No Finding'})
    # Save dataframe
    df.to_csv("./local_data/partitions/transferability/vindr_train.csv")

    df = df_test
    # Filter samples with more than one annotated category
    df = df[df[categories_vindr_recurrent].values.sum(-1) == 1]
    # Select n samples per category
    indexes_all = []
    for iCat in categories_vindr_recurrent:
        # Select category index
        idx_category = (np.argwhere(df[iCat].values)[0:n_test]).flatten()
        indexes_all.extend(idx_category)
    df = df.iloc[indexes_all, :]
    df = df.reset_index()
    # Rename column
    df = df.rename(columns={'No finding': 'No Finding'})
    # Save dataframe
    df.to_csv("./local_data/partitions/transferability/vindr_test.csv")


def preprocess_mimic_5x200_data():
    only_frontal = True
    tasks = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

    # Load tables with labels and metadata
    df_labels = pd.read_csv(PATH_DATASETS + "MIMIC-CXR-2/2.0.0/mimic-cxr-2.0.0-chexpert.csv").fillna(0)
    table_partitions = pd.read_csv(PATH_DATASETS + "MIMIC-CXR-2/2.0.0/" + "mimic-cxr-2.0.0-split.csv")
    table_metadata = pd.read_csv(PATH_DATASETS + "MIMIC-CXR-2/2.0.0/" + "mimic-cxr-2.0.0-metadata.csv")

    # Select only chest frontal views
    if only_frontal:
        table_metadata = table_metadata[(table_metadata["ViewPosition"] == "PA") | (table_metadata["ViewPosition"] == "AP")]

    # Align parition and metadata tables
    table_partitions = table_partitions[np.in1d(list(table_partitions["dicom_id"]), list(table_metadata["dicom_id"]))]
    table_partitions = table_partitions.reset_index()

    # Filter labels dataset: targeting samples with only one annotated category
    df_labels = df_labels[np.sum(df_labels[tasks].values == 1, -1) == 1]
    table_partitions = table_partitions[np.in1d(list(table_partitions["study_id"]), list(df_labels["study_id"]))]
    table_partitions = table_partitions.reset_index()

    # Select one image per subject and retrieve labels
    images, studies, subjects = table_partitions["dicom_id"], table_partitions["study_id"], table_partitions["subject_id"]
    data, subjects_included = [], []
    for i in range(len(table_partitions)):
        image_id, study_id, subject_id, folder = images[i], studies[i], subjects[i], str(subjects[i])[:2]
        relative_path = "files/p" + folder + "/p" + str(subject_id) + "/s" + str(study_id) + "/" + image_id + ".jpg"
        if os.path.isfile(PATH_DATASETS + "MIMIC-CXR-2/2.0.0/" + relative_path) and (subject_id not in subjects_included):
            subjects_included.append(subject_id)
            labels_id = df_labels[df_labels["study_id"] == study_id][tasks].values[0]

            data.append({"Path": "MIMIC-CXR-2/2.0.0/" + relative_path, "Atelectasis": labels_id[0],
                         "Cardiomegaly": labels_id[1], "Consolidation": labels_id[2], "Edema": labels_id[3],
                         "Pleural Effusion": labels_id[4]})

    # Create dataframe
    df = pd.DataFrame(data)

    # Create balanced dataset
    task_dfs = []
    for i, t in enumerate(tasks):
        index = np.zeros(5)
        index[i] = 1
        df_task = df[
            (df["Atelectasis"] == index[0])
            & (df["Cardiomegaly"] == index[1])
            & (df["Consolidation"] == index[2])
            & (df["Edema"] == index[3])
            & (df["Pleural Effusion"] == index[4])
        ]
        df_task = df_task.sample(n=200, random_state=42)
        task_dfs.append(df_task)
    df_200 = pd.concat(task_dfs)
    df_200 = df_200.reset_index()

    # Save dataframe
    df_200.to_csv("./local_data/partitions/transferability/mimic_5x200.csv")


def main(create_pretraining=False):

    # Create folders
    if not os.path.isdir("./local_data/partitions/pretraining/"):
        os.makedirs("./local_data/partitions/pretraining/")
    if not os.path.isdir("./local_data/partitions/transferability/"):
        os.makedirs("./local_data/partitions/transferability/")

    # 1. Create MIMIC200x5 partition.
    print("Creating MIMIC5x200...")
    preprocess_mimic_5x200_data()

    # 2. Create MIMIC pre-training partition
    if create_pretraining:
        print("Creating MIMIC pre-training partition...")
        prepare_mimic_partition_pretrain()

    # 3. Create CheXpert pretraining partition
    if create_pretraining:
        print("Creating CheXpert pre-training partition...")
        prepare_chexpert_partition_pretrain()

    # 4. Create PadChest partition
    if create_pretraining:
        print("Creating PadChest partition...")
        prepare_padchest_partition_pretrain()

    # 5. Create RSNA partition
    print("Creating RSNA partition...")
    prepare_rsna_pneumonia_partition() # Proposed partition based on detailed class information
    prepare_rsna_pneumonia_partition_gloria()  # GlorIA partition based on local segmentation

    # 6. Create SSIM partition
    print("Creating SSIM partition...")
    prepare_siim_pneumothorax_partition()

    # 7. Create COVID partition
    print("Creating COVID partition...")
    prepare_covid_partition()

    # 8. Create NIH-LT partition
    print("Creating NIH partition...")
    prepare_nih_lt_partition()

    # 9. Create VinDR partition
    print("Creating VinDR partition...")
    prepare_vindr_partition()


if __name__ == "__main__":
    main()