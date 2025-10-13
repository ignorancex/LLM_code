"""
Script to retrieve transferability experiments setting
(i.e. dataframe path, target classes, and task type)
"""


def get_experiment_setting(experiment):

    # Chexpert
    if experiment == "chexpert_5x200":
        setting = {"experiment": "chexpert5x200",
                   "targets": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
                   "dataframe": "./local_data/partitions/transferability/chexpert_5x200.csv",
                   "base_samples_path": "CheXpert/CheXpert-v1.0/"}

    # MIMIC
    elif experiment == "mimic_5x200":
        setting = {"experiment": "mimic5x200",
                   "targets": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
                   "dataframe": "./local_data/partitions/transferability/mimic_5x200.csv",
                   "base_samples_path": ""}

    # SSIM
    elif experiment == "ssim_train":
        setting = {"experiment": "ssim",
                   "targets": ["Normal", "Pneumothorax"],
                   "dataframe": "./local_data/partitions/transferability/SIIM_train.csv",
                   "base_samples_path": "SIIM_Pneumothorax/"}
    elif experiment == "ssim_test":
        setting = {"experiment": "ssim",
                   "targets": ["Normal", "Pneumothorax"],
                   "dataframe": "./local_data/partitions/transferability/SIIM_test.csv",
                   "base_samples_path": "SIIM_Pneumothorax/"}

    # RNSA
    elif experiment == "rsna_train":
        setting = {"experiment": "rsna",
                   "targets": ["Normal", "Pneumonia"],
                   "dataframe": "./local_data/partitions/transferability/RSNA_train.csv",
                   "base_samples_path": "RSNA_PNEUMONIA/"}
    elif experiment == "rsna_test":
        setting = {"experiment": "rsna",
                   "targets": ["Normal", "Pneumonia"],
                   "dataframe": "./local_data/partitions/transferability/RSNA_test.csv",
                   "base_samples_path": "RSNA_PNEUMONIA/"}
    elif experiment == "rsna_gloria_train":
        setting = {"experiment": "rsna",
                   "targets": ["Normal", "Pneumonia"],
                   "dataframe": "./local_data/partitions/transferability/RSNA_gloria_train.csv",
                   "base_samples_path": "RSNA_PNEUMONIA/"}
    elif experiment == "rsna_gloria_test":
        setting = {"experiment": "rsna",
                   "targets": ["Normal", "Pneumonia"],
                   "dataframe": "./local_data/partitions/transferability/RSNA_gloria_test.csv",
                   "base_samples_path": "RSNA_PNEUMONIA/"}

    # COVID datasets
    elif experiment == "covid_train_2class":
        setting = {"experiment": "covid2class",
                   "targets": ["Normal", "COVID"],
                   "dataframe": "./local_data/partitions/transferability/covid_train.csv",
                   "base_samples_path": "COVID-19_Radiography_Dataset/"}
    elif experiment == "covid_test_2class":
        setting = {"experiment": "covid2class",
                   "targets": ["Normal", "COVID"],
                   "dataframe": "./local_data/partitions/transferability/covid_test.csv",
                   "base_samples_path": "COVID-19_Radiography_Dataset/"}
    elif experiment == "covid_train_3class":
        setting = {"experiment": "covid3class",
                   "targets": ["Normal", "COVID", "Pneumonia"],
                   "dataframe": "./local_data/partitions/transferability/covid_train.csv",
                   "base_samples_path": "COVID-19_Radiography_Dataset/"}
    elif experiment == "covid_test_3class":
        setting = {"experiment": "covid3class",
                   "targets": ["Normal", "COVID", "Pneumonia"],
                   "dataframe": "./local_data/partitions/transferability/covid_test.csv",
                   "base_samples_path": "COVID-19_Radiography_Dataset/"}
    elif experiment == "covid_train_4class":
        setting = {"experiment": "covid4class",
                   "targets": ["Normal", "COVID", "Pneumonia", "Lung Opacity"],
                   "dataframe": "./local_data/partitions/transferability/covid_train.csv",
                   "base_samples_path": "COVID-19_Radiography_Dataset/"}
    elif experiment == "covid_test_4class":
        setting = {"experiment": "covid4class",
                   "targets": ["Normal", "COVID", "Pneumonia", "Lung Opacity"],
                   "dataframe": "./local_data/partitions/transferability/covid_test.csv",
                   "base_samples_path": "COVID-19_Radiography_Dataset/"}

    # NIH-LT
    elif experiment == "nihlt_train":
        setting = {"experiment": "nihlt",
                   "targets": ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
                               "Nodule", "Pneumonia", "No Finding", "Pneumothorax", "Consolidation",
                               "Edema", "Emphysema", "Fibrosis", "Pleural Thickening", "Pneumoperitoneum",
                               "Pneumomediastinum", "Subcutaneous Emphysema", "Tortuous Aorta",
                               "Calcification of the Aorta"],
                   "dataframe": "./local_data/partitions/transferability/nih_train.csv",
                   "base_samples_path": "NIH/"}
    elif experiment == "nihlt_test":
        setting = {"experiment": "nihlt",
                   "targets": ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
                               "Nodule", "Pneumonia", "No Finding", "Pneumothorax", "Consolidation",
                               "Edema", "Emphysema", "Fibrosis", "Pleural Thickening", "Pneumoperitoneum",
                               "Pneumomediastinum", "Subcutaneous Emphysema", "Tortuous Aorta",
                               "Calcification of the Aorta"],
                   "dataframe": "./local_data/partitions/transferability/nih_test.csv",
                   "base_samples_path": "NIH/"}
    elif experiment == "nihlt_train_base":
        setting = {"experiment": "nihlt_base",
                   "targets": ["Atelectasis", "Cardiomegaly", "Effusion", "Pneumonia", "No Finding", "Pneumothorax",
                               "Consolidation", "Edema"],
                   "dataframe": "./local_data/partitions/transferability/nih_train.csv",
                   "base_samples_path": "NIH/"}
    elif experiment == "nihlt_test_base":
        setting = {"experiment": "nihlt_base",
                   "targets": ["Atelectasis", "Cardiomegaly", "Effusion", "Pneumonia", "No Finding", "Pneumothorax",
                               "Consolidation", "Edema"],
                   "dataframe": "./local_data/partitions/transferability/nih_test.csv",
                   "base_samples_path": "NIH/"}
    elif experiment == "nihlt_train_new":
        setting = {"experiment": "nihlt_new",
                   "targets": ["Infiltration", "Mass", "Nodule", "Emphysema", "Fibrosis", "Pleural Thickening",
                               "Pneumoperitoneum", "Pneumomediastinum", "Subcutaneous Emphysema", "Tortuous Aorta",
                               "Calcification of the Aorta"],
                   "dataframe": "./local_data/partitions/transferability/nih_train.csv",
                   "base_samples_path": "NIH/"}
    elif experiment == "nihlt_test_new":
        setting = {"experiment": "nihlt_new",
                   "targets": ["Infiltration", "Mass", "Nodule", "Emphysema", "Fibrosis", "Pleural Thickening",
                               "Pneumoperitoneum", "Pneumomediastinum", "Subcutaneous Emphysema", "Tortuous Aorta",
                               "Calcification of the Aorta"],
                   "dataframe": "./local_data/partitions/transferability/nih_test.csv",
                   "base_samples_path": "NIH/"}

    # VinDR
    elif experiment == "vindr_train":
        setting = {"experiment": "vindr",
                   "targets": ["No Finding", "Pneumonia", "Bronchitis", "Brocho-pneumonia", "Bronchiolitis"],
                   "dataframe": "./local_data/partitions/transferability/vindr_train.csv",
                   "base_samples_path": "VinDr-PCXR/"}
    elif experiment == "vindr_test":
        setting = {"experiment": "vindr",
                   "targets": ["No Finding", "Pneumonia", "Bronchitis", "Brocho-pneumonia", "Bronchiolitis"],
                   "dataframe": "./local_data/partitions/transferability/vindr_test.csv",
                   "base_samples_path": "VinDr-PCXR/"}
    elif experiment == "vindr_train_base":
        setting = {"experiment": "vindr_base",
                   "targets": ["No Finding", "Pneumonia"],
                   "dataframe": "./local_data/partitions/transferability/vindr_train.csv",
                   "base_samples_path": "VinDr-PCXR/"}
    elif experiment == "vindr_test_base":
        setting = {"experiment": "vindr_base",
                   "targets": ["No Finding", "Pneumonia"],
                   "dataframe": "./local_data/partitions/transferability/vindr_test.csv",
                   "base_samples_path": "VinDr-PCXR/"}
    elif experiment == "vindr_train_new":
        setting = {"experiment": "vindr_new",
                   "targets": ["Bronchitis", "Brocho-pneumonia", "Bronchiolitis"],
                   "dataframe": "./local_data/partitions/transferability/vindr_train.csv",
                   "base_samples_path": "VinDr-PCXR/"}
    elif experiment == "vindr_test_new":
        setting = {"experiment": "vindr_test_new",
                   "targets": ["Bronchitis", "Brocho-pneumonia", "Bronchiolitis"],
                   "dataframe": "./local_data/partitions/transferability/vindr_test.csv",
                   "base_samples_path": "VinDr-PCXR/"}
    else:
        setting = None
        print("Experiment not prepared...")

    return setting