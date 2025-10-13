"""
drawn from:
 Gloria github: https://github.com/marshuang80/gloria
and
 MedCLIP github: https://github.com/RyanWangZf/MedCLIP
"""

import random

CATEGORIES_ALL = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion", "Lung Opacity",
                  "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
                  "Pleural Other", "Fracture", "Support Devices", "Normal", "COVID", "Infiltration", "Mass",
                  "Nodule", "Emphysema", "Fibrosis", "Pleural Thickening", "Pneumoperitoneum", "Pneumomediastinum",
                  "Subcutaneous Emphysema", "Tortuous Aorta", "Calcification of the Aorta", "Bronchitis",
                  "Brocho-pneumonia", "Bronchiolitis", "Situs Inversus", "Pleuropneumonia", "Diafragmatic hernia",
                  "Tuberculosis", "Congenital Pulmonary Airwat Malformation", "Hyaline Membrane Disease",
                  "Mediastinal Tumor", "Lung Tumor", "Effusion"]

ASSEMBLE_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": ["subsegmental atelectasis", "linear atelectasis", "trace atelectasis", "bibasilar atelectasis",
                    "retrocardiac atelectasis", "bandlike atelectasis", "residual atelectasis"],
        "location": ["at the mid lung zone", "at the upper lung zone", "at the right lung zone",
                     "at the left lung zone", "at the lung bases", "at the right lung base", "at the left lung base",
                     "at the bilateral lung bases", "at the left lower lobe", "at the right lower lobe"],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": ["cardiac silhouette size is upper limits of normal", "cardiomegaly which is unchanged",
                    "mildly prominent cardiac silhouette", "portable view of the chest demonstrates stable cardiomegaly",
                    "portable view of the chest demonstrates mild cardiomegaly", "persistent severe cardiomegaly",
                    "heart size is borderline enlarged", "cardiomegaly unchanged",
                    "heart size is at the upper limits of normal", "redemonstration of cardiomegaly",
                    "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
                    "cardiac silhouette size is mildly enlarged",
                    "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
                    "heart size remains at mildly enlarged",
                    "persistent cardiomegaly with prominent upper lobe vessels"],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": ["bilateral consolidation", "reticular consolidation", "retrocardiac consolidation",
                    "patchy consolidation", "airspace consolidation", "partial consolidation"],
        "location": ["at the lower lung zone", "at the upper lung zone", "at the left lower lobe",
                     "at the right lower lobe", "at the left upper lobe", "at the right uppper lobe",
                     "at the right lung base", "at the left lung base"],
    },
    "Edema": {
        "severity": ["", "mild", "improvement in", "presistent", "moderate", "decreased"],
        "subtype": ["edema", "pulmonary edema", "trace interstitial edema", "pulmonary interstitial edema"],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": ["pleural effusion", "bilateral pleural effusion", "subpulmonic pleural effusion",
                    "bilateral pleural effusion"],
    },
    "Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": ["pleural effusion", "bilateral effusion", "subpulmonic effusion",
                    "bilateral effusion"],
    },
    "Pneumothorax": {
        "severity": [""],
        "subtype": ["pneumothorax"],
        "location": ["at the mid lung zone", "at the upper lung zone", "at the right lung zone",
                     "at the left lung zone", "at the lung bases", "at the right lung base", "at the left lung base",
                     "at the bilateral lung bases", "at the left lower lobe", "at the right lower lobe",
                     "at the left middle lobe", "at the right middle lobe"],
    },
    "Pneumonia": {
        "severity": ['round', 'early', 'focal', 'multifocal', 'small', ''],
        "subtype": ["pneumonia", 'bacterial', 'viral', 'mycoplasma', ''],
        "location": ["at the mid lung zone", "at the upper lung zone", "at the right lung zone",
                     "at the left lung zone", "at the lung bases", "at the right lung base", "at the left lung base",
                     "at the bilateral lung bases", "at the left lower lobe", "at the right lower lobe",
                     "at the left middle lobe", "at the right middle lobe"],
    },
    "COVID": {
        "severity": ["patchy", "confluent"],
        "description": ["ground glass"],
        "subtype": ["covid", "opacity", "consolidation"],
        "location": ["in peripheral", "in mid", "in lower"],
    },
    "Normal": {
        "severity": [""],
        "description": [""],
        "subtype": ["normal", "no findings"],
        "location": [""],
    },
}

DESCRIPTIONS_PROMPTS = {"No Finding": ["no findings"],
                        "Enlarged Cardiomediastinum": ["enlarged cardiomediastinum"],
                        "Cardiomegaly": ["the heart is enlarged", "cardiomegaly"],
                        "Lung Lesion": ["lung lesion"],
                        "Lung Opacity": ["area of hazy opacification due to air displacement by fluid, airway collapse,"
                                         " fibrosis, or a neoplastic process." "It is causes include infections,"
                                         " interstitial lung disease, and pulmonary edema", "lung opacity"],
                        "Edema": ["pulmonary congestion", "excessive liquid accumulation in the tissue and air spaces"
                                  " of the lungs", "fluid in the alveolar walls", "edema"],
                        "Consolidation": ["region of normally compressible lung tissue that has filled with "
                                          " instead of air", "consolidation"],
                        "Pneumonia": ["pneumonia is an inflammatory condition of the lung primarily small air sacs"
                                      " known as alveoli", "pneumonia may present with opacities", "Complications such"
                                      " as pleural effusion may also be found increasing the diagnostic accuracy of "
                                      "lung consolidation and pleural effusion", "pneumonia"],
                        "Atelectasis": ["collapse or closure of a lung resulting in reduced or absent gas exchange",
                                        "Findings can include lung opacification and loss of lung volume",
                                        "atelectasis"],
                        "Pneumothorax": ["abnormal collection of air in the pleural space between the lung and the "
                                         "chest wall", "it may be caused by pneumonia or fibrosis and other diseases",
                                         "pneumothorax"],
                        "Pleural Effusion": ["pleural Effusion"],
                        "Pleural Other": ["pleural lesion"],
                        "Fracture": ["break in a rib bone", "fracture"],
                        "Support Devices": ["support devices"],
                        "Normal": ["absence of diseases and infirmity findings, indicating the structure is normal.",
                                   "no findings"],
                        "COVID": ["it is a contagious disease caused by a virus.", "ground-glass opacities,"
                                  " consolidation, thickening, pleural effusions commonly appear in infection."],
                        "Infiltration": ["infiltration", "substance denser than air, such as pus, blood, or protein,"
                                         " which lingers within the parenchyma of the lungs."],
                        "Mass": ["mass"],
                        "Nodule": ["nodule"],
                        "Emphysema": ["emphysema", "lower respiratory tract disease, characterized by air-filled spaces"
                                      "in the lungs, that can vary in size and may be very large."],
                        "Fibrosis": ["fibrosis"],
                        "Pleural Thickening": ["pleural thickening"],
                        "Pneumoperitoneum": ["pneumoperitoneum"],
                        "Pneumomediastinum": ["pneumomediastinum"],
                        "Subcutaneous Emphysema": ["subcutaneous emphysema"],
                        "Tortuous Aorta": ["tortuous aorta", "aorta is slightly tortuous", "varicose veins"],
                        "Calcification of the Aorta": ["calcification of the aorta"],
                        "Bronchitis": ["bronchitis"],
                        "Broncho-pneumonia": ["broncho-pneumonia"],
                        "Bronchiolitis": ["bronchiolitis"],
                        "Situs Inversus": ["situs inversus"],
                        "Pleuropneumonia": ["pleuropneumonia"],
                        "Diafragmatic hernia": ["diafragmatic hernia"],
                        "Tuberculosis": ["tuberculosis"],
                        "Congenital Pulmonary Airwat Malformation": ["congenital pulmonary airwat malformation"],
                        "Hyaline Membrane Disease": ["hyaline membrane disease"],
                        "Mediastinal Tumor": ["mediastinal tumor"],
                        "Lung Tumor": ["lung tumor"],
                        "Effusion": ["pleural Effusion"],
                        }


def generate_prompt_ensemble(n=100):

    # Generate missing categories
    for iCategory in CATEGORIES_ALL:
        if iCategory not in list(ASSEMBLE_PROMPTS.keys()):
            ASSEMBLE_PROMPTS[iCategory] = {}
            ASSEMBLE_PROMPTS[iCategory]["severity"] = {""}
            ASSEMBLE_PROMPTS[iCategory]["description"] = {""}
            ASSEMBLE_PROMPTS[iCategory]["subtype"] = {iCategory}
            ASSEMBLE_PROMPTS[iCategory]["location"] = {""}

    # Generate prompts
    prompts = {}
    for k, v in ASSEMBLE_PROMPTS.items():
        cls_prompts = []
        keys = list(v.keys())

        # severity
        for k0 in v[keys[0]]:
            # subtype
            for k1 in v[keys[1]]:
                # location
                for k2 in v[keys[2]]:
                    prompt = f"{k0} {k1} {k2}".replace("   ", " ").replace("  ", " ")
                    if prompt[0] == " ":
                        prompt = prompt[1:]
                    if prompt[-1] == " ":
                        prompt = prompt[:-1]
                    cls_prompts.append(prompt.lower())

        # randomly sample n prompts for zero-shot classification
        if n is not None and n < len(cls_prompts):
            prompts[k] = random.sample(cls_prompts, n)
        else:
            prompts[k] = cls_prompts
        print(f'sample {len(prompts[k])} num of prompts for {k} from total {len(cls_prompts)}')
    return prompts


def generate_name_prompt(template="A radiology image of [CLS]"):
    prompts = {}
    for iCat in CATEGORIES_ALL:
        prompts[iCat] = template.replace("[CLS]", iCat).lower()
    return prompts