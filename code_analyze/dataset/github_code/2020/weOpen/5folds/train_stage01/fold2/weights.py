import numpy as np


def get_class_weights(total_counts, class_positive_counts, multiply):
    """
    Calculate class_weight used in training

    Arguments:
    total_counts - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    multiply - int, positve weighting multiply
    use_class_balancing - boolean 

    Returns:
    class_weight - dict of dict, ex: {"Effusion": { 0: 0.01, 1: 0.99 }, ... }
    """
    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }

    class_names = list(class_positive_counts.keys())
    label_counts = np.array(list(class_positive_counts.values()))
    class_weights = []
    for i, class_name in enumerate(class_names):
        if(str(class_name) == 'Hernia' or str(class_name) == 'hernia'): #1
            class_weights.append({1:0.7260310122144928,0:0.27396898778550716})
        if(str(class_name) == 'Pneumonia' or str(class_name) == 'pneumonia'): #2
            class_weights.append({0:0.887921058482114,1:0.11207894151788601})
        if(str(class_name) == 'Fibrosis' or str(class_name) == 'fibrosis'): #3
            class_weights.append({0:0.9031142272380817,1:0.09688577276191833})
        if(str(class_name) == 'Edema' or str(class_name) == 'edema'): #4
            class_weights.append({0:0.9258290211260599,1:0.07417097887394011})
        if(str(class_name) == 'Emphysema' or str(class_name) == 'emphysema'): #5
            class_weights.append({0:0.9309729671935325,1:0.06902703280646749})
        if(str(class_name) == 'Cardiomegaly' or str(class_name) == 'cardiomegaly'): #6
            class_weights.append({0:0.9400531586640466,1:0.0599468413359534})
        if(str(class_name) == 'Pleural_Thickening' or str(class_name) == 'pleural_thickening' or str(class_name) == 'Pleural_thickening' or str(class_name) == 'pleural_Thickening'): #7
            class_weights.append({0:0.9486392720795189,1:0.05136072792048107})
        if(str(class_name) == 'Pneumothorax' or str(class_name) == 'pneumothorax'): #8
            class_weights.append({0:0.9594233806041276,1:0.04057661939587243})
        if(str(class_name) == 'Consolidation' or str(class_name) == 'consolidation'): #9
            class_weights.append({0:0.9624678616588797,1:0.03753213834112039})
        if(str(class_name) == 'Mass' or str(class_name) == 'mass'): #10
            class_weights.append({0:0.9650454792339842,1:0.03495452076601584})
        if(str(class_name) == 'Nodule' or str(class_name) == 'nodule'): #11
            class_weights.append({0:0.967391854805868,1:0.032608145194132035})
        if(str(class_name) == 'Atelectasis' or str(class_name) == 'atelectasis'): #12
            class_weights.append({0:0.9764272698419662,1:0.023572730158033888})
        if(str(class_name) == 'Effusion' or str(class_name) == 'effusion'): #13
            class_weights.append({0:0.9780334749965782,1:0.021966525003421725})
        if(str(class_name) == 'Infiltration' or str(class_name) == 'infiltration'): #4
            class_weights.append({0:0.9807119862897359,1:0.019288013710264135})
    return class_weights
