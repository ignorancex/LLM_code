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
            class_weights.append({1:0.679685206627672,0:0.32031479337232804})
        if(str(class_name) == 'Pneumonia' or str(class_name) == 'pneumonia'): #2
            class_weights.append({0:0.881088142036747,1:0.11891185796325295})
        if(str(class_name) == 'Fibrosis' or str(class_name) == 'fibrosis'): #3
            class_weights.append({0:0.9059375434856511,1:0.09406245651434883})
        if(str(class_name) == 'Edema' or str(class_name) == 'edema'): #4
            class_weights.append({0:0.9233487959688106,1:0.07665120403118937})
        if(str(class_name) == 'Emphysema' or str(class_name) == 'emphysema'): #5
            class_weights.append({0:0.9235885787842366,1:0.0764114212157634})
        if(str(class_name) == 'Cardiomegaly' or str(class_name) == 'cardiomegaly'): #6
            class_weights.append({0:0.9333107477727204,1:0.0666892522272796})
        if(str(class_name) == 'Pleural_Thickening' or str(class_name) == 'pleural_thickening' or str(class_name) == 'Pleural_thickening' or str(class_name) == 'pleural_Thickening'): #7
            class_weights.append({0:0.9431179943581464,1:0.056882005641853646})
        if(str(class_name) == 'Pneumothorax' or str(class_name) == 'pneumothorax'): #8
            class_weights.append({0:0.9557172899373126,1:0.04428271006268737})
        if(str(class_name) == 'Consolidation' or str(class_name) == 'consolidation'): #9
            class_weights.append({0:0.9581388688544957,1:0.04186113114550423})
        if(str(class_name) == 'Mass' or str(class_name) == 'mass'): #10
            class_weights.append({0:0.9622303192607198,1:0.03776968073928026})
        if(str(class_name) == 'Nodule' or str(class_name) == 'nodule'): #11
            class_weights.append({0:0.9644892042146104,1:0.03551079578538967})
        if(str(class_name) == 'Atelectasis' or str(class_name) == 'atelectasis'): #12
            class_weights.append({0:0.974436988289452,1:0.025563011710548})
        if(str(class_name) == 'Effusion' or str(class_name) == 'effusion'): #13
            class_weights.append({0:0.975741603148118,1:0.02425839685188202})
        if(str(class_name) == 'Infiltration' or str(class_name) == 'infiltration'): #4
            class_weights.append({0:0.9785391305166514,1:0.02146086948334855})
    return class_weights
