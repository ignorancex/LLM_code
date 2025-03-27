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
        if(str(class_name) == 'Hernia' or str(class_name) == 'hernia'): #14
            class_weights.append({1:0.720226409263611,0:0.27977359073638897})
        if(str(class_name) == 'Pneumonia' or str(class_name) == 'pneumonia'): #7
            class_weights.append({0:0.8859702012473223,1:0.11402979875267771})
        if(str(class_name) == 'Fibrosis' or str(class_name) == 'fibrosis'): #12
            class_weights.append({0:0.9021976306069932,1:0.09780236939300682})
        if(str(class_name) == 'Edema' or str(class_name) == 'edema'): #10
            class_weights.append({0:0.9298929992036218,1:0.07010700079637826})
        if(str(class_name) == 'Emphysema' or str(class_name) == 'emphysema'): #11
            class_weights.append({0:0.9335352709009039,1:0.06646472909909606})
        if(str(class_name) == 'Cardiomegaly' or str(class_name) == 'cardiomegaly'): #2
            class_weights.append({0:0.9379028967906056,1:0.06209710320939444})
        if(str(class_name) == 'Pleural_Thickening' or str(class_name) == 'pleural_thickening' or str(class_name) == 'Pleural_thickening' or str(class_name) == 'pleural_Thickening'): # 13
            class_weights.append({0:0.9453965277787032,1:
0.05460347222129675})
        if(str(class_name) == 'Pneumothorax' or str(class_name) == 'pneumothorax'): #8
            class_weights.append({0:0.9586866934982315,1:0.04131330650176841})
        if(str(class_name) == 'Consolidation' or str(class_name) == 'consolidation'): #9
            class_weights.append({0:0.9623146440112557,1:0.03768535598874437})
        if(str(class_name) == 'Mass' or str(class_name) == 'mass'): #5
            class_weights.append({0:0.9642020357560434,1:0.03579796424395662})
        if(str(class_name) == 'Nodule' or str(class_name) == 'nodule'): #6
            class_weights.append({0:0.9663727015263743,1:0.033627298473625666})
        if(str(class_name) == 'Atelectasis' or str(class_name) == 'atelectasis'): #1
            class_weights.append({0:0.976060692178489,1:0.023939307821511})
        if(str(class_name) == 'Effusion' or str(class_name) == 'effusion'): #3
            class_weights.append({0:0.977507900874183,1:0.02249209912581691})
        if(str(class_name) == 'Infiltration' or str(class_name) == 'infiltration'): #4
            class_weights.append({0:0.9801862148908839,1:0.01981378510911613})
    return class_weights
