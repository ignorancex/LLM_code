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
            class_weights.append({1:0.7236677571675076,0:0.27633224283249236})
        if(str(class_name) == 'Pneumonia' or str(class_name) == 'pneumonia'): #2
            class_weights.append({0:0.8907219629050631,1:0.10927803709493689})
        if(str(class_name) == 'Fibrosis' or str(class_name) == 'fibrosis'): #3
            class_weights.append({0:0.8919634412359331,1:0.10803655876406691})
        if(str(class_name) == 'Edema' or str(class_name) == 'edema'): #4
            class_weights.append({0:0.9319271929109054,1:0.06807280708909452})
        if(str(class_name) == 'Emphysema' or str(class_name) == 'emphysema'): #5
            class_weights.append({0:0.9337897103865062,1:0.06621028961349384})
        if(str(class_name) == 'Cardiomegaly' or str(class_name) == 'cardiomegaly'): #6
            class_weights.append({0:0.9347068811318509,1:0.06529311886814916})
        if(str(class_name) == 'Pleural_Thickening' or str(class_name) == 'pleural_thickening' or str(class_name) == 'Pleural_thickening' or str(class_name) == 'pleural_Thickening'): #7
            class_weights.append({0:0.9471736691180249,1:0.05282633088197504})
        if(str(class_name) == 'Pneumothorax' or str(class_name) == 'pneumothorax'): #8
            class_weights.append({0:0.9598208840445588,1:0.040179115955441255})
        if(str(class_name) == 'Consolidation' or str(class_name) == 'consolidation'): #9
            class_weights.append({0:0.9634611131859865,1:0.036538886814013516})
        if(str(class_name) == 'Mass' or str(class_name) == 'mass'): #10
            class_weights.append({0:0.966524458089011,1:0.03347554191098903})
        if(str(class_name) == 'Nodule' or str(class_name) == 'nodule'): #11
            class_weights.append({0:0.9675667635564271,1:0.03243323644357292})
        if(str(class_name) == 'Atelectasis' or str(class_name) == 'atelectasis'): #12
            class_weights.append({0:0.9768238833524878,1:0.02317611664751221})
        if(str(class_name) == 'Effusion' or str(class_name) == 'effusion'): #13
            class_weights.append({0:0.9783726554958037,1:0.02162734450419629})
        if(str(class_name) == 'Infiltration' or str(class_name) == 'infiltration'): #4
            class_weights.append({0:0.980815141754949,1:0.019184858245051006})
    return class_weights
