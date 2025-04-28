"""


- The module implements ``scoring_f_max`` method. The method takes a list as input.The List consist of model pipline,real annotation numpy array and protein vector dataframe
The function , call intersection function for true positive and true negative value calculation.It calculates f score.


"""


from imblearn.pipeline import Pipeline

def intersection(real_annot, pred_annot):
    count = 0
    tn = 0
    tp = 0
    for i in range(len(real_annot)):
        if real_annot[i] == pred_annot[i]:
            if real_annot[i] == 0:
                tn += 1
            else:
                tp += 1
            count += 1

    return tn, tp



def scoring_f_max_machine(model_pipline,protein_representation_array,real_annots):

    tn=0
    tp=0
    
    pred_annots=model_pipline.predict(protein_representation_array)
   
    tn,tp=intersection(real_annots, pred_annots)
    fp = list(pred_annots).count(1) - tp
    fn = list(real_annots).count(0) - tn
    recall = tp /(1.0 + (tp + fn))
    precision = tp / (1.0 + (tp + fp))
    f = 0.0
    if precision + recall > 0:
        f = 2 * precision * recall / (precision + recall)
    
    return f

# f_max scoring function
def evaluate_annotation_f_max(real_annots, pred_annots):

    tn = 0
    tp = 0

    tn, tp = intersection(real_annots, pred_annots)
    fp = list(pred_annots).count(1) - tp
    fn = list(real_annots).count(0) - tn

    recall = tp / (1.0 + (tp + fn))
    precision = tp / (1.0 + (tp + fp))

    f = 0.0
    if precision + recall > 0:
        f = 2 * precision * recall / (precision + recall)

    return f