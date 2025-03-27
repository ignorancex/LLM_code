

import os 
import numpy as np
import pickle

def compute_performance_metrics(predictions,mask):

    tp = np.sum((predictions == 1) & (mask == 1)).item()
    fp = np.sum((predictions == 1) & (mask == 0)).item()
    tn = np.sum((predictions == 0) & (mask == 0)).item()
    fn = np.sum((predictions == 0) & (mask == 1)).item()
    
    acc = (tp + tn) / (tp + fp + tn + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1_score = (2 * precision * recall) / (precision + recall + 1e-7)

    intersection = np.sum((predictions == 1) & (mask == 1)).item()
    union = np.sum((predictions == 1) | (mask == 1)).item()
    iou = intersection / (union + 1e-7)

    return{'acc':acc,'f1':f1_score,'iou':iou,'precision':precision,'recall':recall}


def main(root,model,dataset):
    
    NDVI_dir = os.path.join(root,f"{model}_{dataset}_NDVI","pred_masks")
    RGB_dir =  os.path.join(root,f"{model}_{dataset}_RGB","pred_masks")

    weight_RGB = 0.5
    weight_NDVI = 0.5

    weight_RGB = weight_RGB / (weight_RGB + weight_NDVI)
    weight_NDVI = weight_NDVI / (weight_RGB + weight_NDVI)

    assert os.path.isdir(NDVI_dir), "NDVI path not recognized"
    assert os.path.isdir(RGB_dir), "NDVI path not recognized"

    ndvi_files = os.listdir(NDVI_dir)
    print(len(ndvi_files))


    rgb_files = os.listdir(RGB_dir)
    print(len(rgb_files))

    check = np.array([1 if a == b else 0 for a,b in zip(ndvi_files,rgb_files)])
    assert np.sum(check) == len(rgb_files)
        
    accuracies, f1_scores, recalls, precisions, dice_scores, ious = [], [], [], [], [], []

    for fileRGB,fileNDVI in zip(ndvi_files,rgb_files):
        with open(os.path.join(RGB_dir,fileRGB), 'rb') as handle:
            RGB_data = pickle.load(handle)
        
        with open(os.path.join(NDVI_dir,fileNDVI), 'rb') as handle:
            NDVI_data = pickle.load(handle)

        RGB_mask = RGB_data['pred']
        NDVI_mask = NDVI_data['pred']

        gt_mask = RGB_data['mask']

        late_fused_pred = weight_RGB*RGB_mask + weight_NDVI*NDVI_mask
        predictions = (late_fused_pred > 0.5)
        metricts = compute_performance_metrics(predictions,gt_mask)


        recalls.append(metricts['recall'])
        precisions.append(metricts['precision'])
        f1_scores.append(metricts['f1'])
        ious.append(metricts['iou'])
        accuracies.append(metricts['acc'])
    
    mean_precision = round(np.mean(precisions),3)
    mean_recall = round(np.mean(recalls),3)
    mean_f1_score = round(np.mean(f1_scores),3)
    mean_iou = round(np.mean(ious),3)
    mean_accuracy = round(np.mean(accuracies),3)
    
    content =  f"LF| -Model:{model} dataset {dataset}- Acc: {mean_accuracy}%. Prec: {mean_precision}%. Recall: {mean_recall}%. F1: {mean_f1_score}%. IoU: {mean_iou}%.\n"
    print(content)

    return content
    

if __name__=='__main__':

    root = "/home/deep/Mario/multiclass_segmentation/weights"
    models = ["deeplab","segnet"]
    datasets = ["t1","t2","t3","vg"]
    overal_perfm = "global_perfm.txt"

    for model in models:
        for dataset in datasets:
            content = main(root,model,dataset)

            with open(overal_perfm, 'a') as appender:
                appender.write(content)
                appender.write('\n')
            
   
    #content += f"| Accuracy of each class: {acc}%\n"
        




        

        









