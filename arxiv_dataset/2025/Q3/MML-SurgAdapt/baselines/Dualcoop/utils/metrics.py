import numpy as np
import os, json
from statistics import mean
from sklearn.metrics import f1_score, average_precision_score
from datetime import datetime
import ivtmetrics

def process_cholec80(true,pred,video_ids):

    true = true[:,:7]
    pred = pred[:,:7]

    unique_ids = np.unique(video_ids)
    video_f1s = {}

    for vid in unique_ids:
        indices = np.where(video_ids==vid)[0]
        y_true_video = true[indices]  
        y_pred_video = pred[indices]

        y_pred_video = np.argmax(y_pred_video,axis=1)
        y_true_video = np.argmax(y_true_video,axis=1)

        f1 = f1_score(y_true_video, y_pred_video, average="macro",labels=np.unique(y_true_video))*100

        video_f1s[vid] = f1

    f1_score_reg = mean(video_f1s.values())

    return f1_score_reg, video_f1s

def process_endo(true,pred,video_ids):

    true = true[:,7:10]
    pred = pred[:,7:10]

    num_classes = true.shape[1]

    per_class_map = {}

    for label in range(num_classes):
        avg_precision = average_precision_score(
            true[:, label], pred[:, label]
        )
        per_class_map[f"C{label}"] = avg_precision * 100.0
    
    mean_ap = mean(per_class_map.values())

    return mean_ap, per_class_map

def resolve_nan(classwise):
        classwise[classwise==-0.0] = np.nan
        return classwise

def process_cholect50(true,pred,video_ids):

    true = true[:,10:]
    pred = pred[:,10:]
    unique_vids = np.unique(video_ids)

    ap_i_list = []
    ap_v_list = []
    ap_t_list = []
    ap_iv_list = []
    ap_it_list = []
    ap_ivt_list = []

    for vid in unique_vids:
            indices = np.where(video_ids == vid)[0]
            ivt_labels = true[indices]
            ivt_preds = pred[indices]

            filter = ivtmetrics.Disentangle()

            i_labels = filter.extract(inputs=ivt_labels, component="i")
            v_labels = filter.extract(inputs=ivt_labels, component="v")
            t_labels = filter.extract(inputs=ivt_labels, component="t")
            iv_labels = filter.extract(inputs=ivt_labels, component="iv")
            it_labels = filter.extract(inputs=ivt_labels, component="it")

            i_preds = filter.extract(inputs=ivt_preds, component="i")
            v_preds = filter.extract(inputs=ivt_preds, component="v")
            t_preds = filter.extract(inputs=ivt_preds, component="t")
            iv_preds = filter.extract(inputs=ivt_preds, component="iv")
            it_preds = filter.extract(inputs=ivt_preds, component="it")

            ap_i = average_precision_score(i_labels,i_preds,average=None)*100
            ap_v = average_precision_score(v_labels,v_preds,average=None)*100
            ap_t = average_precision_score(t_labels,t_preds,average=None)*100
            ap_iv = average_precision_score(iv_labels,iv_preds,average=None)*100
            ap_it = average_precision_score(it_labels,it_preds,average=None)*100
            ap_ivt = average_precision_score(ivt_labels,ivt_preds,average=None)*100

            ap_i = resolve_nan(ap_i)
            ap_v = resolve_nan(ap_v)
            ap_t = resolve_nan(ap_t)
            ap_iv = resolve_nan(ap_iv)
            ap_it = resolve_nan(ap_it)
            ap_ivt = resolve_nan(ap_ivt)

            ap_i_list.append(ap_i.reshape([1,-1]))
            ap_v_list.append(ap_v.reshape([1,-1]))
            ap_t_list.append(ap_t.reshape([1,-1]))
            ap_iv_list.append(ap_iv.reshape([1,-1]))
            ap_it_list.append(ap_it.reshape([1,-1]))
            ap_ivt_list.append(ap_ivt.reshape([1,-1]))
    
    ap_i_list = np.concatenate(ap_i_list,axis=0)
    ap_i_list = np.nanmean(ap_i_list,axis=0)
    map_i = np.nanmean(ap_i_list)
    ap_v_list = np.concatenate(ap_v_list,axis=0)
    ap_v_list = np.nanmean(ap_v_list,axis=0)
    map_v = np.nanmean(ap_v_list)
    ap_t_list = np.concatenate(ap_t_list,axis=0)
    ap_t_list = np.nanmean(ap_t_list,axis=0)
    map_t = np.nanmean(ap_t_list)
    ap_iv_list = np.concatenate(ap_iv_list,axis=0)
    ap_iv_list = np.nanmean(ap_iv_list,axis=0)
    map_iv = np.nanmean(ap_iv_list)
    ap_it_list = np.concatenate(ap_it_list,axis=0)
    ap_it_list = np.nanmean(ap_it_list,axis=0)
    map_it = np.nanmean(ap_it_list)
    ap_ivt_list = np.concatenate(ap_ivt_list,axis=0)
    ap_ivt_list = np.nanmean(ap_ivt_list,axis=0)
    map_ivt = np.nanmean(ap_ivt_list)

    aps = {
        "AP_i" : map_i,
        "AP_v" : map_v,
        "AP_t" : map_t,
        "AP_iv" : map_iv,
        "AP_it" : map_it,
        "AP_ivt" : map_ivt
    }

    return aps

def save_results(a,b,c,test):

    cholec80_data = {
        "F1_score" : a[0],
        "Per_video_f1" : a[1],
    }

    endo_data = {
        "mAP" : b[0],
        "mAP_per_class" : b[1],
    }

    cholect50_data = {
        "AP" : c,
    }

    data = {
        "Test" : test,
        "Cholec80" : cholec80_data,
        "Endoscapes" : endo_data,
        "CholecT50" : cholect50_data
    }

    folder_name = f"results/trial"
    os.makedirs(folder_name, exist_ok=True)

    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    if test == True:
        file_name = f"{folder_name}/result_{timestamp}_test.json"
    else:
        file_name = f"{folder_name}/result_{timestamp}.json"

    try:
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print("Results saved successfully")
    except Exception as e:
        print(f"Error in saving results : {e}")


def calculate_metrics(labels,preds,video_ids,test):

    labels = np.round(labels)

    datasets = ["cholec80","endoscapes",'cholect50']

    for dataset in datasets:

        print(f"Processing dataset: {dataset}")

        mask = np.array([dataset in v for v in video_ids])

        filtered_labels  = labels[mask]
        filtered_preds = preds[mask]
        filtered_video_ids = video_ids[mask]

        if dataset == "cholec80":
            a = process_cholec80(filtered_labels,filtered_preds,filtered_video_ids)
        if dataset == "endoscapes":
            b = process_endo(filtered_labels,filtered_preds,filtered_video_ids) 
        if dataset == 'cholect50':
            c = process_cholect50(filtered_labels,filtered_preds,filtered_video_ids)    

    save_results(a,b,c,test)

    print("Calculating metrics done")