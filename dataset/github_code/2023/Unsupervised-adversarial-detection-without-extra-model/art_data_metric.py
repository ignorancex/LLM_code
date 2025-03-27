from proto_train import *
import sklearn.metrics

class ModelHandle(TrainHandle):
    def dataset_result_customized(self, batch_size, test_data_path, test_label_path, ComputeHandle, **extra_setting_dict):
        test_imgs = np.load(test_data_path)
        test_labels = np.load(test_label_path)
        data_len = len(test_imgs)
        test_steps = data_len//batch_size

        #start computing
        compute_handle = ComputeHandle(self.model, self._get_compute_dict(), **extra_setting_dict)
        result_dict = None
        for test_step in range(test_steps):
            temp_test_imgs = test_imgs[test_step:test_step+batch_size]
            temp_test_labels = test_labels[test_step:test_step+batch_size]
            print(f'{(test_step+1)/test_steps:.1%}    ', end="\r")
            batch_result_dict = compute_handle.result(temp_test_imgs, temp_test_labels)
            if isinstance(result_dict, type(None)):
                result_dict = {key: [] for key in batch_result_dict.keys()}
            for result_key, results in batch_result_dict.items():
                result_dict[result_key].extend(results)
        return result_dict

    def statistics(self, batch_size, raw_data_path, test_data_path, test_label_path, ComputeHandle, save_dir,
        **extra_setting_dict):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        raw_result_dict = self.dataset_result_customized(batch_size, raw_data_path, test_label_path, ComputeHandle, 
            **extra_setting_dict)

        result_dict = self.dataset_result_customized(batch_size, test_data_path, test_label_path, ComputeHandle, 
            **extra_setting_dict)

        correct_index_list = [index for index, correct in enumerate(raw_result_dict["correct"]) if correct==1]
        raw_correct_result_dict = {data_key:[data_list[index] for index in correct_index_list] \
            for data_key, data_list in raw_result_dict.items()}
        atk_correct_result_dict = {data_key:[data_list[index] for index in correct_index_list] \
            for data_key, data_list in result_dict.items()}

        origin_acc = np.mean(raw_result_dict["correct"])
        print("origin acc=", origin_acc)
        acc = np.mean(result_dict["correct"])
        print("attack acc=", acc)

        data_result_dict = {
            "atk_false":atk_correct_result_dict["raw_false"],
            "atk_true":atk_correct_result_dict["raw_true"],
            "false":raw_correct_result_dict["raw_false"],
            "true":raw_correct_result_dict["raw_true"],
        }
        color_dict = {
            "atk_false":"red",
            "atk_true":"blue",
            "false":"black",
            "true":"green",
        }
        plot_info = {
            "bin_width":0.1, 
            # x_range:(-10, 10),
            "x_label":"raw output",
            "y_label":"numbers in ratio",
        }
        
        distribution_draw(PltQuantityPlot, data_result_dict, color_dict, 
            save_dir+"raw_distrbution.png", **plot_info)


        #get auroc for maximum threshold
        data_result_dict = {
            "atk_false_max":atk_correct_result_dict["raw_false_max"],
            "true":raw_correct_result_dict["raw_true"],
        }
        metric_dict = detect_metric(data_result_dict, pos_key="true", neg_key="atk_false_max")

        #get threshold
        max_threshold, min_threshold = threshold_choosing(raw_correct_result_dict)
        print("max_threshold=", max_threshold)
        print("min_threshold=", min_threshold)

        #get the results of atk detection
        raw_max_detect_result, raw_both_detect_result = atk_detect(raw_correct_result_dict["raw"], max_threshold, min_threshold)
        atk_max_detect_result, atk_both_detect_result = atk_detect(atk_correct_result_dict["raw"], max_threshold, min_threshold)
        

        max_detect_TPR = np.mean(atk_max_detect_result)
        max_detect_FPR = np.mean(raw_max_detect_result)
        metric_dict["max_detect_TPR"] = max_detect_TPR
        metric_dict["max_detect_FPR"] = max_detect_FPR
        
        both_detect_TPR = np.mean(atk_both_detect_result)
        both_detect_FPR = np.mean(raw_both_detect_result)
        metric_dict["both_detect_TPR"] = both_detect_TPR
        metric_dict["both_detect_FPR"] = both_detect_FPR
        #save message
        message_handle = MessageHandle(save_dir+"metric.txt")
        for metric_key, metric in metric_dict.items():
            message_handle.add(metric_key+f':{metric: .3f}', interval=", ")
        message_handle.save()

def threshold_choosing(raw_result_dict):
    raw_false = raw_result_dict["raw_false"]
    sort_raw_false = np.sort(np.reshape(raw_false, -1))
    data_len = len(sort_raw_false)
    max_thre = sort_raw_false[(int)(data_len*0.99)]
    min_thre = sort_raw_false[0]
    return max_thre, min_thre
def distribution_draw(draw_func, data_result_dict, color_dict, save_path, normalized=True, cumsum=False,
    **plot_info):
    if not os.path.isdir(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    #do not load weight here
    #set default parameter
    plot = draw_func(**plot_info)
    for data_key, data_result in data_result_dict.items():
        color = color_dict[data_key]
        plot.add_data(data_result, color=color, normalized=normalized, cumsum=cumsum, label=data_key)
    plot.save(save_path, add_legend=True)
    plt.close('all')
def compute_fpr(labels, scores, true_positive_rate=0.95):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores, pos_label=1)
    #get the range for specified true_positive_rate
    buffer_for_lower_bound = np.where(tpr>true_positive_rate, np.zeros_like(tpr), tpr)
    arg_lower_bound = np.argmax(buffer_for_lower_bound)

    buffer_for_upper_bound = np.where(tpr<true_positive_rate, np.ones_like(tpr), tpr)
    arg_upper_bound = np.argmin(buffer_for_upper_bound)
    #get specified fpr
    if arg_lower_bound==arg_upper_bound:
        out_fpr = fpr[arg_lower_bound]
    else:
        ratio = (true_positive_rate-tpr[arg_lower_bound])/(tpr[arg_upper_bound]-tpr[arg_lower_bound])
        out_fpr = fpr[arg_lower_bound]+ratio*(fpr[arg_upper_bound]-fpr[arg_lower_bound])
    return out_fpr
def detect_metric(data_result_dict, pos_key, neg_key):
    label_list = []
    result_list = []
    for data_key, temp_result_list in data_result_dict.items():
        if data_key==pos_key:
            label_list.extend(np.ones_like(temp_result_list))
        elif data_key==neg_key:
            label_list.extend(np.zeros_like(temp_result_list))
        else:
            continue
        result_list.extend(temp_result_list)
    #AUROC
    auroc = sklearn.metrics.roc_auc_score(label_list, result_list)
    #AUPR
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    aupr = sklearn.metrics.average_precision_score(label_list, result_list)

    #FPR95
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    fpr95 = compute_fpr(label_list, result_list)

    metric_dict = {
        "auroc": auroc,
        "aupr": aupr,
        "fpr95": fpr95,
    }
    return metric_dict

def atk_detect(raw_outputs, max_threshold, min_threshold):
    # thresholds are adjusted based on the distribution of test data.
    raw_max = np.max(raw_outputs, axis=1)
    raw_min = np.min(raw_outputs, axis=1)
    #1 for detect attack
    max_result = np.where(raw_max<max_threshold, np.ones_like(raw_max), np.zeros_like(raw_max))
    both_result = np.where((raw_max<max_threshold) | (raw_min<min_threshold), np.ones_like(raw_max), np.zeros_like(raw_max))
    return max_result, both_result
class ResultCompute(ModelComputeTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.max_threshold = kwargs["max_threshold"]
        # self.min_threshold = kwargs["min_threshold"]
    def result(self, data_list, label_list):
        pred_list = self.model(data_list, training=False)
        result_dict = {}
        #get raw output
        raw_output = pred_list[2].numpy()
        result_dict["raw"] = raw_output

        #get raw true output
        label_list = label_list.astype(np.int32)
        raw_true_list = tf.gather(pred_list[2], label_list, batch_dims=1).numpy()
        result_dict["raw_true"] = raw_true_list

        #get whether answer is correct and index output
        index_output = pred_list[1].numpy()
        index_output.astype(np.int32)
        correct_list = [1 if index_output[label_index]==label else 0 for label_index, label in enumerate(label_list)]
        result_dict["correct"] = correct_list
        result_dict["label"] = label_list

        #get max raw false output
        batch_size, class_count = pred_list[2].shape[:2]
        index_array = np.array(range(class_count))
        index_map = np.broadcast_to([index_array], (batch_size, class_count))
        index_map = np.array([np.delete(temp_index_array, label_list[map_index]) for map_index, temp_index_array in enumerate(index_map)])
        raw_false_list = tf.gather(pred_list[2], index_map, batch_dims=1)
        result_dict["raw_false"] = raw_false_list.numpy()
        raw_false_max_list = tf.reduce_max(raw_false_list, axis=-1)
        result_dict["raw_false_max"] = raw_false_max_list.numpy()

        return result_dict
if __name__ == "__main__":
    #info
    dataset_name = "cifar10"
    weight_name = "MyResnet cifar10 proposed"
    config_name = "main"
    model_dir = "./train/"+weight_name+"/"
    raw_data_dir = "./test/"+dataset_name+"/raw data/"
    atk_data_dir = "./test/"+dataset_name+"/"+weight_name+"/"+config_name+"/"
    statistics_dir = atk_data_dir+"statistics/"

    #load model
    model_handle = ModelHandle(10)
    model_handle.load_weight(model_dir, 59)

    #statistics through all data in atk_data_dir
    for atk_data in os.listdir(atk_data_dir):
        if os.path.isdir(atk_data_dir+atk_data):#avoid directory
            continue
        atk_name, end_text = atk_data.split(".")
        if end_text!="npy":#avoid strange data
            continue
        print("atk:", atk_name)
        model_handle.statistics(1000, raw_data_dir+"raw_x.npy",
            atk_data_dir+atk_data, raw_data_dir+"labels.npy", ResultCompute,
            save_dir=statistics_dir+atk_name+"/")
    