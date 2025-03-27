from adver_train import *
import numpy as np
import os
from data import MessageHandle
import sklearn.metrics

class AdvTestEachClassHandle(AdversarialTrainHandle):
    def test_each_output_statistics(self, batch_size, test_data_handle, **extra_setting_dict):
        metric_dict = None
        save_dir = extra_setting_dict["save_dir"]+"each output atk detection statistics (step "+str(extra_setting_dict["update_steps"])+")/"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for chosen_output_index in range(10):
            id_result_dict = self.dataset_result_customized(batch_size, test_data_handle, ChosenAtkResultCompute, 
                test_index=chosen_output_index, **extra_setting_dict)

            true_correct_index_list = [index for index, label in enumerate(id_result_dict["label"])
                if id_result_dict["correct"][index]==1 and label==chosen_output_index]
            true_correct_result_dict = {data_key:[data_list[index] for index in true_correct_index_list] \
                for data_key, data_list in id_result_dict.items()}

            false_correct_index_list = [index for index, label in enumerate(id_result_dict["label"]) 
                if id_result_dict["correct"][index]==1 and label!=chosen_output_index]
            false_correct_result_dict = {data_key:[data_list[index] for index in false_correct_index_list] \
                for data_key, data_list in id_result_dict.items()}

            data_result_dict = {
                "chosen_false":false_correct_result_dict["chosen_raw"],
                "chosen_true":true_correct_result_dict["raw_true"],
                "atk_chosen_false":false_correct_result_dict["atk_chosen_raw"],
            }
            setting_dict = {
                "atk_chosen_false":{
                    "color":"red",
                },
                "chosen_false":{
                    "color":"blue",
                },
                "chosen_true":{
                    "color":"black",
                    "alpha":0.2,
                },
            }
            plot_info = {
                "bin_width":0.1, 
                # x_range:(-10, 10),
                "x_label":"raw output",
                "y_label":"numbers in ratio",
            }
            distribution_draw(PltQuantityPlot, data_result_dict, setting_dict, 
                save_dir+str(chosen_output_index)+"_raw_distrbution.png", **plot_info)


    def test_output_statistics(self, batch_size, test_data_handle, **extra_setting_dict):
        save_dir = extra_setting_dict["save_dir"]+"reverse atk detection statistics (step "+str(extra_setting_dict["update_steps"])+")/"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        id_result_dict = self.dataset_result_customized(batch_size, test_data_handle, AtkResultCompute, 
            **extra_setting_dict)

        correct_index_list = [index for index, correct in enumerate(id_result_dict["correct"]) if correct==1]
        correct_result_dict = {data_key:[data_list[index] for index in correct_index_list] \
            for data_key, data_list in id_result_dict.items()}

        origin_acc = np.mean(id_result_dict["correct"])
        print("origin acc=", origin_acc)
        
        #draw distribution
        data_result_dict = {
            "atk_false":correct_result_dict["atk_raw_false"],
            "atk_true":correct_result_dict["atk_raw_true"],
            "false":correct_result_dict["raw_false"],
            "true":correct_result_dict["raw_true"],
        }
        
        setting_dict = {
                "atk_false":{
                    "color":"red",
                },
                "atk_true":{
                    "color":"blue",
                },
                "false":{
                    "color":"black",
                },
                "true":{
                    "color":"green",
                },
            }
        plot_info = {
            "bin_width":0.1, 
            # x_range:(-10, 10),
            "x_label":"raw output",
            "y_label":"numbers in ratio",
        }
        distribution_draw(PltQuantityPlot, data_result_dict, setting_dict, 
            save_dir+"_raw_distrbution.png", **plot_info)

        data_result_dict = {
            "atk_false_max":correct_result_dict["atk_raw_false_max"],
            "atk_true":correct_result_dict["atk_raw_true"],
            "false_max":correct_result_dict["raw_false_max"],
            "true":correct_result_dict["raw_true"],
        }
        
        setting_dict = {
                "atk_false_max":{
                    "color":"red",
                },
                "atk_true":{
                    "color":"blue",
                },
                "false_max":{
                    "color":"black",
                },
                "true":{
                    "color":"green",
                },
            }
        plot_info = {
            "bin_width":0.1, 
            # x_range:(-10, 10),
            "x_label":"raw output",
            "y_label":"numbers in ratio",
        }
        distribution_draw(PltQuantityPlot, data_result_dict, setting_dict, 
            save_dir+"_detect_raw_distrbution.png", **plot_info)


        #get auroc for maximum threshold
        data_result_dict = {
            "atk_false_max":correct_result_dict["atk_raw_false_max"],
            "true":correct_result_dict["raw_true"],
        }
        metric_dict = detect_metric(data_result_dict, pos_key="true", neg_key="atk_false_max")

        #get threshold
        max_threshold, min_threshold = threshold_choosing(correct_result_dict)
        print("max_threshold=", max_threshold)
        print("min_threshold=", min_threshold)

        #get the results of atk detection
        raw_max_detect_result, raw_both_detect_result = atk_detect(correct_result_dict["raw"], max_threshold, min_threshold)
        atk_max_detect_result, atk_both_detect_result = atk_detect(correct_result_dict["atk_raw"], max_threshold, min_threshold)
        

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
def threshold_choosing(raw_result_dict):
    raw_false = raw_result_dict["raw_false"]
    sort_raw_false = np.sort(np.reshape(raw_false, -1))
    data_len = len(sort_raw_false)
    max_thre = sort_raw_false[(int)(data_len*0.99)]
    min_thre = sort_raw_false[0]
    return max_thre, min_thre
def distribution_draw(draw_func, data_result_dict, setting_dict, save_path, normalized=True, cumsum=False,
    **plot_info):
    if not os.path.isdir(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    #do not load weight here
    #set default parameter
    plot = draw_func(**plot_info)
    for data_key, data_result in data_result_dict.items():
        key_setting = setting_dict[data_key]
        plot.add_data(data_result, normalized=normalized, cumsum=cumsum, label=data_key, **key_setting)
    plot.save(save_path, add_legend=True)
    plt.close('all')
class AtkResultCompute(ModelComputeTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_dict["loss"]["grad"] = kwargs["grad_func"]
        self.compute_dict["loss output index"]["grad"] = kwargs["grad_func_output"]
        self.compute_dict["update_rate"] = kwargs["update_rate"] #how far a step can go
        self.compute_dict["update_steps"] = kwargs["update_steps"] #how many steps i will walk
        self.compute_dict["grad_adjust_func"] = kwargs["grad_adjust_func"] #how to walk
        self.compute_dict["epsilon"] = kwargs["epsilon"]
        
    def result(self, data_list, label_list):
        pred_list = self.model(data_list, training=False)
        result_dict = {}
        raw_output = pred_list[2].numpy()
        result_dict["raw"] = raw_output
        #get raw true output
        raw_true_list = tf.gather(pred_list[2], label_list, batch_dims=1).numpy()
        result_dict["raw_true"] = raw_true_list

        #get whether answer is correct and index output
        index_output = pred_list[1].numpy()
        label_list = label_list.numpy()
        index_output.astype(np.int32)
        correct_list = [1 if index_output[label_index]==label else 0 for label_index, label in enumerate(label_list)]
        result_dict["correct"] = correct_list
        result_dict["label"] = label_list
        
        #get max raw false output
        batch_size, class_count = pred_list[2].shape[:2]
        index_array = np.array(range(class_count))
        index_map = np.broadcast_to([index_array], (batch_size, class_count))
        index_map = np.array([np.delete(temp_index_array, label_list[map_index]) for map_index, temp_index_array in enumerate(index_map)])
        raw_false_max_list = tf.reduce_max(tf.gather(pred_list[2], index_map, batch_dims=1), axis=-1).numpy()
        result_dict["raw_false_max"] = raw_false_max_list
        raw_false = tf.gather(pred_list[2], index_map, batch_dims=1).numpy()
        result_dict["raw_false"] = raw_false

        # attack model
        attack_data_list = generate_atk_data(label_list, data_list, self.model, self.compute_dict, training=False)
        attack_pred_list = self.model(attack_data_list, training=False)
        atk_raw_output = attack_pred_list[2].numpy()
        result_dict["atk_raw"] = atk_raw_output

        #get raw true after attack
        atk_raw_true_list = tf.gather(attack_pred_list[2], label_list, batch_dims=1).numpy()
        result_dict["atk_raw_true"] = atk_raw_true_list

        #get max raw false output after attack
        batch_size, class_count = attack_pred_list[2].shape[:2]
        index_array = np.array(range(class_count))
        index_map = np.broadcast_to([index_array], (batch_size, class_count))
        index_map = np.array([np.delete(temp_index_array, result_dict["label"][map_index]) for map_index, temp_index_array in enumerate(index_map)])
        atk_raw_false_max_list = tf.reduce_max(tf.gather(attack_pred_list[2], index_map, batch_dims=1), axis=-1).numpy()
        result_dict["atk_raw_false_max"] = atk_raw_false_max_list
        atk_raw_false = tf.gather(attack_pred_list[2], index_map, batch_dims=1).numpy()
        result_dict["atk_raw_false"] = atk_raw_false

        return result_dict
class ChosenAtkResultCompute(ModelComputeTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_dict["loss"]["grad"] = kwargs["grad_func"]
        self.compute_dict["loss output index"]["grad"] = kwargs["grad_func_output"]
        self.compute_dict["update_rate"] = kwargs["update_rate"] #how far a step can go
        self.compute_dict["update_steps"] = kwargs["update_steps"] #how many steps i will walk
        self.compute_dict["grad_adjust_func"] = kwargs["grad_adjust_func"] #how to walk
        self.compute_dict["epsilon"] = kwargs["epsilon"]
        self.test_index = kwargs["test_index"]
        
    def result(self, data_list, label_list):
        pred_list = self.model(data_list, training=False)
        result_dict = {}
        #get raw true output
        raw_true_list = tf.gather(pred_list[2], label_list, batch_dims=1).numpy()
        result_dict["raw_true"] = raw_true_list

        #get whether answer is correct and index output
        index_output = pred_list[1].numpy()
        label_list = label_list.numpy()
        index_output.astype(np.int32)
        correct_list = [1 if index_output[label_index]==label else 0 for label_index, label in enumerate(label_list)]
        result_dict["correct"] = correct_list
        result_dict["label"] = label_list

        #get chosen false
        result_dict["chosen_raw"] = pred_list[2].numpy()[:, self.test_index]

        # attack model
        test_indexs = np.broadcast_to([self.test_index], len(label_list))
        attack_data_list = generate_atk_data(test_indexs, data_list, self.model, self.compute_dict, training=False)
        attack_pred_list = self.model(attack_data_list, training=False)

        #get chosen false after attack
        result_dict["atk_chosen_raw"] = attack_pred_list[2].numpy()[:, self.test_index]

        return result_dict
if __name__ == "__main__":
    #set up data
    dataset_name = "cifar10"
    train_data_handle = TFDataGenerator(dataset_name, "train", shuffle=True)
    test_data_handle = TFDataGenerator(dataset_name, "test")

    train_info = train_data_handle.info
    input_shape = train_info.features["image"].shape
    num_classes = train_info.features["label"].num_classes

    #set up model
    train_handle = AdvTestEachClassHandle(class_count=num_classes, input_shape=input_shape)
    save_dir = "./train/"+train_handle.model_name+" "+dataset_name+"/"
    train_handle.load_weight(save_dir, 59)

    info = {
        "save_dir":save_dir,
        "grad_func":TargetDirLoss(),
        "grad_func_output":2,#raw output
        "grad_adjust_func":FGSM,
        "update_rate":2,
        "update_steps":5,
        "epsilon":8,
    }
    train_handle.test_each_output_statistics(1000, test_data_handle, **info)

    info = {
        "save_dir":save_dir,
        "grad_func":ReverseLoss(tf.keras.losses.SparseCategoricalCrossentropy()),
        "grad_func_output":0,#softmax output
        # "grad_func":ReverseLoss(TargetDirLoss()),
        # "grad_func_output":2,#raw output
        "grad_adjust_func":FGSM,
        "update_rate":2,
        "update_steps":5,
        "epsilon":8,
    }
    train_handle.test_output_statistics(1000, test_data_handle, **info)