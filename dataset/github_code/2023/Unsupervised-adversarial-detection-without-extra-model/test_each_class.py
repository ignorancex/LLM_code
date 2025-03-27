import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from data import *
import tensorflow as tf

def each_class_statistics(train_handle, test_data_handle, save_dir):
    if not os.path.isdir(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))
    #get results for statistics of this model
    result_dict = train_handle.dataset_result_customized(1000, test_data_handle, SimpleResultCompute)
    
    correct_index_list = [index for index, label in enumerate(result_dict["label"]) 
        if label==result_dict["max_index"][index]]

    #get acc for each class
    correct_label_sum_dict = {}
    for index in correct_index_list:
        label = result_dict["label"][index]
        if label not in correct_label_sum_dict:
            correct_label_sum_dict[label] = 0
        correct_label_sum_dict[label] += 1
    label_sum_dict = {}
    for label in result_dict["label"]:
        if label not in label_sum_dict:
            label_sum_dict[label] = 0
        label_sum_dict[label] += 1
    acc_dict = {label:correct_label_sum_dict[label]/label_sum 
        for label, label_sum in label_sum_dict.items()}
    message_handle = MessageHandle(save_dir+"statistics each class.txt")
    for label in range(10):#only test cifar10
        acc = acc_dict[label]
        message_handle.add(f'{label}: acc={acc:.2f}')
        message_handle.save()

    for chosen_output_index in range(10):#only test cifar10
        true_correct_index_list = [index for index, label in enumerate(result_dict["label"]) 
            if label==result_dict["max_index"][index] and label==chosen_output_index]
        true_correct_result_dict = {data_key:[data_list[index] for index in true_correct_index_list] \
            for data_key, data_list in result_dict.items()}
        true_incorrect_index_list = [index for index, label in enumerate(result_dict["label"]) 
            if label!=result_dict["max_index"][index] and label==chosen_output_index]
        true_incorrect_result_dict = {data_key:[data_list[index] for index in true_incorrect_index_list] \
            for data_key, data_list in result_dict.items()}
        false_correct_index_list = [index for index, label in enumerate(result_dict["label"]) 
            if label==result_dict["max_index"][index] and label!=chosen_output_index]
        false_correct_result_dict = {data_key:[data_list[index] for index in false_correct_index_list] \
            for data_key, data_list in result_dict.items()}
        false_incorrect_index_list = [index for index, label in enumerate(result_dict["label"]) 
            if label!=result_dict["max_index"][index] and label!=chosen_output_index]
        false_incorrect_result_dict = {data_key:[data_list[index] for index in false_incorrect_index_list] \
            for data_key, data_list in result_dict.items()}

        #get statistics
        data_result_dict = {
            "false_incorrect":false_incorrect_result_dict["raw_output"],
            "false_correct":false_correct_result_dict["raw_output"],
            "true_incorrect":true_incorrect_result_dict["raw_output"],
            "true_correct":true_correct_result_dict["raw_output"],
        }
        color_dict = {
            "true_correct":"green",
            "true_incorrect":"red",
            "false_correct":"blue",
            "false_incorrect":"black",
        }
        plot_info = {
            "bin_width":0.1, 
            # "x_range":(-10, 10),
            "x_label":"chosen raw output",
            "y_label":"numbers in ratio",
        }
        distribution_draw(PltQuantityPlot, chosen_output_index, data_result_dict, color_dict, 
            save_dir+"raw distrbution_"+str(chosen_output_index)+".png", **plot_info)


class SimpleResultCompute(ModelComputeTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def result(self, data_list, label_list, training=False):
        pred_list = self.model(data_list, training=training)
        result_dict = {}
        
        #get max index
        max_index = pred_list[1].numpy()
        result_dict["max_index"] = max_index

        # get raw output
        raw_output = pred_list[2].numpy()
        result_dict["raw_output"] = raw_output

        #get whether answer is true
        result_dict["label"] = label_list.numpy()

        return result_dict
def distribution_draw(draw_func, chosen_output_index, data_result_dict, color_dict, save_path, normalized=True, cumsum=False,
    **plot_info):
    if not os.path.isdir(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    #do not load weight here
    raw_total_plot = draw_func(**plot_info)

    for data_key, data_result in data_result_dict.items():
        color = color_dict[data_key]

        test_indexs = np.broadcast_to([chosen_output_index], len(data_result))

        chosen_data_result = tf.gather(data_result, test_indexs, batch_dims=1).numpy()
        raw_total_plot.add_data(chosen_data_result, color=color, normalized=normalized, cumsum=cumsum, label=data_key)
    raw_total_plot.save(save_path, add_legend=True)

    plt.close('all')
