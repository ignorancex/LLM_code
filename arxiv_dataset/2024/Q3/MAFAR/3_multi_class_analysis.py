import csv
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
test_file = ['DJI_20231202102104_0001_D.npy', 'DJI_20231203012836_0002_D.npy', 'DJI_20231203015158_0003_D.npy', 'DJI_20231203015646_0004_D.npy', 'DJI_20231203020240_0005_D.npy', 'DJI_20231203022232_0006_D.npy', 'DJI_20231203025216_0007_D.npy', 'DJI_20231203051423_0008_D.npy', 'DJI_20231203051949_0009_D.npy', 'DJI_20231203052327_0010_D.npy', 'DJI_20231212164538_0038_D.npy', 'DJI_20231214154115_0040_D.npy', 'DJI_20231214175432_0041_D.npy', 'DJI_20231216095837_0042_D.npy', 'DJI_20231216100845_0043_D.npy', 'DJI_20231216164512_0044_D.npy', 'DJI_20231216165534_0045_D.npy', 'DJI_20231216174224_0046_D.npy', 'DJI_20231216175753_0047_D.npy', 'DJI_20231216180413_0048_D.npy', 'DJI_20231216181520_0049_D.npy', 'DJI_20231216184838_0050_D.npy', 'DJI_20231216200711_0052_D.npy', 'DJI_20231216201917_0053_D.npy', 'DJI_20231217123556_0054_D.npy']
print(len(test_file))
each_labeller_results = np.load("each_labeller_results.npy",allow_pickle=True).item()
each_model_results = np.load("each_model_results.npy",allow_pickle=True).item()
result_add = "predictions_10s"
true_label = "{}_label.json".format(result_add)
with open(true_label, 'r') as file:
    true_label = json.load(file)

label2text = []
with open('class_labels_indices.csv', 'r') as file:
    csvreader = csv.reader(file)
    next(csvreader)
    for row in csvreader:
        label2text.append(row[2])


def kl_divergence(p, q):
    return np.sum(p * (np.log(p + 1e-8) - np.log(q + 1e-8)), axis=1)


def js_divergence(p, q):
    m = (p + q) / 2
    return kl_divergence(p, m) + kl_divergence(q, m)


def analyse_predict_distribution(resutls):
    final_data = []
    for file in test_file:
        file_name = file[:-4]
        agents = resutls[file_name]
        all_result = []
        for agent_name in list(agents.keys()):
            agent = agents[agent_name]
            all_result.append(agent)

        all_result = np.array(all_result)
        final_data.append(all_result)
    final_data = np.concatenate(final_data,axis=1)

    final_data[final_data>0.5] = 1
    final_data[final_data<=0.5] = 0

    print(final_data.shape)
    mean_result = np.mean(final_data, axis=0)

    return mean_result,final_data

def sorted_array(arr,removed_index):
    sorted_arr_with_indices = sorted(enumerate(arr), key=lambda x: x[1])
    sorted_arr = []
    original_indices = []
    for x in sorted_arr_with_indices:
        if x[0] not in removed_index:
            sorted_arr.append(x[1])
            original_indices.append(x[0])

    return sorted_arr, original_indices

def draw_bar(divergence_distribution,removed_index,name,y,red=None):
    font_size = 14
    sorted_arr, original_indices = sorted_array(divergence_distribution,removed_index)
    topk = 40
    sorted_arr_t = []
    # sorted_arr_t.extend(sorted_arr[:topk])
    sorted_arr_t.extend(sorted_arr[-topk:])
    original_indices_t = []
    # original_indices_t.extend(original_indices[:topk])
    original_indices_t.extend(original_indices[-topk:])

    # plt.figure(figsize=(12, 4))
    # plt.bar(range(len(sorted_arr_t)), sorted_arr_t, color='skyblue')
    #
    # plt.xticks(range(len(original_indices_t)), [label2text[x].split(",")[0].split(" (")[0] for x in original_indices_t],rotation=60, ha="right",va='top')
    # plt.subplots_adjust(bottom=0.5)
    #
    #
    # plt.xlabel('Class name',fontsize=font_size)
    # plt.ylabel("{}".format(y),fontsize=font_size)
    #
    # plt.title('Top {} Classes in the {}'.format(topk,name),fontsize=font_size)

    # plt.show()
    # plt.savefig("picture/{}.pdf".format(name),bbox_inches='tight')
    return original_indices_t


def draw_bar_red(divergence_distribution,removed_index,name,y,red):
    font_size = 14
    sorted_arr, original_indices = sorted_array(divergence_distribution,removed_index)
    topk = 40
    sorted_arr_t = []
    # sorted_arr_t.extend(sorted_arr[:topk])
    sorted_arr_t.extend(sorted_arr[-topk:])
    original_indices_t = []
    # original_indices_t.extend(original_indices[:topk])
    original_indices_t.extend(original_indices[-topk:])

    red_x = []
    red_y = []
    for i in range(len(original_indices_t)):
        if original_indices_t[i] in red:
            red_x.append(i)
            red_y.append(sorted_arr_t[i])

    plt.figure(figsize=(12, 3))
    plt.bar(range(len(sorted_arr_t)), sorted_arr_t, color='skyblue')
    plt.bar(red_x, red_y, color='orange')
    plt.xticks(range(len(original_indices_t)), [label2text[x].split(",")[0].split(" (")[0] for x in original_indices_t],rotation=60, ha="right",va='top')
    plt.subplots_adjust(bottom=0.5)


    # plt.xlabel('Class name',fontsize=font_size)
    plt.ylabel("{}".format(y),fontsize=font_size)

    # plt.title('Top {} Classes in the {}'.format(topk,name),fontsize=font_size)

    plt.show()
    # plt.savefig("picture/{}.png".format(name),bbox_inches='tight',dpi=500)
    return original_indices_t


def draw_bar_dual(divergence_distribution,removed_index,name,y):
    font_size = 14
    sorted_arr, original_indices = sorted_array(divergence_distribution,removed_index)
    topk = 20
    sorted_arr_t = []
    sorted_arr_t.extend(sorted_arr[:topk])
    sorted_arr_t.extend(sorted_arr[-topk:])
    original_indices_t = []
    original_indices_t.extend(original_indices[:topk])
    original_indices_t.extend(original_indices[-topk:])

    plt.figure(figsize=(12, 3))
    plt.bar(range(len(sorted_arr_t)), sorted_arr_t, color='skyblue')

    plt.xticks(range(len(original_indices_t)), [label2text[x].split(",")[0].split(" (")[0] for x in original_indices_t],rotation=60, ha="right",va='top')
    plt.subplots_adjust(bottom=0.5)


    # plt.xlabel('Class name',fontsize=font_size)
    plt.ylabel("{}".format(y),fontsize=font_size)

    # 设置图表标题
    # plt.title('Top and bottom  {} Classes of {}'.format(topk,name),fontsize=font_size)

    # 显示图表
    plt.show()
    # plt.savefig("picture/{}.png".format(name),bbox_inches='tight',dpi=500)




def remove_unintereting_events(model_pre,human_pre):
    removed_index = []
    for i in range(527):
        model_classes = np.sum(model_pre[:,:,i]>0.5)
        human_classes = np.sum(human_pre[:,:,i]>0.5)
        num_classes = model_classes + human_classes
        if num_classes==0:
            removed_index.append(i)

    return removed_index

def find_each_class_variance(pre):
    result = np.zeros(527)
    mean = np.mean(pre, axis=0)
    var = np.var(pre, axis=0)
    for i in range(527):
        var_i = var[:,i]
        mean_i = mean[:,i]
        index = mean_i > 0.00001
        if var_i[index].shape[0] != 0:
            result[i] = np.mean(var_i[index])
    return result

def find_rank(arr):
    sorted_indices = np.argsort(-arr) #decrease
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(arr) + 1)
    return ranks


def overall_detect(model_distribution,human_distribution):

    model_distribution = np.max(model_distribution,axis=1)
    human_distribution = np.sum(human_distribution,axis=1)
    all_p = []
    all_r = []
    all_f = []
    # print(human_distribution[human_distribution>1])
    for threshold in range(20):
        y_true = np.zeros(human_distribution.shape[0])
        y_true[human_distribution>threshold/10+0.0001] = 1
        y_pred = np.zeros(model_distribution.shape[0])
        y_pred[model_distribution>0.5] = 1
        # print("======= {} ========".format(threshold+1))
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f_measure = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        # print(f"Precision: {precision:.4f}")
        # print(f"Recall: {recall:.4f}")
        # print(f"F-measure (F1 Score): {f_measure:.4f}")
        # print(f"Accuracy: {accuracy:.4f}")
        all_p.append(precision)
        all_r.append(recall)
        all_f.append(f_measure)
    epochs = range(1,len(all_p)+1)
    # epochs = np.array(epochs)/10
    plt.plot(epochs, all_p, marker='o', label='Precision', color='blue')
    plt.plot(epochs, all_r, marker='s', label='Recall', color='green')
    plt.plot(epochs, all_f, marker='^', label='F-measure', color='red')
    plt.subplots_adjust(bottom=0.2)
    # 添加标题和标签
    font_size = 12
    plt.title('Performance of AudioSet pre-trained model',fontsize=font_size+2)
    plt.xlabel('Threshold of ground truth \n(amount of labelled events) ',fontsize=font_size)
    plt.ylabel('Score',fontsize=font_size)
    plt.xticks(epochs)
    plt.legend(loc='best')
    # plt.show()

    plt.savefig("picture/existence.pdf",bbox_inches='tight',dpi=400)


model_distribution, model_pre = analyse_predict_distribution(each_model_results)
human_distribution, human_pre = analyse_predict_distribution(each_labeller_results)
removed_index = remove_unintereting_events(model_pre,human_pre)
print(removed_index)
print(len(removed_index))
print(human_pre.shape)
divergence_distribution = np.mean(np.abs(model_distribution-human_distribution),axis=0)
# divergence = np.mean(divergence_distribution,axis=0)

# model_var = np.mean(np.var(model_pre,axis=0),axis=0)
# human_var = np.mean(np.var(human_pre,axis=0),axis=0)

model_var = find_each_class_variance(model_pre)
human_var = find_each_class_variance(human_pre)

model_rank = find_rank(model_var)
human_rank = find_rank(human_var)

overall_detect(model_distribution,human_distribution)

print(divergence_distribution.shape)

# draw_bar(divergence_distribution,removed_index,"disagreement between human and model",'Divergence')

red_m = draw_bar(model_var,removed_index,"disagreement of models",'Variance')
red_h = draw_bar(human_var,removed_index,"disagreement of humans",'Variance')
draw_bar_dual(human_var - model_var,removed_index,"the variance difference between human and model (h-m)",'Divergence')
#
draw_bar_red(model_var,removed_index,"disagreement of models",'Variance',red_h)
draw_bar_red(human_var,removed_index,"disagreement of humans",'Variance',red_m)