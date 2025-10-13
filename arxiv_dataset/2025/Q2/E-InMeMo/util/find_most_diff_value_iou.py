
# iou_values_file1 = read_file('I:/Pycharmprojects/ICLVP/trainer/prompt_selection/pad_output_examples_prompt_selection/no_vp_fold_1/no_vp_fold1_segmentation_a1_0/log.txt')
# iou_values_file2 = read_file('I:/Pycharmprojects/ICLVP/trainer/prompt_selection/pad_output_examples_prompt_selection/no_vp_fold_1/no_vp_fold1_segmentation_a1_-1/log.txt')

import re
import ast
from operator import itemgetter

# 读取文件函数
def read_file(file_name):
    with open(file_name, 'r') as file:
        data = file.readlines()
    iou_values = {}
    for line in data:
        if line[0].isdigit():
            key, value = line.split("\t")
            value = ast.literal_eval(value)
            iou_values[int(key)] = value['iou']
    return iou_values

fold = 0
iou_values_file1 = read_file('/data/jiahao/PycharmProjects/Enhanced_InMeMo/TNNLS_med_output_samples/ISIC/pad_output_examples/no_vp/no_vp_fold0_segmentation_a1/log.txt')
iou_values_file2 = read_file('/data/jiahao/PycharmProjects/Enhanced_InMeMo/TNNLS_med_output_samples/ISIC/pad_output_examples/spimg_spmask/spimg_spmask_fold0_segmentation_a1/log.txt')
# # 读取两个文件
# iou_values_file1 = read_file(
#     f'I:/Pycharmprojects/ICLVP/trainer/detection/pad_output_examples/no_vp_fold_{fold}/no_vp_fold{fold}_segmentation_a1/log.txt')
# iou_values_file2 = read_file(
#     f'I:/Pycharmprojects/ICLVP/trainer/detection/pad_output_examples/spimg_spmask_fold_{fold}/spimg_spmask_fold{fold}_segmentation_a1/log.txt')

# 计算'iou'的差值
iou_diff = {key: iou_values_file2[key] - iou_values_file1[key] for key in iou_values_file1.keys()}

# #打印满足条件的值
# for index, value in iou_diff.items():
#     # print(iou_values_file1[index])
#     if abs(value) < 0.02 and (iou_values_file1[index] > 0.5 or iou_values_file2[index] > 0.5):
#         print(f"序号: {index}, IOU差值: {value}, 原始IOU值: {iou_values_file1[index]}, {iou_values_file2[index]}")

# 按照'iou'的差值从大到小排序，并获取前10个
top_10_diff = sorted(iou_diff.items(), key=itemgetter(1), reverse=True)[:50]

for index, value in top_10_diff:
    print(f"序号: {index}, IOU差值: {value}, 原始IOU值: {iou_values_file1[index]}, {iou_values_file2[index]}")