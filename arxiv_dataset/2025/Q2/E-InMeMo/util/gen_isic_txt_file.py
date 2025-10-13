# import os
#
# # 图像文件夹路径
# base_path = "../med_dataset/ISIC2016/ISBI2016_ISIC_Part1_Test_Data"
#
# # 输出文件路径
# output_file_path = "../evaluate/splits/isic/val/fold0.txt"
#
# # 打开输出文件准备写入
# with open(output_file_path, 'w') as output_file:
#     # 遍历文件夹中的所有文件
#     for image_name in os.listdir(base_path):
#         # 检查是否为文件（排除子文件夹）
#         if os.path.isfile(os.path.join(base_path, image_name)):
#             # 从文件名中去除扩展名
#             image_name_wo_ext = os.path.splitext(image_name)[0]
#             # 构造输出格式，并写入文件
#             output_line = f"{image_name_wo_ext}__01\n"
#             output_file.write(output_line)
#
# print("文件处理完成。")


# import os
# from sklearn.model_selection import KFold
#
# # 图像文件夹路径
# base_path = "../med_dataset/Kvasir-SEG/images"
#
# # 获取图像文件列表
# images = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
#
# # 初始化5-Fold
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
#
# # 开始5-Fold分割
# for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
#     # 分别为训练集和验证集创建文件
#     train_file_path = f"../evaluate/splits/kvasir/trn/fold{fold}.txt"
#     val_file_path = f"../evaluate/splits/kvasir/val/fold{fold}.txt"
#
#     # 保存训练集文件名
#     with open(train_file_path, 'w') as train_file:
#         for idx in train_idx:
#             image_name_wo_ext = os.path.splitext(images[idx])[0]
#             train_file.write(f"{image_name_wo_ext}__01\n")
#
#     # 保存验证集文件名
#     with open(val_file_path, 'w') as val_file:
#         for idx in val_idx:
#             image_name_wo_ext = os.path.splitext(images[idx])[0]
#             val_file.write(f"{image_name_wo_ext}__01\n")
#
# print("5-Fold交叉验证文件已生成。")


import os

# 图像文件夹路径
base_path = "../low_level_dataset/defocus_dataset/test_data/DUT/dut500-source"

# 输出文件路径
output_file_path = "../evaluate/splits/cuhk/val/fold1.txt"

# 打开输出文件准备写入
with open(output_file_path, 'w') as output_file:
    # 遍历文件夹中的所有文件
    for image_name in os.listdir(base_path):
        # 检查是否为文件（排除子文件夹）
        if os.path.isfile(os.path.join(base_path, image_name)):
            # 从文件名中去除扩展名
            image_name_wo_ext = os.path.splitext(image_name)[0]
            # 构造输出格式，并写入文件
            output_line = f"{image_name_wo_ext}__01\n"
            output_file.write(output_line)

print("文件处理完成。")




