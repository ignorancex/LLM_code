# import os
# import shutil
#
# # 源文件夹路径
# source_folder = r"D:\test project\InMeMo_Atlantis\atlantis\images\test"
#
# # 目标文件夹路径
# target_folder = r"D:\test project\InMeMo_Atlantis\atlantis\images\new_test"
#
# # 如果目标文件夹不存在，则创建它
# if not os.path.exists(target_folder):
#     os.makedirs(target_folder)
#
# # 遍历源文件夹中的所有子文件夹和文件
# for root, dirs, files in os.walk(source_folder):
#     for file in files:
#         # 构建源文件的完整路径
#         file_path = os.path.join(root, file)
#
#         # 构建目标文件的完整路径
#         dest_path = os.path.join(target_folder, file)
#
#         # 复制文件
#         shutil.copy(file_path, dest_path)
#
# print("图片复制完成。")


import os
import shutil

# 源文件夹路径
src_dir = 'D:/test project/InMeMo_Atlantis/atlantis/masks'
# 目标文件夹路径
dest_dir = 'D:/test project/InMeMo_Atlantis/atlantis/masks/SegmentationClassAug'

# 确保目标文件夹存在，如果不存在则创建
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# 遍历源文件夹中的所有子文件夹和文件
for root, dirs, files in os.walk(src_dir):
    for file in files:
        # 检查文件是否是图片，这里假设图片格式为jpg和jpeg，你可以根据需要添加更多的图片格式
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            # 构造源文件的完整路径
            src_file = os.path.join(root, file)
            # 构造目标文件的完整路径
            dest_file = os.path.join(dest_dir, file)

            # 复制文件
            shutil.copy(src_file, dest_file)
            print(f'Copied "{src_file}" to "{dest_file}"')

print('All images have been copied to the destination folder.')
