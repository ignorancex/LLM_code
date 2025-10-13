import os
import random
import math


def process_files(file_paths, output_dir, ratios):
    random.seed(0)
    for file_path in file_paths:
        # 读取文件内容
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 按类别分类
        categories = {}
        for line in lines:
            line = line.strip()
            if line:
                filename, category = line.split('__')
                if category not in categories:
                    categories[category] = []
                categories[category].append(line)

        # 遍历每个比例，随机选择样本
        for ratio in ratios:
            selected_samples = []
            for category, samples in categories.items():
                sample_count = math.ceil(len(samples) * ratio)
                selected_samples.extend(random.sample(samples, sample_count))

            # 构造输出路径
            base_filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, f"pascal_{int(ratio * 1000):03d}", "trn", base_filename)

            print('output_path: ', output_path)
            print('len selected_samples: ', len(selected_samples))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 写入新文件
            with open(output_path, 'w') as output_file:
                output_file.write('\n'.join(selected_samples) + '\n')


# 输入文件夹路径
input_folder = "../evaluate/splits/pascal/trn"
# 输出文件夹路径
output_folder = "../evaluate/splits"
# 比例列表
# ratios = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8]
ratios = [0.02, 0.03, 0.04, 0.05]


# 获取所有文件路径
file_paths = [os.path.join(input_folder, f"fold{i}.txt") for i in range(4)]

# 处理文件
process_files(file_paths, output_folder, ratios)
