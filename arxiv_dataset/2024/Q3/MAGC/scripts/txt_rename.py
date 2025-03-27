# 导入必要的模块
import os

# 定义输入和输出文件的路径
input_file_path = '/mnt/massive/wangce/SGDM/dataset/Synthetic-v18-45k/test/test_hr_original.txt'  # 替换为你的输入文件路径
output_file_path = '/mnt/massive/wangce/SGDM/dataset/Synthetic-v18-45k/test/test_hr.txt'  # 替换为你想要保存的新文件路径

# 确保输入文件存在
if not os.path.exists(input_file_path):
    print("输入文件不存在，请检查路径。")
else:
    # 打开输入文件和输出文件
    with open(input_file_path, 'r') as file, open(output_file_path, 'w') as output_file:
        # 逐行读取文件
        for line in file:
            # 替换字符串
            new_line = line.replace('../dataset', '/mnt/massive/wangce/SGDM/dataset')
            # 将修改后的行写入新文件
            output_file.write(new_line)

    print(f"文件已成功保存到：{output_file_path}")