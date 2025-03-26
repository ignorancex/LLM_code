import os

def delete_folders_with_only_time_info(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 检查该目录中是否只包含一个文件，且是 time_info.txt
        if len(filenames) == 1 and filenames[0] == 'time_info.txt' and not dirnames:
            print(f"Deleting folder: {dirpath}")
            try:
                os.remove(os.path.join(dirpath, 'time_info.txt'))  # 删除文件
                os.rmdir(dirpath)  # 删除空文件夹
            except Exception as e:
                print(f"Error deleting {dirpath}: {e}")

# 设置你要查找的根目录
root_directory = 'LLM_code/dataset/github_code/2020'
delete_folders_with_only_time_info(root_directory)
