import os
import glob
import subprocess
import threading
import queue
import sys
from pathlib import Path
import yaml


def get_num_gpus():
    """
    获取系统中可用的 GPU 数量。
    优先尝试使用 PyTorch，如果未安装，则回退到使用 nvidia-smi 命令。
    """
    try:
        import torch
        num_gpus = torch.cuda.device_count()
        return num_gpus
    except ImportError:
        # 使用 nvidia-smi 命令
        try:
            result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                sys.exit("无法通过 nvidia-smi 获取 GPU 信息，请确保 nvidia-smi 已安装并且驱动正常。")
            num_gpus = len(result.stdout.strip().split('\n'))
            return num_gpus
        except FileNotFoundError:
            sys.exit("nvidia-smi 命令未找到，请确保 NVIDIA 驱动已正确安装。")

def worker(gpu_id, task_queue, main_path, run_type):
    """
    工作线程函数，在指定的 GPU 上执行任务队列中的配置文件。
    """
    while True:
        try:
            config_file = task_queue.get_nowait()
        except queue.Empty:
            break  # 队列为空，退出线程

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


        if run_type == 'attack':
            command = ['python', os.path.join(main_path, 'main.py'), '--config', config_file, '--tqdm_log', 'True', '--run_type', 'attack']
        else:
            command = ['python', os.path.join(main_path, 'main.py'), '--config', config_file, '--tqdm_log', 'True', '--run_type', 'inference']

        # 输出当前任务的信息
        print(f"GPU {gpu_id}: 开始运行 {config_file}")

        try:
            # 启动子进程，抑制其标准输出和错误输出
            process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # 等待子进程完成
            process.communicate()

            if process.returncode == 0:
                print(f"GPU {gpu_id}: 完成 {config_file}")
            else:
                print(f"GPU {gpu_id}: 运行 {config_file} 时出错，返回码 {process.returncode}")
        except Exception as e:
            print(f"GPU {gpu_id}: 运行 {config_file} 时发生异常: {e}")

        task_queue.task_done()

def main(config_dir, run_type, gpu_ids):
    """
    主函数，设置任务队列并启动工作线程。
    """
    
    if config_dir is None:
        config_files = [
        "/home/fit/huangml/WORK/chenrenmiao/project/Einstein_project/JPS/config/advbench/internvl2/full_con32|255_lr1|255.yaml"
    ]
    else:
        path = Path(config_dir)
        if path.is_file() and path.suffix == '.yaml':
            config_files = [path]  # 返回一个包含该文件路径的列表

        # 如果 config_dir 是目录，则使用 rglob 递归查找所有 YAML 文件
        elif path.is_dir():
            # 收集所有 .yaml 文件
            config_files = list(path.rglob('*.yaml'))
            
        else:
            raise ValueError(f"{config_dir} 既不是文件也不是目录!")

    
        if not config_files:
            sys.exit(f"在目录 {config_dir} 中未找到任何 YAML 配置文件。")

    # 定义一个函数来获取 attack_config.total_img_attn_weight
    def get_total_img_attn_weight(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            # 提取 attack_config.total_img_attn_weight，默认值为1
            return data.get('attack_config', {}).get('total_img_attn_weight', 1)
        except Exception as e:
            # 读取或解析失败时，返回默认值1
            print(f"Error reading {file_path}: {e}")
            return 1
    
    # 使用 sorted 对文件进行排序
    # 第一个排序键：是否 total_img_attn_weight == 0（0 为 True，优先）
    # 第二个排序键：文件名（按字母顺序）
    config_files_sorted = sorted(
        config_files,
        key=lambda x: (0 if get_total_img_attn_weight(x) == 0 else 1, x.name)
    )
    
    # 将 Path 对象转换为字符串路径
    config_files = [str(file) for file in config_files_sorted]
        
    print(f"找到 {len(config_files)} 个配置文件。")

    # 创建任务队列
    task_queue = queue.Queue()
    for config in config_files:
        task_queue.put(config)

    # 获取 main.py 的路径（上一级目录）
    main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # 检查 main.py 是否存在
    main_py_path = os.path.join(main_path, 'main.py')
    if not os.path.isfile(main_py_path):
        sys.exit(f"未在 {main_path} 目录中找到 main.py。")

    # 创建并启动线程
    threads = []
    for gpu_id in gpu_ids:
        thread = threading.Thread(target=worker, args=(gpu_id, task_queue, main_path, run_type))
        thread.start()
        threads.append(thread)

    # 等待所有任务完成
    for thread in threads:
        thread.join()

    print("所有任务已完成。")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="在多张 GPU 上并行运行多个配置文件。")
    parser.add_argument('--config_dir', type=str, default = '', help="包含 YAML 配置文件的文件夹路径。")
    parser.add_argument('--type', type=str, required=True, help="类型")
    parser.add_argument('--gpus', type=str, default=None, help="使用的 GPU ID 列表，以逗号分隔。例如：'4,7'")

    args = parser.parse_args()

    if args.gpus:
        try:
            gpu_ids = [int(x) for x in args.gpus.split(',')]
        except ValueError:
            sys.exit("GPU ID 必须是整数，并以逗号分隔。例如：'4,7'")
    else:
        gpu_ids = [0,1,2,3,4,5,6,7]

    if not os.path.exists(args.config_dir):
        sys.exit(f"指定的配置文件夹 {args.config_dir} 不存在或不是一个文件夹。")

    main(args.config_dir, args.type, gpu_ids)
