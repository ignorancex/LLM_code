#!/usr/bin/env python3
# batch_modify_yaml.py

import sys
import argparse
from pathlib import Path
import shutil
from ruamel.yaml import YAML

def interpret_value(value):
    """
    尝试将字符串值转换为适当的 Python 类型。
    """
    if isinstance(value, bool) or isinstance(value, int) or isinstance(value, float):
        return value  # 已经是合适的类型
    if isinstance(value, str):
        lower = value.lower()
        if lower in ['true', 'yes', 'on']:
            return True
        elif lower in ['false', 'no', 'off']:
            return False
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value  # 保持为字符串
    return value  # 其他类型保持不变

def load_yaml(file_path):
    """
    加载 YAML 文件，并返回数据和 YAML 对象以保持格式。
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.allow_duplicate_keys = False  # 默认不允许重复键
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.load(f)
    return data, yaml

def save_yaml(data, file_path, yaml):
    """
    将修改后的数据保存回 YAML 文件，保持原有格式。
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)

def modify_yaml(data, modifications, add_missing=False):
    """
    根据修改字典修改 YAML 数据。
    """
    for key_path, new_value in modifications.items():
        keys = key_path.split('.')
        current = data
        for key in keys[:-1]:
            if key in current:
                if isinstance(current[key], dict):
                    current = current[key]
                else:
                    print(f"警告: 路径 '{key_path}' 中的 '{key}' 不是字典。")
                    if add_missing:
                        current[key] = {}
                        current = current[key]
                        print(f"已创建字典: '{key}'")
                    else:
                        print("跳过此键。")
                        break
            else:
                if add_missing:
                    current[key] = {}
                    current = current[key]
                    print(f"已创建字典: '{key}'")
                else:
                    print(f"警告: 路径 '{key_path}' 中的 '{key}' 不存在。跳过此键。")
                    break
        else:
            last_key = keys[-1]
            if last_key in current:
                old_value = current[last_key]
                interpreted_value = interpret_value(new_value)
                print(f"修改 '{key_path}': '{old_value}' -> '{interpreted_value}'")
                current[last_key] = interpreted_value
            else:
                if add_missing:
                    interpreted_value = interpret_value(new_value)
                    current[last_key] = interpreted_value
                    print(f"添加 '{key_path}': '{interpreted_value}'")
                else:
                    print(f"警告: 键 '{last_key}' 在路径 '{key_path}' 中不存在。跳过此键。")

def create_backup(file_path):
    """
    创建文件的备份，后缀为 .bak
    """
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    shutil.copyfile(file_path, backup_path)
    print(f"备份创建: {backup_path}")

def parse_arguments():
    """
    使用 argparse 解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="批量修改 YAML 文件中的指定键值对。"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径。"
    )
    parser.add_argument(
        "--add_missing",
        action='store_true',
        help="如果键路径不存在，则添加新的键值对。"
    )
    return parser.parse_args()

def load_config(config_path):
    """
    加载 YAML 格式的配置文件。
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f)
    return config

def main():
    args = parse_arguments()
    config_path = Path(args.config)
    
    if not config_path.is_file():
        print(f"错误: 配置文件 '{config_path}' 不存在。")
        sys.exit(1)
    
    # 加载配置文件
    config = load_config(config_path)
    
    # 获取配置参数
    target_dir = Path(config.get('target_directory', '.'))
    search_strings = config.get('search_strings', [])
    modifications = config.get('modifications', {})
    backup = config.get('backup', False)
    
    add_missing = args.add_missing
    
    if not target_dir.is_dir():
        print(f"错误: '{target_dir}' 不是一个有效的文件夹路径。")
        sys.exit(1)
    
    # 遍历文件夹，查找符合条件的 YAML 文件
    yaml_files = list(target_dir.rglob('*.yaml')) + list(target_dir.rglob('*.yml'))
    filtered_files = []
    for f in yaml_files:
        try:
            relative_path = f.relative_to(target_dir).as_posix()
        except ValueError:
            # 文件不在目标目录下
            continue
        if any(s in relative_path for s in search_strings):
            filtered_files.append(f)
    
    if not filtered_files:
        print(f"未找到包含指定字符串 {search_strings} 的 YAML 文件。")
        return
    
    print(f"找到 {len(filtered_files)} 个符合条件的 YAML 文件。")
    
    for file_path in filtered_files:
        print(f"\n处理文件: {file_path}")
        if backup:
            create_backup(file_path)
        
        try:
            data, yaml = load_yaml(file_path)
        except Exception as e:
            print(f"错误: 无法加载文件 '{file_path}': {e}")
            continue
        
        modify_yaml(data, modifications, add_missing)
        
        try:
            save_yaml(data, file_path, yaml)
            print(f"文件已保存: {file_path}")
        except Exception as e:
            print(f"错误: 无法保存文件 '{file_path}': {e}")
    
    print("\n所有文件处理完成。")

if __name__ == "__main__":
    main()
