import os
import json

def read_txt(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f'File not found: {path}')

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            content = json.load(file)
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f'File not found: {path}')

def write_json(path, content):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(content, file, indent=4)