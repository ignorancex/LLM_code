from PIL import Image, ImageFont, ImageDraw
import os
import time
import datetime
import random
def fit_text(text, max_width, font_path='arial.ttf'):
    """动态调整字体大小以适应指定宽度的文本并支持自动换行。
    
    Args:
        text (str): 要格式化的文本。
        max_width (int): 最大允许的文本宽度。
        font_path (str): 字体文件路径。
    
    Returns:
        tuple: (调整后的字体大小, 行数, 总高度)
    """
    font_size = 26  # 初始字体大小
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))  # 创建一个临时画布来计算文本尺寸
    words = text.split()
    lines = []
    current_line = []
    current_line_width = 0

    for word in words:
        word_width = draw.textlength(word, font=font)
        if current_line_width + word_width + len(current_line) * 5 <= max_width:
            current_line.append(word)
            current_line_width += word_width + 5  # 添加单词间距
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_line_width = word_width
    if current_line:
        lines.append(' '.join(current_line))

    # 计算所需的高度
    line_height = font_size  # 使用字体大小作为行高
    total_height = len(lines) * line_height

    return font_size, lines, total_height

def create_image_for_text(text, max_width, font_path='arial.ttf'):
    """创建一个足够大的图像以适应文本，并绘制文本。
    
    Args:
        text (str): 要绘制的文本。
        max_width (int): 图像的最大宽度。
        font_path (str): 字体文件路径。
    
    Returns:
        Image: 包含文本的图像。
    """
    font_size, lines, total_height = fit_text(text, max_width, font_path)
    font = ImageFont.truetype(font_path, font_size)
    img = Image.new('RGB', (max_width, total_height + 20), (255, 255, 255))  # 添加一些边距
    draw = ImageDraw.Draw(img)
    
    # 绘制每一行文本
    y = 10
    for line in lines:
        draw.text((10, y), line, font=font, fill=(0, 0, 0))
        y += font_size  # 增加行间距

    return img

def draw_text_on_image(image_path, text, font_path='arial.ttf'):
    """在原始图片下方垂直拼接文本。
    
    Args:
        image_path (str): 原始图像路径。
        text (str): 要添加的文本。
        font_path (str): 字体文件路径。
    
    Returns:
        Image: 垂直拼接后的图像。
    """
    img = Image.open(image_path)
    max_width = img.width
    font_size, lines, total_height = fit_text(text, max_width, font_path)
    text_img = create_image_for_text(text, max_width, font_path)
    result_img = Image.new('RGB', (img.width, img.height + text_img.height), (255, 255, 255))
    result_img.paste(img, (0, 0))
    result_img.paste(text_img, (0, img.height))
    return result_img

def save_image(img, output_path):
    """保存图像到指定路径。
    
    Args:
        img (Image): 要保存的图像。
        output_path (str): 输出路径。
    """
    img.save(output_path)

import pandas as pd
import os
import json
import jsonlines
from tqdm import tqdm
def find_matching_files(base_dir, cat, file_id):
    """
    在指定的基目录下查找是否存在与给定类别和文件名匹配的文件。
    :param base_dir: 基目录路径
    :param cat: 类别名称
    :param file_id: 文件名
    :return: 匹配的文件路径列表
    """
    matching_files = []
    # 遍历基目录下的所有子目录
    for root, dirs, files in os.walk(base_dir):
        # 检查当前目录是否包含指定类别和文件名
        if cat in dirs:
            cat_path = os.path.join(root, cat, 'images', file_id)
            if os.path.exists(cat_path):
                matching_files.append(cat_path)
    return matching_files


def read_data(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for line in tqdm(reader, desc="Loading data..."):
            data.append(line)
        return data


file_path = "/Datasets/mmvet/mm-vet/mm-vet.json"  # 替换为你的文件路径
with open(file_path, 'r') as file:
    data = json.load(file)
save_path_dir = '/Datasets/mmvet/mm-vet/new_image'
# 对每个对象的 instruction 进行处理，并显示进度条
for i in tqdm(range(len(data)), desc="处理数据", unit="条"):
    id = f"v1_{i}"

    imagename = data[id]['imagename']

    instruction = data[id]['question']
    image_path =  '/Datasets/mmvet/mm-vet/images/' + imagename
    font_path_list = [
                    "/Datasets/english/ARIAL.TTF",
                    "/Datasets/english/FreeMonoBoldOblique.ttf",
                    "/Datasets/english/ventura_times.ttf"
                ]

    font_path = random.choice(font_path_list)
    image = draw_text_on_image(image_path,instruction,font_path)
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
        print(f"目录已创建: {save_path_dir}")
    # file_id = img_path.split('/')[-1]

    save_path = os.path.join(save_path_dir,imagename)
    # print(save_path )
    # exit(0)
    image.save(save_path)
 
            