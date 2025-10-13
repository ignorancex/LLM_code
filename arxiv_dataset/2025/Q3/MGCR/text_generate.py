import torch
import argparse
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

from llava.llm_agent import LLavaAgent
from utils.CKPT_PTH import LLAVA_MODEL_PATH  # 只加载 LLaVA 模型路径

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", default='Dataset/LEVIR/train/B', type=str)
parser.add_argument("--save_dir", default='LEVIR', type=str)
parser.add_argument("--load_4bit_llava", action='store_true', default=True)
parser.add_argument("--load_8bit_llava", action='store_true', default=False)
parser.add_argument('--gpu', '-g', default='3,4,5', metavar='cuda', type=str, help='device id')
args = parser.parse_args()
print(args)

# 设备分配（多 GPU）
device_ids = [int(i) for i in args.gpu.split(',')]
device = torch.device(f'cuda:{device_ids[0]}')

# 加载 LLaVA 模型
llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=device, load_4bit=args.load_8bit_llava, load_8bit=args.load_8bit_llava)

# 生成保存的 Excel 文件名
dataset_name = os.path.basename(args.img_dir)  # 使用文件夹名称作为数据集名称
output_file = os.path.join(args.save_dir, f"{dataset_name}_I2T.xlsx")

# 确保输出目录存在
os.makedirs(args.save_dir, exist_ok=True)

# 存储图像名称和对应描述
data_list = []

# 关闭梯度计算，节省显存
with torch.no_grad():
    # 获取文件列表并排序（按文件名升序）
    img_files = sorted(os.listdir(args.img_dir),key=str.lower)

    for file_name in tqdm(img_files, desc="Processing Images", unit="image"):
        img_path = os.path.join(args.img_dir, file_name)

        # 仅处理图片格式文件
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
            continue

        # 加载图像
        image = Image.open(img_path).convert('RGB')

        # 使用 LLaVA 生成图像描述
        captions = llava_agent.gen_image_caption([image])

        # 存入列表
        data_list.append([file_name, captions[0]])

        # **打印图像名称及描述**
        print(f"🖼️ {file_name}: {captions[0]}")

        # 立即保存为 Excel 文件，确保实时更新
        df = pd.DataFrame(data_list, columns=["Image", "Description"])
        df.to_excel(output_file, index=False)

print(f"✅ 图像描述已保存到 {output_file}")
