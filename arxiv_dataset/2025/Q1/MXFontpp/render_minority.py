from PIL import Image
import numpy as np
import cv2
import glob
from tqdm import tqdm


def calculate_char_spacing(image_path):
    # 打开图像
    img = Image.open(image_path)

    # 转换为灰度图像
    gray_img = img.convert('L')

    # 转换为 NumPy 数组
    img_array = np.array(gray_img)

    # 将图像二值化，黑色部分为字符
    binary_img = (img_array < 128).astype(np.uint8)

    # 找到字符的左边缘和右边缘位置
    left_edges = np.argmax(binary_img, axis=1)
    right_edges = binary_img.shape[1] - np.argmax(binary_img[:, ::-1], axis=1) - 1

    # 计算字符之间的距离
    char_spacing = left_edges[1:] - right_edges[:-1]

    return char_spacing


def refine_char(img_fp):
    im = cv2.imread(img_fp, cv2.IMREAD_GRAYSCALE)
    im = im > 0
    im = 1 - im
    line = im.sum(axis=0) > 0
    line = line + 0   # [0,0,1,1,0,0,1,1,...] 0 indicate white, 1 indicate black
    
    kernel = np.array([-1,1])
    for i in range(len(line)-2):
        line[i] = (line[i:i+2] * kernel).sum()
    
    res = []
    for x in range(len(line)):
        if line[x] != 1:
            continue
        for y in range(x+1, len(line)):
            if line[y] == -1:
                res.append((x, y))
                break
                
    return res


def render_char_to_word(img_fp, interval=10):
    loc = refine_char(img_fp)
    im = cv2.imread(img_fp, 0)
    
    h, w = im.shape
    
    tmp = [np.ones((h, interval))*255]
    
    for left, right in loc:
        tmp.append(im[:, left:right]) 
        tmp.append(np.ones((h, interval))*255)

    return np.concatenate(tmp, axis=1)


if __name__ == '__main__':
    images = glob.glob('/DATA/bvac/personal/reserach/font_gen/mxfont/synth_minority/weizu_use2train_v2/images/*.png')
    for each in tqdm(images):
        image = render_char_to_word(each)
        cv2.imwrite(each, image)