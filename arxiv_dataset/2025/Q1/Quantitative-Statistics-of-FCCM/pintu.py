import os
import cv2
from PIL import Image


def is_prime(number):
    # 0和1不是质数
    if number < 2:
        return False
    # 2和3是质数
    elif number == 2 or number == 3:
        return True
    # 如果能整除2或3，不是质数
    elif number % 2 == 0 or number % 3 == 0:
        return False
    # 从5开始检查是否能被整除，只检查奇数
    i = 5
    while i * i <= number:
        if number % i == 0 or number % (i + 2) == 0:
            return False
        i += 6
    return True


def closest_factors(n):
    # 初始化因子
    factor1 = 1
    factor2 = n

    # 寻找最接近平方根的因子
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            factor1 = i
            factor2 = n // i

    return factor1, factor2


## 拼接图像 ##
def tp_pinjie_row(images_path, save_path, gap):
    test_imgs = os.listdir(images_path)
    test_imgs.sort(key=lambda x: (int(x.split('.')[0])))
    pj_imgs_path = []
    pj_imgs_path.append(images_path)

    image_save_path = save_path  # 图片转换后的绝对路径

    img_num = len(test_imgs)     # 测试图片张数
    a = is_prime(img_num)
    b = True
    image_row, image_column = 0, 0
    while b:
        if a:
            img_num = img_num - 1
            a = is_prime(img_num)
        else:
            b = False
            image_row, image_column = closest_factors(img_num)

    pj_imgs_abspath = []         # 保存测试图片的绝对路径
    # for i in range(len(test_imgs)):
    #     for j in pj_imgs_path:
    #         pj_imgs_abspath.append(os.path.join(j, test_imgs[i]))

    for j in pj_imgs_path:
        for i in range(img_num):
            pj_imgs_abspath.append(os.path.join(j, test_imgs[i]))

    lenth = len(pj_imgs_abspath)
    # 简单的对于参数的设定和实际图片集的大小进行数量判断
    if lenth != image_row * image_column:
        raise ValueError("合成图片的参数和要求的数量不能匹配！")

    c = cv2.imread(pj_imgs_abspath[0])
    h1, w1, channels = c.shape
    h = h1 * image_row
    w = w1 * image_column

    to_image = Image.new('RGB', (w + (image_column - 1) * gap, h + (image_row - 1) * gap), (255, 255, 255))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    a = int(h / image_row)
    b = int(w / image_column)
    for x in range(image_row):
        for y in range(image_column):
            img = Image.open(pj_imgs_abspath[image_column * x + y])
            img_1 = img.resize((b, a))
            to_image.paste(img_1, (y * b + y * gap, x * a + x * gap))

    to_image.save(image_save_path)  # 保存新图

