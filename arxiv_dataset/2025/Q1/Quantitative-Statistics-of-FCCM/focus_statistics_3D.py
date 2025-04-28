import os
import ants
import skimage.measure
import torch
import argparse
import time
import cv2
import numpy as np
import nibabel as nib  # 用nibabel包打开nii文件
import imageio  # 图像io
import pandas as pd
import multiprocessing
import subprocess
import segmentation_models_pytorch as smp
import albumentations as A
from torch.utils.data import DataLoader
from datasets.create_dataset import Mydataset_test
from albumentations.pytorch import ToTensorV2
from skimage.measure import label, regionprops
from Quantitative_Statistics_Module import Quantitative_Statistics, Quantitative_Statistics_1
from pintu import tp_pinjie_row
from Comparison_chart import Comparison_chart_1, Comparison_chart_2, Volume_proportion_statistics


def get_rectangle(ori_label_path, yuzhi):
    # 加载原始图像的分割标签
    original_labels = nib.load(ori_label_path).get_fdata()

    # 阈值化图像，将非白色像素设置为0，白色像素设置为1
    thresholded_image = np.where(original_labels > 0, 1, 0)

    # 使用连通组件分析获取连通区域的标签
    labeled_image, num_labels = label(thresholded_image, connectivity=1, return_num=True)

    # 获取区域的属性
    regions = regionprops(labeled_image)

    # 计算每个白色立体区域的体积
    region_volumes = [int(region.area) for region in regions if region.area > yuzhi]

    # 存储每个白色像素区域的最小外接立方体坐标
    bounding_boxes = []

    # 给每个白色立体区域编号
    for region in regions:
        if region.area > yuzhi:
            # 获取最小外接立方体坐标
            minr, minc, minz, maxr, maxc, maxz = region.bbox
            bounding_boxes.append([[minr, minc, minz], [maxr, maxc, maxz]])

    return region_volumes, bounding_boxes


def set_specific_pixels(ori_image_path, bounding_box, result_path):
    if not os.path.exists(result_path):
        os.makedirs(result_path)  # 新建文件夹

    # 加载原始脑部图像
    original_image = nib.load(ori_image_path).get_fdata()

    # 创建一个与original_image相同的数组，但数据类型为uint16
    copy_image = original_image.astype('uint16')

    mins = bounding_box[0]
    maxs = bounding_box[1]

    # 计算中心坐标
    center_x = (mins[1] + maxs[1]) // 2
    center_y = (mins[0] + maxs[0]) // 2
    center_z = (mins[2] + maxs[2]) // 2

    # 将中心点坐标的像素值设置为2000
    copy_image[center_y, center_x, center_z] = 1000

    # 创建一个 NIfTI 图像对象，使用 copy_image 和原始的 NIfTI 文件头信息
    output_nifti = nib.Nifti1Image(copy_image, affine=nib.load(ori_image_path).affine)

    # 保存 NIfTI 图像到文件
    output_path = result_path + '/' + ori_image_path.split('/')[-1]
    nib.save(output_nifti, output_path)

    return output_path


def get_specific_pixel_coordinates(peizhun_after_path):
    # 加载图像
    peizhun_after_image = nib.load(peizhun_after_path).get_fdata()

    # max_pixel_coordinates = np.argwhere((peizhun_after_image > 255) & (peizhun_after_image <= 1000))
    # pixel_values = peizhun_after_image[max_pixel_coordinates[:, 0], max_pixel_coordinates[:, 1], max_pixel_coordinates[:, 2]]

    # 找到像素值的最小值和最大值
    # min_pixel_value = np.min(peizhun_after_image)
    max_pixel_value = np.max(peizhun_after_image)

    # print(f"最小像素值: {min_pixel_value}")
    # print(f"最大像素值: {max_pixel_value}")

    # 找到像素值最大的像素点的数量
    # max_pixel_value_count = np.sum(peizhun_after_image == max_pixel_value)

    # print(f"像素值最大的像素点数量: {max_pixel_value_count}")

    # 找到像素值最大的像素点的坐标
    max_pixel_coordinates = np.argwhere(peizhun_after_image == max_pixel_value)

    # max_pixel_coordinates 是一个包含坐标的NumPy数组，每一行表示一个个像素点的坐标，例如 [(x1, y1, z1), (x2, y2, z2), ...]
    # 打印像素点的坐标
    x, y, z = max_pixel_coordinates[0]
    # print(f"像素点 {1} 的坐标: ({x}, {y}, {z})")
    a = [x, y, z]
    return a


def registration(fixed_image_path, moving_image_path, peizhun_result):
    # 读取待处理的图像
    fix_img = ants.image_read(fixed_image_path, pixeltype='float')
    move_img = ants.image_read(moving_image_path, pixeltype='float')
    # Rigid Similarity
    result = peizhun_result
    if not os.path.exists(result):
        os.makedirs(result)  # 新建文件夹
    out = ants.registration(fix_img, move_img, type_of_transform='Similarity', outprefix=result + '/')
    reg_img = out['warpedmovout']  # 获取配准结果
    reg_img.to_file(result + '/' + moving_image_path.split('/')[-1])
    _ = result + '/' + moving_image_path.split('/')[-1]
    return _


def nii_to_image(filepath, img_f_pathz):
    # 开始读取nii文件
    img = nib.load(filepath, )  # 读取nii
    img_fdata = img.get_fdata()

    if not os.path.exists(img_f_pathz):
        os.makedirs(img_f_pathz)  # 新建文件夹

    # 开始转换为图像
    # 可能用到的图像变换
    # 旋转操作利用numpy 的rot90（a,b）函数 a为数据 b为旋转90度的多少倍 ！正数逆时针 负数顺时针
    # 左右翻转 ： img_lr = np.fliplr(img) 上下翻转： img_ud = np.flipud(img)
    (x, y, z) = img.shape  # 获取图像的3个方向的维度

    for i in range(z):  # z方向
        slice = np.fliplr(np.rot90(img_fdata[:, :, i], 1))
        slice[slice == 0] = 1e-6
        slice = slice / slice.max()  # normalizes img_grey in range 0 - 255
        slice = 255 * slice
        slice = slice.astype(np.uint8)
        imageio.imwrite(os.path.join(img_f_pathz, '{}.png'.format(i)), slice)  # 保存图像
    return img_f_pathz


def _2d_to_3d(nii_2d, img_save_path, img_3d_name):
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    a = os.listdir(nii_2d)
    a.sort(key=lambda x: (int(x.split('.')[0])))

    img_nii_2d_path = []
    for i in a:
        b = os.path.join(nii_2d, i)
        img_nii_2d_path.append(b)

    allImg = np.zeros([768, 768, len(img_nii_2d_path)], dtype='uint8')
    for h in range(len(img_nii_2d_path)):
        single_image_name = img_nii_2d_path[h]
        img_as_img = cv2.imread(single_image_name)
        img_as_img = cv2.rotate(img_as_img, cv2.ROTATE_90_CLOCKWISE)
        img_as_img = cv2.flip(img_as_img, 0)
        img_as_img = cv2.cvtColor(img_as_img, cv2.COLOR_BGR2GRAY)
        allImg[:, :, h] = img_as_img

    # 创建一个新的NIfTI图像对象，新生成的图像作为数据
    copy_image_3d = nib.Nifti1Image(allImg, affine=np.eye(4))

    # 保存图像为新的NIfTI格式文件
    nib.save(copy_image_3d, img_save_path + '/' + img_3d_name)
    # print(np.shape(copy_image_3d))
    _ = img_save_path + '/' + img_3d_name
    return _


def to_csv(path, save_path):
    list_name = []
    imgs = os.listdir(path)
    imgs.sort(key=lambda x: (int(x.split('.')[0])))
    for img_name in imgs:  # 遍历每张图片
        list_name.append(img_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df = pd.DataFrame()
    df['image_name'] = list_name
    df.to_csv(save_path + '/' + 'infer_statistics.csv', index=False)
    result = save_path + '/' + 'infer_statistics.csv'
    return result


def Lesion_Visualization(path_2d, region_volumes_P, region_volumes, box_label, biaozhun, biaozhun_mask, region_volumes_final_2,
                         region_volumes_final, bounding_boxes_pair, copy_image, save_path, numbering, yuzhi):
    if not os.path.exists(save_path + '/' + '2d_previous'):
        os.makedirs(save_path + '/' + '2d_previous')
    path_2d_previous = nii_to_image(biaozhun, save_path + '/2d_previous')
    img_2d_previous = os.listdir(path_2d_previous)
    img_2d_previous.sort(key=lambda x: (int(x.split('.')[0])))
    img_2d_path_previous = []
    for j in img_2d_previous:
        b = os.path.join(path_2d_previous, j)
        img_2d_path_previous.append(b)

    img_2d = os.listdir(path_2d)
    img_2d.sort(key=lambda x: (int(x.split('.')[0])))
    img_2d_path = []
    for j in img_2d:
        b = os.path.join(path_2d, j)
        img_2d_path.append(b)

    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX  # 字体样式
    # font = cv2.FONT_HERSHEY_SIMPLEX

    V_P, V_F = [], []
    for i in region_volumes_final_2:
        if type(i) == list:
            V_P.append(i[0])
            V_F.append(i[1])

    bounding_boxes_P, bounding_boxes_F, box_box = [], [], {}
    for i in bounding_boxes_pair:
        if len(i) != 0:
            bounding_boxes_P.append(i[0])
            bounding_boxes_F.append(i[1])
            if str(i[0]) not in list(box_box.keys()):
                box_box[str(i[0])] = i[1]
            else:
                if V_F[bounding_boxes_F.index(i[1])] > V_F[bounding_boxes_F.index(box_box[str(i[0])])]:
                    box_box[str(i[0])] = i[1]

    # 加载3D mask图像
    image_3d = nib.load(biaozhun_mask)
    image_data = image_3d.get_fdata()
    thresholded_image = np.where(image_data > 0, 1, 0)
    labeled_image, num_labels = skimage.measure.label(thresholded_image, connectivity=1, return_num=True)
    regions = regionprops(labeled_image)
    copy_image_1 = np.zeros(image_data.shape, dtype='uint16')

    for region in regions:
        label_value = region.label
        minr, minc, minz, maxr, maxc, maxz = region.bbox
        if [[minr, minc, minz], [maxr, maxc, maxz]] in bounding_boxes_P:
            label_value_1 = box_label[str(box_box[str([[minr, minc, minz], [maxr, maxc, maxz]])])]
            copy_image_1[labeled_image == label_value] = label_value_1

    indexed_original_list = list(enumerate(region_volumes_final))

    non_list_elements = []
    list_elements = []
    for i in indexed_original_list:
        a, b = i
        if type(b) == list:
            list_elements.append(i)
        else:
            non_list_elements.append(i)

    # 对非列表元素排序
    sorted_non_list_elements = sorted(non_list_elements, key=lambda x: x[1])
    # 对列表元素排序
    sorted_list_elements = sorted(list_elements, key=lambda x: x[1])

    for i in sorted_list_elements:
        sorted_non_list_elements.append(i)

    region_volumes_sorted = []
    difference_list = []
    for i in range(len(sorted_non_list_elements)):
        if type(sorted_non_list_elements[i][1]) != list:
            region_volumes_sorted.append(sorted_non_list_elements[i][1])
            difference_list.append('')
        else:
            difference = sorted_non_list_elements[i][1][1] - sorted_non_list_elements[i][1][0]
            if i != 0:
                if type(sorted_non_list_elements[i - 1][1]) == list:
                    if sorted_non_list_elements[i][1][0] == sorted_non_list_elements[i - 1][1][0]:
                        difference = int(difference_list[i - 1]) + sorted_non_list_elements[i][1][1]
                        difference_list[i - 1] = ''
            if difference > 0:
                difference = '+' + str(difference)
            else:
                difference = str(difference)
            difference_list.append(difference)
            y = str(sorted_non_list_elements[i][1])
            y = y.replace('[', '')
            y = y.replace(']', '')
            region_volumes_sorted.append(y)

    # 创建字典，将原始列表中元素的索引与排序后的索引对应起来
    # 此处original_index + 1要与box_label[str([[minr, minc, minz], [maxr, maxc, maxz]])] = label_num中的label_num的赋值和顺序一致
    index_mapping = {original_index + 1: sorted_index + 1 for sorted_index, (original_index, _) in
                     enumerate(sorted_non_list_elements)}

    (x, y, z) = copy_image.shape  # 获取图像的3个方向的维度
    for i in range(z):  # z方向
        silce = copy_image[:, :, i]
        silce = cv2.rotate(silce, cv2.ROTATE_90_CLOCKWISE)
        silce = cv2.flip(silce, 0)
        silce = silce.astype(np.uint16)
        # 创建一个全黑的3通道彩色图像，大小和切片一致
        color_image = np.zeros((silce.shape[0], silce.shape[1], 3), dtype=np.uint8)

        position = {}
        for a in range(len(region_volumes_final)):
            color_image[silce == a + 1] = [255, 255, 255]

            # 找到当前区域的像素点坐标
            points = np.column_stack(np.where(silce == a + 1))

            if len(points) > 0:
                # 计算标签位置
                text_position = (points[:, 1].mean(), points[:, 0].mean())
                position[a + 1] = text_position

        gray_1 = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # 二值化处理
        _, binary_image = cv2.threshold(gray_1, 0, 255, cv2.THRESH_BINARY)
        # 找到当前区域的轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_2d = cv2.imread(img_2d_path[i])

        # # 绘制病灶区域的轮廓
        # cv2.drawContours(image_2d, contours, -1, (0, 0, 255), 1)
        # index_mapping[list(position.keys())[c]] 是排序后体积的索引， list(position.keys())[c]是排序前的体积的索引，用他区调用bounding_boxes

        if numbering:
            for c in range(len(position)):
                cv2.putText(image_2d, str(index_mapping[list(position.keys())[c]]),
                            (int(list(position.values())[c][0]), int(list(position.values())[c][1])),
                            font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


        # 绘制病灶区域的轮廓
        cv2.drawContours(image_2d, contours, -1, (0, 0, 255), 1)

        if not os.path.exists(save_path + '/' + 'Lesion_Visualization_Followup'):
            os.makedirs(save_path + '/' + 'Lesion_Visualization_Followup')
        cv2.imwrite(save_path + '/' + 'Lesion_Visualization_Followup' + '/' + str(i) + '.png', image_2d)  # 保存图像

    (x, y, z) = copy_image_1.shape  # 获取图像的3个方向的维度
    for i in range(z):  # z方向
        silce = copy_image_1[:, :, i]
        silce = cv2.rotate(silce, cv2.ROTATE_90_CLOCKWISE)
        silce = cv2.flip(silce, 0)
        silce = silce.astype(np.uint16)
        # 创建一个全黑的3通道彩色图像，大小和切片一致
        color_image_1 = np.zeros((silce.shape[0], silce.shape[1], 3), dtype=np.uint8)

        position = {}
        for a in range(len(region_volumes_final)):
            color_image_1[silce == a + 1] = [255, 255, 255]

            # 找到当前区域的像素点坐标
            points = np.column_stack(np.where(silce == a + 1))

            if len(points) > 0:
                # 计算标签位置
                text_position = (points[:, 1].mean(), points[:, 0].mean())
                position[a + 1] = text_position

        gray_1 = cv2.cvtColor(color_image_1, cv2.COLOR_BGR2GRAY)
        # 二值化处理
        _, binary_image = cv2.threshold(gray_1, 0, 255, cv2.THRESH_BINARY)
        # 找到当前区域的轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_2d_previous = cv2.imread(img_2d_path_previous[i])

        for c in range(len(position)):
            cv2.putText(image_2d_previous, str(index_mapping[list(position.keys())[c]]),
                        (int(list(position.values())[c][0]), int(list(position.values())[c][1])),
                        font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # 绘制病灶区域的轮廓
        cv2.drawContours(image_2d_previous, contours, -1, (0, 0, 255), 1)

        if not os.path.exists(save_path + '/' + 'Lesion_Visualization_Previous'):
            os.makedirs(save_path + '/' + 'Lesion_Visualization_Previous')
        cv2.imwrite(save_path + '/' + 'Lesion_Visualization_Previous' + '/' + str(i) + '.png', image_2d_previous)  # 保存图像

    label = [m + 1 for m in range(len(region_volumes_sorted))]
    # 创建DataFrame
    df = pd.DataFrame()
    df['label'] = label
    df['Focal_volume'] = region_volumes_sorted
    df['Volume_change'] = difference_list

    def align_center(x):
        return ['text-align: right' for x in x]

    with pd.ExcelWriter(save_path + "/Focal_volume.xlsx") as writer:
        df.style.apply(align_center, axis=1).to_excel(
            writer,
            index=False
        )
    tp_pinjie_row(save_path + '/' + 'Lesion_Visualization_Followup', save_path + '/' + 'Followup.png', gap=0)
    tp_pinjie_row(save_path + '/' + 'Lesion_Visualization_Previous', save_path + '/' + 'Previous.png', gap=0)
    Comparison_chart_1(region_volumes_final, save_path)
    Comparison_chart_2(region_volumes_final, region_volumes_P, save_path)
    Volume_proportion_statistics(region_volumes, save_path)

def Lesion_Visualization_1(path_2d, region_volumes_final, copy_image, save_path, numbering, yuzhi):
    img_2d = os.listdir(path_2d)
    img_2d.sort(key=lambda x: (int(x.split('.')[0])))
    img_2d_path = []
    for j in img_2d:
        b = os.path.join(path_2d, j)
        img_2d_path.append(b)

    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX  # 字体样式
    # font = cv2.FONT_HERSHEY_SIMPLEX

    indexed_original_list = list(enumerate(region_volumes_final))

    non_list_elements = []
    list_elements = []
    for i in indexed_original_list:
        a, b = i
        if type(b) == list:
            list_elements.append(i)
        else:
            non_list_elements.append(i)

    # 对非列表元素排序
    sorted_non_list_elements = sorted(non_list_elements, key=lambda x: x[1])
    # 对列表元素排序
    sorted_list_elements = sorted(list_elements, key=lambda x: x[1])

    for i in sorted_list_elements:
        sorted_non_list_elements.append(i)

    region_volumes_sorted = []
    difference_list = []
    for i in range(len(sorted_non_list_elements)):
        if type(sorted_non_list_elements[i][1]) != list:
            region_volumes_sorted.append(sorted_non_list_elements[i][1])
            difference_list.append('')
        else:
            difference = sorted_non_list_elements[i][1][1] - sorted_non_list_elements[i][1][0]
            if i != 0:
                if type(sorted_non_list_elements[i - 1][1]) == list:
                    if sorted_non_list_elements[i][1][0] == sorted_non_list_elements[i - 1][1][0]:
                        difference = int(difference_list[i - 1]) + sorted_non_list_elements[i][1][1]
                        difference_list[i - 1] = ''
            if difference > 0:
                difference = '+' + str(difference)
            else:
                difference = str(difference)
            difference_list.append(difference)
            y = str(sorted_non_list_elements[i][1])
            y = y.replace('[', '')
            y = y.replace(']', '')
            region_volumes_sorted.append(y)

    # 创建字典，将原始列表中元素的索引与排序后的索引对应起来
    index_mapping = {original_index + 1: sorted_index + 1 for sorted_index, (original_index, _) in
                     enumerate(sorted_non_list_elements)}

    (x, y, z) = copy_image.shape  # 获取图像的3个方向的维度
    for i in range(z):  # z方向
        silce = copy_image[:, :, i]
        silce = cv2.rotate(silce, cv2.ROTATE_90_CLOCKWISE)
        silce = cv2.flip(silce, 0)
        silce = silce.astype(np.uint16)
        # 创建一个全黑的3通道彩色图像，大小和切片一致
        color_image = np.zeros((silce.shape[0], silce.shape[1], 3), dtype=np.uint8)

        position = {}
        for a in range(len(region_volumes_final)):
            color_image[silce == a + 1] = [255, 255, 255]

            # 找到当前区域的像素点坐标
            points = np.column_stack(np.where(silce == a + 1))

            if len(points) > 0:
                # 计算标签位置
                text_position = (points[:, 1].mean(), points[:, 0].mean())
                position[a + 1] = text_position

        gray_1 = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # 二值化处理
        _, binary_image = cv2.threshold(gray_1, 0, 255, cv2.THRESH_BINARY)
        # 找到当前区域的轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_2d = cv2.imread(img_2d_path[i])

        # # 绘制病灶区域的轮廓
        # cv2.drawContours(image_2d, contours, -1, (0, 0, 255), 1)

        if numbering:
            for c in range(len(position)):
                cv2.putText(image_2d, str(index_mapping[list(position.keys())[c]]),
                            (int(list(position.values())[c][0]), int(list(position.values())[c][1])),
                            font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # 绘制病灶区域的轮廓
        cv2.drawContours(image_2d, contours, -1, (0, 0, 255), 1)

        if not os.path.exists(save_path + '/' + 'Lesion_Visualization'):
            os.makedirs(save_path + '/' + 'Lesion_Visualization')
        cv2.imwrite(save_path + '/' + 'Lesion_Visualization' + '/' + str(i) + '.png', image_2d)  # 保存图像

    label = [m + 1 for m in range(len(region_volumes_sorted))]
    # 创建DataFrame
    df = pd.DataFrame()
    df['label'] = label
    df['Focal_volume'] = region_volumes_sorted
    df['Volume_change'] = difference_list

    def align_center(x):
        return ['text-align: right' for x in x]

    with pd.ExcelWriter(save_path + "/Focal_volume.xlsx") as writer:
        df.style.apply(align_center, axis=1).to_excel(
            writer,
            index=False
        )
    tp_pinjie_row(save_path + '/' + 'Lesion_Visualization', save_path + '/' + 'Lesion_Visualization.png', gap=0)

def infer(test_img_path, result_path, csv_path):
    labels_test_path = '/mnt/zrg/Image_segmentation/2D_segmentation/ccm_duo_fa_segmentation/result/test_mask'
    batch_size = 1
    weight = './model-weight/Unet_effi_b3.pth'

    pre_result_path = result_path
    if not os.path.exists(pre_result_path):
        os.makedirs(pre_result_path)

    df_test = pd.read_csv(csv_path)  # [0:10]
    test_imgs, test_masks = test_img_path, labels_test_path
    test_imgs = [''.join([test_imgs, '/', i]) for i in df_test['image_name']]
    test_masks = [''.join([test_masks, '/', i]) for i in df_test['image_name']]

    test_transform = A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)

    test_ds = Mydataset_test(test_imgs, test_masks, test_transform)
    test_dl = DataLoader(test_ds, batch_size=batch_size, pin_memory=False, num_workers=0, )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = smp.Unet(encoder_name='efficientnet-b3', encoder_weights=None, classes=2).to(device)

    state_dict = torch.load(weight)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        for batch_idx, (name, inputs, targets) in enumerate(test_dl):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            out = model(inputs)
            predicted = out.argmax(1)

            predict = predicted.squeeze(0)
            mask_np = predict.cpu().numpy()
            mask_np = (mask_np * 255).astype('uint8')
            mask_np[mask_np > 160] = 255
            mask_np[mask_np <= 160] = 0

            cv2.imwrite(pre_result_path + '/' + name[0], mask_np)

    return pre_result_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous_img_path', type=str,
                        default='/mnt/zrg/dataset/yingxiang/ccm_duo_fa_nii/ccm_duo_fa_nii_3d/LI_YU_RUAN-F12_01_LYRU-2021.03.06.nii',
                        help='previous img data path.')
    parser.add_argument('--test_img_path', type=str,
                        default='/mnt/zrg/dataset/yingxiang/ccm_duo_fa_nii/ccm_duo_fa_nii_3d/LI_YU_LUAN-F12_01_LYRU-2022.04.23.nii',
                        help='follow-up img data path.')
    parser.add_argument('--statistics_guodu_file', type=str,
                        default='/mnt/zrg/Image_segmentation/2D_segmentation/FCCM_TBME_1/result/QS2/F12_01_LYRU',
                        help='statistics guodu file.')
    parser.add_argument('--comparative_statistics', type=str, default=True, help='comparative_statistics ?')
    parser.add_argument('--numbering', type=str, default=True, help='numbering ?')
    parser.add_argument('--yuzhi', default='10', type=int, help='Threshold for statistical volume')
    parser.add_argument('--threshold', default='200', type=int, help='Volume threshold for matching lesions')
    parser.add_argument('--devicenum', default='0', type=str, )

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devicenum

    begin_time = time.time()

    image_3d_name_P = args.previous_img_path.split('/')[-1]
    image_3d_name_F = args.test_img_path.split('/')[-1]
    result_path = args.statistics_guodu_file
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    region_volumes_final, region_volumes_final_1, bounding_boxes_pair, bounding_boxes_pair_1 = [], [], [], []
    if args.comparative_statistics:
        biaozhun = args.previous_img_path

        # Previous inspection的统计数据
        path_2d_P = nii_to_image(biaozhun, result_path + '/2d_P')
        csv_path_P = to_csv(path_2d_P, result_path)
        path_2d_mask_P = infer(path_2d_P, result_path + '/2d_mask_P', csv_path_P)
        path_3d_mask_P = _2d_to_3d(path_2d_mask_P, result_path + '/3d_mask_P', image_3d_name_P)
        region_volumes_P, bounding_boxes_P, relevant_label_image_P, box_label_P = Quantitative_Statistics(path_3d_mask_P,
                                                                                                  result_path,
                                                                                                  yuzhi=args.yuzhi)

        # 展示配准前的统计数据
        path_2d = nii_to_image(args.test_img_path, result_path + '/2d_before')
        csv_path = to_csv(path_2d, result_path)
        path_2d_mask = infer(path_2d, result_path + '/2d_mask_before', csv_path)
        path_3d_mask = _2d_to_3d(path_2d_mask, result_path + '/3d_mask_before', image_3d_name_F)
        region_volumes, bounding_boxes, relevant_label_image, box_label = Quantitative_Statistics(path_3d_mask,
                                                                                                  result_path,
                                                                                                  yuzhi=args.yuzhi)

        region_volumes_low, region_volumes_high = [], []
        bounding_boxes_low, bounding_boxes_high = [], []

        for i in range(len(region_volumes)):
            if region_volumes[i] >= args.threshold:
                region_volumes_high.append(region_volumes[i])
                bounding_boxes_high.append(bounding_boxes[i])
            else:
                region_volumes_low.append(region_volumes[i])
                bounding_boxes_low.append(bounding_boxes[i])

        with open(os.path.join(result_path, 'output.txt'), 'w') as file:
            file.writelines(str(region_volumes_P))
            file.write("\n")
            file.writelines(str(bounding_boxes_P))
            file.write("\n")
            file.writelines(str(region_volumes_high))
            file.write("\n")
            file.writelines(str(bounding_boxes_high))
            file.write("\n")
            file.writelines(str(box_label))

        if len(bounding_boxes_high) > 0:
            if len(bounding_boxes_high) <= 9:
                chan_num = '1'
                QS_result_txtlist = ['QS_0_result_0.txt','QS_1_result_0.txt']

                process_a = subprocess.Popen(
                    ['python', 'process_region.py', '--QS_result_txt', QS_result_txtlist[0], '--channel_num', chan_num,
                     '--channeled', '0',
                     '--previous_img_path', args.previous_img_path, '--test_img_path', args.test_img_path,
                     '--statistics_guodu_file', args.statistics_guodu_file, '--yuzhi', str(args.yuzhi),
                     '--threshold', str(args.threshold), '--devicenum', '2'])

                process_d = subprocess.Popen(
                    ['python', 'process_region.py', '--QS_result_txt', QS_result_txtlist[1], '--channel_num', chan_num,
                     '--channeled', '0',
                     '--previous_img_path', args.previous_img_path, '--test_img_path', args.test_img_path,
                     '--statistics_guodu_file', args.statistics_guodu_file, '--yuzhi', str(args.yuzhi),
                     '--threshold', str(args.threshold), '--devicenum', '3'])

                # 等待两个脚本完成
                process_a.wait()
                process_d.wait()
            else:
                chan_num = '4'
                QS_result_txtlist = ['QS_0_result_0.txt', 'QS_0_result_1.txt', 'QS_0_result_2.txt', 'QS_0_result_3.txt',
                                     'QS_1_result_0.txt', 'QS_1_result_1.txt', 'QS_1_result_2.txt', 'QS_1_result_3.txt']
                process_a = subprocess.Popen(
                    ['python', 'process_region.py', '--QS_result_txt', QS_result_txtlist[0], '--channel_num', chan_num,
                     '--channeled', '0',
                     '--previous_img_path', args.previous_img_path, '--test_img_path', args.test_img_path,
                     '--statistics_guodu_file', args.statistics_guodu_file, '--yuzhi', str(args.yuzhi),
                     '--threshold', str(args.threshold), '--devicenum', '0'])

                process_b = subprocess.Popen(
                    ['python', 'process_region.py', '--QS_result_txt', QS_result_txtlist[1], '--channel_num', chan_num,
                     '--channeled', '1',
                     '--previous_img_path', args.previous_img_path, '--test_img_path', args.test_img_path,
                     '--statistics_guodu_file', args.statistics_guodu_file, '--yuzhi', str(args.yuzhi),
                     '--threshold', str(args.threshold), '--devicenum', '1'])

                process_c = subprocess.Popen(
                    ['python', 'process_region.py', '--QS_result_txt', QS_result_txtlist[2], '--channel_num', chan_num,
                     '--channeled', '2',
                     '--previous_img_path', args.previous_img_path, '--test_img_path', args.test_img_path,
                     '--statistics_guodu_file', args.statistics_guodu_file, '--yuzhi', str(args.yuzhi),
                     '--threshold', str(args.threshold), '--devicenum', '2'])

                process_d = subprocess.Popen(
                    ['python', 'process_region.py', '--QS_result_txt', QS_result_txtlist[3], '--channel_num', chan_num,
                     '--channeled', '3',
                     '--previous_img_path', args.previous_img_path, '--test_img_path', args.test_img_path,
                     '--statistics_guodu_file', args.statistics_guodu_file, '--yuzhi', str(args.yuzhi),
                     '--threshold', str(args.threshold), '--devicenum', '3'])

                process_e = subprocess.Popen(
                    ['python', 'process_region.py', '--QS_result_txt', QS_result_txtlist[4], '--channel_num', chan_num,
                     '--channeled', '0',
                     '--previous_img_path', args.previous_img_path, '--test_img_path', args.test_img_path,
                     '--statistics_guodu_file', args.statistics_guodu_file, '--yuzhi', str(args.yuzhi),
                     '--threshold', str(args.threshold), '--devicenum', '4'])

                process_f = subprocess.Popen(
                    ['python', 'process_region.py', '--QS_result_txt', QS_result_txtlist[5], '--channel_num', chan_num,
                     '--channeled', '1',
                     '--previous_img_path', args.previous_img_path, '--test_img_path', args.test_img_path,
                     '--statistics_guodu_file', args.statistics_guodu_file, '--yuzhi', str(args.yuzhi),
                     '--threshold', str(args.threshold), '--devicenum', '5'])

                process_g = subprocess.Popen(
                    ['python', 'process_region.py', '--QS_result_txt', QS_result_txtlist[6], '--channel_num', chan_num,
                     '--channeled', '2',
                     '--previous_img_path', args.previous_img_path, '--test_img_path', args.test_img_path,
                     '--statistics_guodu_file', args.statistics_guodu_file, '--yuzhi', str(args.yuzhi),
                     '--threshold', str(args.threshold), '--devicenum', '6'])

                process_h = subprocess.Popen(
                    ['python', 'process_region.py', '--QS_result_txt', QS_result_txtlist[7], '--channel_num', chan_num,
                     '--channeled', '3',
                     '--previous_img_path', args.previous_img_path, '--test_img_path', args.test_img_path,
                     '--statistics_guodu_file', args.statistics_guodu_file, '--yuzhi', str(args.yuzhi),
                     '--threshold', str(args.threshold), '--devicenum', '7'])

                # 等待两个脚本完成
                process_a.wait()
                process_b.wait()
                process_c.wait()
                process_d.wait()
                process_e.wait()
                process_f.wait()
                process_g.wait()
                process_h.wait()

            print('QS result gen finish')
            print('*******************************')

            region_volumes_final = region_volumes_final + region_volumes_low
            for low_num in range(len(region_volumes_low)):
                bounding_boxes_pair.append([])

            region_volumes_final_1 = region_volumes_final_1 + region_volumes_low
            for low_num in range(len(region_volumes_low)):
                bounding_boxes_pair_1.append([])

            for txt_num in range(len(QS_result_txtlist)):
                with open(os.path.join(result_path, QS_result_txtlist[txt_num]), 'r') as f:
                    content = f.read().splitlines()
                if QS_result_txtlist[txt_num].split('_')[1] == '0':
                    for i_0 in range(len(eval(content[0]))):
                        region_volumes_final.append(eval(content[0])[i_0])
                        bounding_boxes_pair.append(eval(content[1])[i_0])
                else:
                    for i_1 in range(len(eval(content[0]))):
                        region_volumes_final_1.append(eval(content[0])[i_1])
                        bounding_boxes_pair_1.append(eval(content[1])[i_1])

        else:
            region_volumes_final = region_volumes_final + region_volumes_low
            for low_num in range(len(region_volumes_low)):
                bounding_boxes_pair.append([])

            region_volumes_final_1 = region_volumes_final_1 + region_volumes_low
            for low_num in range(len(region_volumes_low)):
                bounding_boxes_pair_1.append([])

        region_volumes_final_2 = []
        for i in region_volumes_final:
            region_volumes_final_2.append(i)

        for m in range(len(region_volumes_final)):
            if type(region_volumes_final[m]) != list:
                region_volumes_final[m] = region_volumes_final_1[m]

        for m in region_volumes_final_1:
            if m not in region_volumes_final_2 and type(m) == list:
                region_volumes_final_2.append(m)

        for m in bounding_boxes_pair_1:
            if m not in bounding_boxes_pair:
                bounding_boxes_pair.append(m)
        Lesion_Visualization(path_2d, region_volumes_P, region_volumes, box_label, biaozhun, path_3d_mask_P, region_volumes_final_2,
                             region_volumes_final, bounding_boxes_pair, relevant_label_image, result_path, args.numbering, yuzhi=args.yuzhi)
        print("每个白色立体区域的体积(列表的两个元素分别代表上一次和现在的病灶体积)：", region_volumes_final)
    else:
        # 展示配准前的统计数据
        path_2d = nii_to_image(args.test_img_path, result_path + '/2d_before')
        csv_path = to_csv(path_2d, result_path)
        path_2d_mask = infer(path_2d, result_path + '/2d_mask_before', csv_path)
        path_3d_mask = _2d_to_3d(path_2d_mask, result_path + '/3d_mask_before', image_3d_name_F)
        region_volumes, bounding_boxes, relevant_label_image, box_label = Quantitative_Statistics_1(path_3d_mask,
                                                                                                  result_path,
                                                                                                  yuzhi=args.yuzhi)
        Lesion_Visualization_1(path_2d, region_volumes, relevant_label_image, result_path, args.numbering, yuzhi=args.yuzhi)

    end_time = time.time()
    print('时间:\t', (end_time - begin_time) / 60.0)
