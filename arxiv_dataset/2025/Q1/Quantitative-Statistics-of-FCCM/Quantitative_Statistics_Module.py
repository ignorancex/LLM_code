import os
import time

import ants
import json
import argparse
import cv2
import numpy as np
from skimage.measure import label, regionprops
import nibabel as nib  # 用nibabel包打开nii文件


def save_bounding_coordinates(images_path, filename, filename_1):
    images = os.listdir(images_path)
    images.sort(key=lambda x: (int(x.split('-')[1].split('_')[0].split('F')[1]), int(x.split('-')[1].split('_')[1]),
                               int(x.split('-')[2].split('.')[0]), int(x.split('-')[2].split('.')[1]),
                               int(x.split('-')[2].split('.')[2])))

    # 存储每个白色像素区域的最小外接立方体坐标
    bounding_boxes_1, bounding_boxes_2, bounding_boxes_3, bounding_boxes_4 = {}, [], {}, {}
    lesion_volume_1, lesion_volume_2, lesion_volume_3, lesion_volume_4 = {}, [], {}, {}
    name_0, name_1, name_2 = '0', '0', '0'
    for num, i in enumerate(images):
        image_path = os.path.join(images_path, i)
        # 加载3D NIfTI图像
        image_3d = nib.load(image_path)

        # 获取图像数据
        image_data = image_3d.get_fdata()

        # 阈值化图像，将非白色像素设置为0，白色像素设置为1
        thresholded_image = np.where(image_data > 0, 1, 0)

        # 使用连通组件分析获取连通区域的标签
        labeled_image, num_labels = label(thresholded_image, connectivity=1, return_num=True)

        # 获取区域的属性
        regions = regionprops(labeled_image)

        # 计算每个白色立体区域的体积
        region_volumes = [int(region.area) for region in regions]

        # 创建一个3D图像，与原始图像具有相同的形状和数据类型
        copy_image = np.zeros(image_data.shape, dtype='uint8')

        each_ball_label_color = {}

        # 存储每个白色像素区域的最小外接立方体坐标
        bounding_boxes = []

        name0 = i.split('-')[0]
        name1 = i.split('-')[1]
        name2 = i.split('-')[2].split('.')[0] + '.' + i.split('-')[2].split('.')[1] + '.' + \
                i.split('-')[2].split('.')[
                    2]

        # 给每个白色立体区域编号
        for region in regions:
            # 获取区域的标签值
            label_value = region.label

            while True:
                # 为当前区域生成随机颜色
                color = np.random.randint(0, 255, size=3).tolist()
                if color not in list(each_ball_label_color.values()):
                    each_ball_label_color[label_value] = color  # 保存区域对应的标签值颜色字典
                    break

            # 将当前区域的像素值设置为对应的编号
            copy_image[labeled_image == label_value] = label_value

            # 获取最小外接立方体坐标
            minr, minc, minz, maxr, maxc, maxz = region.bbox
            bounding_boxes.append([[minr, minc, minz], [maxr, maxc, maxz]])

        # 输出白色像素区域的最小外接立方体坐标
        for n, bbox in enumerate(bounding_boxes):
            print(f"Region {n + 1} Bounding Box: {bbox}")

        bounding_boxes_1[name2] = bounding_boxes

        if name1 != name_1:  # 判断name1是否变化
            if len(bounding_boxes_2) != 0:
                bounding_boxes_3[name_1] = bounding_boxes_2
                bounding_boxes_2 = []

                if name_0 in list(bounding_boxes_4.keys()):
                    guodu = []
                    for j in range(len(bounding_boxes_4[name_0])):
                        guodu.append(bounding_boxes_4[name_0][j])
                    guodu.append(bounding_boxes_3)
                    bounding_boxes_4[name_0] = guodu
                else:
                    guodu = []
                    guodu.append(bounding_boxes_3)
                    bounding_boxes_4[name_0] = guodu
                bounding_boxes_3 = {}

        if num == len(images) - 1:  # 遍历到最后一张图片
            if name1 != name_1:  # 判断name1是否变化
                if len(bounding_boxes_2) == 0:
                    bounding_boxes_2.append(bounding_boxes_1)
                    bounding_boxes_3[name1] = bounding_boxes_2
                    bounding_boxes_2 = []

                    if name0 in list(bounding_boxes_4.keys()):
                        guodu = []
                        for j in range(len(bounding_boxes_4[name0])):
                            guodu.append(bounding_boxes_4[name0][j])
                        guodu.append(bounding_boxes_3)
                        bounding_boxes_4[name0] = guodu
                    else:
                        guodu = []
                        guodu.append(bounding_boxes_3)
                        bounding_boxes_4[name0] = guodu
                    bounding_boxes_3 = {}

            else:
                if len(bounding_boxes_2) != 0:
                    bounding_boxes_2.append(bounding_boxes_1)
                    bounding_boxes_3[name1] = bounding_boxes_2
                    bounding_boxes_2 = []

                    if name0 in list(bounding_boxes_4.keys()):
                        guodu = []
                        for j in range(len(bounding_boxes_4[name0])):
                            guodu.append(bounding_boxes_4[name0][j])
                        guodu.append(bounding_boxes_3)
                        bounding_boxes_4[name0] = guodu
                    else:
                        guodu = []
                        guodu.append(bounding_boxes_3)
                        bounding_boxes_4[name0] = guodu
                    bounding_boxes_3 = {}

        bounding_boxes_2.append(bounding_boxes_1)  # 依次保存同一病人3D图片的连通组件的最小外接立方体坐标
        bounding_boxes_1 = {}

        # 以和病灶最小外接立方体坐标相同的格式和顺序存储各个病灶的体积
        lesion_volume_1[name2] = region_volumes

        if name1 != name_1:  # 判断name1是否变化
            if len(lesion_volume_2) != 0:
                lesion_volume_3[name_1] = lesion_volume_2
                lesion_volume_2 = []

                if name_0 in list(lesion_volume_4.keys()):
                    guodu = []
                    for j in range(len(lesion_volume_4[name_0])):
                        guodu.append(lesion_volume_4[name_0][j])
                    guodu.append(lesion_volume_3)
                    lesion_volume_4[name_0] = guodu
                else:
                    guodu = []
                    guodu.append(lesion_volume_3)
                    lesion_volume_4[name_0] = guodu
                lesion_volume_3 = {}

        if num == len(images) - 1:  # 遍历到最后一张图片
            if name1 != name_1:  # 判断name1是否变化
                if len(lesion_volume_2) == 0:
                    lesion_volume_2.append(lesion_volume_1)
                    lesion_volume_3[name1] = lesion_volume_2
                    lesion_volume_2 = []

                    if name0 in list(lesion_volume_4.keys()):
                        guodu = []
                        for j in range(len(lesion_volume_4[name0])):
                            guodu.append(lesion_volume_4[name0][j])
                        guodu.append(lesion_volume_3)
                        lesion_volume_4[name0] = guodu
                    else:
                        guodu = []
                        guodu.append(lesion_volume_3)
                        lesion_volume_4[name0] = guodu
                    lesion_volume_3 = {}

            else:
                if len(lesion_volume_2) != 0:
                    lesion_volume_2.append(lesion_volume_1)
                    lesion_volume_3[name1] = lesion_volume_2
                    lesion_volume_2 = []

                    if name0 in list(lesion_volume_4.keys()):
                        guodu = []
                        for j in range(len(lesion_volume_4[name0])):
                            guodu.append(lesion_volume_4[name0][j])
                        guodu.append(lesion_volume_3)
                        lesion_volume_4[name0] = guodu
                    else:
                        guodu = []
                        guodu.append(lesion_volume_3)
                        lesion_volume_4[name0] = guodu
                    lesion_volume_3 = {}

        lesion_volume_2.append(lesion_volume_1)  # 依次保存同一病人3D图片的连通组件的最小外接立方体坐标
        lesion_volume_1 = {}

        name_0 = i.split('-')[0]
        name_1 = i.split('-')[1]
        name_2 = i.split('-')[2].split('.')[0] + '.' + i.split('-')[2].split('.')[1] + '.' + \
                 i.split('-')[2].split('.')[
                     2]

    # 保存连通组件的最小外接矩形坐标字典列表到JSON文件
    with open(filename, 'w') as file:
        json.dump(bounding_boxes_4, file)

    # 保存病灶体积字典列表到JSON文件
    with open(filename_1, 'w') as file_1:
        json.dump(lesion_volume_4, file_1)


# 计算两个立方体重合部分体积和占比
def calculate_overlap_volume(cube1, cube2):
    # 提取两个立方体的坐标信息
    min1, max1 = cube1
    min2, max2 = cube2

    # 判断两个立方体是否重合
    if (max1[0] < min2[0] or min1[0] > max2[0] or
            max1[1] < min2[1] or min1[1] > max2[1] or
            max1[2] < min2[2] or min1[2] > max2[2]):
        return 0, 0  # 不重合，重合部分体积为0，占比为0

    # 计算重合部分的体积
    overlap_x = min(max1[0], max2[0]) - max(min1[0], min2[0])
    overlap_y = min(max1[1], max2[1]) - max(min1[1], min2[1])
    overlap_z = min(max1[2], max2[2]) - max(min1[2], min2[2])

    volume = overlap_x * overlap_y * overlap_z

    # 计算占比
    volume1 = (max1[0] - min1[0]) * (max1[1] - min1[1]) * (max1[2] - min1[2])
    volume2 = (max2[0] - min2[0]) * (max2[1] - min2[1]) * (max2[2] - min2[2])

    if volume1 == 0 or volume2 == 0:
        percentage = 0  # 避免分母为0的情况
    else:
        percentage = (volume / max(volume1, volume2)) * 100

    return volume, percentage


def Quantitative_Statistics(test_img_path, result_path, save, yuzhi):

    # 加载3D NIfTI图像
    image_3d = nib.load(test_img_path)

    # 获取图像数据
    image_data = image_3d.get_fdata()

    # 计算总像素点
    pixel_sum = image_data.shape[0] * image_data.shape[1] * image_data.shape[2]

    # 阈值化图像，将非白色像素设置为0，白色像素设置为1
    thresholded_image = np.where(image_data > 0, 1, 0)

    # 使用连通组件分析获取连通区域的标签
    labeled_image, num_labels = label(thresholded_image, connectivity=1, return_num=True)

    # 获取区域的属性
    regions = regionprops(labeled_image)

    # 计算每个白色立体区域的体积
    region_volumes = [int(region.area) for region in regions if region.area > yuzhi]
    label_ori = [int(region.label) for region in regions if region.area > yuzhi]

    combined = list(zip(region_volumes, label_ori))
    combined.sort(key=lambda x: x[0])
    region_volumes_sorted, label_ori_sorted = zip(*combined)
    label_ori_sorted = list(label_ori_sorted)

    # 创建一个3D图像，与原始图像具有相同的形状和数据类型
    copy_image = np.zeros(image_data.shape, dtype='uint16')

    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX  # 字体样式

    # 存储每个白色像素区域的最小外接立方体坐标
    bounding_boxes = []
    box_label = {}

    # 给每个白色立体区域编号
    for region in regions:
        if region.area > yuzhi:
            # 获取区域的标签值
            label_value = region.label

            # 将当前区域的像素值设置为对应的编号
            # ii = label_ori_sorted.index(label_value) + 1
            copy_image[labeled_image == label_value] = label_ori_sorted.index(label_value) + 1

            # 获取最小外接立方体坐标
            minr, minc, minz, maxr, maxc, maxz = region.bbox
            bounding_boxes.append([[minr, minc, minz], [maxr, maxc, maxz]])
            box_label[str([[minr, minc, minz], [maxr, maxc, maxz]])] = label_ori_sorted.index(label_value) + 1

    combined = list(zip(region_volumes, bounding_boxes))
    combined.sort(key=lambda x: x[0])
    region_volumes_sorted, bounding_boxes_sorted = zip(*combined)
    region_volumes_sorted = list(region_volumes_sorted)
    bounding_boxes_sorted = list(bounding_boxes_sorted)

    if save:
        copy_image_3d = nib.Nifti1Image(copy_image, affine=np.eye(4))
        nib.save(copy_image_3d, result_path + '/' + 'relevant_label_image' + '.nii')


    print("像素点总数：", pixel_sum)
    print("白色像素点的个数：", np.sum(region_volumes_sorted))
    print("白色立体区域的个数：", len(region_volumes_sorted))
    print("每个白色立体区域的体积：", region_volumes_sorted)

    return region_volumes_sorted, bounding_boxes_sorted, copy_image, box_label


def Quantitative_Statistics_pixel(test_img_path, special_pixel_coordinate_after, num, bounding_boxes_ori, region_volumes_ori, region_volumes_P, bounding_boxes_P, yuzhi):

    # 加载3D NIfTI图像
    image_3d = nib.load(test_img_path)

    # 获取图像数据
    image_data = image_3d.get_fdata()

    # 阈值化图像，将非白色像素设置为0，白色像素设置为1
    thresholded_image = np.where(image_data > 0, 1, 0)

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

    image_3d_name = test_img_path.split('/')[-1]

    time.sleep(3)
    before_1, now_1, coordinate, coordinate_1 = [], [], [], []
    a, before, now, bounding_box_previous = 0, 0, 0, 0
    b, c, bounding_boxes_previous = [], [], []
    d = False

    for h in range(len(bounding_boxes)):
        for j in range(len(bounding_boxes_P)):
            overlap_volume, overlap_percentage = calculate_overlap_volume(bounding_boxes[h], bounding_boxes_P[j])
            if (overlap_percentage > 0 and overlap_percentage >= a):
                before = region_volumes_P[j]
                now = region_volumes[h]
                coordinate = bounding_boxes[h]
                a = overlap_percentage
                bounding_box_previous = bounding_boxes_P[j]
                d = True
        if d == True:
            b.append(before)
            b.append(now)
            c.append(b)
            coordinate_1.append(coordinate)
            bounding_boxes_previous.append(bounding_box_previous)
            a = 0
            b = []
            d = False

    time.sleep(3)
    region_volumes_1 = []
    for l in range(len(region_volumes)):
        if bounding_boxes[l] in coordinate_1:
            region_volumes_1.append(c[coordinate_1.index(bounding_boxes[l])])
        else:
            region_volumes_1.append(region_volumes[l])

    volumes, bounding_box_pairs = [], []
    if (num + 1) * 6 <= len(bounding_boxes_ori):
        nums = range(num * 6, (num + 1) * 6)
    else:
        nums = range(num * 6, len(bounding_boxes_ori))

    for special_pixel_num in range(len(special_pixel_coordinate_after)):
        x, y, z = special_pixel_coordinate_after[special_pixel_num]
        ddd = 0
        bounding_box_pair = []
        # 遍历 coordinate_1 列表
        for index, box in enumerate(coordinate_1):
            min_corner = box[0]  # 矩形框的最小角坐标 (minr, minc, minz)
            max_corner = box[1]  # 矩形框的最大角坐标 (maxr, maxc, maxz)

            # 检查坐标是否在矩形框的范围内
            if (min_corner[0] <= x <= max_corner[0] and
                    min_corner[1] <= y <= max_corner[1] and
                    min_corner[2] <= z <= max_corner[2]):
                c[index][1] = region_volumes_ori[nums[special_pixel_num]]
                ddd = c[index]
                bounding_box_pair.append(bounding_boxes_previous[index])
                bounding_box_pair.append(bounding_boxes_ori[nums[special_pixel_num]])
                volumes.append(ddd)
                bounding_box_pairs.append(bounding_box_pair)
                break
        if ddd == 0:
            volumes.append(region_volumes_ori[nums[special_pixel_num]])
            bounding_box_pairs.append(bounding_box_pair)
    return volumes, bounding_box_pairs


def Quantitative_Statistics_1(test_img_path, result_path, yuzhi):

    # 加载3D NIfTI图像
    image_3d = nib.load(test_img_path)

    # 获取图像数据
    image_data = image_3d.get_fdata()

    # 计算总像素点
    pixel_sum = image_data.shape[0] * image_data.shape[1] * image_data.shape[2]

    # 阈值化图像，将非白色像素设置为0，白色像素设置为1
    thresholded_image = np.where(image_data > 0, 1, 0)

    # 使用连通组件分析获取连通区域的标签
    labeled_image, num_labels = label(thresholded_image, connectivity=1, return_num=True)

    # 获取区域的属性
    regions = regionprops(labeled_image)

    # 计算每个白色立体区域的体积
    region_volumes = [int(region.area) for region in regions if region.area > yuzhi]

    # 创建一个3D图像，与原始图像具有相同的形状和数据类型
    copy_image = np.zeros(image_data.shape, dtype='uint16')

    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX  # 字体样式

    # 存储每个白色像素区域的最小外接立方体坐标
    bounding_boxes = []
    box_label = {}
    label_num = 1

    # 给每个白色立体区域编号
    for region in regions:
        if region.area > yuzhi:
            # 获取区域的标签值
            label_value = region.label

            # 将当前区域的像素值设置为对应的编号
            copy_image[labeled_image == label_value] = label_num

            # 获取最小外接立方体坐标
            minr, minc, minz, maxr, maxc, maxz = region.bbox
            bounding_boxes.append([[minr, minc, minz], [maxr, maxc, maxz]])
            box_label[str([[minr, minc, minz], [maxr, maxc, maxz]])] = label_num

            label_num = label_num + 1

    print("像素点总数：", pixel_sum)
    print("白色像素点的个数：", np.sum(region_volumes))
    print("白色立体区域的个数：", len(region_volumes))
    print("每个白色立体区域的体积：", region_volumes)

    return region_volumes, bounding_boxes, copy_image, box_label


def Quantitative_Statistics_pixel_1(test_img_path, special_pixel_coordinate_after, bounding_box, region_volume, region_volumes_P, bounding_boxes_P, yuzhi):

    # 加载3D NIfTI图像
    image_3d = nib.load(test_img_path)

    # 获取图像数据
    image_data = image_3d.get_fdata()

    # 阈值化图像，将非白色像素设置为0，白色像素设置为1
    thresholded_image = np.where(image_data > 0, 1, 0)

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

    image_3d_name = test_img_path.split('/')[-1]

    time.sleep(3)
    before_1, now_1, coordinate, coordinate_1 = [], [], [], []
    a, before, now, bounding_box_previous = 0, 0, 0, 0
    b, c, bounding_boxes_previous = [], [], []
    d = False

    for h in range(len(bounding_boxes)):
        for j in range(len(bounding_boxes_P)):
            overlap_volume, overlap_percentage = calculate_overlap_volume(bounding_boxes[h], bounding_boxes_P[j])
            if (overlap_percentage > 0 and overlap_percentage >= a):
                before = region_volumes_P[j]
                now = region_volumes[h]
                coordinate = bounding_boxes[h]
                a = overlap_percentage
                bounding_box_previous = bounding_boxes_P[j]
                d = True
        if d == True:
            b.append(before)
            b.append(now)
            c.append(b)
            coordinate_1.append(coordinate)
            bounding_boxes_previous.append(bounding_box_previous)
            a = 0
            b = []
            d = False

    time.sleep(3)
    region_volumes_1 = []
    for l in range(len(region_volumes)):
        if bounding_boxes[l] in coordinate_1:
            region_volumes_1.append(c[coordinate_1.index(bounding_boxes[l])])
        else:
            region_volumes_1.append(region_volumes[l])

    x, y, z = special_pixel_coordinate_after
    ddd = 0
    bounding_box_pair = []
    # 遍历 coordinate_1 列表
    for index, box in enumerate(coordinate_1):
        min_corner = box[0]  # 矩形框的最小角坐标 (minr, minc, minz)
        max_corner = box[1]  # 矩形框的最大角坐标 (maxr, maxc, maxz)

        # 检查坐标是否在矩形框的范围内
        if (min_corner[0] <= x <= max_corner[0] and
                min_corner[1] <= y <= max_corner[1] and
                min_corner[2] <= z <= max_corner[2]):
            c[index][1] = region_volume
            ddd = c[index]
            bounding_box_pair.append(bounding_boxes_previous[index])
            bounding_box_pair.append(bounding_box)
            break
    if ddd == 0:
        ddd = region_volume
    return ddd, bounding_box_pair
