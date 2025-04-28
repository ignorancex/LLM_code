import os
import ast
import argparse
import multiprocessing
from focus_statistics_3D import set_specific_pixels, registration, get_specific_pixel_coordinates, nii_to_image
from focus_statistics_3D import to_csv, infer, _2d_to_3d
from Quantitative_Statistics_Module import Quantitative_Statistics_pixel_1


def process_region(args, num, bounding_box, region_volumes, region_volumes_P, bounding_boxes_P, biaozhun, image_3d_name_F, result_path):
    if region_volumes[num] > args.threshold:
        # print(args.channeled, region_volumes[num])
        specific_pixel_img_path = set_specific_pixels(args.test_img_path, bounding_box,
                                                      result_path + '/specific_pixel_3d_' + str(num))
        peizhun_after_path = registration(biaozhun, specific_pixel_img_path, result_path + '/peizhun_' + str(num))
        special_pixel_coordinate_after = get_specific_pixel_coordinates(peizhun_after_path)
        path_2d_after = nii_to_image(peizhun_after_path, result_path + '/2d_after_' + str(num))
        csv_path_after = to_csv(path_2d_after, result_path + '/' + 'csv_' + str(num))
        path_2d_mask_after = infer(path_2d_after, result_path + '/2d_mask_after_' + str(num), csv_path_after)
        path_3d_mask_after = _2d_to_3d(path_2d_mask_after, result_path + '/3d_mask_after_' + str(num), image_3d_name_F)
        ddd, bounding_box_pair = Quantitative_Statistics_pixel_1(path_3d_mask_after, special_pixel_coordinate_after,
                                            bounding_box, region_volumes[num], region_volumes_P, bounding_boxes_P, yuzhi=args.yuzhi)
        return ddd, bounding_box_pair
    else:
        # print(args.channeled, region_volumes[num])
        bounding_box_pair = []
        return region_volumes[num], bounding_box_pair


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--QS_result_txt', type=str,
                        default='QS_result.txt',
                        help='Volumns Boxes data path.')
    parser.add_argument('--channel_num', type=int, default='4', help='channel num.')
    parser.add_argument('--channeled', type=int, default='1', help='channeled.')
    parser.add_argument('--multiprocess_num', type=int, default=9, help='multiprocess_num.')
    parser.add_argument('--previous_img_path', type=str,
                            default='/mnt/zrg/dataset/yingxiang/ccm_duo_fa_nii/ccm_duo_fa_nii_3d/XIE_YU_SONG-F36_01_XYSO-2022.06.14.nii',
                            help='previous img data path.')
    parser.add_argument('--test_img_path', type=str,
                            default='/mnt/zrg/dataset/yingxiang/ccm_duo_fa_nii/ccm_duo_fa_nii_3d/XIE_YU_SONG-F36_01_XYSO-2022.09.16.nii',
                            help='follow-up img data path.')
    parser.add_argument('--statistics_guodu_file', type=str,
                        default='/mnt/zrg/Image_segmentation/2D_segmentation/FCCM_TBME_1/result/QS/F36_01_XYSO_new',
                        help='statistics guodu file.')
    parser.add_argument('--yuzhi', default='10', type=int, help='Threshold for statistical volume')
    parser.add_argument('--threshold', default='200', type=int, help='Volume threshold for matching lesions')
    parser.add_argument('--devicenum', default='0', type=str, )

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devicenum

    biaozhun = args.previous_img_path
    result_path = args.statistics_guodu_file + '/' + args.QS_result_txt.split('_')[1] + '_' + str(args.channeled)
    image_3d_name_F = args.test_img_path.split('/')[-1]

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    print('*******************************')
    print(args.QS_result_txt)
    print('*******************************')

    with open(os.path.join(args.statistics_guodu_file, 'output.txt'), 'r') as f:
        content = f.read().splitlines()

    region_volumes_P = eval(content[0])
    bounding_boxes_P = eval(content[1])
    region_volumes = eval(content[2])
    bounding_boxes = eval(content[3])
    box_label = ast.literal_eval(content[4])

    if len(bounding_boxes) < args.channel_num:
        num_every = 1
    else:
        num_every = int(len(bounding_boxes) / args.channel_num)
    channeled = args.channeled
    if channeled == args.channel_num - 1:
        bounding_boxes_every = bounding_boxes[channeled * num_every:len(bounding_boxes)]
    elif ((channeled) + 1) * num_every <= len(bounding_boxes):
        bounding_boxes_every = bounding_boxes[channeled * num_every:(channeled + 1) * num_every]
    else:
        bounding_boxes_every = bounding_boxes[channeled * num_every:len(bounding_boxes)]

    # 设置启动方法为'spawn'
    multiprocessing.set_start_method("spawn", force=True)

    # 创建进程池，可以指定线程数量
    if 0 < len(bounding_boxes_every) < args.multiprocess_num:
        multiprocess_num = len(bounding_boxes_every)
    else:
        multiprocess_num = args.multiprocess_num

    pool = multiprocessing.Pool(multiprocess_num)
    # 使用map函数并行运行任务
    results = pool.starmap(process_region,
                           [(args, channeled * num_every + num, bounding_box, region_volumes, region_volumes_P, bounding_boxes_P, biaozhun,
                             image_3d_name_F, result_path) for
                            num, bounding_box in enumerate(bounding_boxes_every)])
    # 关闭进程池，等待所有任务完成
    pool.close()

    region_volumes_final, bounding_boxes_pair = [], []
    for z in results:
        region_volumes_final.append(z[0])
        bounding_boxes_pair.append(z[1])

    with open(os.path.join(args.statistics_guodu_file, args.QS_result_txt), 'w') as file:
        file.writelines(str(region_volumes_final))
        file.write("\n")
        file.writelines(str(bounding_boxes_pair))

    print('*******************************')
    print('结束:', args.QS_result_txt)
    print('*******************************')

