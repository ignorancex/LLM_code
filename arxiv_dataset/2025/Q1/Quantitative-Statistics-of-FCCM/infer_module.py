import os
import cv2
import segmentation_models_pytorch as smp
import time
import torch
from torch.utils.data import DataLoader
import argparse
import Miou
import pandas as pd
from datasets.create_dataset import Mydataset_test
from albumentations.pytorch import ToTensorV2
import albumentations as A


def infer(test_img_path, result_path, csv_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_test_path', type=str,
                        default=test_img_path,
                        help='imgs test data path.')
    parser.add_argument('--labels_test_path', type=str,
                        default='/mnt/ai2022/zrg/Image_segmentation/2D_segmentation/ccm_duo_fa_segmentation/result/statistics_guodu_file/test_mask/',
                        help='labels test data path.')
    parser.add_argument('--csv_dir_test', type=str,
                        default=csv_path,
                        help='labels test data path.')
    parser.add_argument('--pre_save_dir', type=str,
                        default=result_path,
                        help='labels pre save path.')

    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--devicenum', default='0', type=str, )

    parser.add_argument('--model_name', default='UNet_efficientnet-b3', type=str, )
    parser.add_argument('--weight', default='./checkpoint/train/UNet_efficientnet-b3/UNet_efficientnet-b3.pth',
                        type=str, )

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devicenum

    pre_result_path = args.pre_save_dir
    if not os.path.exists(pre_result_path):
        os.makedirs(pre_result_path)

    # begin_time = time.time()

    df_test = pd.read_csv(args.csv_dir_test)  # [0:10]
    test_imgs, test_masks = args.imgs_test_path, args.labels_test_path
    test_imgs = [''.join([test_imgs, '/', i]) for i in df_test['image_name']]
    test_masks = [''.join([test_masks, '/', i]) for i in df_test['image_name']]

    test_transform = A.Compose([
        # A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)

    # test_number = len(test_imgs)
    test_ds = Mydataset_test(test_imgs, test_masks, test_transform)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, pin_memory=False, num_workers=4, )
    # print('==> Preparing data..')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model
    # print('==> Building model..')

    model = smp.Unet(encoder_name='efficientnet-b3', encoder_weights=None, classes=2).to(device)

    state_dict = torch.load(args.weight)
    model.load_state_dict(state_dict)
    model.eval()

    # test_dice, test_miou, test_Pre, test_recall, test_F1score, test_pa = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for batch_idx, (name, inputs, targets) in enumerate(test_dl):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            out = model(inputs)
            predicted = out.argmax(1)

            # test_dice += Miou.calculate_mdice(predicted, targets, 2).item()
            # test_miou += Miou.calculate_miou(predicted, targets, 2).item()
            # test_Pre += Miou.pre(predicted, targets).item()
            # test_recall += Miou.recall(predicted, targets).item()
            # test_F1score += Miou.F1score(predicted, targets).item()
            # test_pa += Miou.Pa(predicted, targets).item()

            predict = predicted.squeeze(0)
            mask_np = predict.cpu().numpy()
            mask_np = (mask_np * 255).astype('uint8')
            mask_np[mask_np > 160] = 255
            mask_np[mask_np <= 160] = 0

            cv2.imwrite(pre_result_path + '/' + name[0], mask_np)

    # average_test_dice = test_dice / test_number
    # average_test_miou = test_miou / test_number
    # average_test_Pre = test_Pre / test_number
    # average_test_recall = test_recall / test_number
    # average_test_F1score = test_F1score / test_number
    # average_test_pa = test_pa / test_number
    #
    # dice, miou, pre, recall, f1_score, pa = \
    #     '%.4f' % average_test_dice, '%.4f' % average_test_miou, '%.4f' % average_test_Pre, '%.4f' % average_test_recall, '%.4f' % average_test_F1score, '%.4f' % average_test_pa

    # end_time = time.time()

    # print("时间")
    # print(end_time - begin_time)
    # print("dice:" + '\t' + str(dice))
    # print("miou:" + '\t' + str(miou))
    # print("pre:" + '\t' + str(pre))
    # print("recall:" + '\t' + str(recall))
    # print("f1_score:" + '\t' + str(f1_score))
    # print("pa:" + '\t' + str(pa))

    return pre_result_path

