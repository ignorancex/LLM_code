import argparse
import sys
import os
import cv2
import glob
import math
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader, default_collate
from torchvision.utils import save_image
import numpy as np
from random import random
from torchvision import transforms
import imutils

from warp_utils import (
    RGBDRenderer,
    image_to_tensor,
    disparity_to_tensor,
    transformation_from_parameters,
)


def resize_and_center_crop(image, disparity):
    # 获取图像和视差图的尺寸
    h, w = image.shape[:2]

    # 计算最短边的尺寸
    shortest_edge = min(h, w)

    # 按最短边缩放
    if h < w:
        new_h = shortest_edge
        new_w = int(shortest_edge * (w / h))
    else:
        new_w = shortest_edge
        new_h = int(shortest_edge * (h / w))

    # 缩放图像
    image_resized = cv2.resize(image, (new_w, new_h))
    disparity_resized = cv2.resize(disparity, (new_w, new_h))

    # 计算裁剪区域，使得图像变为正方形
    crop_size = min(image_resized.shape[:2])  # 取缩放后图像的最短边作为裁剪大小
    start_x = (new_w - crop_size) // 2
    start_y = (new_h - crop_size) // 2

    # 裁剪图像和视差图
    image_cropped = image_resized[
        start_y : start_y + crop_size, start_x : start_x + crop_size
    ]
    disparity_cropped = disparity_resized[
        start_y : start_y + crop_size, start_x : start_x + crop_size
    ]

    return image_cropped, disparity_cropped


class WarpBackStage1Dataset(Dataset):
    def __init__(
        self,
        data_root,
        disp_root,
        width=512,
        height=512,
        device="cuda",  # device of mesh renderer
        trans_range={"x": 0.4, "y": 0.4, "z": 0.8, "a": 18, "b": 18, "c": 18},
        # trans_range={"x": -1, "y": -1, "z": -1, "a": -1, "b": -1, "c": -1},
    ):
        self.data_root = data_root
        self.disp_root = disp_root

        self.renderer = RGBDRenderer(device)
        self.width = width
        self.height = height
        self.device = device
        self.trans_range = trans_range
        self.image_path_list = [
            os.path.join(data_root, img) for img in os.listdir(data_root)
        ]
        self.img2tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_path_list)

    def rand_tensor(self, r, l):
        if (
            r < 0
        ):  # we can set a negtive value in self.trans_range to avoid random transformation
            return torch.zeros((l, 1, 1))
        rand = torch.rand((l, 1, 1))
        sign = 2 * (torch.randn_like(rand) > 0).float() - 1
        return sign * (r / 2 + r / 2 * rand)

    def get_rand_ext(self, bs):
        x, y, z = self.trans_range["x"], self.trans_range["y"], self.trans_range["z"]
        a, b, c = self.trans_range["a"], self.trans_range["b"], self.trans_range["c"]
        cix = self.rand_tensor(x, bs)
        ciy = self.rand_tensor(y, bs)
        ciz = self.rand_tensor(z, bs)

        aix = self.rand_tensor(math.pi / a, bs)
        aiy = self.rand_tensor(math.pi / b, bs)
        aiz = self.rand_tensor(math.pi / c, bs)

        axisangle = torch.cat([aix, aiy, aiz], dim=-1)  # [b,1,3]
        translation = torch.cat([cix, ciy, ciz], dim=-1)

        cam_ext = transformation_from_parameters(axisangle, translation)  # [b,4,4]
        cam_ext_inv = torch.inverse(cam_ext)  # [b,4,4]

        print(axisangle, translation)

        return cam_ext[:, :-1], cam_ext_inv[:, :-1]

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        disp_path = os.path.join(self.disp_root, "%s.npy" % image_name)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, ::-1]  # [3,h,w]
        disp = np.load(disp_path)  # [1,h,w]

        H, W = image.shape[:2]

        if H > W:
            image = imutils.resize(image, width=512)
            disp = imutils.resize(disp, width=512)
        else:
            image = imutils.resize(image, height=512)
            disp = imutils.resize(disp, height=512)

        image, disp = resize_and_center_crop(image, disp)

        max_d, min_d = disp.max(), disp.min()
        disp = (disp - min_d) / (max_d - min_d)

        image = torch.tensor(image).permute(2, 0, 1) / 255
        disp = torch.tensor(disp).unsqueeze(0) + 0.001

        self.focal = 0.45 + np.random.random() * 0.3
        # set intrinsics
        self.K = torch.tensor(
            [[self.focal, 0, 0.5], [0, self.focal, 0.5], [0, 0, 1]]
        ).to(self.device)

        image = image.to(self.device).unsqueeze(0).float()
        disp = disp.to(self.device).unsqueeze(0).float()
        rgbd = torch.cat([image, disp], dim=1)  # [b,4,h,w]
        b = image.shape[0]

        cam_int = self.K.repeat(b, 1, 1)  # [b,3,3]

        # warp to a random novel view
        mesh = self.renderer.construct_mesh(
            rgbd, cam_int, torch.ones_like(disp), normalize_depth=True
        )
        cam_ext, cam_ext_inv = self.get_rand_ext(b)  # [b,3,4]
        cam_ext = cam_ext.to(self.device)
        cam_ext_inv = cam_ext_inv.to(self.device)

        warp_image, warp_disp, warp_mask, object_mask = self.renderer.render_mesh(
            mesh, cam_int, cam_ext
        )
        warp_mask = (warp_mask < 0.5).float()

        warp_image = torch.clip(warp_image, 0, 1)
        
        cam_int[0, :2, :] *= 512

        return {
            "rgb": image,
            "disp": disp,
            "warp_mask": warp_mask,
            "warp_rgb": warp_image,
            "warp_disp": warp_disp,
            "image_name": image_name,
            "cam_int": cam_int[0],
            "cam_ext": cam_ext[0],
        }


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    from diffusers import StableDiffusionInpaintPipeline
    import torch
    import torchvision
    import matplotlib.pyplot as plt

    def project_point_to_3d(x, y, depth, K):
        """根据相机内参和深度值，计算像素在相机坐标系中的 3D 坐标"""
        inv_K = torch.linalg.inv(K)
        pixel = torch.tensor([x, y, 1.0]).to(K.device)
        normalized_coords = inv_K @ pixel * depth
        return normalized_coords

    def transform_to_another_camera(point_3d, T):
        """根据变换矩阵将 3D 点从相机 1 转到相机 2"""
        point_3d_homogeneous = torch.cat([point_3d, torch.tensor([1.0]).to(T.device)])
        transformed_point = T @ point_3d_homogeneous
        return transformed_point[:3]

    def project_to_image_plane(point_3d, K):
        """将 3D 点投影到图像平面"""
        point_2d_homogeneous = K @ point_3d
        point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
        return point_2d

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/davis/raw_images",
    )
    parser.add_argument(
        "--disp_path",
        type=str,
        default="data/davis/disps",
    )
    parser.add_argument("--output_path", type=str, default="data/mixed_datasets/l2m_davis")
    opt, _ = parser.parse_known_args()

    # 指定模型文件路径
    model_path = "stabilityai/stable-diffusion-2-inpainting"

    # 加载模型
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16
    )
    pipe.to("cuda")  # 如果有 GPU，可以将模型加载到 GPU 上
    prompt = "a realistic photo"

    setup_seed(0)

    output = opt.output_path
    from tqdm import tqdm

    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
        os.mkdir(os.path.join(output, "image1"))
        os.mkdir(os.path.join(output, "image2"))
        os.mkdir(os.path.join(output, "depth1"))
        os.mkdir(os.path.join(output, "depth2"))
        os.mkdir(os.path.join(output, "cams"))
        os.mkdir(os.path.join(output, "ext"))
        os.mkdir(os.path.join(output, "debug"))

    data = WarpBackStage1Dataset(data_root=opt.data_path, disp_root=opt.disp_path)

    for loop in range(2):
        for idx in tqdm(range(len(data))):

            batch = data.__getitem__(idx)

            image, disp = batch["rgb"], batch["disp"]
            w_image, w_disp = batch["warp_rgb"], batch["warp_disp"]
            warp_mask = batch["warp_mask"]

            w_disp = torch.clip(w_disp, 0.01, 100)

            init_image = torchvision.transforms.functional.to_pil_image(w_image[0])
            mask_image = torchvision.transforms.functional.to_pil_image(warp_mask[0])
            image = torchvision.transforms.functional.to_pil_image(image[0])

            W, H = init_image.size

            inpaint_image = pipe(
                prompt=prompt, image=init_image, mask_image=mask_image, h=512, w=512
            ).images[0]

            image.save(os.path.join(output, "image1", batch["image_name"] + ".png"))
            inpaint_image.save(
                os.path.join(output, "image2", batch["image_name"] + ".png")
            )

            np.save(
                os.path.join(output, "depth1", batch["image_name"] + ".npy"),
                1 / disp.squeeze().cpu().numpy(),
            )
            np.save(
                os.path.join(output, "depth2", batch["image_name"] + ".png"),
                1 / w_disp.squeeze().cpu().numpy(),
            )
            
            w_depth = 1 / (w_disp + 1e-4) * (1 - warp_mask)

            cam_int = batch["cam_int"].cpu().numpy()
            cam_ext = batch["cam_ext"].cpu().numpy()

            cam_ext = np.concatenate(
                [cam_ext, np.array([[0.0000, 0.0000, 0.0000, 1.0000]])], 0
            )

            np.save(os.path.join(output, "cams", batch["image_name"] + ".npy"), cam_int)
            np.save(os.path.join(output, "ext", batch["image_name"] + ".png"), cam_ext)
            
            # 可视化前20个生成的数据
            if idx > 19:
                continue

            im_A_depth = 1 / disp.squeeze()
            im_B_depth = 1 / w_disp.squeeze()

            K1 = batch["cam_int"].float()
            K2 = batch["cam_int"].float()
            T_1to2 = torch.tensor(cam_ext).cuda().float()

            im_A_cv = np.array(image)
            im_B_cv = np.array(inpaint_image)

            # 拼接图像用于显示
            im_combined = np.hstack(
                (im_A_cv.astype(np.uint8), im_B_cv.astype(np.uint8))
            )

            # 选择一部分像素来计算映射关系
            matches_A = []
            matches_B = []

            # 选择具有非零深度的像素点
            for y in range(0, im_A_depth.shape[0], 10):  # 步长为10，减少计算量
                for x in range(0, im_A_depth.shape[1], 10):
                    depth_A = im_A_depth[y, x].item()
                    if depth_A > 0:  # 只处理有深度信息的像素
                        # 计算相机1中的3D坐标
                        point_3d_A = project_point_to_3d(x, y, depth_A, K1)
                        # 转换到相机2坐标系
                        point_3d_B = transform_to_another_camera(
                            point_3d_A.float(), T_1to2
                        )
                        # 投影到相机2的图像平面
                        point_2d_B = project_to_image_plane(point_3d_B, K2)

                        # 将2D匹配点加入列表
                        matches_A.append((x, y))
                        matches_B.append((point_2d_B[0].item(), point_2d_B[1].item()))
                        

            # 转换为 numpy 数组以便绘图
            matches_A = np.array(matches_A)
            matches_B = np.array(matches_B)

            H, W = im_combined.shape[:2]

            selected_index = np.random.choice(range(len(matches_A)), 20, replace=False)

            # 绘制匹配点及连接线
            for i in selected_index:

                # 在合并图像中绘制匹配点
                x_A = int(matches_A[i, 0])
                y_A = int(matches_A[i, 1])
                x_B = int(matches_B[i, 0]) + im_A_cv.shape[1]  # 加上图像A的宽度偏移
                y_B = int(matches_B[i, 1])

                if y_B > H or x_B > W or y_B < 0:
                    continue

                # 绘制匹配点
                cv2.circle(
                    im_combined, (x_A, y_A), 5, (0, 255, 0), -1
                )  # 图像A上的匹配点
                cv2.circle(
                    im_combined, (x_B, y_B), 5, (0, 0, 255), -1
                )  # 图像B上的匹配点

                # 绘制匹配点之间的连线
                cv2.line(im_combined, (x_A, y_A), (x_B, y_B), (255, 0, 0), 1)

            # 显示拼接后的图像
            plt.figure(figsize=(12, 6))
            plt.imshow(im_combined)
            plt.title("Image A and B with Matches")
            plt.axis("off")
            plt.savefig(os.path.join(output, "debug", batch["image_name"] + ".png"))