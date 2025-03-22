import time
import torch
import math
import cv2
import numpy as np
import os
from lib.datasets import build_dataset
from lib.datasets.mix_dataset import MixDataset
from lib.utils.config import CN
from lib.utils.etqdm import etqdm
from lib.utils.heatmap import sample_with_heatmap
from lib.utils.transform import bchw_2_bhwc, denormalize, persp_project
from lib.viztools.draw import plot_hand
from lib.utils.config import get_config
from lib.viztools.opendr_renderer import OpenDRRenderer
from lib.viztools.utils import ColorsMap as CMap, get_color_map, ColorMode
from manotorch.manolayer import ManoLayer
from lib.utils.transform import gen_random_rotation, gen_random_direction


def viz_hodataset(args):
    np.random.seed(1)
    cfg_all = get_config(config_file="config/datasets.yml")
    if "mix" in args.dataset:
        this_cfg = cfg_all[args.dataset]
        for i in range(len(this_cfg.MIX)):
            ds = cfg_all[args.dataset].MIX[i]
            ds.DATA_SPLIT = args.split
            ds.DATA_MODE = args.mode
            this_cfg.MIX[i] = ds
        this_cfg.TRANSFORM = cfg_all.TRANSFORM.TRAIN if args.split == "train" else cfg_all.TRANSFORM.TEST
    else:
        this_cfg = cfg_all[args.dataset]
        this_cfg.DATA_SPLIT = args.split
        this_cfg.TRANSFORM = cfg_all.TRANSFORM.TRAIN if args.split == "train" else cfg_all.TRANSFORM.TEST
        this_cfg.DATA_MODE = args.mode

    if this_cfg.TYPE == "MIX":
        dataset = MixDataset(this_cfg.MIX,
                             data_preset=cfg_all.DATA_PRESET,
                             transform=this_cfg.TRANSFORM,
                             length=100_000)
    else:
        dataset = build_dataset(this_cfg, data_preset=cfg_all.DATA_PRESET)

    renderer = OpenDRRenderer()
    manolayer_left = ManoLayer(
        center_idx=cfg_all.DATA_PRESET.CENTER_IDX,
        mano_assets_root="assets/mano_v1_2",
        flat_hand_mean=True,
        side="left",
    )
    manolayer_right = ManoLayer(
        center_idx=cfg_all.DATA_PRESET.CENTER_IDX,
        mano_assets_root="assets/mano_v1_2",
        flat_hand_mean=True,
        side="right",
    )

    for i in range(len(dataset)):
        if args.shuffle:
            i = np.random.randint(0, len(dataset))

        output: dict = dataset[i]
        image = denormalize(output["image"], [0.5, 0.5, 0.5], [1, 1, 1]).numpy().transpose(1, 2, 0)
        image = (image * 255.0).astype(np.uint8)

        img_list = []

        if args.mode == "2d":
            joints_heatmap = output["target_joints_heatmap"]
            img_hm = sample_with_heatmap(image, joints_heatmap)
            img_list.append(img_hm)
        elif args.mode in ["3d", "3d_hand_obj", "3d_hand_obj_contact"]:
            img_list.append(image)
            manolayer = manolayer_left if output["hand_side"] == "left" else manolayer_right
            verts_3d = output["target_verts_3d"]
            joints_2d = output["target_joints_2d"]
            hand_faces = manolayer.get_mano_closed_faces().numpy()
            joints_3d = output["target_joints_3d"]
            cam_intr = output["target_cam_intr"]
            mano_pose = torch.from_numpy(output["target_mano_pose"]).reshape(1, -1)
            mano_shape = torch.from_numpy(output["target_mano_shape"]).unsqueeze(0)
            mano_out = manolayer(mano_pose, mano_shape)
            hand_tsl = verts_3d - mano_out.verts.squeeze(0).numpy()
            hand_tsl = np.mean(hand_tsl, axis=0, keepdims=True)  # (1, 3)
            mano_verts = mano_out.verts.squeeze(0).numpy() + hand_tsl
            joints_2d = persp_project(joints_3d, cam_intr)

            try:
                frame_ske = plot_hand(image.copy(), joints_2d, linewidth=2)
                img_list.append(frame_ske)
            except Exception as e:
                pass

            h_color = np.array(CMap["lightblue"])
            frame_h = renderer([verts_3d], [hand_faces], cam_intr, img=image.copy(), vertex_color=[h_color])
            img_list.append(frame_h)

        if args.mode in ["3d_hand_obj", "3d_hand_obj_contact"]:
            obj_verts_3d = output["target_obj_verts_3d"]
            if args.perturb:
                maximal_ang = math.pi / 12
                mean_offset = .010
                std_offset = .01
                obj_verts_3d = torch.from_numpy(obj_verts_3d.astype(np.float32)).unsqueeze(0)
                rand_rot = gen_random_rotation(maximal_ang).expand(1, -1, -1)
                mean_obj_v = torch.mean(obj_verts_3d, dim=1, keepdim=True)  # TENSOR[NBATCH, 1, 3]
                obj_verts_3d = (torch.bmm(rand_rot,
                                          (obj_verts_3d - mean_obj_v).permute(0, 2, 1)).permute(0, 2, 1) + mean_obj_v)
                # generate a random_direction
                dir_vec = gen_random_direction()
                rand_dist = torch.normal(torch.Tensor([mean_offset]), torch.Tensor([std_offset]))
                offset = rand_dist * dir_vec
                obj_verts_3d = obj_verts_3d + offset
                obj_verts_3d = obj_verts_3d.squeeze(0).numpy()

            obj_faces = output["obj_faces"]
            o_color = output["obj_verts_color"] if "obj_verts_color" in output else np.array(CMap["lime"])

            if args.mode == "3d_hand_obj":
                frame_o = renderer([obj_verts_3d, mano_verts], [obj_faces, hand_faces],
                                   cam_intr,
                                   img=image.copy(),
                                   vertex_color=[o_color, h_color])
                img_list.append(frame_o)
            elif args.mode == "3d_hand_obj_contact":
                vertex_contact = output["vertex_contact"]
                contact_region_id = output["contact_region_id"]
                contact_region_id[vertex_contact == 0] = 17
                anchor_elasti = output["anchor_elasti"]
                anchor_elasti = np.max(anchor_elasti, axis=1)  # find the maximum
                vc_color = get_color_map(vertex_contact, ColorMode.VERTEX_CONTACT)
                cr_color = get_color_map(contact_region_id, ColorMode.CONTACT_REGION)
                ae_color = get_color_map(anchor_elasti, ColorMode.ANCHOR_ELASTI)
                frame_vc = renderer([obj_verts_3d], [obj_faces], cam_intr, img=image.copy(), vertex_color=[vc_color])
                frame_cr = renderer([obj_verts_3d], [obj_faces], cam_intr, img=image.copy(), vertex_color=[cr_color])
                frame_ae = renderer([obj_verts_3d], [obj_faces], cam_intr, img=image.copy(), vertex_color=[ae_color])
                img_list.extend([frame_vc, frame_cr, frame_ae])

        frame = np.hstack(img_list)
        cv2.imshow(f"hello {args.mode}", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split", type=str, default="train", help="data split type")
    parser.add_argument("-m", "--mode", type=str, default="3d_hand_obj", help="data split type")
    parser.add_argument("-d", "--dataset", type=str, default="ho3dv2", help="dataset name")
    parser.add_argument("--perturb", action="store_true", help="perturb the object")
    parser.add_argument("--shuffle", action="store_true", help="shuffle the dataset")

    args, _ = parser.parse_known_args()
    viz_hodataset(args)
