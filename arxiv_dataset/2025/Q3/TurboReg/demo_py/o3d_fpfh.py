import os
import numpy as np
import open3d as o3d
import torch
import turboreg_gpu
from glob import glob
from sklearn.neighbors import NearestNeighbors
from utils_pcr import compute_transformation_error
import time

data_dir = "../demo_data"
voxel_size = 1

def numpy_to_torch32(device, *arrays):
    return [torch.tensor(array, device=device, dtype=torch.float32) for array in arrays]

def load_data(idx_str):
    src = o3d.io.read_point_cloud(os.path.join(data_dir, f"{idx_str}_pts_src.ply"))
    dst = o3d.io.read_point_cloud(os.path.join(data_dir, f"{idx_str}_pts_dst.ply"))
    trans_gt = np.loadtxt(os.path.join(data_dir, f"{idx_str}_trans.txt"))
    return src, dst, trans_gt

def preprocess(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*3, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100)
    )
    return pcd_down, fpfh

def draw_registration(src, dst, trans, title=""):
    src_temp = src.transform(trans.copy())
    src_temp.paint_uniform_color([1, 1, 0])
    dst.paint_uniform_color([0, 1, 1])
    o3d.visualization.draw_geometries([src_temp, dst], window_name=title)

idx_list = sorted(set([os.path.basename(f).split('_')[0] for f in glob(os.path.join(data_dir, "*_pts_src.ply"))]))


reger = turboreg_gpu.TurboRegGPU(6000, 0.1, 2500, 0.15, 0.4, "IN")
while True:
    for idx_str in idx_list:
        print(f"\nProcessing index: {idx_str}")
        src, dst, trans_gt = load_data(idx_str)

        tmp_kpts_src = os.path.join(data_dir, f"{idx_str}_fpfh_kpts_src.txt")
        tmp_kpts_dst = os.path.join(data_dir, f"{idx_str}_fpfh_kpts_dst.txt")
        if os.path.exists(tmp_kpts_src) and os.path.exists(tmp_kpts_dst):
            kpts_src = np.loadtxt(tmp_kpts_src)
            kpts_dst = np.loadtxt(tmp_kpts_dst)
        else:
            src_down, src_fpfh = preprocess(src, voxel_size)
            dst_down, dst_fpfh = preprocess(dst, voxel_size)

            src_feats = np.array(src_fpfh.data).T
            dst_feats = np.array(dst_fpfh.data).T

            nn = NearestNeighbors(n_neighbors=1).fit(dst_feats)
            _, indices = nn.kneighbors(src_feats)

            kpts_src = np.asarray(src_down.points)
            kpts_dst = np.asarray(dst_down.points)[indices[:, 0]]

            np.savetxt(tmp_kpts_src, kpts_src)
            np.savetxt(tmp_kpts_dst, kpts_dst)

        # TurboReg
        T0 = time.time()
        kpts_src_torch, kpts_dst_torch = numpy_to_torch32("cuda:0", kpts_src, kpts_dst)
        TIME_data_to_gpu = time.time() - T0
        T1 = time.time()
        trans = reger.run_reg(kpts_src_torch, kpts_dst_torch).cpu().numpy()
        TIME_reg = time.time() - T1
        print(trans, '\n', trans_gt)
        rre, rte = compute_transformation_error(trans_gt, trans)
        is_succ = (rre < 5) & (rte < 0.6)
        print("SUCC: ", is_succ, " TIME_data_to_gpu: {:.3f} (ms)".format(TIME_data_to_gpu * 1000), " TIME_reg: {:.3f} (ms)".format(TIME_reg * 1000))