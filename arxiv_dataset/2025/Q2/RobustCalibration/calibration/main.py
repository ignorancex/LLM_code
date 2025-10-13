from easydict import EasyDict as edict
from argparse import ArgumentParser
from pathlib import Path
import sys, os, yaml, random, torch
from loader import Loader, VODB
from pose_model import Pose
from gaussian_model import GaussianModel
from tqdm import trange
import numpy as np
from matplotlib import pyplot as plt
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import json
from scipy.spatial.transform import Rotation
import time


def depths_to_points(intrins, pose, W, H, depthmap, scale=3, mask=None):
    c2w = pose.invert()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    uv = torch.stack([grid_x+0.5, grid_y+0.5, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = uv @ intrins.invert().mat.T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    depthmap = F.interpolate(depthmap.unsqueeze(0).unsqueeze(0),  (H, W), mode="bilinear")
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points.reshape(H, W, 3)[mask], uv.reshape(H, W, 3)[mask]

def world_to_pixel(pose, intr, points):
    uv = intr @ (pose @ points)
    return uv[..., :2], uv[..., 2]

def get_tensor_values(tensor, p,  mode='nearest',
                      scale=True, detach=True, detach_p=True, align_corners=False):
    '''
        MIT License

        Copyright (c) 2022 Wenjing Bian

        Reference: https://github.com/ActiveVisionLab/nope-nerf
    ''' 
    
    batch_size, _, h, w = tensor.shape
    
    if detach_p:
        p = p.detach()
    if scale:
        p[:, :, 0] = 2.*p[:, :, 0]/w - 1
        p[:, :, 1] = 2.*p[:, :, 1]/h - 1
    p = p.unsqueeze(1)
    values = torch.nn.functional.grid_sample(tensor, p, mode=mode, align_corners=align_corners)
    values = values.squeeze(2)

    if detach:
        values = values.detach()
    values = values.permute(0, 2, 1)

    return values

def reproject_loss(cam1, cam2, ext, depthmap, depth2, reachable_mask, it):
    if it < 2000:
        scale = 2
    else:
        scale = 1  
    pose1 = ext @ cam1.pose
    pose2 = ext @ cam2.pose
    W = cam1.width
    H = cam1.height
    points, uv = depths_to_points(cam1.intr, pose1, cam1.width, cam1.height, depthmap, scale, reachable_mask)
    uv2, z = world_to_pixel(pose2, cam2.intr, points)
    uv /= scale
    f = z>0
    uv2 /= (z[..., None] + 1e-6)
    uv2 /= scale
    f = f & ((uv2[..., 0]>0) & (uv2[..., 0] < W//scale - 1) & (uv2[..., 1] > 0) & (uv2[..., 1] < H//scale - 1))
    
    getdepth2 = get_tensor_values(depth2.unsqueeze(0).unsqueeze(0), uv2[f][None, ..., :2]*scale, mode="bilinear", scale="True", detach=True, detach_p=True, align_corners=True).squeeze()
    f[f.clone()] = ((getdepth2 - z[f]).abs() < 0.5)


    image1 = cam1.image.cuda().permute(2, 0, 1).unsqueeze(0)
    image2 = cam2.image.cuda().permute(2, 0, 1).unsqueeze(0)
    img1 = F.interpolate(image1, (H//scale, W//scale) ,mode='bilinear')
    img2 = F.interpolate(image2, (H//scale, W//scale) ,mode='bilinear')

    rbg1 = get_tensor_values(img1, uv[f][None,..., :2], mode="bilinear", scale=True, detach=False, detach_p=False, align_corners=True)
    rbg2 = get_tensor_values(img2, uv2[f][None,..., :2], mode="bilinear", scale=True, detach=False, detach_p=False, align_corners=True)
    if rbg1.shape[1] == 0:
        return torch.tensor(0).float().cuda()
    return l2(rbg1, rbg2)

def to_hom(a):
    return torch.column_stack((a, torch.ones_like(a[:, 0])))

def cal_tri_depth(p1, p2, pose):
    p2x = p2[:, 0]
    p2y = p2[:, 1]
    R, t = pose.get_Rt
    R1 = R.mat[0]
    R2 = R.mat[1]
    R3 = R.mat[2]
    t1 = t[0]
    t2 = t[1]
    t3 = t[2]
    zx = (p2x * t3 - t1) / ((R1 * p1).sum(dim=1) - (R3 * p1).sum(dim=1) * p2x)
    zy = (p2y * t3 - t2) / ((R2 * p1).sum(dim=1) - (R3 * p1).sum(dim=1) * p2y)
    return zx, zy

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    rays, points = depths_to_points2(view, depth)
    rays = rays.reshape(*depth.shape, 3)
    points = points.reshape(*depth.shape, 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, rays

def depths_to_points2(view, depthmap):
    W, H = view.width, view.height
    intrins = view.intr.mat
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T
    points = depthmap.reshape(-1, 1) * rays_d
    return rays_d, points

def triangulation_loss(it, colmap_db, cam1, cam2, ext, depth, reachable_mask):
    surf_normal, rays = depth_to_normal(cam1, depth)

    pose1 = (ext @ cam1.pose).invert()
    pose2 = ext @ cam2.pose
    pose = pose2 @ pose1

    pixel1, pixel2 = colmap_db.load_corres_pixels(cam1.image_name, cam2.image_name)
    
    a, b = reachable_mask.shape
    pixel_f = (pixel1[:, 1] < a) & (pixel1[:, 0] < b) & (pixel2[:, 1] < a) & (pixel2[:, 0] < b)
    pixel1 = pixel1[pixel_f]
    pixel2 = pixel2[pixel_f]
    pixel1 = torch.from_numpy(pixel1).float().cuda()
    f = reachable_mask[pixel1.int()[:, 1], pixel1.int()[:, 0]]
    pixel1 = pixel1[f]
    pixel2 = torch.from_numpy(pixel2).float().cuda()[f]

    z1, z2 = cal_tri_depth(cam1.intr.invert()@to_hom(pixel1), cam1.intr.invert()@to_hom(pixel2), pose)
    rd = get_tensor_values(depth.unsqueeze(0).unsqueeze(0), pixel1.clone().unsqueeze(0), mode="bilinear", scale=True, detach=False, detach_p=True, align_corners=True).squeeze()
    res = 0
    n = 0
    normals = get_tensor_values(surf_normal.permute(2,0,1).unsqueeze(0), pixel1.clone().unsqueeze(0), mode="bilinear", scale=True, detach=True, detach_p=True, align_corners=True).squeeze()
    rot = ext.get_R.mat.detach()
    xyz_score = (normals[..., None] * rot[None]).sum(1).abs()
    xyz_score_inv = (normals[..., None] * rot.t()[None]).sum(1).abs()
    res_filter = torch.zeros((len(xyz_score)), dtype=torch.float, device="cuda")
    if  ((z1rdf := huber(z1, rd)) is not None):
        z1rd, z1f = z1rdf
        res_filter += z1f.float()
        res += z1rd
        n += 1
    if  ((z2rdf := huber(z2, rd)) is not None):
        z2rd, z2f = z2rdf
        res_filter += z2f.float()
        res += z2rd
        n += 1
    if n==0:
        return None, None, None
    res_filter /= n
    res_score = (res_filter[..., None] * xyz_score).sum(0)
    res_score_inv = (res_filter[..., None] * xyz_score_inv).sum(0)
    return res / n, res_score, res_score_inv

def huber(x, y):
    c = 1
    diff = (x - y).abs() / c
    f = diff<=c
    if (f).sum() == 0:
        return None
    diff[~f] = c
    s = (((1 - (1 - diff * diff) ** 3) * c * c/ 6))
    res = s.mean()
    return res, f

def parse(file_name):
    assert os.path.exists(file_name), "option file does not exist"
    opt = yaml.safe_load(Path(file_name).read_text())
    return edict(opt)

def create_pose(opt, gt_ext):
    ext = Pose(opt, gt_ext).cuda()
    R_params = {"params": ext.R.parameters(), "lr": 1e-2}
    t_params = {"params": ext.t.parameters(), "lr": 5e-4}
    optimizer = torch.optim.Adam([R_params, t_params])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)
    return ext, optimizer, scheduler

def load_gaussians(opt, color=False):
    gaussians = GaussianModel(color=color)
    fn = "2dgs-test.pth"
    checkpoint_path = os.path.join(
            opt.output, 
            opt.g,
            fn
            )
    (model_params, first_iter) = torch.load(checkpoint_path)
    gaussians.restore(model_params)
    if color:
        field_params = [{"params": gaussians._colors, "lr": 1e-2},]
        return gaussians, torch.optim.Adam(field_params)
    return gaussians

def l2(a, b, w=None):
    if w is None:
        return ((a - b) ** 2).mean()
    else:
        return (((a-b)**2)*w[:, None]).sum()/w.sum()

def l1(a, b):
    return (a - b).abs().mean()

def train_gaussian(opt, train_cams, feature_db, gt_ext, writer):
    total = 15000
    lambda_tri = 1
    lambda_repr = 200
    

    gaussians, optimizer_color = load_gaussians(opt, color=True)
    ext, optimizer_pose, sched = create_pose(opt, gt_ext)
    with torch.no_grad():
        print(ext().mat.tolist())
        error_R, error_t = ext.get_error
        obj = {
            "inital": {
                "mat": ext().mat.tolist(),
                "R": error_R.tolist(),
                "t": error_t.tolist(),
                "R_abs": np.linalg.norm(error_R).tolist(),
                "t_abs": np.linalg.norm(error_t).tolist()
            }
            }

    progress = trange(total)
    stack = []

    output_dir = f"./{opt.output}/{opt.g}"
    os.makedirs(output_dir, exist_ok=True)

    flag1, flag2, flagR = False, False, float("inf")
    score_sum = torch.zeros(3, dtype=torch.float, device="cuda")
    score_num = 0
    score_inv_sum = torch.zeros(3, dtype=torch.float, device="cuda")
    loss_sum = torch.zeros(3, dtype=torch.float, device="cuda")
    loss_num = torch.zeros(3, dtype=torch.float, device="cuda")
    start_time = time.time()
    for it in progress:
        if not stack:
            stack = list(range(opt.data.num))
            random.shuffle(stack)
        idx = stack.pop()
        camera = train_cams[idx]

        pose = ext() @ camera.pose
        pose._mat.retain_grad()
        ext.R.weight.retain_grad()
        ext.t.weight.retain_grad()
        depth, normal, color, reachable_mask, var, uncertain = gaussians.render(pose, camera)
        uncertaind = torch.exp(-uncertain.detach())
        gt_colors = camera.image.cuda()[reachable_mask]

        repr_loss = 0
        rc = 0
        rcp = 0
        tri_loss = 0
        loss = l2(gt_colors, color.permute(1, 2, 0)[reachable_mask], w=uncertaind[reachable_mask])
        if it > 10000:
            loss_sum[0] += loss.detach()
            loss_num[0] += 1
        score = None
        if idx > 0:
            camera2 = train_cams[idx - 1]
            with torch.no_grad():
                pose2 = ext() @ camera2.pose
                rot = (pose2.invert() @ pose).get_R.mat.detach().cpu().numpy()
                rot_deg = np.linalg.norm(Rotation.from_matrix(rot).as_rotvec(degrees=True))
            if ((flagR==float("inf"))) or (rot_deg > 1):
                rcp += 1
                with torch.no_grad():
                    depth2, _, _, _, _, _ = gaussians.render(pose2, camera2)
                repr_loss += reproject_loss(camera, camera2, ext(), depth, depth2, reachable_mask, it)
            if (rot_deg < 10): 
                tri_loss1, score, score_inv = triangulation_loss(it, feature_db, camera, camera2, ext(), depth, reachable_mask)
                if tri_loss1 is not None:
                    rc += 1
                    tri_loss += tri_loss1
                    score_sum += score.detach()
                    score_inv_sum += score_inv.detach()
                    score_num += 1
        
        if (idx < opt.data.num - 1):
            camera2 = train_cams[idx + 1]
            with torch.no_grad():
                pose2 = ext() @ camera2.pose
                rot = (pose2.invert() @ pose).get_R.mat.detach().cpu().numpy()
                rot_deg = np.linalg.norm(Rotation.from_matrix(rot).as_rotvec(degrees=True))
            if ((flagR==float("inf"))) or (rot_deg > 1):
                rcp += 1
                with torch.no_grad():
                    depth2, _, _, _, _, _ = gaussians.render(pose2, camera2)
                repr_loss += reproject_loss(camera, camera2, ext(), depth, depth2, reachable_mask, it)
            if (rot_deg < 10): 
                tri_loss1 ,_,_= triangulation_loss(it, feature_db, camera, camera2, ext(), depth, reachable_mask)
                if tri_loss1 is not None:
                    rc += 1
                    tri_loss += tri_loss1
        
        if rc > 0:
            tri_loss = tri_loss / rc * lambda_tri
            loss += tri_loss
            if it > 10000:
                loss_sum[1] += tri_loss.detach()
                loss_num[1] += 1
        if rcp > 0:
            repr_loss = repr_loss / rcp * lambda_repr
            loss += repr_loss
            if it > 10000:
                loss_sum[2] += repr_loss.detach()
                loss_num[2] += 1
        loss.backward()
        if gaussians._colors.grad.isnan().any():
            optimizer_color.zero_grad()
            continue
        optimizer_color.step()
        optimizer_color.zero_grad()
        if ext.R.weight.grad.isnan().any() or ext.t.weight.grad.isnan().any():
            optimizer_pose.zero_grad()
            optimizer_color.zero_grad()
            continue
        changed = ext.change_rate()
        g = np.linalg.norm(ext.R.weight.grad.detach().cpu().numpy())
        if it > flagR+1000 and not flag1  and changed[0] < 1e-5:
            optimizer_pose.param_groups[-1]["lr"] *= 0.5
            flag1 = True
        if it > flagR+1000 and not flag2 and changed[0] < 5e-6:
            optimizer_pose.param_groups[-1]["lr"] *= 0.5
            flag2 = True
        if it > 1000 and flagR==float("inf")  and changed[1] < 0.1:
            optimizer_pose.param_groups[0]["lr"] = 1e-3
            optimizer_pose.param_groups[-1]["lr"] = 1e-2
            flagR = it        

        error_R, error_t = ext.get_error
        writer.add_scalar('Pose/error_R', np.linalg.norm(error_R), it)
        writer.add_scalar('Pose/error_t', np.linalg.norm(error_t), it)
        writer.add_scalar('Pose/error_tx', error_t[0], it)
        writer.add_scalar('Pose/error_ty', error_t[1], it)
        writer.add_scalar('Pose/error_tz', error_t[2], it)
        writer.add_scalar('Pose/change_t', changed[0], it)
        writer.add_scalar('Pose/change_R', changed[1], it)
        if score is not None:
            writer.add_scalar('Score/x', score[0], it)
            writer.add_scalar('Score/y', score[1], it)
            writer.add_scalar('Score/z', score[2], it)
            writer.add_scalar('Score/x_inv', score_inv[0], it)
            writer.add_scalar('Score/y_inv', score_inv[1], it)
            writer.add_scalar('Score/z_inv', score_inv[2], it)
        writer.add_scalar('Pose/grad_R', g, it)
        writer.add_scalar('Loss/loss', loss.detach(), it)
        progress.set_postfix(loss=loss.tolist())
        if (it > 200 ):
            optimizer_pose.step()
            optimizer_pose.zero_grad()
    with torch.no_grad():
        print(ext().mat.tolist())
        error_R, error_t = ext.get_error
        obj["res"] = {
                "mat": ext().mat.tolist(),
                "R": error_R.tolist(),
                "t": error_t.tolist(),
                "R_abs": np.linalg.norm(error_R).tolist(),
                "t_abs": np.linalg.norm(error_t).tolist()
            }
        obj["scores"] = {
            "tri_score" : ( score_sum / (score_num + 1e-6)).tolist(),
            "tri_score_inv": ( score_inv_sum / (score_num + 1e-6)).tolist(),
            "photo_loss": (loss_sum[0] / (loss_num[0] + 1e-6)).tolist(),
            "tri_loss": (loss_sum[1] / (loss_num[1] + 1e-6)).tolist(),
            "repr_loss": (loss_sum[2] / (loss_num[2] + 1e-6)).tolist(),
            "loss_numbers": loss_num.tolist()
        }
    end_time = time.time()
    with open("times.txt", "a") as f:
        f.write(f"{end_time - start_time}\n")
    with open(f"{output_dir}/res.json", "w") as f:
        json.dump(obj, f)
    
    if opt.render:
        for p in ext.parameters():
            p.requires_grad_(False)
        gaussians.capture(output_dir)
        optimizer = gaussians.double()
        # specially for color rendering
        progress2 = trange(3000)
        stack = []
        count = 0
        for it in progress2:
            if not stack:
                stack = list(range(opt.data.num))
                random.shuffle(stack)
            idx = stack.pop()
            camera = train_cams[idx]
            pose = ext() @ camera.pose
            depth, normal, color, reachable_mask, var, uncertain = gaussians.render(pose, camera)
            if camera.idx ==8:
                if it > 2950:
                    with torch.no_grad():
                        render_colors = color
                        image = render_colors.permute(1,2,0).clamp_max(1).detach().cpu().numpy()
                        plt.imsave(f"{output_dir}/{it+15000}-color.png", image)
                count += 1
                continue
            gt_colors = camera.image.cuda().clone()[reachable_mask]
            loss = l2(gt_colors, color.permute(1, 2, 0)[reachable_mask], w=None)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress2.set_postfix({"loss": f"{loss:.02f}"})
        gaussians.capture(output_dir)

def parse_arg(arg, opt):
    parser = ArgumentParser()
    parser.add_argument("-g")
    parser.add_argument("--render", action="store_true")
    opt_agrs = parser.parse_args(arg)
    opt.update(vars(opt_agrs))
    scene, frame, name = opt.g.split("-")
    opt.data.scene = int(scene)
    opt.data.start_frame = int(frame)
    opt.data.num = 50
    opt.data.name = name


if __name__ == "__main__":
    opt = parse("option.yml")
    parse_arg(sys.argv[1:], opt)
    random.seed(0)
    os.makedirs(f"./{opt.output}/{opt.g}", exist_ok=True)
    writer = SummaryWriter(f"{opt.output}/{opt.g}")
    train_cams, gt_ext = Loader.load_cams(opt)
    feature_db = VODB(opt)

    train_gaussian(opt, train_cams, feature_db, gt_ext, writer)
    print("finish")

        
