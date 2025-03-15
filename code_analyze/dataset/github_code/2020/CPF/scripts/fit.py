import os
import pickle
import time
import traceback
import numpy as np
import torch
import trimesh
from lib.datasets import create_dataset
from lib.utils.config import CN, get_config
from lib.postprocess.geo_optim import GeOptimizer
from lib.utils.collision import penetration_loss_hand_in_obj, solid_intersection_volume, region_disjointness_metric
from lib.utils.transform import aa_to_rotmat, aa_to_quat
from lib.utils.logger import logger
from lib.datasets.hdata import HODataset
from lib.datasets.cionline import CIDumpedData, CIDumpedQueries
from joblib import Parallel, delayed

from manotorch.utils.anchorutils import masking_load_driver
from termcolor import colored, cprint

hand_closed_path = "assets/closed_hand/hand_mesh_close.obj",
metric_keys = [
    "hand_mpvpe",
    "hand_mpjpe",
    "obj_mpvpe",
    "penetration_depth",
    "solid_intersection_volume",
    "disjointness_tip_only",
    "disjointness_palm",
]


def extract_score_from_res(res):
    scores = {}
    scores["exclude_from_score"] = res["exclude_from_score"]

    # if the key in res contains any of the metric_keys, then add it to the scores
    for k, v in res.items():
        if any([metric_key in k for metric_key in metric_keys]):
            scores[k] = v
    return scores


def collapse_score_list(score_list_list):
    scores_list = []
    for item in score_list_list:
        score, use_honet_cnt = item[0], item[1]
        scores_list.extend(score)
    return scores_list


def merge_score_list(score_list):
    if len(score_list) < 1:
        return dict()

    keys_list = list(score_list[0].keys())
    keys_list.remove("exclude_from_score")

    # create init dict
    scores = {k: 0.0 for k in keys_list}
    include = 0
    exclude = 0

    # iterate
    for item_id, item in enumerate(score_list):
        has_nan = False
        for k in keys_list:
            if np.isnan(item[k]):
                cprint(f"encountered nan in {item_id} key {k}", "red")
                has_nan = True
                break

        if has_nan:
            exclude += 1
            continue

        for k in keys_list:
            if item["exclude_from_score"] is False:
                scores[k] += item[k]

        if item["exclude_from_score"] is False:
            include += 1
        else:
            exclude += 1

    # avg
    for k in keys_list:
        scores[k] /= include

    return scores, include, exclude


def summarize(score_dict):
    for k, v in score_dict.items():
        print("mean " + str(k), v)


def mppe(prd, gt):
    ''' prd: (NP, 3), gt (NP, 3) '''
    if isinstance(prd, np.ndarray):
        prd = torch.from_numpy(prd).float()
    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt).float()

    mppe = torch.mean(torch.norm(prd - gt, p=2, dim=1)).item()
    return mppe


def run_sample_by_idx(
    device: torch.device,
    dataset: HODataset,
    optim: GeOptimizer,
    mode: str,
    index: int,
    data_prefix: str,
    save_path: str,
    minimal_contact_ratio: float,
    hand_region_assignment,
    hand_palm_vertex_mask,
):
    sample_id = dataset.get_sample_identifier(index)
    save_file = os.path.join(save_path, f"{sample_id}_optim.pkl")
    if os.path.exists(save_file):
        with open(save_file, "rb") as fstream:
            res = pickle.load(fstream)
            scores = extract_score_from_res(res)
        return False, scores, res["use_honet"]

    # data GT
    gt_obj_verts = torch.from_numpy(dataset.get_obj_verts_3d(index))  # (NO, 3)
    obj_id = dataset.get_obj_id(index)
    n_o_verts = len(gt_obj_verts)
    obj_faces = torch.from_numpy(dataset.get_obj_faces(index)).long()  # (NF, 3)
    obj_normals_can = torch.from_numpy(dataset.get_obj_normals_can(index))  # (NO, 3)
    obj_verts_can = torch.from_numpy(dataset.get_obj_verts_can(index))  # (NO, 3)
    obj_vox_can = torch.from_numpy(dataset.get_obj_vox_can(index))  # (32, 32, 32)
    obj_vox_el_vol = dataset.get_obj_vox_element_volume(index)
    gt_hand_verts = torch.from_numpy(dataset.get_verts_3d(index))  # (778, 3)
    gt_hand_joints = torch.from_numpy(dataset.get_joints_3d(index))  # (21, 3)
    gt_hand_joints_0 = gt_hand_joints[0]  # @NOTE: Explain.

    # hand: close faces => "data/info/closed_hand/hand_mesh_close.obj"
    hand_closed_path = "assets/closed_hand/hand_mesh_close.obj"
    hand_closed_trimesh = trimesh.load(hand_closed_path, process=False)
    hand_closed_faces = torch.from_numpy(np.asarray(hand_closed_trimesh.faces))

    # data from dumpted prediction:
    ci_dumped = CIDumpedData(sample_id, data_prefix).get()
    prd_obj_verts = ci_dumped[CIDumpedQueries.OBJ_VERTS_3D]
    prd_obj_verts = prd_obj_verts[:n_o_verts, :]  # incase the padding is not removed
    prd_obj_rot = ci_dumped[CIDumpedQueries.OBJ_ROT]
    prd_obj_tsl = ci_dumped[CIDumpedQueries.OBJ_TSL]
    prd_obj_rotmat = aa_to_rotmat(prd_obj_rot)  # (3, 3)
    prd_obj_normals = (prd_obj_rotmat @ obj_normals_can.T).T  # (NO, 3)

    prd_hand_pose = ci_dumped[CIDumpedQueries.HAND_POSE].reshape(-1, 3)  # (16, 3)
    prd_hand_pose_quat = aa_to_quat(prd_hand_pose)  # (16, 4)
    prd_hand_verts = ci_dumped[CIDumpedQueries.HAND_VERTS_3D]
    prd_hand_joints = ci_dumped[CIDumpedQueries.HAND_JOINTS_3D]
    prd_hand_joints_0 = prd_hand_joints[0, ...]

    # region ====== eval before fit >>>>>>
    prd_vertex_contact = ci_dumped[CIDumpedQueries.VERTEX_CONTACT]
    contact_ratio = np.sum(prd_vertex_contact.numpy()) / len(prd_vertex_contact.numpy())

    hand_mpvpe_before = mppe(prd_hand_verts, gt_hand_verts)
    hand_mpjpe_before = mppe(prd_hand_joints, gt_hand_joints)

    if dataset.mode_split == "ho3d_paper":
        # center wrt hand root when eval in ho3d paper version
        obj_mpvpe_before = mppe(prd_obj_verts - prd_hand_joints_0, gt_obj_verts - gt_hand_joints_0)
    else:
        obj_mpvpe_before = mppe(prd_obj_verts, gt_obj_verts)

    penetration_depth_before = torch.sqrt(penetration_loss_hand_in_obj(prd_hand_verts, prd_obj_verts, obj_faces)).item()
    intersection_before, _, _ = solid_intersection_volume(
        prd_hand_verts.numpy(),
        hand_closed_faces.numpy(),
        obj_vox_can.numpy(),
        prd_obj_tsl.numpy(),
        prd_obj_rot.numpy(),
        obj_vox_el_vol,
    )

    dj_vec_before, dj_tip_only_before, dj_palm_before = region_disjointness_metric(
        prd_hand_verts.numpy(),
        prd_obj_verts.numpy(),
        hand_region_assignment,
    )
    # endregion <<<<<<

    # region ====== optimize >>>>>>
    exclude_from_score = False
    use_honet = False
    if contact_ratio < minimal_contact_ratio:
        # return honet result
        use_honet = True
        opt_hand_verts = prd_hand_verts
        opt_hand_joints = prd_hand_joints
        opt_hand_pose = prd_hand_pose
        opt_obj_verts = prd_obj_verts

    if dataset.mode_split == "ho3d_paper":
        seq, frame_idx = dataset.get_seq_frame(index)
        from lib.datasets.ho3d import CPF_TRAIN_SEQS, CPF_TEST_SEQS, CPF_GRASP_LIST
        CPF_SEQS = CPF_TRAIN_SEQS + CPF_TEST_SEQS
        if seq not in CPF_SEQS or int(frame_idx) not in CPF_GRASP_LIST[seq]:
            # return honet result
            # cprint(f"  x    [{seq},{frame_idx}] not in CPF-official HO3D test list, do not optimzie", "green")
            use_honet = True
            exclude_from_score = True
            opt_hand_verts = prd_hand_verts
            opt_hand_joints = prd_hand_joints
            opt_hand_pose = prd_hand_pose
            opt_obj_verts = prd_obj_verts

    if use_honet is False:
        # prepare kwargs according to mode
        opt_val_kwargs = dict(
            # static
            mode=mode,
            vertex_contact=ci_dumped[CIDumpedQueries.VERTEX_CONTACT].long().to(device),
            contact_region_id=ci_dumped[CIDumpedQueries.CONTACT_REGION_ID].long().to(device),
            anchor_id=ci_dumped[CIDumpedQueries.CONTACT_ANCHOR_ID].long().to(device),
            anchor_elasti=ci_dumped[CIDumpedQueries.CONTACT_ANCHOR_ELASTI].float().to(device),
            anchor_padding_mask=ci_dumped[CIDumpedQueries.CONTACT_ANCHOR_PADDING_MASK].long().to(device),
            # hand_region_assignment=torch.from_numpy(hand_region_assignment).long().to(device),
            # hand_palm_vertex_mask=torch.from_numpy(hand_palm_vertex_mask).long().to(device),
            obj_faces=obj_faces.long().to(device),
        )
        if mode == "hand":
            opt_val_kwargs.update(
                dict(
                    # hand
                    hand_shape_init=ci_dumped[CIDumpedQueries.HAND_SHAPE].float().to(device),
                    hand_tsl_init=ci_dumped[CIDumpedQueries.HAND_TSL].float().to(device),
                    hand_pose_gt=([0], prd_hand_pose_quat[0:1, :].to(device)),
                    hand_pose_init=(list(range(1, 16)), prd_hand_pose_quat[1:, :].to(device)),
                    # obj
                    obj_verts_gt=prd_obj_verts.float().to(device),
                    obj_normals_gt=prd_obj_normals.float().to(device),
                ))
        elif mode == "obj":
            opt_val_kwargs.update(
                dict(
                    # hand
                    hand_shape_gt=ci_dumped[CIDumpedQueries.HAND_SHAPE].float().to(device),
                    hand_tsl_gt=ci_dumped[CIDumpedQueries.HAND_TSL].float().to(device),
                    hand_pose_gt=(list(range(0, 16)), prd_hand_pose_quat[0:, :].to(device)),
                    # obj
                    obj_verts_can=obj_verts_can.float().to(device),
                    obj_normals_can=obj_normals_can.float().to(device),
                    obj_tsl_init=ci_dumped[CIDumpedQueries.OBJ_TSL].float().to(device),
                    obj_rot_init=ci_dumped[CIDumpedQueries.OBJ_ROT].float().to(device),
                ))
        elif mode == "hand_obj":
            opt_val_kwargs.update(
                dict(
                    # hand
                    hand_shape_init=ci_dumped[CIDumpedQueries.HAND_SHAPE].float().to(device),
                    hand_tsl_init=ci_dumped[CIDumpedQueries.HAND_TSL].float().to(device),
                    hand_pose_gt=([0], prd_hand_pose_quat[0:1, :].to(device)),
                    hand_pose_init=(list(range(1, 16)), prd_hand_pose_quat[1:, :].to(device)),
                    # obj
                    obj_verts_can=obj_verts_can.float().to(device),
                    obj_normals_can=obj_normals_can.float().to(device),
                    obj_tsl_init=ci_dumped[CIDumpedQueries.OBJ_TSL].float().to(device),
                    obj_rot_init=ci_dumped[CIDumpedQueries.OBJ_ROT].float().to(device),
                ))
        else:
            raise KeyError(f"unknown optimization mode {mode}")
        opt_val_kwargs.update(dict(
            # hand compensate
            hand_compensate_root=prd_hand_joints_0.float().to(device),))

        optim.set_opt_val(**opt_val_kwargs)
        optim.optimize(progress=False)

        opt_hand_verts, opt_hand_joints, opt_hand_transf = optim.recover_hand()
        opt_hand_verts = opt_hand_verts.cpu()
        opt_hand_joints = opt_hand_joints.cpu()

        opt_hand_pose = optim.recover_hand_pose().cpu()
        opt_obj_verts = optim.recover_obj().cpu()
    # endregion <<<<<<

    # region ====== eval after fit >>>>>>
    hand_mpvpe_after = mppe(opt_hand_verts, gt_hand_verts)
    hand_mpjpe_after = mppe(opt_hand_joints, gt_hand_joints)
    if dataset.mode_split == "ho3d_paper":
        # center wrt hand root when eval in ho3d paper version
        obj_mpvpe_after = mppe(opt_obj_verts - opt_hand_joints[0, ...], gt_obj_verts - gt_hand_joints_0)
    else:
        obj_mpvpe_after = mppe(opt_obj_verts, gt_obj_verts)

    penetration_depth_after = torch.sqrt(penetration_loss_hand_in_obj(opt_hand_verts, opt_obj_verts, obj_faces)).item()
    # ! dispatch given mode
    if use_honet:
        opt_obj_tsl = prd_obj_tsl
        opt_obj_rot = prd_obj_rot
    else:
        if mode == "obj" or mode == "hand_obj":
            # obj optimiziing option on
            opt_obj_tsl = optim.recover_obj_tsl()
            opt_obj_rot = optim.recover_obj_rot()
        else:
            opt_obj_tsl = prd_obj_tsl
            opt_obj_rot = prd_obj_rot

    intersection_after, _, _ = solid_intersection_volume(
        opt_hand_verts.numpy(),
        hand_closed_faces.numpy(),
        obj_vox_can.numpy(),
        opt_obj_tsl.numpy(),
        opt_obj_rot.numpy(),
        obj_vox_el_vol,
    )
    dj_vec_after, dj_tip_only_after, dj_palm_after = region_disjointness_metric(
        opt_hand_verts.numpy(),
        opt_obj_verts.numpy(),
        hand_region_assignment,
    )
    # endregion <<<<<<

    # res dict
    res = {
        "sample_id": sample_id,
        "use_honet": use_honet,
        "exclude_from_score": exclude_from_score,
        "opt_hand_verts": opt_hand_verts.numpy(),
        "opt_hand_joints": opt_hand_joints.numpy(),
        "opt_hand_pose": opt_hand_pose.numpy(),
        "opt_obj_verts": opt_obj_verts.numpy(),
        "opt_obj_rot": opt_obj_rot.numpy(),
        "opt_obj_tsl": opt_obj_tsl.numpy(),
        "hand_mpvpe_before": hand_mpvpe_before,
        "hand_mpvpe_after": hand_mpvpe_after,
        "hand_mpjpe_before": hand_mpjpe_before,
        "hand_mpjpe_after": hand_mpjpe_after,
        "obj_mpvpe_before": obj_mpvpe_before,
        "obj_mpvpe_after": obj_mpvpe_after,
        "penetration_depth_before": penetration_depth_before,
        "penetration_depth_after": penetration_depth_after,
        "solid_intersection_volume_before": intersection_before * 1e6,
        "solid_intersection_volume_after": intersection_after * 1e6,
        "disjointness_tip_only_before": dj_tip_only_before,
        "disjointness_palm_before": dj_palm_before,
        "disjointness_tip_only_after": dj_tip_only_after,
        "disjointness_palm_after": dj_palm_after,
    }

    # save result
    with open(save_file, "wb") as fstream:
        pickle.dump(res, fstream)

    # retrieve scores
    scores = extract_score_from_res(res)

    return True, scores, use_honet
    # endregion


def worker(
    cfg,
    data_prefix,
    save_prefix,
    device,
    worker_id,
    n_workers,
    lr,
    n_iter,
    mode,
    minimal_contact_ratio,
    lambda_contact_loss,
    lambda_repulsion_loss,
    repulsion_query,
    repulsion_threshold,
    use_fhb_adaptor,
    compenstate_tsl,
    verbose,
    runtime_viz,
    hand_region_assignment,
    hand_palm_vertex_mask,
):
    dataset = create_dataset(cfg.DATASET.TEST, data_preset=cfg.DATA_PRESET)  # perworker dataset

    if save_prefix is None:
        # get dir of data_path
        save_prefix = os.path.join(os.path.dirname(data_prefix), "optimized")
        os.makedirs(save_prefix, exist_ok=True)

    begin_index = worker_id * len(dataset) // n_workers
    end_index = (worker_id + 1) * len(dataset) // n_workers
    cprint(f"{worker_id:>3} begin: {begin_index:0>4} end: {end_index:0>4} len: {len(dataset)}", "cyan")
    cprint(f"{worker_id:>3} using device: {device}", "cyan")

    optim = GeOptimizer(
        device=device,
        lr=lr,
        n_iter=n_iter,
        verbose=verbose,
        lambda_contact_loss=lambda_contact_loss,
        lambda_repulsion_loss=lambda_repulsion_loss,
        repulsion_query=repulsion_query,
        repulsion_threshold=repulsion_threshold,
        use_fhb_adaptor=use_fhb_adaptor,
        compensate_tsl=compenstate_tsl,
        runtime_viz=runtime_viz and worker_id == 0,
    )
    cprint(f"optimizer created on device: {device}", "cyan")
    time.sleep(5)

    score_list = []
    use_honet_cnt = 0  # then use honet as results
    for i in range(begin_index, end_index):
        try:
            print_line = colored(f"       {worker_id:>3} index: {i:0>4}, mode: {mode}\n", "yellow")
            time_start = time.time()
            flag, scores, use_honet = run_sample_by_idx(
                device=device,
                dataset=dataset,
                optim=optim,
                mode=mode,
                index=i,
                data_prefix=data_prefix,
                save_path=save_prefix,
                minimal_contact_ratio=minimal_contact_ratio,
                hand_region_assignment=hand_region_assignment,
                hand_palm_vertex_mask=hand_palm_vertex_mask,
            )
            if flag is False:
                print_line += colored(f"  x    {worker_id:>3} skip: {i:0>4}", "yellow")
            score_list.append(scores)
            time_end = time.time()
            use_honet_cnt += 1 if use_honet else 0
            if not use_honet:
                # region ====== print scores  >>>>>>
                better = "blue"
                worse = "red"
                print_line += f"   x   {worker_id:>3} processed: {i:0>5} elapsed {round(time_end - time_start):>4}s result: "
                print_line += colored(f"HD:bf={scores['hand_mpvpe_before']:.4f}, ", "white")
                print_line += colored(f"OD:bf={scores['obj_mpvpe_before']:.4f}, ", "white")
                print_line += colored(f"PD:bf={scores['penetration_depth_before']:.4f}, ", "white")
                print_line += colored(f"SI:bf={scores['solid_intersection_volume_before']:.4f}, ", "white")
                print_line += "\n"
                print_line += f"   |   {worker_id:>3} processed: {i:0>5}           continue:  "
                print_line += colored(f"DJ_TO:bf={scores['disjointness_tip_only_before']:.4f}, ", "white")
                print_line += colored(f"DJ_PM:bf={scores['disjointness_palm_before']:.4f}, ", "white")
                print_line += "\n"
                print_line += f"   |   {worker_id:>3} processed: {i:0>5}           continue:  "
                cflag = better if scores["hand_mpvpe_after"] < scores["hand_mpvpe_before"] else worse
                print_line += colored(f"HD:af={scores['hand_mpvpe_after']:.4f}, ", cflag)
                cflag = better if scores["obj_mpvpe_after"] < scores["obj_mpvpe_before"] else worse
                print_line += colored(f"OD:af={scores['obj_mpvpe_after']:.4f}, ", cflag)
                cflag = better if scores["penetration_depth_after"] < scores["penetration_depth_before"] else worse
                print_line += colored(f"PD:df={scores['penetration_depth_after']:.4f}, ", cflag)
                cflag = better if scores["solid_intersection_volume_after"] < scores[
                    "solid_intersection_volume_before"] else worse
                print_line += colored(f"SI:af={scores['solid_intersection_volume_after']:.4f}, ", cflag)
                cflag = better if scores["disjointness_tip_only_after"] < scores[
                    "disjointness_tip_only_before"] else worse
                print_line += colored(f"DJ_TO:af={scores['disjointness_tip_only_after']:.4f}, ", cflag)
                cflag = better if scores["disjointness_palm_after"] < scores["disjointness_palm_before"] else worse
                print_line += colored(f"DJ_PM:af={scores['disjointness_palm_after']:.4f}, ", cflag)
                print(print_line)
                # endregion <<<<<<
        except Exception as e:
            exc_trace = traceback.format_exc()
            err_msg = f"  x    {worker_id:>3}: sample {i:0>4}, \n{exc_trace}"
            cprint(err_msg, "red")
            assert False

    optim.reset()  # make sure the viz window is distoryed
    cprint(f"{worker_id:>3} conclude", "cyan")
    return score_list, use_honet_cnt


def main(cfg, args, exp_time, word_size):
    # dataset = create_dataset(cfg.DATASET.TEST, data_preset=cfg.DATA_PRESET)
    hand_region_assignment, hand_palm_vertex_mask = \
        masking_load_driver("assets/anchor", "assets/hand_palm_full.txt")

    # create device for each worker
    device_list = []
    for worker_id in range(args.workers):
        device_list.append(torch.device(f"cuda:{worker_id % word_size}"))

    collected = Parallel(n_jobs=args.workers)(
        delayed(worker)(
            # dataset=dataset,  # which hodataset you want use
            cfg=cfg,  # TODO: tmp fix
            data_prefix=args.data_prefix,  # where the HoNet + PiCR prediction is dumped to
            save_prefix=args.save_prefix,  # where the fitting result will be saved
            device=device_list[worker_id],
            worker_id=worker_id,
            n_workers=args.workers,
            lr=args.lr,
            n_iter=args.n_iter,
            mode=args.mode,
            minimal_contact_ratio=args.minimal_contact_ratio,
            lambda_contact_loss=args.lambda_contact_loss,
            lambda_repulsion_loss=args.lambda_repulsion_loss,
            repulsion_query=args.repulsion_query,
            repulsion_threshold=args.repulsion_threshold,
            use_fhb_adaptor=args.use_fhb_adaptor,
            compenstate_tsl=args.compensate_tsl,
            verbose=args.verbose,
            runtime_viz=args.runtime_viz,
            hand_region_assignment=hand_region_assignment,
            hand_palm_vertex_mask=hand_palm_vertex_mask,
        ) for worker_id in list(range(args.workers)))

    collapsed = collapse_score_list(collected)
    merged, include, exclude = merge_score_list(collapsed)
    summarize(merged)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="experiment configure file name", type=str, default=None)
    parser.add_argument("-g", "--gpu_id", type=str, default="0", help="env var CUDA_VISIBLE_DEVICES")
    parser.add_argument("-w", "--workers", help="worker number from data loader", type=int, default=8)
    parser.add_argument("--data_prefix", type=str, required=True)
    parser.add_argument("--save_prefix", type=str, default=None)

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n_iter", type=int, default=400)
    parser.add_argument("--mode", type=str, choices=["hand", "obj", "hand_obj"], default="hand")
    parser.add_argument("--minimal_contact_ratio", type=float, default=0.01)
    parser.add_argument("--repulsion_query", type=float, default=0.030)
    parser.add_argument("--repulsion_threshold", type=float, default=0.080)
    parser.add_argument("--lambda_contact_loss", type=float, default=10.0)
    parser.add_argument("--lambda_repulsion_loss", type=float, default=0.5)
    parser.add_argument("--use_fhb_adaptor", action="store_true", default=False)
    parser.add_argument("--compensate_tsl", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--runtime_viz", action="store_true", default=False)

    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    exp_time = time.time()
    world_size = torch.cuda.device_count()
    cfg = get_config(config_file=args.config, arg=args, merge=True)
    main(cfg, args, exp_time, world_size)
