import argparse
import os
import numpy as np
from scipy.spatial.distance import cdist
import pickle
from joblib import Parallel, delayed
from lib.utils.config import get_config
from lib.datasets import build_dataset
from lib.datasets.hdata import HODataset
from lib.utils.etqdm import etqdm
from termcolor import cprint
from manotorch.utils.anchorutils import anchor_load_driver, recover_anchor, region_select_and_mask, get_rev_anchor_mapping


def elasti_fn(x, range_th=20.0):
    x = x.copy()
    np.putmask(x, x > range_th, range_th)
    res = 0.5 * np.cos((np.pi / range_th) * x) + 0.5
    np.putmask(res, res < 1e-8, 0)
    return res


def get_mode_list(assignment, n_bins, weights=None):
    # weights: weight of each vertex
    res = np.zeros((n_bins,))
    for bin_id in range(n_bins):
        res[bin_id] = np.sum((assignment == bin_id).astype(np.float32) * weights)
    maxidx = np.argmax(res)
    return maxidx


def worker(
    selec_hand_verts: np.ndarray,  # ARRAY[NHND, 3]
    obj_verts: np.ndarray,  # ARRAY[NOBJ, 3]
    selec_verts_assignment: np.ndarray,  # ARRAY[NOBJ]
    n_regions: int,
    anchor_posi: np.ndarray,
    anchor_mapping: dict,
    dst_path: str,
    replace: bool = True,
    n_samples: int = 32,
    range_threshold: float = 20.0,
    elasti_threshold: float = 45.0,
    elasti_cutoff: float = 0.1,
):
    if os.path.exists(dst_path) and replace is False:
        return False

    vertex_blob_list = []
    obj_n_verts = obj_verts.shape[0]
    rev_anchor_mapping = get_rev_anchor_mapping(anchor_mapping, n_region=n_regions)
    all_dist = cdist(obj_verts, selec_hand_verts) * 1000  # ARRAY[NOBJ, NHND_SEL]

    # ========== STAGE 1: COMPUTE CROSS DISTANCE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    all_dist = cdist(obj_verts, selec_hand_verts) * 1000  # ARRAY[NOBJ, NHND_SEL]
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ========== STAGE 2: Get 20 Closest Points>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    pts_idx = np.arange(all_dist.shape[0])[:, None]
    order_idx = np.argsort(all_dist, axis=1)  # ARRAY[NOBJ, NHND_SEL]
    sorted_dist = all_dist[pts_idx, order_idx]
    sorted_dist_sampled = sorted_dist[:, :n_samples]  # ARRAY[NOBJ, 20]
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ========== STAGE 3: iterate over all points >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    for point_id in range(obj_n_verts):
        dist_vec = sorted_dist_sampled[point_id, :]  # ARRAY[20]
        valid_mask = dist_vec < range_threshold
        masked_dist_vec = dist_vec[valid_mask]  # ARRAY[NSELECTED]
        if len(masked_dist_vec) == 0:
            # there is no contact
            vertex_blob_list.append({"contact": 0})
        else:
            # there is contact
            # ======= get region assignment >>>>>>
            # first we need to use np.where to collect the indexes of vertices in dist_vec
            valid_idx = np.where(valid_mask)[0]  # LIST[INT; NSELECTED]
            # then we now the samples keep the same order as in order_idx
            origin_valid_idx = order_idx[point_id, valid_idx]  # ARRAY[NSELECTED]
            # index them for region information
            origin_valid_points_region = selec_verts_assignment[origin_valid_idx]  # ARRAY[NSELECTED]
            # compute the mode
            # if there are ties, we will try to use the one with most mode
            mode_weight = range_threshold - masked_dist_vec
            target_region = get_mode_list(origin_valid_points_region, n_regions, weights=mode_weight)
            # <<<<<<<<<<<<

            # ====== get anchor distance (by indexing dist mat) >>>>>>
            # get the anchors
            anchor_list = rev_anchor_mapping[target_region]
            # get the distances
            obj_point = obj_verts[point_id:point_id + 1, :]  # ARRAY[1, 3]
            anchor_points = anchor_posi[anchor_list, :]  # ARRAY[NAC, 3]
            dist_mat = cdist(obj_point, anchor_points).squeeze(0) * 1000.0
            elasti_mat = elasti_fn(dist_mat, range_th=elasti_threshold)
            np.putmask(elasti_mat, elasti_mat < elasti_cutoff, elasti_cutoff)
            # <<<<<<<<<<<<

            # store the result
            vertex_blob_list.append({
                "contact": 1,
                "region": target_region,
                "anchor_id": anchor_list,
                "anchor_dist": dist_mat.tolist(),
                "anchor_elasti": elasti_mat.tolist(),
            })

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "wb") as fout:
        pickle.dump(vertex_blob_list, fout)

    return True


def main(args):
    cfg_all = get_config(config_file="config/datasets.yml")
    assert "mix" not in args.dataset, "MixDataset is not supported in this script"

    this_cfg = cfg_all[args.dataset]
    this_cfg.DATA_SPLIT = args.split
    this_cfg.DATA_MODE = args.mode
    this_cfg.TRANSFORM = cfg_all.TRANSFORM.TEST

    assert args.mode == "3d_hand_obj", "Only support 3d_hand_obj mode"
    assert args.split == "train", "Only support train split"

    dataset = build_dataset(this_cfg, data_preset=cfg_all.DATA_PRESET)
    palm_vid = np.loadtxt(os.path.join("assets", "hand_palm_full.txt"), dtype=int)
    face_vertex_index, anchor_weight, merged_vertex_assignment, anchor_mapping = anchor_load_driver("assets")
    n_regions = len(np.unique(merged_vertex_assignment))

    todo_list = []

    for i in etqdm(range(len(dataset)), desc="Preparing data for launch"):
        img_path = dataset.get_image_path(i)

        if args.dataset == "ho3dv2":
            dst_path = img_path.replace("HO3D", "HO3D_supp_v2")
            dst_path = dst_path.replace("png", "pkl")
            dst_path = dst_path.replace("rgb", "contact_info")
        elif args.dataset == "fphab":
            dst_path = img_path.replace("fhbhands", "fhbhands_supp")
            dst_path = dst_path.replace("Video_files_480", "Object_contact_region_annotation_v512")
            dst_path = dst_path.replace("jpeg", "pkl")
            dst_path = dst_path.replace("color/color", "contact_info")
        elif args.dataset == "dexycb":
            dst_path = img_path.replace("DexYCB", "DexYCB_supp")
            dst_path = dst_path.replace("jpg", "pkl")
            dst_path = dst_path.replace("color", "contact_info")
        elif args.dataset == "ho3dycba":
            dst_path = img_path.replace("rgb", "contact_info")
            dst_path = dst_path.replace("png", "pkl")

        hand_verts = dataset.get_verts_3d(i)
        obj_verts = dataset.get_obj_verts_3d(i)
        selec_hand_verts, selec_verts_assignment = \
            region_select_and_mask(hand_verts, merged_vertex_assignment, palm_vid)
        anchor_posi = recover_anchor(hand_verts, face_vertex_index, anchor_weight)

        todo_list.append({
            "selec_hand_verts": selec_hand_verts,
            "obj_verts": obj_verts,
            "selec_verts_assignment": selec_verts_assignment,
            "n_regions": n_regions,
            "anchor_posi": anchor_posi,
            "anchor_mapping": anchor_mapping,
            "dst_path": dst_path,
            "replace": args.replace,
        })

    cprint(f"Launching {len(todo_list)} samples start", color="yellow")
    collected = Parallel(n_jobs=args.n_jobs, verbose=True)(delayed(worker)(**item) for item in todo_list)

    n_skips = 0
    n_success = 0
    for suc in collected:
        n_skips += 1 if suc is False else 0
        n_success += 1 if suc is True else 0

    cprint(f"Skipped {n_skips} samples, success {n_success} samples", color="green")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split", type=str, default="train", help="data split type")
    parser.add_argument("-m", "--mode", type=str, default="3d_hand_obj", help="data split type")
    parser.add_argument("-d", "--dataset", type=str, default="ho3dv2", help="dataset name")
    parser.add_argument("--replace", action="store_true", default=False)
    parser.add_argument("--n_jobs", default=8, type=int)

    args, _ = parser.parse_known_args()
    main(args)
