import os
import numpy as np
import trimesh
import imageio

from anakin.utils.renderer import PointLight, load_bg
from anakin.artiboost.render_infra import RendererProvider
from anakin.utils.misc import CONST

from lib.utils.transform import center_vert_bbox
from lib.utils.config import get_config
from lib.utils.etqdm import etqdm
from lib.datasets import build_dataset

YCB_TEXTURED_MODEL_PATH = "./data/DexYCB/models"


def get_HTML_mesh(n_hands=51):
    hand_mesh = []
    for i in range(n_hands):
        hand_path = f"data/HTML_supp/html_{i + 1:03d}/hand.obj"
        if not os.path.exists(hand_path):
            continue
        hand_mesh.append(trimesh.load(hand_path, process=False))

    return hand_mesh


def setup_render_infra():
    query_obj_list = [
        "010_potted_meat_can",
        "006_mustard_bottle",
        "021_bleach_cleanser",
        # ...
        # "003_cracker_box",
        # "004_sugar_box",
        # "011_banana",
        # "019_pitcher_base",
        # "025_mug",
        # "035_power_drill",
        # "037_scissors",
    ]
    obj_meshes = []
    obj_names = []
    for on in query_obj_list:
        obj_path = os.path.join(YCB_TEXTURED_MODEL_PATH, on, "textured.obj")
        omesh = trimesh.load(obj_path, process=False)
        verts = np.asfarray(omesh.vertices)
        verts_can, bbox_center, bbox_scale = center_vert_bbox(verts, scale=False)
        omesh.vertices = verts_can
        obj_meshes.append(omesh)
        obj_names.append(on)

    obj_trimeshes_mapping = {name: m for name, m in zip(obj_names, obj_meshes)}
    render_intr = np.array([
        [617.343, 0.0, 312.42],
        [0.0, 617.343, 241.42],
        [0.0, 0.0, 1.0],
    ]).astype(np.float32)
    cam_extr = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]).astype(np.float32)
    render_size = [640, 480]  # [width, height]
    render_hand_meshes = get_HTML_mesh()
    render_bgs = load_bg("data/synth_bg")
    render_lights = [PointLight(color=np.array([0.9, 0.9, 0.9]), intensity=5.0, pose=np.eye(4))]
    render_gpu_ids = [0]
    render_provider = RendererProvider(num_workers=1,
                                       gpu_render_id=render_gpu_ids,
                                       render_size=render_size,
                                       cam_intr=render_intr,
                                       cam_extr=CONST.PYRENDER_EXTRINSIC,
                                       obj_meshes=obj_trimeshes_mapping,
                                       hand_meshes=render_hand_meshes,
                                       bgs=render_bgs,
                                       lights=render_lights,
                                       random_seed=2)
    render_provider.begin()
    return render_provider


def main(args):
    cfg_all = get_config(config_file="config/datasets.yml")
    assert args.dataset in ["ho3dycba", "ho3dsyntht"], "Only support ho3dycba and ho3dsyntht"
    assert args.mode == "3d_hand_obj", "Only support 3d_hand_obj mode"
    assert args.split == "train", "Only support train split"

    this_cfg = cfg_all[args.dataset]
    this_cfg.DATA_SPLIT = args.split
    this_cfg.DATA_MODE = args.mode
    this_cfg.TRANSFORM = cfg_all.TRANSFORM.TEST
    dataset = build_dataset(this_cfg, data_preset=cfg_all.DATA_PRESET)

    render_provider = setup_render_infra()
    message_queue = render_provider.get_message_queue()
    image_queue_list = render_provider.get_image_queue_list()

    for i in etqdm(range(len(dataset)), desc="Rendering"):
        save_path = dataset.get_image_path(i)
        if os.path.exists(save_path) and not args.replace:
            continue

        obj_id = dataset.get_obj_id(i)
        obj_transf = dataset.get_obj_transf(i)
        hand_verts = dataset.get_verts_3d(i)

        # id: use which render_id to render
        msg = {"id": 0, "objname": obj_id, "pose": obj_transf, "hand_verts": hand_verts}
        message_queue.put(msg)
        img = image_queue_list[0].get()
        img = img[:, :, ::-1]  # BGR -> RGB

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.imwrite(save_path, img)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split", type=str, default="train", help="data split type")
    parser.add_argument("-m", "--mode", type=str, default="3d_hand_obj", help="data split type")
    parser.add_argument("-d", "--dataset", type=str, default="ho3dycba", help="dataset name")
    parser.add_argument("--replace", action="store_true", default=False)

    args, _ = parser.parse_known_args()
    main(args)
