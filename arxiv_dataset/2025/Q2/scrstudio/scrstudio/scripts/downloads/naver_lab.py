# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
# Modified by Xudong Jiang (ETH Zurich)

"""Download datasets and specific captures from the datasets."""

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import kapture
import lmdb
import numpy as np
import quaternion
import tyro
from kapture.io import csv
from tqdm import tqdm

from scrstudio.scripts.downloads.utils import DatasetDownload
from scrstudio.utils.scripts import run_command


def to_4x4mat(pose):
    out=np.eye(4, dtype=np.float64)
    out[:3,:3]=quaternion.as_rotation_matrix(pose.r)
    out[:3,3]=pose.t.flatten()
    return out

def to_3x3mat(inrinsic):
    sensorparams= inrinsic.sensor_params
    assert sensorparams[0] == 'OPENCV'
    fx, fy, cx, cy = sensorparams[3:7]
    out = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]], dtype=np.float64)
    return out

naver_buildings = {
    "dept": "HyundaiDepartmentStore",
    "station": "GangnamStation",
}
naver_captures = ["dept_1F", "dept_4F", "dept_B1", "station_B1","station_B2"]

if TYPE_CHECKING:
    NaverCaptureName = str
else:
    NaverCaptureName = tyro.extras.literal_type_from_choices(naver_captures)

@dataclass
class NaverDownload(DatasetDownload):
    """Download the naver labs dataset."""

    capture_name: NaverCaptureName = "dept_1F"


    def download(self, save_dir: Path):
        """Download the naver labs dataset."""

        os.chdir(save_dir)
        building,floor = self.capture_name.split('_')
        os.makedirs(f"{building}/{floor}", exist_ok=True)
        building_name = naver_buildings[building]
        run_command("kapture_download_dataset.py update")

        split_names= [('train', 'mapping'), ('val', 'validation'), ('test', 'test')]
        for split, name in split_names:
            cmd = f"kapture_download_dataset.py install {building_name}_{floor}_release_{name}"
            run_command(cmd,verbose=True)

            kapture_dir= Path(f"{building_name}/{floor}/release/{name}")
            kapture_data = csv.kapture_from_dir(kapture_dir_path=kapture_dir.as_posix())
            assert kapture_data.records_camera is not None
            ts_cam_fn = list(kapture.flatten(kapture_data.records_camera))

            ts_cam_fn.sort(key=lambda x: x[-1])
            cameras=kapture_data.cameras

            data= {
                "calibration": [],
                "file_list": [],
                "image_shapes": [],
            }
            if kapture_data.trajectories is not None:
                data["poses"] = []

            out_path=Path(f"{building}/{floor}/{split}")
            lmdb_path=out_path/'rgb_lmdb'
            if not os.path.isdir(lmdb_path):
                os.makedirs(lmdb_path)
            else:
                shutil.rmtree(lmdb_path)
                print('lmdb path exist, remove it')
                os.makedirs(lmdb_path)
            env = lmdb.open(str(lmdb_path), subdir=os.path.isdir(lmdb_path),
                        map_size=2**36, readonly=False,
                        meminit=False, map_async=True,max_dbs=1)
            db=env.open_db(b'images',integerkey=True)

            with env.begin(write=True, db=db) as txn:
                for i, ( ts, cam,  fn) in enumerate(tqdm(ts_cam_fn)):
                    image_path=kapture_dir/"sensors/records_data"/fn
                    intrinsics=cameras[cam]
                    k = to_3x3mat(intrinsics)
                    assert intrinsics.sensor_params[0] == 'OPENCV'
                    dist=np.array([float(x) for x in intrinsics.sensor_params[7:]])
                    img=cv2.imread(str(image_path))
                    img_undistorted=cv2.undistort(img,k,dist)
                    image_bytes=cv2.imencode('.jpg',img_undistorted)[1].tobytes()
                    txn.put(key=i.to_bytes(4,sys.byteorder), value=image_bytes,append=True)
                    data["file_list"].append(str(fn))
                    data["calibration"].append(k)
                    data["image_shapes"].append(img.shape[:2])
                    if kapture_data.trajectories is not None:
                        pose=to_4x4mat(kapture_data.trajectories[ts,cam].inverse())
                        data["poses"].append(pose)
            env.close()

            np.save(out_path/'calibration.npy',data["calibration"])
            np.save(out_path/'image_shapes.npy',data["image_shapes"])
            with open(out_path/'file_list.txt','w') as f:
                f.write('\n'.join(data["file_list"]))
            with open(lmdb_path/'file_list.txt','w') as f:
                f.write('\n'.join(data["file_list"]))

            if kapture_data.trajectories is not None:   
                all_poses=np.stack(data["poses"])
                np.save(out_path/'poses.npy',all_poses)
                