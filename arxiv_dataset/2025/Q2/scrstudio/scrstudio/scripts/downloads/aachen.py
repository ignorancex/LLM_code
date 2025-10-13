# Based on https://github.com/vislearn/esac/blob/master/datasets/setup_aachen.py

# The images of the Aachen Day-Night dataset are licensed under a 
# [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) and are intended for non-commercial academic use only. 
# All other data also provided as part of the Aachen Day-Night dataset, including the 3D model 
# and the camera calibrations, is derived from these images. Consequently, all other data is 
# also licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) and intended for non-commercial academic use only.

import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import gdown
import numpy as np
import quaternion

from scrstudio.data.utils.readers import folder2lmdb, lmdb_image_shapes
from scrstudio.scripts.downloads.utils import DatasetDownload
from scrstudio.utils.install_checks import check_colmap_installed
from scrstudio.utils.scripts import run_command


def dl_file(urls, file):
	if os.path.isfile(file):
			return True
	for url in urls:
		print(f"Try downloading to {file} from {url}")
		if "drive.google.com" in url:
			try:
				gdown.download(url, file, quiet=False)
			except gdown.exceptions.FileURLRetrievalError as e:
				print(f"Failed to download {file}: {e}")
				continue
		else:
			urlretrieve(url, file)
		if os.path.isfile(file):
			return True
	sys.exit(f"Failed to download {file}")

@dataclass
class AachenDownload(DatasetDownload):
    """Download the Aachen Day Night dataset."""

    def download(self, save_dir: Path):
        """Download the naver labs dataset."""
        print("\n###############################################################################")
        print("# Please make sure to check this dataset's license before using it!           #")
        print("# https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/README_Aachen-Day-Night.md #")
        print("###############################################################################\n\n")

        license_response = input('Please confirm with "yes" or abort. ')
        if license_response not in {"yes", "y"}:
            sys.exit(f"Your response: {license_response}. Aborting.")

        root = save_dir / "aachen"
        root.mkdir(parents=True, exist_ok=True)
        # change to the root directory
        os.chdir(root)

        image_zip = 'database_and_query_images.zip'
        image_folder = 'images_upright'
        recon_file = 'aachen_cvpr2018_db.nvm'
        db_file = 'database_intrinsics.txt'
        day_file = 'day_time_queries_with_intrinsics.txt'
        night_file = 'night_time_queries_with_intrinsics.txt'
        if not os.path.exists(image_folder):
            dl_file(["https://drive.google.com/uc?id=18vz-nCBFhyxiX7s-zynGczvtPeilLBRB",
                    "https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/images/database_and_query_images.zip"], image_zip)
            f = zipfile.PyZipFile(image_zip)
            f.extractall()
            os.remove(image_zip)
        dl_file(["https://drive.google.com/uc?id=0B7s5ESv70mytamRSY0J1dWs4aE0",
                "https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/3D-models/aachen_cvpr2018_db.nvm"], recon_file)
        dl_file(["https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/3D-models/database_intrinsics.txt"], db_file)
        dl_file(["https://drive.google.com/uc?id=0B7s5ESv70mytQS1MSmlIVVZzaGM",
                "https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/queries/day_time_queries_with_intrinsics.txt"], day_file)
        dl_file(["https://drive.google.com/uc?id=0B7s5ESv70mytTWZmTFoxUkNYZW8",
                "https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/queries/night_time_queries_with_intrinsics.txt"], night_file)
        
        # create folders (colmap will not create them)
        test_dir =  Path("test")
        train_dir =  Path("train")
        test_dir.mkdir(exist_ok=True)
        train_dir.mkdir(exist_ok=True)
        check_colmap_installed("colmap")

        fn_f = []
        for file in (day_file, night_file):
            all_subdirs = set()
            with open( file, 'r') as f:
                for line in f:
                    line = line.split()
                    fn_f.append((line[0], line[4]))
                    all_subdirs.add((test_dir/"rgb"/line[0]).parent)
            for subdir in all_subdirs:
                os.makedirs(subdir, exist_ok=True)
            cmd=f"colmap image_undistorter_standalone --input_file {file} --image_path images_upright --output_path test/rgb"
            run_command(cmd, verbose=True)
        if not os.path.exists(test_dir / "rgb_lmdb"):
            folder2lmdb(test_dir / "rgb", test_dir / "rgb_lmdb")
        fn_f.sort(key=lambda x: x[0])
        calibrations = np.array([float(f) for _, f in fn_f])
        np.save(test_dir/"calibration.npy", calibrations)

        fn_f = []
        all_subdirs = set()
        with open( db_file, 'r') as f:
            for line in f:
                line = line.split()
                fn_f.append((line[0], line[4]))
                all_subdirs.add((train_dir/"rgb"/line[0]).parent)

        if not os.path.exists(train_dir / "rgb"):
            for subdir in all_subdirs:
                os.makedirs(subdir, exist_ok=True)
            cmd = f"colmap image_undistorter_standalone --input_file {db_file} --image_path images_upright --output_path train/rgb"
            run_command(cmd, verbose=True)
        if not os.path.exists(train_dir / "rgb_lmdb"):
            folder2lmdb(train_dir / "rgb", train_dir / "rgb_lmdb")
        if not os.path.exists(train_dir / "image_shapes.npy"):
            lmdb_image_shapes(train_dir)
        fn_f.sort(key=lambda x: x[0])
        calibrations = np.array([float(f) for _, f in fn_f])
        np.save(train_dir/"calibration.npy", calibrations)

        with open(recon_file, 'r') as f:
            reconstruction = f.readlines()

        num_cams = int(reconstruction[2])
        camera_list = [x.split() for x in reconstruction[3:3+num_cams]]
        camera_list.sort(key=lambda x: x[0])


        all_poses = []
        for cam_idx,camera in enumerate(camera_list):
            pose= np.eye(4, dtype=np.float64)
            pose[:3,:3]=quaternion.as_rotation_matrix(
                  quaternion.quaternion(*[float(r) for r in camera[2:6]]).inverse())
            pose[:3,3]=[float(r) for r in camera[6:9]]
            all_poses.append(pose)
        all_poses = np.stack(all_poses)
        np.save(train_dir/'poses.npy', all_poses)


