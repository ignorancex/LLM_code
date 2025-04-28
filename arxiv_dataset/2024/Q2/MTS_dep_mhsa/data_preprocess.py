import os
import numpy as np
import os
import re
import PIL.Image as Image
from pydicom import dcmread
from tqdm import tqdm
from joblib import Parallel, delayed




def match_ppl(str_):
    return re.match('\w+', str_)

def list_images(dir):
    res = []
    if isinstance(dir, str):
        res = os.listdir(dir)
        res = list(filter(match_ppl, res))
        res = [os.path.join(dir, x) for x in res]
    else:
        for d in dir:
            tmp = os.listdir(d)
            tmp = list(filter(match_ppl, tmp))
            tmp = [os.path.join(d, x) for x in tmp]
            res.extend(tmp)
    res = [x for x in res if x != ".DS_Store"]
    return res

def load_scan(path):
    dir_list = os.listdir(path)
    dir_list = sorted(dir_list)
    dir_list = dir_list[:12]

    slices = [dcmread(os.path.join(path, x)) for x in dir_list if match_ppl(x)]
    if slices == []:
        print(path)
    # ImagePositionPatient: [-222.700, -140.000, -498.750] x, y, z ? sorted by z

    # slices.sort(key=lambda x: float(x.ImagePositionPatient[2]), reverse=True)
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # 转换为int16，int16是ok的，因为所有的数值都应该 <32k
    image = image.astype(np.int16)
    # 设置边界外的元素为0
    image[image <= -2000] = 0
    # 转换为HU单位
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int32)

        image[slice_number] += np.int32(intercept)
    return np.array(image, dtype=np.int16)

def merge_images(person_data):
    pass

# Adjust the window width and window level of CT image and scale its value to 0~255
def set_dicom_window_width_and_center(img_data, window_width=200, window_center=50, img_height=512, img_width=512):
    img_temp = img_data.copy()
    # img_temp.flags.writeable = True
    min = (2 * window_center - window_width) / 2.0 + 0.5
    max = (2 * window_center + window_width) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    for i in np.arange(img_height):
        for j in np.arange(img_width):
            img_temp[i, j] = int((img_temp[i, j]-min)*dFactor)

    min_index = img_temp < 0
    img_temp[min_index] = 0
    max_index = img_temp > 255
    img_temp[max_index] = 255
    return img_temp

def patient_helper(param_):
    path_patient_name, folder, patient_pixels, idx, img = param_
    # grab image name and its label from the path and create
    # a placeholder corresponding to the separate label folder

    imageName = path_patient_name.split(os.path.sep)[-1] + "_" + img.split(".")[0] + '.png'
    # image_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, path_patient_name.split(os.path.sep)[-1])
    
    # label = path_patient_name.split(os.path.sep)[-2]
    # if label != "无狭窄及轻度狭窄":
    #     label = "有狭窄"

    targetFolder = os.path.join(folder, path_patient_name.split("/")[-1])
    # targetFolder = os.path.join(folder, image_uuid)
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder, exist_ok=True)
    destination = os.path.join(targetFolder, imageName)
    dcm_normalized_array = set_dicom_window_width_and_center(patient_pixels[idx])  # H, W
    dcm_img = Image.fromarray(dcm_normalized_array)
    dcm_img = dcm_img.convert('L')
    dcm_img.save(destination, format="PNG")
    
def process_one_subject(source_patient_dir, target_data_dir, 
                        min_frames=10, n_frames=12):
    subdir = [x for x in os.listdir(source_patient_dir) if x != ".DS_Store"]
    if len(subdir) < min_frames:
        return
    patient_scans = load_scan(os.path.join(source_patient_dir))
    patient_pixels = get_pixels_hu(patient_scans)
    deepest_dir = os.listdir(os.path.join(source_patient_dir))
    deepest_dir = [x for x in deepest_dir if x != ".DS_Store"]
    deepest_dir = sorted(deepest_dir)[:n_frames]
    param_pack = [(source_patient_dir, target_data_dir, patient_pixels, idx, img) for idx, img in enumerate(sorted(deepest_dir))]
    thread_num= 12
    res_ = Parallel(n_jobs=thread_num)(delayed(patient_helper)(param) for param in param_pack)


def copy_images(imagePaths, target_folder, min_frames=10, n_frames=12):
    # check if the destination folder exists and if not create it
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    skip_record = []
    # loop over the image paths
    pbar = tqdm(sorted(imagePaths))
    thread_num=14
    Parallel(n_jobs=thread_num)(delayed(process_one_subject)(source_patient_dir, target_folder, min_frames, n_frames) for source_patient_dir in pbar)
    # for path_patient_name in pbar:
        # subdir = [x for x in os.listdir(path_patient_name) if x != ".DS_Store"]
        # if len(subdir) < min_frames:
        #     continue
        # patient_scans = load_scan(os.path.join(path_patient_name, postfix))
        # patient_pixels = get_pixels_hu(patient_scans)
        # pbar.set_description(f"Processing: {path_patient_name}")
        # deepest_dir = os.listdir(os.path.join(path_patient_name, postfix))
        # deepest_dir = [x for x in deepest_dir if x != ".DS_Store"]
        # deepest_dir = sorted(deepest_dir)[:n_frames]
        # param_pack = [(path_patient_name, folder, 'ps', patient_pixels, idx, img) for idx, img in enumerate(sorted(deepest_dir))]
        # thread_num = mp.cpu_count()
        # res_ = Parallel(n_jobs=thread_num)(delayed(patient_helper)(param) for param in param_pack)
    print(skip_record)


if __name__ == "__main__":
    # source_normal_dir = ["./ct_scans/light_to_moderate/"]
    # source_anomaly_dir = ["./ct_scans/moderate_to_severe/"]
    source_normal_dir = ["./source_data/enhanced_ct_scans/light_to_moderate/"]
    source_anomaly_dir = ["./source_data/enhanced_ct_scans/moderate_to_severe/"]
    target_dir = './processed_data/enhanced_ct_scans/'
    continue_ = input("continue?")
    if continue_ in ('true', 'y', 'yes'):
        copy_images(list_images(source_normal_dir), target_dir)
        copy_images(list_images(source_anomaly_dir), target_dir)
