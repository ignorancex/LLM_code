import os
import glob
from natsort import natsorted
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils.utils_torch import torch_interp_1d, torch_conv
from utils.set_root_paths import root_data_path, root_idif_path

def reduce_to_600(values):
    step = len(values) // 600  
    reduced_values = [max(values[i:i+step]) for i in range(0, len(values), step)]
    return torch.tensor(reduced_values[:600], dtype=torch.float32)

class KineticModel_2TC_curve_fit():
    def __init__(self, patient):
        self.patient = patient
    
    def read_idif(self, sample_time, t):
        aorta_idif_txt_path = os.path.join(root_data_path, "DynamicPET/IDIF", f"DynamicFDG_{self.patient}_IDIF.txt")
        data = pd.read_csv(aorta_idif_txt_path, sep="\t")
        self.aorta_idif = torch.Tensor(data["plasma[kBq/cc]"])
        self.aorta_idif_interp = reduce_to_600(torch_interp_1d(t, sample_time, self.aorta_idif))

        portal_idif_txt_path = os.path.join(root_idif_path, "pulmonary_artery_IDIF_ownSeg", f"IDIF_Patient_{self.patient}.txt")
        data = pd.read_csv(portal_idif_txt_path, sep="\t")
        self.portal_idif = torch.Tensor(data["plasma[kBq/cc]"])
        self.portal_idif_interp = reduce_to_600(torch_interp_1d(t, sample_time, self.portal_idif))

        return self.aorta_idif_interp, self.portal_idif_interp
    
    def PET_2TC_KM(self, t, k1, k2, k3, Vb, alpha, beta):
        k1, k2, k3, Vb = map(lambda x: torch.tensor(x, dtype=torch.float32), [k1, k2, k3, Vb])
        step = 0.1
        a = alpha * self.aorta_idif_interp + beta * self.portal_idif_interp
        e = (k2 + k3) * t
        b = k1 / (k2 + k3) * (k3 + k2 * torch.exp(-e))
        c = torch_conv(a, b) * step
        PET = (1-Vb) * c + Vb * a
        return PET
    
    def PET_normal(self, t, k1, k2, k3, Vb):
        k1, k2, k3, Vb = map(lambda x: torch.tensor(x, dtype=torch.float32), [k1, k2, k3, Vb])
        step = 0.1
        a = self.aorta_idif_interp 
        e = (k2 + k3) * t
        b = k1 / (k2 + k3) * (k3 + k2 * torch.exp(-e))
        c = torch_conv(a, b) * step
        PET = (1-Vb) * c + Vb * a
        return PET

def process_pet_slice(patient, slice_idx, slice_orientation="axial"):
    print(f"whole slice for aorta and pulmonary input {patient}, {slice_orientation}")

    time_stamp = torch.load(root_data_path + "/DynamicPET/time_stamp.pt")
    t_file = torch.load(root_data_path + "/DynamicPET/t.pt")

    t = torch.linspace(0, torch.load(root_data_path + "/DynamicPET/t.pt")[-1], 2000)

    label_map_path = glob.glob(root_data_path + f"/DynamicPET/*DynamicFDG_{patient}/NIFTY/Resampled/labels.nii.gz")[0]
    PET_list = natsorted(glob.glob(root_data_path + f"/DynamicPET/*DynamicFDG_{patient}/NIFTY/Resampled/PET_*.nii.gz"))
    save_path = os.path.join("/xxx/xxx/xxx/data/plots/organ_curve", "voxel", f"lung_Vb_{slice_orientation}_{patient}_slice_{slice_idx}")
    os.makedirs(save_path, exist_ok=True)
    
    label_map_ = sitk.GetArrayFromImage(sitk.ReadImage(label_map_path))

    if slice_orientation == "axial":
        liver_mask = np.isin(label_map_[slice_idx, :, :], [4, 5])

    elif slice_orientation == "coronal":
        liver_mask = np.isin(label_map_[:, slice_idx, :], [4, 5])

    else:
        raise ValueError("Invalid slice orientation. Choose 'axial' or 'coronal'.")

    pet_images = [sitk.GetArrayFromImage(sitk.ReadImage(pet_path)) for pet_path in PET_list]
    
    if slice_orientation == "axial":
        pet_slice_1 = pet_images[0][slice_idx, :, :]
        pet_slice_2 = pet_images[-1][slice_idx, :, :]
    else:
        pet_slice_1 = pet_images[0][:, slice_idx, :]
        pet_slice_2 = pet_images[-1][:, slice_idx, :]


    KM_2TC = KineticModel_2TC_curve_fit(patient)
    aorta_idif, portal_idif = KM_2TC.read_idif(time_stamp, t_file)

    shape = pet_slice_1.shape
    param_maps = {param: torch.full(shape, 0, dtype=torch.float32) for param in ["k1", "k2", "k3", "Vb", "alpha", "beta"]}  # Schwarz für Nicht-Leber-Pixel

    if slice_orientation == "axial":
        slice_shape = pet_images[0].shape[1:]  # (Höhe, Breite)
    elif slice_orientation == "coronal":
        slice_shape = (pet_images[0].shape[0], pet_images[0].shape[2])  # (Tiefe, Breite)
    else:
        raise ValueError("Invalid slice orientation. Choose 'axial' or 'coronal'.")

    all_cords = np.array([(x, y) for x in range(slice_shape[0]) for y in range(slice_shape[1])])

    liver_coords = np.argwhere(liver_mask)  
    predicted_tac_matrix = []
    liver_tac_matrix = []
    real_liver_tac_matrix = []

    liver_coords_list = [tuple(c) for c in liver_coords]

    for coord in all_cords:
        x, y = coord
        if slice_orientation == "axial":
            tac_values = [pet_image[slice_idx, x, y] / 1000 for pet_image in pet_images]  # Bq/ml → kBq/ml
        else:
            tac_values = [pet_image[x, slice_idx, y] / 1000 for pet_image in pet_images]
        auc = np.trapz(tac_values, x=time_stamp)

        if auc < 10:
            continue

        tac_tensor = torch.tensor(tac_values, dtype=torch.float32)
        tac_interp = reduce_to_600(torch_interp_1d(t, time_stamp, tac_tensor))


        if tuple(coord) in liver_coords_list:
            real_liver_tac_matrix.append(tac_interp)

            try:
                p, _ = curve_fit(KM_2TC.PET_2TC_KM, t_file, tac_interp,
                                p0=[0.1, 0.1, 0.001, 0.001, 0, 0],
                                bounds=([0.01, 0.01, 0.001, 0.001, 0, 0], [10, 10, 1, 0.2, 1, 1]),
                                diff_step=0.001)
                k1, k2, k3, Vb, alpha, beta = p

                predicted_tac = KM_2TC.PET_2TC_KM(t_file, k1, k2, k3, Vb, alpha, beta).detach().numpy()
                predicted_tac_matrix.append(predicted_tac)
                if liver_mask[x, y]:
                    liver_tac_matrix.append(predicted_tac)
            
                param_maps["k1"][x, y] = k1
                param_maps["k2"][x, y] = k2
                param_maps["k3"][x, y] = k3
                param_maps["Vb"][x, y] = Vb
                param_maps["alpha"][x, y] = alpha
                param_maps["beta"][x, y] = beta

            except RuntimeError:
                print("error")
                continue
        else:
            try:
                p, _ = curve_fit(KM_2TC.PET_2TC_KM, t_file, tac_interp,
                                p0=[0.1, 0.1, 0.001, 0.001, 0, 0],
                                bounds=([0.01, 0.01, 0.001, 0.001, 0, 0], [10, 10, 1, 1, 1, 1]),
                                diff_step=0.001)
                k1, k2, k3, Vb, alpha, beta = p

                predicted_tac = KM_2TC.PET_2TC_KM(t_file, k1, k2, k3, Vb, alpha, beta).detach().numpy()
                predicted_tac_matrix.append(predicted_tac)
                if liver_mask[x, y]:
                    liver_tac_matrix.append(predicted_tac)
            
                param_maps["k1"][x, y] = k1
                param_maps["k2"][x, y] = k2
                param_maps["k3"][x, y] = k3
                param_maps["Vb"][x, y] = Vb
                param_maps["alpha"][x, y] = alpha
                param_maps["beta"][x, y] = beta

            except RuntimeError:
                print("error")
                continue

    vmax = {
    "k1": 10, "k2": 10, "k3": 1, "Vb": 1, "alpha": 1, "beta": 1,
    }

    param_save_path = os.path.join(save_path, f"{patient}_slice_{slice_idx}_params.npz")
    np.savez_compressed(param_save_path, **param_maps)


    for param, data in param_maps.items():
 
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap="jet", vmin=0, vmax=vmax[param])
        plt.colorbar(label=param)
        plt.title(f"{param} für Slice {slice_idx} (nur Lunge)")
        plt.savefig(os.path.join(save_path, f"{patient}_slice_{slice_idx}_{param}_lung.png"))
        plt.close()

    plt.figure(figsize=(8, 6))
    plt.imshow(pet_slice_1)
    plt.colorbar(label="PET Signal")
    plt.title(f"PET Slice 1 {slice_idx}")
    plt.savefig(os.path.join(save_path, f"{patient}_slice_{slice_idx}_PET_01.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.imshow(pet_slice_2)
    plt.colorbar(label="PET Signal")
    plt.title(f"PET Slice 62 {slice_idx}")
    plt.savefig(os.path.join(save_path, f"{patient}_slice_{slice_idx}_PET_60.png"))
    plt.close()

    if(len(liver_tac_matrix) > 0):
        predicted_tac_mean = np.mean(liver_tac_matrix, axis=0)
        real_liver_tac_matrix = np.mean(real_liver_tac_matrix, axis=0)
        plt.figure(figsize=(8, 6))
        plt.plot(t_file, predicted_tac_mean, label="Vorhergesagte TAC", color='red')
        plt.plot(t_file, real_liver_tac_matrix, label="Reale TAC", color='blue')
        plt.xlabel("Zeit (s)")
        plt.ylabel("kBq/ml")
        plt.title("Time-Activity-Curve für Lunge")
        plt.legend()
        plt.xlim([0, 5])
        plt.savefig(os.path.join(save_path, f"{patient}_slice_{slice_idx}_TAC_comparison.png"))
        plt.close()

def process_pet_slice_normal(patient, slice_idx, slice_orientation="axial"):
    print(f"whole slice for aorta input {patient}, {slice_orientation}")

    time_stamp = torch.load(root_data_path + "/DynamicPET/time_stamp.pt")
    t_file = torch.load(root_data_path + "/DynamicPET/t.pt")

    t = torch.linspace(0, torch.load(root_data_path + "/DynamicPET/t.pt")[-1], 2000)

    label_map_path = glob.glob(root_data_path + f"/DynamicPET/*DynamicFDG_{patient}/NIFTY/Resampled/labels.nii.gz")[0]
    PET_list = natsorted(glob.glob(root_data_path + f"/DynamicPET/*DynamicFDG_{patient}/NIFTY/Resampled/PET_*.nii.gz"))
    save_path = os.path.join("/xxx/xxx/xxx/data/plots/organ_curve", "voxel", f"lung_normal_{slice_orientation}_{patient}_slice_{slice_idx}")
    os.makedirs(save_path, exist_ok=True)
    
    label_map_ = sitk.GetArrayFromImage(sitk.ReadImage(label_map_path))

    if slice_orientation == "axial":
        liver_mask = np.isin(label_map_[slice_idx, :, :], [4, 5])

    elif slice_orientation == "coronal":
        liver_mask = np.isin(label_map_[:, slice_idx, :], [4, 5])

    else:
        raise ValueError("Invalid slice orientation. Choose 'axial' or 'coronal'.")

    pet_images = [sitk.GetArrayFromImage(sitk.ReadImage(pet_path)) for pet_path in PET_list]
    
    if slice_orientation == "axial":
        pet_slice_1 = pet_images[0][slice_idx, :, :]
        pet_slice_2 = pet_images[-1][slice_idx, :, :]
    else:
        pet_slice_1 = pet_images[0][:, slice_idx, :]
        pet_slice_2 = pet_images[-1][:, slice_idx, :]


    KM_2TC = KineticModel_2TC_curve_fit(patient)
    aorta_idif, portal_idif = KM_2TC.read_idif(time_stamp, t_file)

    shape = pet_slice_1.shape
    param_maps = {param: torch.full(shape, 0, dtype=torch.float32) for param in ["k1", "k2", "k3", "Vb"]}  

    if slice_orientation == "axial":
        slice_shape = pet_images[0].shape[1:] 
    elif slice_orientation == "coronal":
        slice_shape = (pet_images[0].shape[0], pet_images[0].shape[2])  
    else:
        raise ValueError("Invalid slice orientation. Choose 'axial' or 'coronal'.")


    all_cords = np.array([(x, y) for x in range(slice_shape[0]) for y in range(slice_shape[1])])

    liver_coords = np.argwhere(liver_mask)  
    predicted_tac_matrix = []
    liver_tac_matrix = []
    real_liver_tac_matrix = []

    #all_cords = all_cords[:5] 
    liver_coords_list = [tuple(c) for c in liver_coords] 

    for coord in all_cords:
        x, y = coord
        if slice_orientation == "axial":
            tac_values = [pet_image[slice_idx, x, y] / 1000 for pet_image in pet_images]  # Bq/ml → kBq/ml
        else:
            tac_values = [pet_image[x, slice_idx, y] / 1000 for pet_image in pet_images]
        auc = np.trapz(tac_values, x=time_stamp)

        if auc < 10:
            continue

        tac_tensor = torch.tensor(tac_values, dtype=torch.float32)
        tac_interp = reduce_to_600(torch_interp_1d(t, time_stamp, tac_tensor))


        if tuple(coord) in liver_coords_list:
            real_liver_tac_matrix.append(tac_interp)

            try:
                p, _ = curve_fit(KM_2TC.PET_normal, t_file, tac_interp,
                                p0=[0.1, 0.1, 0.001, 0.001],
                                bounds=([0.01, 0.01, 0.001, 0.001], [10, 10, 1, 0.2]),
                                diff_step=0.001)
                k1, k2, k3, Vb = p

                predicted_tac = KM_2TC.PET_normal(t_file, k1, k2, k3, Vb).detach().numpy()
                predicted_tac_matrix.append(predicted_tac)
                if liver_mask[x, y]:
                    liver_tac_matrix.append(predicted_tac)
            
                param_maps["k1"][x, y] = k1
                param_maps["k2"][x, y] = k2
                param_maps["k3"][x, y] = k3
                param_maps["Vb"][x, y] = Vb

            except RuntimeError:
                print("error")
                continue
        else:
            try:
                p, _ = curve_fit(KM_2TC.PET_normal, t_file, tac_interp,
                                p0=[0.1, 0.1, 0.001, 0.001],
                                bounds=([0.01, 0.01, 0.001, 0.001], [10, 10, 1, 1]),
                                diff_step=0.001)
                k1, k2, k3, Vb = p

                predicted_tac = KM_2TC.PET_normal(t_file, k1, k2, k3, Vb).detach().numpy()
                predicted_tac_matrix.append(predicted_tac)
                if liver_mask[x, y]:
                    liver_tac_matrix.append(predicted_tac)
            
                param_maps["k1"][x, y] = k1
                param_maps["k2"][x, y] = k2
                param_maps["k3"][x, y] = k3
                param_maps["Vb"][x, y] = Vb

            except RuntimeError:
                print("error")
                continue

    vmax = {
    "k1": 10, "k2": 10, "k3": 1, "Vb": 1
    }

    param_save_path = os.path.join(save_path, f"{patient}_slice_{slice_idx}_params.npz")
    np.savez_compressed(param_save_path, **param_maps)


    for param, data in param_maps.items():
 
        plt.figure(figsize=(8, 6))
        plt.imshow(np.flipud(data), cmap="jet", vmin=0, vmax=vmax[param])
        plt.colorbar(label=param)
        plt.title(f"{param} für Slice {slice_idx} (nur Lunge)")
        plt.savefig(os.path.join(save_path, f"{patient}_slice_{slice_idx}_{param}_lung.png"))
        plt.close()

    plt.figure(figsize=(8, 6))
    plt.imshow(np.flipud(pet_slice_1))
    plt.colorbar(label="PET Signal")
    plt.title(f"PET Slice 1 {slice_idx}")
    plt.savefig(os.path.join(save_path, f"{patient}_slice_{slice_idx}_PET_01.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.imshow(np.flipud(pet_slice_2))
    plt.colorbar(label="PET Signal")
    plt.title(f"PET Slice 62 {slice_idx}")
    plt.savefig(os.path.join(save_path, f"{patient}_slice_{slice_idx}_PET_60.png"))
    plt.close()

    if(len(liver_tac_matrix) > 0):
        predicted_tac_mean = np.mean(liver_tac_matrix, axis=0)
        real_liver_tac_matrix = np.mean(real_liver_tac_matrix, axis=0)
        plt.figure(figsize=(8, 6))
        plt.plot(t_file, predicted_tac_mean, label="Vorhergesagte TAC", color='red')
        plt.plot(t_file, real_liver_tac_matrix, label="Reale TAC", color='blue')
        plt.xlabel("Zeit (s)")
        plt.ylabel("kBq/ml")
        plt.title("Time-Activity-Curve für Lunge")
        plt.legend()
        plt.xlim([0, 5])
        plt.savefig(os.path.join(save_path, f"{patient}_slice_{slice_idx}_TAC_comparison.png"))
        plt.close()

if __name__ == '__main__':
    process_pet_slice(patient="06", slice_idx=273, slice_orientation="axial")
    process_pet_slice_normal(patient="06", slice_idx=273, slice_orientation="axial")

