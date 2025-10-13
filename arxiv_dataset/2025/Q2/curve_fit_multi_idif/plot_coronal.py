import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import matplotlib.image as mpimg


# Datei-Pfade
file_path_1 = "/xxx/xxx/xxx/data/plots/organ_curve/voxel/all_new_coronal_06_slice_161/06_slice_161_params.npz"
file_path_2 = "/xxx/xxx/xxx/data/plots/organ_curve/voxel/normal_coronal_06_slice_161/06_slice_161_params.npz"

pet_file_path = "/xxx/xxx-data/xxx/DynamicPET/04_DynamicFDG_06/NIFTY/Resampled/PET_61.nii.gz"
ct_file_path = "/xxx/xxx-data/xxx/DynamicPET/04_DynamicFDG_06/NIFTY/Resampled/CT.nii.gz"  
mip_image_path = "/xxx/xxx/xxx/data/plots/organ_curve/voxel/all_new_coronal_06_slice_161/patientPET06_sum_mip.png"

# Lade PET-Bild
pet_img = nib.load(pet_file_path).get_fdata()
slice_161 = pet_img[:, 161, :]

# Erstelle MIP (Maximum Intensity Projection)
mip_img = mpimg.imread(mip_image_path)

# Single-Input Parameter
single_data = np.load(file_path_2)
single_variables = ["k1", "k2", "k3", "Vb"]

# Multi-Input Parameter
multi_data = np.load(file_path_1)
multi_variables = ["k1", "k2", "k3", "Vb"]

# Zusatzparameter
extra_variables = ["alpha", "beta", "gamma", "delta"]

# Erstelle die Gesamtgrafik
fig, axes = plt.subplots(3, 5, figsize=(15, 10))

# Erste Zeile: Single-Input-Heatmaps + PET-Bild
for i, var in enumerate(single_variables):
    if var in single_data:
        if var == "k1":
            im = axes[0, i].imshow(np.flipud(single_data[var]), cmap="jet", vmin=0, vmax=1)
        elif var == "k2":
            im = axes[0, i].imshow(np.flipud(single_data[var]), cmap="jet", vmin=0, vmax=1)
        elif var == "k3":
            im = axes[0, i].imshow(np.flipud(single_data[var]), cmap="jet", vmin=0, vmax=0.6)
        else:
            im = axes[0, i].imshow(np.flipud(single_data[var]), cmap="jet")
        fig.colorbar(im, ax=axes[0, i])
        if var == "k1":
            axes[0, i].set_title("K1")
        else:
            axes[0, i].set_title(var)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

# PET-Bild in Position (0,4)
axes[0, 4].imshow(np.rot90(slice_161), cmap="gray_r", vmax=40000)
axes[0, 4].set_title(f"PET Slice 161")
axes[0, 4].set_xticks([])
axes[0, 4].set_yticks([])

# Zweite Zeile: Multi-Input-Heatmaps + MIP-Bild
for i, var in enumerate(multi_variables):
    if var in multi_data:
        if var == "k1":
            im = axes[1, i].imshow(np.flipud(multi_data[var]), cmap="jet", vmin=0, vmax=6)
        elif var == "k2":
            im = axes[1, i].imshow(np.flipud(multi_data[var]), cmap="jet", vmin=0, vmax=6)
        elif var == "k3":
            im = axes[1, i].imshow(np.flipud(multi_data[var]), cmap="jet", vmin=0, vmax=0.6)
        else:
            im = axes[1, i].imshow(np.flipud(multi_data[var]), cmap="jet")
        fig.colorbar(im, ax=axes[1, i])
        if var == "k1":
            axes[1, i].set_title("K1")
        else:
            axes[1, i].set_title(var)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

# MIP-Bild in Position (1,4)
axes[1, 4].imshow(mip_img, cmap="gray_r")
axes[1, 4].set_title("MIP PET")
axes[1, 4].set_xticks([])
axes[1, 4].set_yticks([])

# Dritte Zeile: Extra-Heatmaps + CT-Bild
for i, var in enumerate(extra_variables):
    if var in multi_data:
        im = axes[2, i].imshow(np.flipud(multi_data[var]), cmap="jet")
        fig.colorbar(im, ax=axes[2, i])
        if var == "alpha":
            axes[2, i].set_title(r"$\alpha$ for aorta")
        elif var == "beta":
            axes[2, i].set_title(r"$\beta$ for portal vein")
        elif var == "gamma":    
            axes[2, i].set_title(r"$\gamma$ for pulm. artery")
        elif var == "delta":
            axes[2, i].set_title(r"$\delta$ for ureter")
        else:
            axes[2, i].set_title(var)
        axes[2, i].set_xticks([])
        axes[2, i].set_yticks([])

# CT-Bild in Position (2,4) (falls vorhanden, sonst Dummy)
if ct_file_path and os.path.exists(ct_file_path):
    ct_img = nib.load(ct_file_path).get_fdata()
    slice_ct = ct_img[:, 161, :]
    axes[2, 4].imshow(np.rot90(slice_ct), cmap="gray")
    axes[2, 4].set_title("CT Slice 161")
    axes[2, 4].set_xticks([])
    axes[2, 4].set_yticks([])
else:
    axes[2, 4].axis("off")

plt.subplots_adjust(wspace=0.1, hspace=0.2)
row_labels = ["Single-Input", "Multi-Input", "Multi-Input"]

for i, label in enumerate(row_labels):
    axes[i, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=20)



# Speicherpfad für den Gesamtplot
save_dir = os.path.join("/", *file_path_1.split("/")[:-1], "plot")
os.makedirs(save_dir, exist_ok=True)
overall_save_path = os.path.join(save_dir, "combined_plot.eps")
plt.savefig(overall_save_path)
plt.savefig(overall_save_path.replace("eps", "png"))
plt.close()

print(f"Gesamtplot gespeichert unter: {overall_save_path}")
