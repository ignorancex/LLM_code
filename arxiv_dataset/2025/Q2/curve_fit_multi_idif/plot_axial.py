import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib

# Datei-Pfade
vb_file_path = "/xxx/xxx/xxx/data/plots/organ_curve/voxel/kidney_Vb_axial_06_slice_205/06_slice_205_params.npz"
normal_file_path = "/home/xxx/xxx/data/plots/organ_curve/voxel/kidney_normal_axial_06_slice_205/06_slice_205_params.npz"
pet_file_path = "/xxx/xxx-data/xxx/DynamicPET/04_DynamicFDG_06/NIFTY/Resampled/PET_61.nii.gz"
ct_file_path = "/xxx/xxx-data/xxx/DynamicPET/04_DynamicFDG_06/NIFTY/Resampled/CT.nii.gz"  

slice = 205

# Funktion zum Zoomen auf die zentralen 85 % des Bildes
def zoom_image(image, zoom_factor=0.8):
    if "lung" in vb_file_path:
        zoom_factor = 0.35
    if "kidney" in vb_file_path:
        zoom_factor = 0.4
    h, w = image.shape
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    start_h, start_w = (h - new_h) // 2, (w - new_w) // 2
    return image[start_h:start_h + new_h, start_w:start_w + new_w]

# Lade PET-Bild
pet_img = nib.load(pet_file_path).get_fdata()
slice_161 = zoom_image(pet_img[:, :, slice])

# Lade CT-Bild
ct_img = nib.load(ct_file_path).get_fdata()
slice_ct = zoom_image(ct_img[:, :, slice])

# Stelle sicher, dass das Verzeichnis existiert
save_dir = os.path.join(os.path.dirname(vb_file_path), "plot")
os.makedirs(save_dir, exist_ok=True)

# Lade die Daten
Vb_data = np.load(vb_file_path)
normal_data = np.load(normal_file_path)

if "lung" in vb_file_path:
    Vb_data_dict = {key: Vb_data[key].copy() for key in Vb_data.files}  # Jede Variable einzeln kopieren

    Vb_data_dict["k3"][Vb_data_dict["k3"] > 0.3] = 0
    print("Lung data")
    Vb_data = Vb_data_dict

print(Vb_data["k3"].max())

# Variablen
variables_normal = ["k1", "k2", "k3", "Vb"]
variables_vb = ["k1", "k2", "k3", "Vb", "alpha", "beta"]

# Erstelle die Plots
fig, axes = plt.subplots(2, 6, figsize=(20, 6))

# Erste Reihe: "normal" Plots
for i, var in enumerate(variables_normal):
    if var in normal_data:
        ax = axes[0, i]
        zoomed_data = zoom_image(normal_data[var])
        if var == "k1":
            im = ax.imshow(zoomed_data, cmap="jet", vmin=0, vmax=1)
        elif var == "k2":
            im = ax.imshow(zoomed_data, cmap="jet", vmin=0, vmax=1)
        elif var == "k3":
            im = ax.imshow(zoomed_data, cmap="jet", vmin=0, vmax=0.6)  
            if "lung" in vb_file_path:
                im = ax.imshow(zoomed_data, cmap="jet", vmin=0, vmax=0.2)
        else:
            im = ax.imshow(zoomed_data, cmap="jet", vmax=0.6)
        fig.colorbar(im, ax=ax, shrink=0.86)
        ax.set_title(var if var != "k1" else "K1")
        ax.set_xticks([])
        ax.set_yticks([])

# PET und CT Bilder in der ersten Reihe
ax = axes[0, 4]
im = ax.imshow(np.rot90(np.flipud(slice_161), k=-1), cmap="gray_r", vmax=40000)
cbar = fig.colorbar(im, ax=ax, shrink=0.9)
cbar.ax.set_visible(False)
ax.set_title(f"PET Axial Slice {slice}")
ax.set_xticks([])
ax.set_yticks([])

ax = axes[0, 5]
im = ax.imshow(np.rot90(np.flipud(slice_ct), k=-1), cmap="gray")
cbar = fig.colorbar(im, ax=ax, shrink=0.86)
cbar.ax.set_visible(False)
ax.set_title(f"CT Axial Slice {slice}")
ax.set_xticks([])
ax.set_yticks([])

# Zweite Reihe: "Vb" Plots
for i, var in enumerate(variables_vb):
    if var in Vb_data:
        ax = axes[1, i]
        zoomed_data = zoom_image(Vb_data[var])
        if var == "k1":
            im = ax.imshow(zoomed_data, cmap="jet", vmin=0, vmax=6)
        elif var == "k2":
            im = ax.imshow(zoomed_data, cmap="jet", vmin=0, vmax=6)
        elif var == "k3":
            im = ax.imshow(zoomed_data, cmap="jet", vmin=0, vmax=0.6)
            if "lung" in vb_file_path:
                im = ax.imshow(zoomed_data, cmap="jet", vmin=0, vmax=0.2)
        elif var == "Vb":
            im = ax.imshow(zoomed_data, cmap="jet", vmin=0, vmax=0.6)
        else:
            im = ax.imshow(zoomed_data, cmap="jet")
        fig.colorbar(im, ax=ax, shrink=0.86)
        if var == "k1":
            ax.set_title("K1")
        elif var == "alpha":
            ax.set_title(r"$\alpha$ for aorta")
        elif "liver" in vb_file_path and var == "beta":
            ax.set_title(r"$\beta$ for portal vein")
        elif "lung" in vb_file_path and var == "beta":
            ax.set_title(r"$\gamma$ for pulm. artery")
        elif "kidney" in vb_file_path and var == "beta":
            ax.set_title(r"$\delta$ for ureter")
        else:
            ax.set_title(var)
        ax.set_xticks([])
        ax.set_yticks([])

plt.subplots_adjust(wspace=0.1, hspace=0.1)

row_labels = ["Single-Input", "Multi-Input"]

for i, label in enumerate(row_labels):
    axes[i, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=20)

# Speicherpfad für den Gesamtplot
overall_save_path = os.path.join(save_dir, "merged_plot.eps")
plt.savefig(overall_save_path)
plt.savefig(overall_save_path.replace(".eps", ".png"))
plt.close()

print(f"Gesamtplot gespeichert unter: {overall_save_path}")
