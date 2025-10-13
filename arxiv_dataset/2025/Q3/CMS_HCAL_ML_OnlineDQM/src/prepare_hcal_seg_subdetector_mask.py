
"""
Created on Mon Dec 07 17:33:21 2023

@author: Mulugeta W. Asres, mulugetawa@uia.no

DESMOD: Segmentation Mask Preparation for HCAL DQM Digioccupancy 

Example: given in main function

"""

import os, sys
import seaborn as sns
import pandas as pd
import numpy as np
from functools import reduce
import numpy as np, pandas as pd
from tqdm import tqdm
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)
data_path = os.path.abspath(os.path.dirname(current_path)+"/data")
sys.path.append(data_path)

import utilities as util

### HCAL DQM Segmentation: UMAP

### utils
def plot_3d_scatter(df,x, y, z, c, figsize=(6, 6), title="", **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    if isinstance(df, pd.DataFrame):
        ax_3d = ax.scatter(df[x], df[y],
                           df[z], c=df[c], marker=".", **kwargs)
    else:
        data_shape = df.shape
        ax_3d = ax.scatter(np.arange(data_shape[0]), np.arange(data_shape[1]),
                           np.arange(data_shape[2]), c=df, marker=".", **kwargs)
    cbar = plt.colorbar(ax_3d)
    cbar.set_label(c)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title("{}_{}".format(title, c))
    fig.show()

def display_columns_values(df):
    for col in df.columns:
        print(df[col].value_counts())

def load_seg_map(hcal_config_datafilepath_csv):

    print("load_seg_map...{hcal_config_datafilepath_csv}")
    hcal_seg_info_df = util.load_csv(hcal_config_datafilepath_csv)
    print(hcal_seg_info_df.shape)
    hcal_seg_info_df.columns = hcal_seg_info_df.columns.str.strip()
    hcal_seg_info_df.drop(columns=["#"], inplace=True)
    for col in hcal_seg_info_df.select_dtypes("object").columns:
        hcal_seg_info_df[col] = hcal_seg_info_df[col].str.strip()
    print(hcal_seg_info_df.shape)
    hcal_seg_info_df.head(10)

    display_columns_values(hcal_seg_info_df)

    hcal_seg_info_df = cleaning_seg_map(hcal_seg_info_df)

    return hcal_seg_info_df

### Cleaning irrelavant segments
def cleaning_seg_map(hcal_seg_info_df):
    
    print("cleaning_seg_map...")
    print(hcal_seg_info_df.shape)
    hcal_seg_info_df = hcal_seg_info_df.loc[~hcal_seg_info_df["Det"].isin(["CALIB_HB", "CALIB_HE"]), :]
    print(hcal_seg_info_df.shape)

    # there are some QIECH belongs to the depth of -999
    hcal_seg_info_df = hcal_seg_info_df.loc[hcal_seg_info_df["type/depth"] != -999 , :] 
    print(hcal_seg_info_df.shape)

    display_columns_values(hcal_seg_info_df)
    
    return hcal_seg_info_df

### Select HE RBXes
def select_subdetector_seg_map(hcal_seg_info_df, subdetector="he"):

    print(f"select_subdetector_seg_map...{subdetector}")
    subdetectors = ["he", "hb", "hf", "ho"]
    assert subdetector in subdetectors, f"subdetector name is not found in {subdetectors}."

    hcal_seg_info_df.shape
    subdetector_seg_info_df = hcal_seg_info_df.loc[hcal_seg_info_df["RBX"].str.startswith(subdetector.upper()), :]
    if subdetector_seg_info_df.empty:

        raise "segmentation info is not found in the hcal_seg_map file for the selected subdetector."
    
    subdetector_seg_info_df["RBX_ID"] = subdetector_seg_info_df["RBX"].apply(lambda x: x[-2:]).astype("int8")
    sorted(subdetector_seg_info_df["RBX"].unique())
    sorted(subdetector_seg_info_df["RBX_ID"].unique())
    # subdetector_seg_info_df["RBX_ID"] = subdetector_seg_info_df["RBX_ID"]*subdetector_seg_info_df["Side"].apply(lambda x: 1 if x==1 else 2)
    # subdetector_seg_info_df.head()
    subdetector_seg_info_df.shape

    subdetector_seg_info_df.head()

    
    display_columns_values(subdetector_seg_info_df)

    print("number of unique values: ", subdetector_seg_info_df.nunique())

    print("detected unique values: ")
    for col in ["Eta", "Phi", "type/depth"]:
        print(f"{col}:", np.sort(subdetector_seg_info_df[col].unique()))
    
    return subdetector_seg_info_df

### Segementation mapper
def subdetector_seg_map_to_seg_mask(subdetector_seg_info_df, ieta_len=64, iphi_len=72, depth_len=7):

    print("select_subdetector_seg_map...")

    subdetector_seg_info_for_mask_df = subdetector_seg_info_df.copy()

    seg_grp = subdetector_seg_info_for_mask_df.groupby(["type/depth", "Eta"]).agg({"Phi": [set, np.size]}).sort_values(["type/depth", "Eta"])

    seg_grp.head()

    hb_depths = seg_grp.index.levels[0]
    hb_depths
    hb_ietas = seg_grp.index.levels[1]
    hb_ietas

    np.sort(subdetector_seg_info_df["Eta"].unique())
    iphi = np.sort(subdetector_seg_info_df["Phi"].unique())
    iphi

    hcal_subdetector_seg_mask = np.zeros((ieta_len, iphi_len, depth_len))
    for depth in hb_depths:
        for ieta in hb_ietas:
            try:
                iphi = np.array(list(seg_grp.loc[((depth, ieta)), ("Phi", "set")]))
                hcal_subdetector_seg_mask[(util.segidx_arrayidx_mapper(-ieta, axis="ieta", istoarray=True), 
                                    util.segidx_arrayidx_mapper(ieta, axis="ieta", istoarray=True)),
                                    util.segidx_arrayidx_mapper(iphi, axis="iphi", istoarray=True)[:, np.newaxis], 
                                    util.segidx_arrayidx_mapper(depth, axis="depth", istoarray=True)] = 1
            except Exception as ex:
                print("non he segement: depth={}, ieta={}, flag: {}".format(depth, ieta, ex))
    print(f"mask: {hcal_subdetector_seg_mask.shape}")

    hcal_subdetector_seg_mask.shape
    # removing top iphi as no data after data loading emap adjusting
    hcal_subdetector_seg_mask[:, -1, :] = False
    # remove from mask the zero data segement
    # hcal_subdetector_seg_mask[np.array([14, 49]), :, 0] = False

    return hcal_subdetector_seg_mask

def plot_subdetector_seg_mask(hcal_subdetector_seg_mask, filepath=None, issave=True):         

    print("plot_subdetector_seg_mask...")     

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    x,y,z = hcal_subdetector_seg_mask.transpose(0, 2, 1).nonzero()
    ax_3d = ax.scatter(x, y, z, c="red", marker=".")
    ax.set_zlabel("iphi")
    ax.set_ylabel("depth")
    ax.set_xlabel("ieta")
    # cbar = plt.colorbar(ax_3d)

    util.save_figure(filepath, fig, isshow=False, issave=issave, dpi=100)

def save_seg_mask(subdetector_name, filedirpath, hcal_subdetector_seg_mask, isemap_adjust=False):

    print("save_seg_mask...")

    # hcal_subdetector_seg_mask.shape
    if isemap_adjust:
        # removing top iphi as no data after data in dqm_util lding shifting
        hcal_subdetector_seg_mask[:, -1, :] = False
        # remove from mask the zero data segement
        hcal_subdetector_seg_mask[np.array([14, 49]), :, 0] = False

    util.save_npdata(rf"{filedirpath}\{subdetector_name}_segmentation_config_mask", hcal_subdetector_seg_mask)
    plot_subdetector_seg_mask(hcal_subdetector_seg_mask, filepath=rf"{filedirpath}\{subdetector_name}_segmentation_config_mask.jpg", issave=True)

def generate_mask():

    # subdetector_name = "ho" # not included in hcal_segmentation_info.csv
    # subdetector_name = "hf" # not included in hcal_segmentation_info.csv

    hcal_config_datafilepath_csv = rf"{data_path}\HCAL_CONFIG\hcal_segmentation_info.csv"
    
    subdetector_name = "he"

    hcal_seg_info_df = load_seg_map(hcal_config_datafilepath_csv)

    subdetector_seg_info_df = select_subdetector_seg_map(hcal_seg_info_df, subdetector=subdetector_name)

    hcal_subdetector_seg_mask = subdetector_seg_map_to_seg_mask(subdetector_seg_info_df, ieta_len=64, iphi_len=72, depth_len=7)

    save_seg_mask(subdetector_name, rf"{data_path}\HCAL_CONFIG", hcal_subdetector_seg_mask, isemap_adjust=False)


    subdetector_name = "hb"
    hcal_seg_info_df = load_seg_map(hcal_config_datafilepath_csv)

    subdetector_seg_info_df = select_subdetector_seg_map(hcal_seg_info_df, subdetector=subdetector_name)

    hcal_subdetector_seg_mask = subdetector_seg_map_to_seg_mask(subdetector_seg_info_df, ieta_len=64, iphi_len=72, depth_len=4)

    save_seg_mask(subdetector_name, rf"{data_path}\HCAL_CONFIG", hcal_subdetector_seg_mask, isemap_adjust=False)

if __name__ == '__main__':
    generate_mask()