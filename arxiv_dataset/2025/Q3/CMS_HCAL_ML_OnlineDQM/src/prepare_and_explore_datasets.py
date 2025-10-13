"""
Created on Mon Dec 07 17:33:21 2023

@author: Mulugeta W. Asres, mulugetawa@uia.no

DESMOD: Working Data Sets Preparation for HCAL DQM Digioccupancy 

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
from hcalanalyzer_root_to_numpy import rootfile_loader


hcal_digimap_3d_meta_template = util.load_json(rf"{data_path}\hcal_digimap_3d_meta_template.json")

## Prepare Data sets

class HcalDQMProfile():
    
    def __init__(self, sys_sel:list="hbhe", collision_year=2022, mask=None, **kwargs) -> None:
        self.sys_sel = sys_sel
        self.collision_year = collision_year
        self.mask = ~mask if mask is not None else mask

        if self.collision_year < 2022:
            raise "collision_year is not >= 2018 for segment_region emap."
        
        if self.sys_sel == "he":
            self.segment_region = kwargs.get("segment_region", {"ieta": np.arange(
                16, 30), "iphi": np.arange(1, 73), "depth": np.arange(1, 8)})
            
        elif self.sys_sel == "hb":
            self.segment_region = kwargs.get("segment_region",  {"ieta": np.arange(
                1, 16), "iphi": np.arange(1, 73), "depth": np.arange(1, 5)})
            
        elif self.sys_sel == "hbhe":
            self.segment_region = kwargs.get("segment_region",  {"ieta": np.arange(
                1, 30), "iphi": np.arange(1, 73), "depth": np.arange(1, 8)})
        
        elif self.sys_sel == "ho":
            raise NotImplementedError
            
        elif self.sys_sel == "hf":
          raise NotImplementedError
            
        else:
            self.segment_region = kwargs.get("segment_region",  {"ieta": np.arange(
                1, 33), "iphi": np.arange(1, 73), "depth": np.arange(1, 8)})

    def TH3_to_Emap_adjust(self, TH3Obj_np):
        ieta_idx = TH3Obj_np.shape[1]//2
        TH3Obj_np[:, ieta_idx:-1, :] = TH3Obj_np[:, ieta_idx+1:, :]
        TH3Obj_np[:, :, :-1] = TH3Obj_np[:, :, 1:]
        TH3Obj_np[:, :, -1] = 0
        return TH3Obj_np

    def  get_segmented_data(self, TH3Obj_np, **kwargs):
        self.isemap_th3_adjust = kwargs.pop("isemap_th3_adjust", True)

        if self.isemap_th3_adjust:
             TH3Obj_np = self.TH3_to_Emap_adjust(TH3Obj_np)
        
        if self.mask is not None:
            print('mask: ', self.mask.shape)
            print('TH3Obj_np: ', TH3Obj_np.shape)

            print("masking...")
            TH3Obj_np = TH3Obj_np.squeeze(-1)
            TH3Obj_np = TH3Obj_np[:, :self.mask.shape[0], :self.mask.shape[1], :self.mask.shape[2],...]
            TH3Obj_np[:, self.mask] = 0

            TH3Obj_np = np.expand_dims(TH3Obj_np, -1)
            print('TH3Obj_np: ', TH3Obj_np.shape)
        
        # else:
        #     pass
            # segment_region_sel = util.copy.deepcopy(self.segment_region)
            # ieta_len, iphi_len = TH3Obj_np.shape[1:]
            # sel_sgement = np.zeros((ieta_len, iphi_len))

            # sel_sgement[util.segidx_arrayidx_mapper(-segment_region_sel["ieta"], axis="ieta", istoarray=True).tolist() + 
            #             util.segidx_arrayidx_mapper(segment_region_sel["ieta"], axis="ieta", istoarray=True).tolist(), 
            #             util.segidx_arrayidx_mapper(
            #     segment_region_sel["iphi"], axis="iphi", istoarray=True)[:, np.newaxis]] = 1
            # mask = ~sel_sgement.astype(bool)
            # TH3Obj_np[:, mask] = 0
        
        print('TH3Obj_np: ', TH3Obj_np.shape)

        return TH3Obj_np

### utils
def digi_precleaner(TH3Obj_np, run_ls_lumi_df):
    print("digi_precleaner...")
    # remove noisy maps such LHC nonlinearity---e.g. very low luminosiy
    return TH3Obj_np

def slice_lumisections_range(data_input, ls_range:list, valid_ls_values:list=None):
    '''
    data_input: multi-dim np.ndarray of maps, 1D list of lumisections

    '''
    print("slice_lumisections_range...", ls_range)
    print(type(data_input))

    if ls_range is None:
        return data_input
    
    if isinstance(data_input, np.ndarray) and (data_input.ndim > 1):
        # input is array of maps for consecutive lumisections

        ls_sel_ranges = [ls_range[0], data_input.shape[0] + ls_range[1] if ls_range[1] < 0 else ls_range[1]]

        if valid_ls_values is not None:
            ls_sel = np.array([ls for ls in np.arange(ls_sel_ranges[0], ls_sel_ranges[1]+1) if ls in valid_ls_values])
        else:
            ls_sel = np.arange(ls_sel_ranges[0], ls_sel_ranges[1]+1)

        return data_input[ls_sel], ls_sel
    
    elif isinstance(data_input, list):
        # input is list of lumisection ids

        ls_sel_ranges = [ls_range[0], np.max(data_input) + ls_range[1] if ls_range[1] < 0 else ls_range[1]]
        
        if valid_ls_values is not None:
            ls_sel = np.array([ls for ls in np.arange(ls_sel_ranges[0], ls_sel_ranges[1]+1) if ls in valid_ls_values])
        else:
            ls_sel = np.arange(ls_sel_ranges[0], ls_sel_ranges[1]+1)

        return data_input[ls_sel], ls_sel
    else:
        raise TypeError

def slice_digimaps_lumisections(TH3Obj_np, ls_range, run_ls_lumi_df=None, run_setting_vars=["NumEvents"]):
    print("slice_digimaps_lumisections...")
    valid_ls_values = run_ls_lumi_df.index.values.tolist()
    TH3Obj_np_sliced, ls_sel_ = slice_lumisections_range(TH3Obj_np, ls_range, valid_ls_values=valid_ls_values)
    print(TH3Obj_np_sliced.shape, (ls_sel_[0], ls_sel_[-1]))
    run_ls_lumi_df_sliced = run_ls_lumi_df.loc[ls_sel_, run_setting_vars]
    return TH3Obj_np_sliced, run_ls_lumi_df_sliced

def digi_renormalizer(TH3Obj_np, normalizer_ls):
    print("digi_renormalizer...")
    print("TH3Obj_np: ", TH3Obj_np.shape)
    print("normalizer_ls: ", normalizer_ls.shape)
    normalizer_ls[normalizer_ls == 0] = 1
    # TH3Obj_np_norm = torch.div(TH3Obj_np, normalizer_ls.unsqueeze(
    #     1).unsqueeze(1).unsqueeze(1).unsqueeze(1))
    TH3Obj_np_norm = np.divide(TH3Obj_np, normalizer_ls[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
    print("TH3Obj_np_norm: ", TH3Obj_np_norm.shape)
    print("total_digi: ", TH3Obj_np_norm.sum())

    return TH3Obj_np_norm

### Master Dataset
def prepare_npmaster_dataset_from_root(root_filedirpath, np_filedirpath, sys_sel="hbhe", runids=[], depth_size=7, subdetector_mask=None):
    print("prepare_npmaster_dataset_from_root...")

    subdetectors = ["he", "hb", "hbhe", "hf", "ho"]
    assert sys_sel in subdetectors, f"sys_sel name is not found in {subdetectors}."

    depth_size = subdetector_mask.shape[-1] if depth_size is None and subdetector_mask is not None else depth_size

    for root_id in tqdm(runids):
        filepath = rf"{root_filedirpath}\Run{root_id}\output__depth.root"
        TH3Obj_np, run_ls_lumi_df, meta_data = rootfile_loader(filepath, depth_size=depth_size, isrun_setting=True, isplot=False)

        hcaldqmObj = HcalDQMProfile(sys_sel=sys_sel, collision_year=2022, mask=subdetector_mask)
        if sys_sel not in ["hbhe", None]:
            TH3Obj_np = hcaldqmObj.get_segmented_data(TH3Obj_np, isemap_th3_adjust=True)

        dst_filedirpath = rf"{np_filedirpath}\{sys_sel}_master_npdataset\Run{root_id}"
        if not os.path.exists(dst_filedirpath):
            os.makedirs(dst_filedirpath)

        # break
        util.save_npdata(rf"{dst_filedirpath}\output__depth.npy", TH3Obj_np)
        util.save_npdata(rf"{dst_filedirpath}\run_ls_lumi_df.npy", run_ls_lumi_df.reset_index(drop=False).to_records(index=False))
        util.save_json(rf"{dst_filedirpath}\meta_data.json", meta_data)
    
    print("done!")

### Explore Dataset
def prepare_explore_datastet(np_filedirpath, sys_sel="hbhe", runids=[], ls_range=[1, -1]):
    print("prepare_explore_datastet...")

    explore_meta_data = {"ls_range_sel": ls_range, "runids_sel": runids}
    explore_data_list = []
    explore_data_summary_list = []
    run_profile_dict = {}
    for root_id in tqdm(runids):
        run_filedirpath = rf"{np_filedirpath}\{sys_sel}_master_npdataset\Run{root_id}"
        
        meta_data = util.load_json(rf"{run_filedirpath}\meta_data.json")
        if (ls_range[0] < meta_data["xaxis"]["valued_range"][0]) or (ls_range[1] > meta_data["xaxis"]["valued_range"][1]):
            continue

        TH3Obj_np = util.load_npdata(rf"{run_filedirpath}\output__depth.npy")
        print(TH3Obj_np.shape)
        run_ls_lumi_df = util.load_npdata(rf"{run_filedirpath}\run_ls_lumi_df.npy")
        run_ls_lumi_df = pd.DataFrame.from_records(run_ls_lumi_df).set_index("ls")
        TH3Obj_np, run_ls_lumi_df = slice_digimaps_lumisections(TH3Obj_np, ls_range, run_ls_lumi_df=run_ls_lumi_df, run_setting_vars=["NumEvents"])
        
        TH3Obj_np = digi_precleaner(TH3Obj_np, run_ls_lumi_df)
        TH3Obj_np_tot = TH3Obj_np.sum(axis=(1,2,3,4))
        TH3Obj_np_tot.shape
        run_ls_lumi_df["Total_DigiOccupancy"] = TH3Obj_np_tot
        run_ls_lumi_df["Max_DigiOccupancy"] = TH3Obj_np.max(axis=(1,2,3,4))
        run_ls_lumi_df["RunId"] = root_id
        run_ls_lumi_df.head()
        explore_data_list.append(TH3Obj_np)
        explore_data_summary_list.append(run_ls_lumi_df.reset_index(drop=False))

        run_profile_dict[f"Run{root_id}"] = {"ls_range": [run_ls_lumi_df.index[0], run_ls_lumi_df.index[-1]], "data_size": TH3Obj_np.shape}

    explore_data = np.concatenate(tuple(explore_data_list), axis=0)
    explore_data.shape
    explore_meta_data["explore_data_size"] = explore_data.shape

    explore_data_summary = pd.concat(explore_data_summary_list, axis=0, ignore_index=True)
    explore_data_summary.shape

    explore_meta_data.update(run_profile_dict)
    explore_meta_data

    dst_filedirpath = rf"{np_filedirpath}\{sys_sel}_explore_dataset"
    if not os.path.exists(dst_filedirpath):
        os.makedirs(dst_filedirpath)
    util.save_npdata(rf"{dst_filedirpath}\explore_data.npy", explore_data)
    util.save_json(rf"{dst_filedirpath}\explore_meta_data.json", explore_meta_data)
    util.save_csv(rf"{dst_filedirpath}\explore_data_summary.csv", explore_data_summary)

    print("done!")
    return dst_filedirpath

### Visualization
def dataset_visualization(data_filedirpath, ls_sel=5):
    print("dataset_visualization...")

    # need "hcal_digimap_3d_meta_template" for plotting purpose to reformat the axes ieta, iphi and depth
    explore_data = util.load_npdata(rf"{data_filedirpath}\explore_data.npy")
    explore_data.shape
    explore_data_summary = util.load_csv(rf"{data_filedirpath}\explore_data_summary.csv", index_col=0)
    explore_data_summary.shape
    explore_data_summary.head()
    explore_meta_data = util.load_json(rf"{data_filedirpath}\\explore_meta_data.json")
    explore_meta_data

    # ls_sel = 5
    for d in range(explore_data.shape[-2]):
        util.plot_2d_heatmap(explore_data[ls_sel, :, :, d, 0], 
                            axes_ticks=[hcal_digimap_3d_meta_template["xaxis"]["values"], hcal_digimap_3d_meta_template["yaxis"]["values"]], 
                            figsize=(4, 4), 
                            mask="positive", 
                            axes_labels=["ieta", "iphi"])
        # break


    util.plot_hist3d_heatmap(explore_data[ls_sel, :, :, :, 0].transpose((0, 2, 1)), 
                            axes_ticks=[hcal_digimap_3d_meta_template["xaxis"]["values"], hcal_digimap_3d_meta_template["zaxis"]["values"], hcal_digimap_3d_meta_template["yaxis"]["values"]],
                            axes_labels=["ieta", "depth", "iphi"],
                            figsize=(6, 6), 
                            mask="positive", 
                            )

    util.plot_hist3d_heatmap(np.sum(explore_data[:100, :, :, :, 0], axis=(3)).transpose((1, 0, 2)), 
                            axes_ticks=[hcal_digimap_3d_meta_template["xaxis"]["values"], np.arange(explore_data.shape[0]), hcal_digimap_3d_meta_template["yaxis"]["values"]],
                            axes_labels=["ieta", "ls", "iphi"],
                            figsize=(6, 6), 
                            mask="positive", 
                            )

    explore_data_summary.columns

    fig, ax = plt.subplots(figsize=(10, 4))
    _ = sns.lineplot(ax=ax, data=explore_data_summary.iloc[:-1, :], x="ls", hue="RunId", palette="tab10", y='Total_DigiOccupancy')
    fig, ax = plt.subplots(figsize=(10, 4))
    _ = sns.lineplot(ax=ax, data=explore_data_summary.iloc[:-1, :], x="ls", hue="RunId", palette="tab10", y='NumEvents')
    fig, ax = plt.subplots(figsize=(10, 4))
    _ = sns.lineplot(ax=ax, data=explore_data_summary.iloc[:-1, :], x="ls", hue="RunId", palette="tab10", y='Max_DigiOccupancy')


    value_var = "Total_DigiOccupancy"
    # ls_len = 100
    # print("show the first {ls_len}")
    # plot_df = explore_data_summary.loc[(explore_data_summary[value_var]>0)].dropna(axis=0).iloc[:ls_len]

    plot_df = explore_data_summary.loc[(explore_data_summary[value_var]>0)].dropna(axis=0)

    fig = plt.figure(figsize=(10, 4), constrained_layout=True)
    axs = fig.subplots(1, 2)
    linewidth=0.1
    alpha=0.8
    size=None
    legend=True
    hue="RunId"
    with sns.axes_style("darkgrid"):

        run_vars = ['Total_DigiOccupancy', 'NumEvents']
        i=0
        axs[i] = sns.scatterplot(x=run_vars[1], y=run_vars[0], 
                    linewidth=linewidth, alpha=alpha, hue=hue, size=size, data=plot_df, ax=axs[i], 
                                palette="tab10", 
                                legend=legend)
        _ = axs[i].set_ylabel(run_vars[0])   
        _ = axs[i].set_xlabel(run_vars[1])

        run_vars = ['Max_DigiOccupancy', 'NumEvents']
        i = i + 1
        axs[i].plot(plot_df[run_vars[1]], plot_df[run_vars[1]], c="red", linewidth=0.5, label="NumEvents (Regression Line)")
        axs[i] = sns.scatterplot(x=run_vars[1], y=run_vars[0], 
                    linewidth=linewidth, alpha=alpha, hue=hue, size=size, data=plot_df, ax=axs[i], 
                                palette="tab10", 
                                legend=legend)
        
        _ = axs[i].set_ylabel(run_vars[0])   
        _ = axs[i].set_xlabel(run_vars[1])
        _ = axs[i].legend()

    fig.show()

    print("done!")

### Training and Testing Data sets
def prepare_modeling_dataset(np_filedirpath, sys_sel="hbhe", ls_range=[1, 500], runids=[], isnormalize=True, datanametag="train"):
    """
    data_filedirpath = rf"{data_path}\HBHE\train_dataset"
    train_data = util.load_npdata(rf"{data_filedirpath}\train_data.npy")
    train_data.shape
    train_meta_data = util.load_json(rf"{data_filedirpath}\\train_meta_data.json")
    train_meta_data

    ls_sel = 5
    for d in range(train_data.shape[-2]):
        util.plot_2d_heatmap(train_data[ls_sel, :, :, d, 0], 
                            axes_ticks=[hcal_digimap_3d_meta_template["xaxis"]["values"], hcal_digimap_3d_meta_template["yaxis"]["values"]], 
                            figsize=(4, 4), 
                            mask="positive", 
                            axes_labels=["ieta", "iphi"])

    util.plot_hist3d_heatmap(train_data[ls_sel, :, :, :, 0].transpose((0, 2, 1)), 
                         axes_ticks=[hcal_digimap_3d_meta_template["xaxis"]["values"], hcal_digimap_3d_meta_template["zaxis"]["values"], hcal_digimap_3d_meta_template["yaxis"]["values"]],
                         axes_labels=["ieta", "depth", "iphi"],
                         figsize=(6, 6), 
                         mask="positive", 
                        )

    """
    print("prepare_modeling_dataset...")
    model_data_list = []
    meta_data = {"ls_range_sel": ls_range, "runids_sel": runids}
    run_profile_dict = {}
    for root_id in runids:
        run_filedirpath = rf"{np_filedirpath}\{sys_sel}_master_npdataset\Run{root_id}"
        meta_data = util.load_json(rf"{run_filedirpath}\meta_data.json")

        if (ls_range[0] < meta_data["xaxis"]["valued_range"][0]) or (ls_range[1] > meta_data["xaxis"]["valued_range"][1]):
            continue

        TH3Obj_np = util.load_npdata(rf"{run_filedirpath}\output__depth.npy")
        print(TH3Obj_np.shape)
        run_ls_lumi_df = util.load_npdata(rf"{run_filedirpath}\run_ls_lumi_df.npy")
        run_ls_lumi_df = pd.DataFrame.from_records(run_ls_lumi_df).set_index("ls")
        TH3Obj_np, run_ls_lumi_df = slice_digimaps_lumisections(TH3Obj_np, ls_range, run_ls_lumi_df=run_ls_lumi_df, run_setting_vars=["NumEvents"])
        
        TH3Obj_np = digi_precleaner(TH3Obj_np, run_ls_lumi_df)
        TH3Obj_np_norm = digi_renormalizer(TH3Obj_np, run_ls_lumi_df["NumEvents"].values.T) if isnormalize else TH3Obj_np
        TH3Obj_np_norm.shape

        model_data_list.append(TH3Obj_np_norm)

        run_profile_dict[f"Run{root_id}"] = {"ls_range": [run_ls_lumi_df.index[0], run_ls_lumi_df.index[-1]], "data_size": TH3Obj_np_norm.shape}

    model_data = np.concatenate(tuple(model_data_list), axis=0)
    model_data.shape
    meta_data["model_data_size"] = model_data.shape

    meta_data.update(run_profile_dict)
    print(meta_data)

    dst_filedirpath = rf"{np_filedirpath}\{sys_sel}_{datanametag}_dataset"
    if not os.path.exists(dst_filedirpath):
        os.makedirs(dst_filedirpath)
    util.save_npdata(rf"{dst_filedirpath}\{datanametag}_data.npy", model_data)
    util.save_json(rf"{dst_filedirpath}\{datanametag}_meta_data.json", meta_data)

    print("done!")
    return dst_filedirpath

# #### Visualization

if __name__ == '__main__':

    # load subdetector mask
    subdetector_name = "hbhe"
    try:
        subdetector_mask = util.load_npdata(rf"{data_path}\HCAL_CONFIG\{subdetector_name}_segmentation_config_mask.npy").astype(bool)
        print(f"hcal segmentation map is loaded: {subdetector_mask.shape}")
    except Exception as ex:
        print(ex)
        subdetector_mask = None

    root_filedirpath = rf"{data_path}\HBHE\raw_roots"
    np_filedirpath = rf"{data_path}\HBHE"

    # runids = [355456, 355680, 356077, 356381, 356615, 357112, 357271, 357329, 357442, 360820, 361240, 361957, 362091, 362696, 362760]
    runids = [355456, 355680]

    prepare_npmaster_dataset_from_root(root_filedirpath, np_filedirpath, sys_sel=subdetector_name, runids=runids, depth_size=7, subdetector_mask=subdetector_mask)

    explore_filedirpath = prepare_explore_datastet(np_filedirpath, sys_sel=subdetector_name, runids=runids, ls_range=[1, -1])

    # visualization
    dataset_visualization(explore_filedirpath, ls_sel=5)
    
    # train_runids = [355456, 355680, 356077, 356381, 356615, 357112, 357271, 357329, 357442, 360820, 361240, 361957, 362091, 362696, 362760]
    train_runids = runids
    train_dataset_filedirpath = prepare_modeling_dataset(np_filedirpath, sys_sel=subdetector_name, ls_range=[1, 500], runids=train_runids, isnormalize=True, datanametag="train")

    # test_runids = [355456, 355680, 356077, 356381, 356615, 357112, 357271, 357329, 357442, 360820, 361240, 361957, 362091, 362696, 362760]
    test_runids = runids
    test_dataset_filedirpath = prepare_modeling_dataset(np_filedirpath, sys_sel=subdetector_name, ls_range=[501, -1], runids=test_runids, isnormalize=True, datanametag="test")


    # visualization
    # test_dataset_filedirpath = rf"{data_path}\HBHE\he_train_dataset"
    train_data = util.load_npdata(rf"{test_dataset_filedirpath}\train_data.npy")
    train_data.shape
    train_meta_data = util.load_json(rf"{test_dataset_filedirpath}\train_meta_data.json")
    train_meta_data

    ls_sel = 5
    for d in range(train_data.shape[-2]):
        util.plot_2d_heatmap(train_data[ls_sel, :, :, d, 0], 
                            axes_ticks=[hcal_digimap_3d_meta_template["xaxis"]["values"], hcal_digimap_3d_meta_template["yaxis"]["values"]], 
                            figsize=(4, 4), 
                            mask="positive", 
                            axes_labels=["ieta", "iphi"])

    util.plot_hist3d_heatmap(train_data[ls_sel, :, :, :, 0].transpose((0, 2, 1)), 
                         axes_ticks=[hcal_digimap_3d_meta_template["xaxis"]["values"], hcal_digimap_3d_meta_template["zaxis"]["values"], hcal_digimap_3d_meta_template["yaxis"]["values"]],
                         axes_labels=["ieta", "depth", "iphi"],
                         figsize=(6, 6), 
                         mask="positive", 
                        )