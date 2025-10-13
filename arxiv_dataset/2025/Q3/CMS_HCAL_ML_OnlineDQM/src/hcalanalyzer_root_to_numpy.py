"""
Created on Mon Dec 07 17:33:21 2023

@author: Mulugeta W. Asres, mulugetawa@uia.no

DESMOD: Root TH3 file (generated using HcalAnalyzer) to numpy conversion for the ML HCAL DQM Digioccupancy 

Example: given in main function

"""

import uproot # tested with uproot-5.0.2
import numpy as np, pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable


class TH3UpRootHandler():

    def __init__(self, dqm_variable="QIEDigiOccupancy", run_profile={"id": None, "year": None, "pd": "ZeroBias"}) -> None:
        self.root_loader = "uproot"
        self.run_profile = run_profile
        self.filepath = None
        self.tree_hist_tag = None
        self.meta = {}
        self.meta["xaxis"] = {}
        self.meta["yaxis"] = {}
        self.meta["zaxis"] = {}
        self.meta["dqm_variable"] = dqm_variable

    def load_TFiledata(self, filepath) -> None:
        self.TFileObj, self.filepath = uproot.open(filepath), filepath 

    def extract_TH(self, tree_hist_tag="HcalDigiAnalyzer/hist3D") -> None:
        '''
        TH3Obj : TH3 object
        tree_hist_tag: tree path of the target TH3I node
        extraction of axes info ls x ieta x iphi
        3D hist root into 3D numpy conversion
        '''

        self.TH3Obj = self.TFileObj[tree_hist_tag]

        self.meta["xaxis"]["label"] = "ls"
        self.meta["yaxis"]["label"] = "ieta"
        self.meta["zaxis"]["label"] = "iphi"

        # extract hist dim info
        self.xaxis_N = self.TH3Obj.axis("x").edges().shape[0]
        self.yaxis_N = self.TH3Obj.axis("y").edges().shape[0]
        self.zaxis_N = self.TH3Obj.axis("z").edges().shape[0]
        print("hist dim: {}={}, {}={}, {}={}".format(
            self.meta["xaxis"]["label"], self.xaxis_N, self.meta["yaxis"]["label"], self.yaxis_N, self.meta["zaxis"]["label"], self.zaxis_N))

        self.xaxis_lim = [self.TH3Obj.axis("x").edges().min(),
                          self.TH3Obj.axis("x").edges().max()]
        self.yaxis_lim = [self.TH3Obj.axis("y").edges().min(),
                          self.TH3Obj.axis("y").edges().max()]
        self.zaxis_lim = [self.TH3Obj.axis("z").edges().min(),
                          self.TH3Obj.axis("z").edges().max()]

        print("hist axis index limits: {}={}, {}={}, {}={}".format(
            self.meta["xaxis"]["label"], self.xaxis_lim, self.meta["yaxis"]["label"], self.yaxis_lim, self.meta["zaxis"]["label"], self.zaxis_lim))

        # ls
        self.meta["xaxis"]["values"] = self.TH3Obj.axis("x").edges().astype(int)

        # ieta
        self.meta["yaxis"]["values"] = self.TH3Obj.axis("y").edges().astype(int)
        self.meta["yaxis"]["values"] = self.meta["yaxis"]["values"][self.meta["yaxis"]["values"] !=0]
        
        # iphi
        self.meta["zaxis"]["values"] = self.TH3Obj.axis("z").edges().astype(int)[1:]

    def convert_TH_to_numpy(self):
        # get numpy values
        self.TH3Obj_np = self.TH3Obj.values()
        
        # remove empty ls at the end
        xaxis_3d_idx, yaxis_3d_idx, zaxis_3d_idx = self.TH3Obj_np.nonzero()

        self.meta["xaxis"]["valued_range"] = [
            xaxis_3d_idx.min(), xaxis_3d_idx.max()]
        self.meta["yaxis"]["valued_range"] = [
            yaxis_3d_idx.min(), yaxis_3d_idx.max()]
        self.meta["zaxis"]["valued_range"] = [
            zaxis_3d_idx.min(), zaxis_3d_idx.max()]

    def plot_TH3_np(self, figsize=(6, 8), title=None, **kwargs) -> None:
        cmap = kwargs.pop("cmap", "Spectral_r")
        xaxis_3d_idx, yaxis_3d_idx, zaxis_3d_idx = self.TH3Obj_np.nonzero()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax_3d = ax.scatter(self.meta["yaxis"]["values"][yaxis_3d_idx],
                           self.meta["xaxis"]["values"][xaxis_3d_idx],  self.meta["zaxis"]["values"][zaxis_3d_idx], c=self.TH3Obj_np[xaxis_3d_idx, yaxis_3d_idx, zaxis_3d_idx], marker=".", cmap=cmap)
       
        cax = fig.add_axes([ax.get_position().x0,
                        ax.get_position().y0-0.04, ax.get_position().width, 0.02])
        cbar = plt.colorbar(ax_3d, orientation="horizontal", cax=cax, pad=1)
        cbar.set_label(self.meta["dqm_variable"])
        
        ax.set_xlabel(self.meta["yaxis"]["label"])
        ax.set_ylabel(self.meta["xaxis"]["label"])
        ax.set_zlabel(self.meta["zaxis"]["label"])

        ax.set_title("{}_{}".format(title, self.meta["dqm_variable"]))
        fig.show()
    
    def get_th3_numpy(self):
        return self.TH3Obj_np[:self.meta["xaxis"]["valued_range"][-1]+1]

    def get_meta_data(self):
        meta_data = self.meta.copy()
        meta_data["xaxis"]["values"] = meta_data["xaxis"]["values"][:meta_data["xaxis"]["valued_range"][-1]+1]
        return meta_data


def root_th3_with_lumi_to_numpy_convert(TH3RootObj,
                tree_hist_tag="Hcal4DQMAnalyzer/hist3D", depth_size=7, **kwargs):

    print("root_th3_with_lumi_to_numpy_convert...", tree_hist_tag)
    th_dim = kwargs.get("th_dim", 3)
    isplot = kwargs.get("isplot", False)

    if "evttree" in tree_hist_tag:
        # "Hcal4DQMAnalyzer/evttree/LumiSec" for event data
        tree_var = tree_hist_tag.split("/")[-1]
        tree_hist_tag = "/".join(tree_hist_tag.split("/")[:-1])
        TH1Obj = TH3RootObj.TFileObj[tree_hist_tag]
        data = np.array(TH1Obj.arrays()[tree_var])
       
        # bin-method-2
        # similar as root plot
        ls = np.unique(data)
        ls = np.concatenate((ls, ls[-1:]+1))
        num_event_np = np.histogram(data, bins=ls)
        num_event_df = pd.Series(
            num_event_np[0], index=num_event_np[1][:-1].astype(int)).to_frame().reset_index(drop=False)

        num_event_df.columns = ["LS#", "NumEvents"]
        print(num_event_df.shape)
        if isplot and ("LumiSec" in tree_var):
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(num_event_df["LS#"], num_event_df["NumEvents"], color='g')
            plt.title("Number of Events")
            plt.xlabel("ls")
            plt.ylabel("value")
            plt.show()

        return num_event_df
    
    elif "hist3D" in tree_hist_tag:
        # "HcalDigiAnalyzer/hist3D" for digioccupancy data
        
        if th_dim == 2:
            # "Hcal4DQMAnalyzer/hist3D" for hist data 
            TH3RootObj.extract_TH(tree_hist_tag=tree_hist_tag)
            TH3RootObj.convert_TH_to_numpy()
            if isplot:
                TH3RootObj.plot_TH3_np(title="Hist3D")
            TH3Obj_np = TH3RootObj.get_th3_numpy()

        elif th_dim == 3:
            # "Hcal4DQMAnalyzer/hist3D_depthX" for hist data 

            roots_per_depth_list = []
            for depth in range(1, depth_size+1):
                print("{}_depth{}".format(tree_hist_tag, depth))
                TH3RootObj.extract_TH(
                    tree_hist_tag="{}_depth{}".format(tree_hist_tag, depth))
                TH3RootObj.convert_TH_to_numpy()
                if isplot:
                    TH3RootObj.plot_TH3_np(title="Hist3D_Depth_{}".format(depth))
                TH3Obj_np_ = TH3RootObj.get_th3_numpy()
                print(TH3Obj_np_.shape)
                roots_per_depth_list.append(np.expand_dims(TH3Obj_np_, axis=3))
                # break
            TH3Obj_np = np.concatenate(tuple(roots_per_depth_list), axis=3) 
        else:
            raise "Undefined tree_hist_tag: {}. please select [HcalDigiAnalyzer/hist3D or Hcal4DQMAnalyzer/evttree/LumiSec].".format(tree_hist_tag)

        print(TH3Obj_np.shape)
        return TH3Obj_np
    

def rootfile_loader(filepath, depth_size=7, isrun_setting=True, **kwargs):
    print("rootfile_loader..., ", filepath)

    try:
        TH3RootObj = TH3UpRootHandler()
        TH3RootObj.load_TFiledata(filepath=filepath)

        # DQM TH3 histogram to numpy conversion
        tree_hist_tag = "Hcal4DQMAnalyzer/hist3D"
        TH3Obj_np = root_th3_with_lumi_to_numpy_convert(TH3RootObj, tree_hist_tag=tree_hist_tag, depth_size=depth_size, th_dim=3, **kwargs)
        meta_data = TH3RootObj.get_meta_data()

        # evttree/LumiSec
        tree_hist_tag = "Hcal4DQMAnalyzer/evttree/LumiSec"  # for event data
        run_ls_lumi_df = root_th3_with_lumi_to_numpy_convert(TH3RootObj, tree_hist_tag=tree_hist_tag, depth_size=depth_size, th_dim=3, **kwargs)

        # evttree/RunNum
        tree_hist_tag = "Hcal4DQMAnalyzer/evttree/RunNum"  # get runId
        run_num_df = root_th3_with_lumi_to_numpy_convert(TH3RootObj, tree_hist_tag=tree_hist_tag, depth_size=depth_size, th_dim=3, **kwargs)

        meta_data["RunId"] = run_num_df.iloc[0, 0]

        meta_data["xaxis"]["values"] = meta_data["xaxis"]["values"].astype(np.int16)
        meta_data["yaxis"]["values"] = meta_data["yaxis"]["values"].astype(np.int8)
        meta_data["zaxis"]["values"] = meta_data["zaxis"]["values"].astype(np.int8)

        if isrun_setting:
            run_ls_lumi_df.set_index("LS#", inplace=True)
            run_ls_lumi_df.index.name = "ls"

        TH3Obj_np = np.expand_dims(TH3Obj_np, axis=TH3Obj_np.ndim)

    except Exception as ex:
        raise ex

    print("#"*100)
    print("CONVERSION SUMMARY REPORT")
    print("RunId: ", meta_data["RunId"])
    print("Digioccupancy data size: ", TH3Obj_np.shape)
    print("Run setting sata size: ", run_ls_lumi_df.shape)
    print("Run setting sata size: ", run_ls_lumi_df.head())
    print("Meta data: ", meta_data)
    print("#"*100)
    
    return TH3Obj_np, run_ls_lumi_df, meta_data


if __name__ == '__main__':
    # Example
    # set file path
    filepath = r"..\data\HBHE\Run355456__hehb_cut20fc__depth.root" # change path to your local path

    # without plots, faster processing
    TH3Obj_np, run_ls_lumi_df, meta_data = rootfile_loader(filepath, depth_size=7, isrun_setting=True, isplot=False)

    # or with plots
    # TH3Obj_np, run_ls_lumi_df, meta_data = rootfile_loader(filepath, depth_size=7, isrun_setting=True, isplot=True) # change depth_size according to the included number of hist3d_depthX histograms in the root file