# runs on conda dqm_py38_x86 envs
# from rootfile_handler_utils import *
import uproot
# import ROOT # cern vm centos 7 installation issue due to
import matplotlib.pyplot as plt
import matplotlib
from operator import index
import sys
import os
import psutil
import gc
import json
import pickle
import re
import copy
import io
# from bokeh.models.markers import Marker
from numpy.core import numeric
import numpy as np
import pandas as pd
import itertools
import functools
import datetime
import time
from tqdm import tqdm

import seaborn as sns
sns.set()

# import torch
current_path = os.path.abspath(os.path.dirname(__file__))
print("current_path: ", current_path)
main_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))




current_path = os.path.abspath(os.path.dirname(__file__))
data_path = "{}//data//".format(os.path.dirname(current_path))
dataset = "HCAL_ONLINE_DQM//DIGI_OCCUPANCY_CUT"
root_dirpath = data_path + "//" + dataset
load_dir = root_dirpath
save_dir = root_dirpath


class TH3RootHandler():

    def __init__(self, dqm_variable="QIE11DigiOccupancy", run_profile={"id": None, "year": None, "pd": "ZeroBias"}) -> None:
        self.root_loader = "ROOT"
        self.run_profile = run_profile
        self.filepath = None
        self.tree_hist_tag = None
        self.meta = {}
        self.meta["xaxis"] = {}
        self.meta["yaxis"] = {}
        self.meta["zaxis"] = {}
        self.meta["dqm_variable"] = dqm_variable
        self.data_export_dirpath = r"../data/HCAL_ONLINE_DQM/DIGI_OCCUPANCY_CUT/{}/{}".format(self.run_profile["pd"],
                                                                                              self.run_profile["year"])

    def load_TFiledata(self, filename=None, dirpath=None, filepath=None) -> None:
        self.TFileObj, self.filepath = load_root_file(
            filename=filename, dirpath=dirpath, filepath=filepath)

    def __hist_formatter(self):
        if "digioccupancy" in self.meta["dqm_variable"].lower():
            #  x- lumi section id, y- ieta, z- iphi
            self.TH3Obj = run_digi_hist_formatter(
                self.TH3Obj, dqm_variable=self.meta["dqm_variable"])

        self.meta["xaxis"]["label"] = self.TH3Obj.GetXaxis().GetTitle()
        self.meta["yaxis"]["label"] = self.TH3Obj.GetYaxis().GetTitle()
        self.meta["zaxis"]["label"] = self.TH3Obj.GetZaxis().GetTitle()

    def extract_TH(self, tree_hist_tag="HcalDigiAnalyzer/hist3D") -> None:
        '''
        TH3Obj : TH3 object
        tree_hist_tag: tree path of the target TH3I node
        extraction of axes info ls x ieta x iphi
        3D hist root into 3D numpy conversion
        :
        '''

        self.tree_hist_tag = tree_hist_tag
        self.TH3Obj = hist_extractor_from_rootfile(
            self.TFileObj, tree_hist_tag=self.tree_hist_tag)

        self.__hist_formatter()  # x- lumi section id, y- ieta, z- iphi

        # extract hist dim info
        self.xaxis_N = self.TH3Obj.GetNbinsX()
        self.yaxis_N = self.TH3Obj.GetNbinsY()
        self.zaxis_N = self.TH3Obj.GetNbinsZ()
        print("hist dim: {}={}, {}={}, {}={}".format(
            self.meta["xaxis"]["label"], self.xaxis_N, self.meta["yaxis"]["label"], self.yaxis_N, self.meta["zaxis"]["label"], self.zaxis_N))

        self.xaxis_lim = [self.TH3Obj.GetXaxis().GetXmin(),
                          self.TH3Obj.GetXaxis().GetXmax()]
        self.yaxis_lim = [self.TH3Obj.GetYaxis().GetXmin(),
                          self.TH3Obj.GetYaxis().GetXmax()]
        self.zaxis_lim = [self.TH3Obj.GetZaxis().GetXmin(),
                          self.TH3Obj.GetZaxis().GetXmax()]
        print("hist axis index limits: {}={}, {}={}, {}={}".format(
            self.meta["xaxis"]["label"], self.xaxis_lim, self.meta["yaxis"]["label"], self.yaxis_lim, self.meta["zaxis"]["label"], self.zaxis_lim))

        # ls
        self.meta["xaxis"]["values"] = np.arange(
            self.xaxis_lim[0], self.xaxis_lim[1]+1, np.diff(self.xaxis_lim)//self.xaxis_N).astype(int)
        # ieta
        # self.meta["yaxis"]["values"] = np.array(
        #     tuple(np.arange(self.yaxis_lim[0], 0, np.diff(self.yaxis_lim)//self.yaxis_N).astype(int)) +
        #     tuple(np.arange(1, self.yaxis_lim[1]+1, np.diff(self.yaxis_lim)//self.yaxis_N).astype(int)))
        self.meta["yaxis"]["values"] = np.arange(self.yaxis_lim[0], np.arange(1, self.yaxis_lim[1]+1, np.diff(self.yaxis_lim)//self.yaxis_N).astype(int)) 
        self.meta["yaxis"]["values"] = self.meta["yaxis"]["values"][self.meta["yaxis"]["values"]!=0]
        self.meta["zaxis"]["values"] = np.arange(
            self.zaxis_lim[0]+1, self.zaxis_lim[1]+1, np.diff(self.zaxis_lim)//self.zaxis_N).astype(int)  # iphi 1-72 but data returns 0-72

        # self.ls_arr = np.arange(self.xaxis_lim[0], self.xaxis_lim[1], np.diff(self.xaxis_lim)//self.xaxis_N).astype(int)
        # self.ieta_arr = np.arange(self.yaxis_lim[0], self.yaxis_lim[1], np.diff(self.yaxis_lim)//self.yaxis_N).astype(int)
        # self.iphi_arr = np.arange(self.zaxis_lim[0], self.zaxis_lim[1], np.diff(self.zaxis_lim)//self.zaxis_N).astype(int)

    def convert_TH_to_numpy(self):
        self.TH3Obj_np = convert_run_digihistroot_to_nparray(self.TH3Obj)
        # remove empty ls at the end
        xaxis_3d_idx, yaxis_3d_idx, zaxis_3d_idx = self.TH3Obj_np.nonzero()
        # ls_range = [xaxis_3d_idx.min(), xaxis_3d_idx.max()]
        self.meta["xaxis"]["valued_range"] = [
            xaxis_3d_idx.min(), xaxis_3d_idx.max()]
        self.meta["yaxis"]["valued_range"] = [
            yaxis_3d_idx.min(), yaxis_3d_idx.max()]
        self.meta["zaxis"]["valued_range"] = [
            zaxis_3d_idx.min(), zaxis_3d_idx.max()]

    def get_axis_labels(self) -> list:
        return self.meta["xaxis"]["label"], self.meta["yaxis"]["label"], self.meta["zaxis"]["label"]

    def plot_TH_np(self, value_cut_thr=None, figsize=(6, 8), title=None) -> None:
        print("value_cut_thr: {}".format(value_cut_thr))
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # digi_occpcy_cut_th = 1e-42
        # xaxis_arr = getattr(self, "xaxis_arr".format(self.meta["xaxis"]["label"]))
        # yaxis_arr = getattr(self, "yaxis_arr".format(self.meta["yaxis"]["label"]))
        # zaxis_arr = getattr(self, "zaxis_arr".format(self.meta["zaxis"]["label"]))

        if value_cut_thr:
            xaxis_3d_idx, yaxis_3d_idx, zaxis_3d_idx = (
                self.TH3Obj_np > value_cut_thr).nonzero()  # gives to coordinate of non-zero
        else:
            # gives to coordinate of non-zero reading
            xaxis_3d_idx, yaxis_3d_idx, zaxis_3d_idx = self.TH3Obj_np.nonzero()

        fig = plt.figure(figsize=figsize)
        # ax = Axes3D(fig)
        ax = fig.add_subplot(111, projection='3d')
        ax_3d = ax.scatter(self.meta["xaxis"]["values"][xaxis_3d_idx], self.meta["yaxis"]["values"][yaxis_3d_idx],
                           self.meta["zaxis"]["values"][zaxis_3d_idx], c=self.TH3Obj_np[xaxis_3d_idx, yaxis_3d_idx, zaxis_3d_idx], marker=".")
        # You can change 0.01 to adjust the distance between the main image and the colorbar.
        # You can change 0.02 to adjust the width of the colorbar.
        # cax = fig.add_axes([ax.get_position().x1+.01, ax.get_position().y0, 0.02, ax.get_position().height])
        # cbar=plt.colorbar(ax_3d, cax=cax)
        cbar = plt.colorbar(ax_3d)
        cbar.set_label(self.meta["dqm_variable"])

        ax.set_xlabel(self.meta["xaxis"]["label"])
        ax.set_ylabel(self.meta["yaxis"]["label"])
        ax.set_zlabel(self.meta["zaxis"]["label"])
        ax.set_title("{}_{}".format(title, self.meta["dqm_variable"]))
        fig.show()
        # # clickplotter = OnClickPlotter(ax, data, source, statData)

    def plot_TH_root(self, **kwargs) -> None:
        plot_run_histroot(self.TH3Obj, **kwargs)

    def plot_TH_2d(self, index_axis: str, column_axis: str, target_idx: int, **kwargs):
        '''
        plots 2d hist across any of the axis selections
        index_axis:  'x', 'y' or 'z'
        column_axis:  'x', 'y' or 'z'
        target_idx: index to select the 2d data along the axis not in the index_axis nor column_axis.

        call self.get_axis_labels() to see the labels for [x, y, z]
        # ls_id=10
        e.g. self.plot_TH_2d(index_axis='y', column_axis='z', target_idx:10)
        '''
        print(self.get_axis_labels())

        index = getattr(self, "{}axis_arr".format(index_axis))
        columns = getattr(self, "{}axis_arr".format(column_axis))
        if 'x' not in [index_axis, column_axis]:
            data_slice = self.TH3Obj_np[target_idx]
        elif 'y' not in [index_axis, column_axis]:
            data_slice = self.TH3Obj_np[:, target_idx, :]
        else:
            data_slice = self.TH3Obj_np[:, :, target_idx]

        if index_axis > column_axis:  # check axis consistency
            data_slice = data_slice.T

        hist_2d_df = pd.DataFrame(data_slice, columns=columns, index=index)
        plot_2d_heatmap(hist_2d_df, **kwargs, return_fig=False)

    def prepare_dataset(self, suffix=None, dst_dirpath=None):
        if not dst_dirpath:
            dst_dirpath = self.data_export_dirpath
        print(dst_dirpath)
        create_dir(dst_dirpath)
        meta_data = self.meta.copy()
        meta_data["xaxis"]["values"] = meta_data["xaxis"]["values"][:meta_data["xaxis"]["valued_range"][-1]+1]
        # remove empty ls at the end
        save_npdata("{}__{}__{}".format(self.meta["dqm_variable"], self.run_profile["id"], suffix).strip(
            "__"), self.TH3Obj_np[: meta_data["xaxis"]["valued_range"][-1]+1], filepath=dst_dirpath)
        save_json("{}__{}__{}__meta".format(self.meta["dqm_variable"], self.run_profile["id"], suffix).strip(
            "__"), self.meta, filepath=dst_dirpath)
    
    def get_th3_numpy(self):
        return self.TH3Obj_np[:self.meta["xaxis"]["valued_range"][-1]+1]

    def get_meta_data(self):
        meta_data = self.meta.copy()
        meta_data["xaxis"]["values"] = meta_data["xaxis"]["values"][:meta_data["xaxis"]["valued_range"][-1]+1]
        return meta_data

class TH3UpRootHandler():

    def __init__(self, dqm_variable="QIE11DigiOccupancy", run_profile={"id": None, "year": None, "pd": "ZeroBias"}) -> None:
        self.root_loader = "uproot"
        self.run_profile = run_profile
        self.filepath = None
        self.tree_hist_tag = None
        self.meta = {}
        self.meta["xaxis"] = {}
        self.meta["yaxis"] = {}
        self.meta["zaxis"] = {}
        self.meta["dqm_variable"] = dqm_variable
        self.data_export_dirpath = r"../data/HCAL_ONLINE_DQM/DIGI_OCCUPANCY_CUT/{}/{}".format(self.run_profile["pd"],
                                                                                              self.run_profile["year"])

    def load_TFiledata(self, filename=None, dirpath=None, filepath=None) -> None:
        self.TFileObj, self.filepath = load_root_file(
            filename=filename, dirpath=dirpath, filepath=filepath, root_loader=self.root_loader)

    def __hist_formatter(self):
        # if "digioccupancy" in self.meta["dqm_variable"].lower():
        #     #  x- lumi section id, y- ieta, z- iphi
        #     self.TH3Obj = run_digi_hist_formatter(
        #         self.TH3Obj, dqm_variable=self.meta["dqm_variable"])

        self.meta["xaxis"]["label"] = "ls"
        self.meta["yaxis"]["label"] = "ieta"
        self.meta["zaxis"]["label"] = "iphi"


    def extract_TH(self, tree_hist_tag="HcalDigiAnalyzer/hist3D") -> None:
        '''
        TH3Obj : TH3 object
        tree_hist_tag: tree path of the target TH3I node
        extraction of axes info ls x ieta x iphi
        3D hist root into 3D numpy conversion
        :
        '''

        self.tree_hist_tag = tree_hist_tag
        self.TH3Obj = hist_extractor_from_rootfile(
            self.TFileObj, tree_hist_tag=self.tree_hist_tag, root_loader=self.root_loader)

        self.__hist_formatter()  # x- lumi section id, y- ieta, z- iphi

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
        self.TH3Obj_np = convert_run_digihistroot_to_nparray(
            self.TH3Obj, root_loader=self.root_loader)
        # remove empty ls at the end
        xaxis_3d_idx, yaxis_3d_idx, zaxis_3d_idx = self.TH3Obj_np.nonzero()
        # ls_range = [xaxis_3d_idx.min(), xaxis_3d_idx.max()]
        self.meta["xaxis"]["valued_range"] = [
            xaxis_3d_idx.min(), xaxis_3d_idx.max()]
        self.meta["yaxis"]["valued_range"] = [
            yaxis_3d_idx.min(), yaxis_3d_idx.max()]
        self.meta["zaxis"]["valued_range"] = [
            zaxis_3d_idx.min(), zaxis_3d_idx.max()]

    def get_axis_labels(self) -> list:
        return self.meta["xaxis"]["label"], self.meta["yaxis"]["label"], self.meta["zaxis"]["label"]

    def plot_TH_np(self, value_cut_thr=None, figsize=(6, 8), title=None) -> None:
        print("value_cut_thr: {}".format(value_cut_thr))
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # digi_occpcy_cut_th = 1e-42
        # xaxis_arr = getattr(self, "xaxis_arr".format(self.meta["xaxis"]["label"]))
        # yaxis_arr = getattr(self, "yaxis_arr".format(self.meta["yaxis"]["label"]))
        # zaxis_arr = getattr(self, "zaxis_arr".format(self.meta["zaxis"]["label"]))

        if value_cut_thr:
            xaxis_3d_idx, yaxis_3d_idx, zaxis_3d_idx = (
                self.TH3Obj_np > value_cut_thr).nonzero()  # gives to coordinate of non-zero
        else:
            # gives to coordinate of non-zero reading
            xaxis_3d_idx, yaxis_3d_idx, zaxis_3d_idx = self.TH3Obj_np.nonzero()

        fig = plt.figure(figsize=figsize)
        # ax = Axes3D(fig)
        ax = fig.add_subplot(111, projection='3d')
        ax_3d = ax.scatter(self.meta["xaxis"]["values"][xaxis_3d_idx], self.meta["yaxis"]["values"][yaxis_3d_idx],
                           self.meta["zaxis"]["values"][zaxis_3d_idx], c=self.TH3Obj_np[xaxis_3d_idx, yaxis_3d_idx, zaxis_3d_idx], marker=".")
        # You can change 0.01 to adjust the distance between the main image and the colorbar.
        # You can change 0.02 to adjust the width of the colorbar.
        # cax = fig.add_axes([ax.get_position().x1+.01, ax.get_position().y0, 0.02, ax.get_position().height])
        # cbar=plt.colorbar(ax_3d, cax=cax)
        cbar = plt.colorbar(ax_3d)
        cbar.set_label(self.meta["dqm_variable"])

        ax.set_xlabel(self.meta["xaxis"]["label"])
        ax.set_ylabel(self.meta["yaxis"]["label"])
        ax.set_zlabel(self.meta["zaxis"]["label"])
        ax.set_title("{}_{}".format(title, self.meta["dqm_variable"]))
        fig.show()
        # # clickplotter = OnClickPlotter(ax, data, source, statData)

    def plot_TH_root(self, **kwargs) -> None:
        raise "plot_TH_root is not implemeted for uproot loader"
        plot_run_histroot(self.TH3Obj, **kwargs)

    def plot_TH_2d(self, index_axis: str, column_axis: str, target_idx: int, **kwargs):
        '''
        plots 2d hist across any of the axis selections
        index_axis:  'x', 'y' or 'z'
        column_axis:  'x', 'y' or 'z'
        target_idx: index to select the 2d data along the axis not in the index_axis nor column_axis.

        call self.get_axis_labels() to see the labels for [x, y, z]
        # ls_id=10
        e.g. self.plot_TH_2d(index_axis='y', column_axis='z', target_idx:10)
        '''
        print(self.get_axis_labels())

        index = getattr(self, "{}axis_arr".format(index_axis))
        columns = getattr(self, "{}axis_arr".format(column_axis))
        if 'x' not in [index_axis, column_axis]:
            data_slice = self.TH3Obj_np[target_idx]
        elif 'y' not in [index_axis, column_axis]:
            data_slice = self.TH3Obj_np[:, target_idx, :]
        else:
            data_slice = self.TH3Obj_np[:, :, target_idx]

        if index_axis > column_axis:  # check axis consistency
            data_slice = data_slice.T

        hist_2d_df = pd.DataFrame(data_slice, columns=columns, index=index)
        plot_2d_heatmap(hist_2d_df, **kwargs, return_fig=False)

    def prepare_dataset(self, suffix=None, dst_dirpath=None):
        if not dst_dirpath:
            dst_dirpath = self.data_export_dirpath
        print(dst_dirpath)
        create_dir(dst_dirpath)
        meta_data = self.meta.copy()
        meta_data["xaxis"]["values"] = meta_data["xaxis"]["values"][:meta_data["xaxis"]["valued_range"][-1]+1]
        # remove empty ls at the end
        save_npdata("{}__{}__{}".format(self.meta["dqm_variable"], self.run_profile["id"], suffix).strip(
            "__"), self.TH3Obj_np[: meta_data["xaxis"]["valued_range"][-1]+1], filepath=dst_dirpath)
        save_json("{}__{}__{}__meta".format(self.meta["dqm_variable"], self.run_profile["id"], suffix).strip(
            "__"), self.meta, filepath=dst_dirpath)
    
    def get_th3_numpy(self):
        return self.TH3Obj_np[:self.meta["xaxis"]["valued_range"][-1]+1]

    def get_meta_data(self):
        meta_data = self.meta.copy()
        meta_data["xaxis"]["values"] = meta_data["xaxis"]["values"][:meta_data["xaxis"]["valued_range"][-1]+1]
        return meta_data


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def removesuffix(s, suffix):
    if s.endswith(suffix):
        return s[:-len(suffix)]
    else:
        return s[:]


def join_path(path_list=[]):
    return os.path.join(*[removesuffix(removesuffix(path, "//"), "/") for path in path_list])+"//"


def save_npdata(filename, data, filepath=None):
    if not filepath:
        filepath = save_dir
    filepath = "{}//".format(removesuffix(filepath, "//"))
    filename = "{}.npy".format(removesuffix(filename, ".npy"))
    print(filepath + filename)
    np.save(filepath + filename, data)


def load_npdata(filename, filepath=None):
    if not filepath:
        filepath = load_dir
    filepath = "{}//".format(removesuffix(filepath, "//"))
    filename = "{}.npy".format(removesuffix(filename, ".npy"))

    print(filepath + filename)
    # to allow Object arrays containing string
    return np.load(filepath + filename, allow_pickle=True)


def save_csv(filename, df, filepath=None, index=True, ignore_format=False):
    if not filepath:
        filepath = save_dir
    filepath = "{}//".format(removesuffix(filepath, "//"))
    filename = "{}.csv".format(removesuffix(filename, ".csv"))

    print('saving ', filepath + filename)
    if not ignore_format:
        df.to_csv(filepath + filename, float_format='%6.5f', index=index)
    else:
        df.to_csv(filepath + filename, index=index)


def save_json(filename, datadic, filepath=None):
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(MyEncoder, self).default(obj)

    if not filepath:
        filepath = save_dir
    filepath = "{}//".format(removesuffix(filepath, "//"))
    filename = "{}.json".format(removesuffix(filename, ".json"))

    print("saving: ", filepath)
    with open(filepath + filename, 'w') as fhandle:
        fhandle.write(json.dumps(datadic, indent=4,
                                 sort_keys=True, cls=MyEncoder))
        fhandle.close()


def load_json(filename, filepath=None):
    if not filepath:
        filepath = load_dir
    filepath = "{}//".format(removesuffix(filepath, "//"))
    filename = "{}.json".format(removesuffix(filename, ".json"))
    print("loading: ", filepath)
    with open(filepath + filename, 'r') as fhandle:
        data_dict = json.load(fhandle)
        fhandle.close()
    # print(data_dict)
    return data_dict


def load_root_file(filename=None, dirpath=None, filepath=None, root_loader="ROOT"):
    print("load_root_file...")
    if not filepath:
        if not dirpath:
            dirpath = data_path
        dirpath = "{}//".format(removesuffix(dirpath, "//"))
        filename = "{}.root".format(removesuffix(filename, ".root"))
        filepath = dirpath + filename
    else:
        # filename and dirpath are overridden by filepath
        pass

    print('loading:', filepath)
    # for cern vm desmod dashboard
    '''
    if root_loader == "ROOT":
        TFileObj = ROOT.TFile.Open(filepath, "READ")
        print(TFileObj)
        print(TFileObj.__dict__)
        return TFileObj, filepath

    elif 
    '''
    if root_loader == "uproot":
        TFileObj_uproot = uproot.open(filepath)
        # e.g. {'HcalDigiAnalyzer;1': 'TDirectory', 'HcalDigiAnalyzer/hist3D;1': 'TH3I'}
        print(TFileObj_uproot.classnames())

        # hist_treepath = [key for key in TFileObj_uproot.classnames().keys()
        #             if "hist3D" in key][0]
        return TFileObj_uproot, filepath


def hist_extractor_from_rootfile(TFileObj, tree_tag="HcalDigiAnalyzer", hist_tag="hist3D", tree_hist_tag="HcalDigiAnalyzer/hist3D", root_loader="ROOT"):
    # {'HcalDigiAnalyzer;1': 'TDirectory', 'HcalDigiAnalyzer/hist3D;1': 'TH3I'}
    # TH3I is 3-D histogram with an int per channel (see TH1 documentation)}
    # https://root.cern.ch/doc/master/classTH3I.html
    print("hist_extractor_from_rootfile...")
    # tree = TFileObj.Get(tree_tag)
    if not tree_hist_tag:
        tree_hist_tag = "{tree}/{hist}".format(tree=tree_tag, hist=hist_tag)
    if root_loader == "ROOT":
        histObj = TFileObj.Get(tree_hist_tag)
        print(histObj)
        print(histObj.Print())
    elif root_loader == "uproot":
        histObj = TFileObj[tree_hist_tag]
        print(histObj)
        print(histObj.all_members)
    # e.g get value at  lumi section 6 in ieta=24 and iphi=36
    # value = histObj.GetBinContent(6, 24, 36)

    '''
    # using uproot
    histObj = TFileObj[tree_hist_tag]
    dqm, ls, ieta, iphi = histObj.to_numpy()
    '''
    return histObj


def run_digi_hist_formatter(TH3Obj, dqm_variable="QIE11DigiOccupancy"):
    '''
    x- lumi section id, y- ieta, z- iphi
    '''

    TH3Obj.SetXTitle("ls")
    TH3Obj.SetYTitle("ieta")
    TH3Obj.SetZTitle("iphi")
    TH3Obj.SetTitle(dqm_variable)
    # it is DigiOccupancyCut or DigiOccupancy?

    # Nx = TH3Obj.GetNbinsX()
    # Ny = TH3Obj.GetNbinsY()
    # Nz = TH3Obj.GetNbinsZ()
    # print("hist dim: ls={}, ieta={}, iphi={}".format(Nx, Ny, Nz))

    # xaxis = [TH3Obj.GetXaxis().GetXmin(), TH3Obj.GetXaxis().GetXmax()]
    # yaxis = [TH3Obj.GetYaxis().GetXmin(), TH3Obj.GetYaxis().GetXmax()]
    # zaxis = [TH3Obj.GetZaxis().GetXmin(), TH3Obj.GetZaxis().GetXmax()]
    # print("hist axis index limites: ls={}, ieta={}, iphi={}".format(xaxis, yaxis, zaxis))

    return TH3Obj


def convert_run_digihistroot_to_nparray(TH3Obj, hist_dim=3, include_overflow=False, iscopy=True, root_loader="ROOT"):
    '''
    TH3Obj : TH3 object
    3D hist root into 3D numpy conversion
    output: digi_hist_np 3D numpy array equivalent to TH3
    '''
    
    # for cern vm desmod dashboard
    '''
    if root_loader == "ROOT":
        if isinstance(TH3Obj, ROOT.TH3):
            shape = (TH3Obj.GetNbinsZ() + 2,
                    TH3Obj.GetNbinsY() + 2,
                    TH3Obj.GetNbinsX() + 2)
        elif isinstance(TH3Obj, ROOT.TH2):
            shape = (TH3Obj.GetNbinsY() + 2, TH3Obj.GetNbinsX() + 2)
        elif isinstance(TH3Obj, ROOT.TH1):
            shape = (TH3Obj.GetNbinsX() + 2,)
        elif isinstance(TH3Obj, ROOT.THnBase):
            shape = tuple([TH3Obj.GetAxis(i).GetNbins() + 2
                        for i in range(TH3Obj.GetNdimensions())])
        else:
            raise TypeError(
                "TH3Obj hist must be an instance of ROOT.TH1, "
                "ROOT.TH2, or ROOT.TH3")

        if hist_dim == 3:
            Nx = TH3Obj.GetNbinsX()+2
            Ny = TH3Obj.GetNbinsY()+2
            Nz = TH3Obj.GetNbinsZ()+2

            # 3D hist root into 3D numpy conversion
            binvals_np = TH3Obj.GetArray()  # numpy array
            binvals_np = np.ndarray((Nx*Ny*Nz,),
                                    dtype=np.int32, buffer=binvals_np)
            print(binvals_np.shape)
            binvals_np = np.transpose(binvals_np.reshape(
                (Nz, Ny, Nx), order='C'), (2, 1, 0))
            print(binvals_np.shape)  # (Nx+2)*(Ny+2)*(Nz+2)
            # Strip off underflow and overflow bins
            digi_hist_np = binvals_np[1:-1, 1:-1, 1:-1]  # Nx*Ny*Nz
        elif hist_dim == 1:
            Nx = TH3Obj.GetNbinsX()
            # 3D hist root into 3D numpy conversion
            binvals_np = TH3Obj.GetArray()  # numpy array
            binvals_np = np.ndarray((Nx,),
                                    dtype=np.int32, buffer=binvals_np)
            print(binvals_np.shape)
            # binvals_np = np.transpose(binvals_np.reshape(
            #     (Nz+2, Ny+2, Nx+2), order='C'), (2, 1, 0))
            # print(binvals_np.shape)  # (Nx+2)*(Ny+2)*(Nz+2)
            # Strip off underflow and overflow bins
            digi_hist_np = binvals_np[1:-1]  # Nx
    elif
    ''' 
    if root_loader == "uproot":
            digi_hist_np = TH3Obj.values()
    print(digi_hist_np.shape)

    '''
    # hist root into numpy conversion
    binvals_np = TH3Obj.GetArray()  # numpy array
    binvals_np = np.ndarray(shape,
                            dtype=np.int32, buffer=binvals_np)
    print(binvals_np.shape)

    # binvals_np = np.transpose(binvals_np.reshape(
    #     (Nz, Ny, Nx), order='C'), (2, 1, 0))
    # print(binvals_np.shape)  # (Nx+2)*(Ny+2)*(Nz+2)
    # # Strip off underflow and overflow bins
    # digi_hist_np = binvals_np[1:-1, 1:-1, 1:-1]
    # print(digi_hist_np.shape)  # Nx*Ny*Nz

    if not include_overflow:
        # Remove overflow and underflow bins
        array = digi_hist_np[tuple([slice(1, -1)
                                    for idim in range(digi_hist_np.ndim)])]

    # Preserve x, y, z -> axis 0, 1, 2 order
    digi_hist_np = np.transpose(digi_hist_np)
    print(digi_hist_np.shape)  # Nx
    if iscopy:
        return np.copy(digi_hist_np)
    '''

    '''
    # using uproot
    # TH3Obj = TFileObj[tree_hist_tag]
    dqm, ls, ieta, iphi = TH3Obj.to_numpy()
    np.equal(dqm, digi_hist_np).all() # True
    dqm.shape: [lsxietaxiphi], [lsx64x72]
    ls: [0, 2001]
    ieta: 65, includes 0, [-32, 32]
    iphi: 73, includes [0, 72]
    '''

    return digi_hist_np


def plot_run_histroot(TH3Obj):
    '''
    assumes ls dmin in the xaxis of TH3Obj
    '''
    xaxis = np.array([TH3Obj.GetXaxis().GetXmin(),
                      TH3Obj.GetXaxis().GetXmax()]).astype(int).tolist()
    # print(xaxis[0], xaxis[1])
    # to have focused zoom plot and avoid empty padded LSs
    # only takes python native int, not np.int. Thus tolist conversts to native int or .item() for non-array np variable
    TH3Obj.GetXaxis().SetRange(xaxis[0], xaxis[1])
    # TH3Obj.GetYaxis().SetRange(-40, 40)
    # TH3Obj.GetZaxis().SetRange(0, 75)
    TH3Obj.Draw("box")


def plot_2d_heatmap(hist_data, x_arr=None, y_arr=None, figsize=(8, 20), return_fig=False, **kwargs):
    '''
    hist_data is 2D numpy array or dataframe
    kwargs: vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt=".2g", annot_kws=None, linewidths=0, linecolor="white", cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels="auto", yticklabels="auto",
    '''
    if not isinstance(hist_data, pd.DataFrame):
        hist_data = pd.DataFrame(hist_data, columns=y_arr, index=x_arr)

    fig, ax = plt.subplots(figsize=figsize)
    hist_data = hist_data.reindex(
        sorted(hist_data.columns, reverse=True), axis=1)
    g = sns.heatmap(hist_data.T, ax=ax, **kwargs)
    if return_fig:
        return g
    plt.show()


def plot_TH3_np(TH3Obj_np, meta, value_cut_thr=None, figsize=(6, 8), title=None) -> None:
    print("value_cut_thr: {}".format(value_cut_thr))
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # digi_occpcy_cut_th = 1e-42
    # xaxis_arr = getattr(self, "xaxis_arr".format(self.meta["xaxis"]["label"]))
    # yaxis_arr = getattr(self, "yaxis_arr".format(self.meta["yaxis"]["label"]))
    # zaxis_arr = getattr(self, "zaxis_arr".format(self.meta["zaxis"]["label"]))

    if value_cut_thr:
        xaxis_3d_idx, yaxis_3d_idx, zaxis_3d_idx = (
            TH3Obj_np > value_cut_thr).nonzero()  # gives to coordinate of non-zero
    else:
        # gives to coordinate of non-zero reading
        xaxis_3d_idx, yaxis_3d_idx, zaxis_3d_idx = TH3Obj_np.nonzero()

    fig = plt.figure(figsize=figsize)
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax_3d = ax.scatter(meta["xaxis"]["values"][xaxis_3d_idx], meta["yaxis"]["values"][yaxis_3d_idx],
                       meta["zaxis"]["values"][zaxis_3d_idx], c=TH3Obj_np[xaxis_3d_idx, yaxis_3d_idx, zaxis_3d_idx], marker=".")
    # You can change 0.01 to adjust the distance between the main image and the colorbar.
    # You can change 0.02 to adjust the width of the colorbar.
    # cax = fig.add_axes([ax.get_position().x1+.01, ax.get_position().y0, 0.02, ax.get_position().height])
    # cbar=plt.colorbar(ax_3d, cax=cax)
    cbar = plt.colorbar(ax_3d)
    cbar.set_label(meta["dqm_variable"])
    ax.set_xlabel(meta["xaxis"]["label"])
    ax.set_ylabel(meta["yaxis"]["label"])
    ax.set_zlabel(meta["zaxis"]["label"])
    ax.set_title("{}_{}".format(title, meta["dqm_variable"]))
    fig.show()
    # # clickplotter = OnClickPlotter(ax, data, source, statData)


def root_th3_with_lumi_to_numpy_convert(TH3RootObj, root_filename=None, root_dirpath=None, filepath=None,
                np_dirpath=None,
                tree_hist_tag="Hcal4DQMAnalyzer/hist3D", depth_size=7, **kwargs):

    print("root_th3_with_lumi_to_numpy_convert...{}".format(tree_hist_tag))
    th_dim = kwargs.get("th_dim", 3)
    isplot = kwargs.get("isplot", False)
    root_loader = kwargs.get("root_loader", "ROOT")

    if filepath is None:
        filepath = root_dirpath + "/" + root_filename

    if "evttree" in tree_hist_tag:
        # "Hcal4DQMAnalyzer/evttree/LumiSec" for event data(regardless of depth)
        # TH1Obj_np = convert_run_digihistroot_to_nparray(self.TH3Obj, hist_dim=1)
        tree_var = tree_hist_tag.split("/")[-1]
        tree_hist_tag = "/".join(tree_hist_tag.split("/")[:-1])
        print(tree_hist_tag, tree_var)

        # TH3RootObj = TH3RootHandler()
        # TH3RootObj.load_TFiledata(filename=root_filename, dirpath=root_dirpath, filepath=filepath)

        print(TH3RootObj.TFileObj, TH3RootObj.filepath)
        TH1Obj = hist_extractor_from_rootfile(
            TH3RootObj.TFileObj, tree_hist_tag=tree_hist_tag, root_loader=TH3RootObj.root_loader)
        # Get branch "tree_var" as numpy array
        if root_loader == "ROOT":
            # data = TH1Obj.GetArray()
            # TH1Obj.GetBranch(tree_var).SetName(tree_var)
            TH1Obj.GetLeaf(tree_var).SetName(tree_var)

            # deprecated in 6.24
            # data = TH1Obj.AsMatrix([tree_var])  # get numpy

            # after 6.24
            data = ROOT.RDataFrame(TH1Obj).AsNumpy()[tree_var]
            print(data.shape)
        elif root_loader == "uproot":
            data = np.array(TH1Obj.arrays()[tree_var])

        # # bin-method-1
        # ls = np.unique(data)
        # ls = np.concatenate((np.array([0]), ls))
        # num_event_np = np.histogram(data, bins=ls)
        # print(len(num_event_np[0]), len(num_event_np[1]))
        # num_event_np
        # num_event_df = pd.Series(
        #     num_event_np[0][1:], index=num_event_np[1][1:].astype(int)).to_frame().reset_index(drop=False)

        # bin-method-2
        # similar as root plot
        ls = np.unique(data)
        ls = np.concatenate((ls, ls[-1:]+1))
        num_event_np = np.histogram(data, bins=ls)
        print(len(num_event_np[0]), len(num_event_np[1]))
        num_event_np
        num_event_df = pd.Series(
            num_event_np[0], index=num_event_np[1][:-1].astype(int)).to_frame().reset_index(drop=False)

        # # bin-method-3
        # ls = np.unique(data)
        # ls = np.concatenate((np.array([0]), ls, ls[-1:]+1))
        # num_event_np = np.histogram(data, bins=ls)
        # print(len(num_event_np[0]), len(num_event_np[1]))
        # num_event_np
        # num_event_df = pd.Series(
        #     num_event_np[0][1:], index=num_event_np[1][1:-1].astype(int)).to_frame().reset_index(drop=False)

        num_event_df.columns = ["LS#", "NumEvents"]
        print(num_event_df.shape)
        print(num_event_df.head())
        if isplot:
            plt.plot(num_event_df["LS#"], num_event_df["NumEvents"])
            plt.show()

        return num_event_df
    
    else:

        print(TH3RootObj.TFileObj, TH3RootObj.filepath)
        # if "depth" not in filepath:
        if th_dim == 2:
            # "Hcal4DQMAnalyzer/hist3D" for hist data ,
            TH3RootObj.extract_TH(tree_hist_tag=tree_hist_tag)
            # TH3RootObj.plot_TH_root()
            TH3RootObj.convert_TH_to_numpy()
            if isplot:
                TH3RootObj.plot_TH_np(title="run")
            # TH3RootObj.plot_TH_2d(index_axis="y", column_axis="z", target_idx=1, figsize=(8, 20))
            TH3Obj_np = TH3RootObj.get_th3_numpy()

        elif th_dim == 3:
            roots_per_depth_list = []
            for depth in range(1, depth_size+1):
                print("{}_depth{}".format(tree_hist_tag, depth))
                TH3RootObj.extract_TH(
                    tree_hist_tag="{}_depth{}".format(tree_hist_tag, depth))
                # TH3RootObj.plot_TH_root()
                TH3RootObj.convert_TH_to_numpy()
                if isplot:
                    TH3RootObj.plot_TH_np(title="run")
                # TH3RootObj.plot_TH_2d(index_axis="y", column_axis="z", target_idx=1, figsize=(8, 20))
                TH3Obj_np_ = TH3RootObj.get_th3_numpy()
                print(TH3Obj_np_.shape)
                roots_per_depth_list.append(np.expand_dims(TH3Obj_np_, axis=3))
            TH3Obj_np = np.concatenate(tuple(roots_per_depth_list), axis=3) 
        print(TH3Obj_np.shape)
        return TH3Obj_np
                

def root_to_numpy_dataset_convert(root_filename=None, root_dirpath=None, filepath=None, np_dirpath=None, run_profile={"id": None, "year": None, "pd": "ZeroBias"}, tree_hist_tag="Hcal4DQMAnalyzer/hist3D", depth_size=7, isplot=False, root_loader="ROOT"):
    
    print("root_to_numpy_dataset_convert...")
    if filepath is not None:
        root_dirpath = join_path(
            [root_dirpath, run_profile["pd"], run_profile["year"], run_profile["id"]])
        print(root_dirpath)

    if "evttree" in tree_hist_tag:
        # "Hcal4DQMAnalyzer/evttree/LumiSec" for event data(regardless of depth)
        # TH1Obj_np = convert_run_digihistroot_to_nparray(self.TH3Obj, hist_dim=1)
        tree_var = tree_hist_tag.split("/")[-1]
        tree_hist_tag = "/".join(tree_hist_tag.split("/")[:-1])
        print(tree_hist_tag, tree_var)
        if root_loader == "ROOT":
            TH1RootObj = TH3RootHandler(
            run_profile=run_profile)
        elif root_loader == "uproot":
            TH1RootObj = TH3UpRootHandler(
            run_profile=run_profile)

        TH1RootObj.load_TFiledata(filename=root_filename, dirpath=root_dirpath, filepath=filepath, root_loader=root_loader)
        print(TH1RootObj.TFileObj, TH1RootObj.filepath)
        
        TH1Obj = hist_extractor_from_rootfile(
            TH1RootObj.TFileObj, tree_hist_tag=tree_hist_tag, root_loader=root_loader)

        # Get branch "tree_var" as numpy array

        # for cern vm desmod dashboard
        '''
        if root_loader == "ROOT":
            # data = TH1Obj.GetArray()
            # TH1Obj.GetBranch(tree_var).SetName(tree_var)
            TH1Obj.GetLeaf(tree_var).SetName(tree_var)
            
            # deprecated in 6.24
            # data = TH1Obj.AsMatrix([tree_var])  # get numpy

            # after 6.24
            data = ROOT.RDataFrame(TH1Obj).AsNumpy()[tree_var]
            print(data.shape)
        elif 
        '''
        if root_loader == "uproot":
            data =  np.array(TH1Obj.arrays()[tree_var])

        # # bin-method-1
        # ls = np.unique(data)
        # ls = np.concatenate((np.array([0]), ls))
        # num_event_np = np.histogram(data, bins=ls)
        # print(len(num_event_np[0]), len(num_event_np[1]))
        # num_event_np
        # num_event_df = pd.Series(
        #     num_event_np[0][1:], index=num_event_np[1][1:].astype(int)).to_frame().reset_index(drop=False)

        # bin-method-2
        # similar as root plot
        ls = np.unique(data)
        ls = np.concatenate((ls, ls[-1:]+1))
        num_event_np = np.histogram(data, bins=ls)
        print(len(num_event_np[0]), len(num_event_np[1]))
        num_event_np
        num_event_df = pd.Series(
            num_event_np[0], index=num_event_np[1][:-1].astype(int)).to_frame().reset_index(drop=False)

        # # bin-method-3
        # ls = np.unique(data)
        # ls = np.concatenate((np.array([0]), ls, ls[-1:]+1))
        # num_event_np = np.histogram(data, bins=ls)
        # print(len(num_event_np[0]), len(num_event_np[1]))
        # num_event_np
        # num_event_df = pd.Series(
        #     num_event_np[0][1:], index=num_event_np[1][1:-1].astype(int)).to_frame().reset_index(drop=False)

        num_event_df.columns = ["LS#", "NumEvents"]
        print(num_event_df.shape)
        print(num_event_df.head())
        plt.plot(num_event_df["LS#"], num_event_df["NumEvents"])
        plt.show()
        # plt.plot(data)
        # plt.show()
        # plt.hist(data, bins=np.unique(data))
        # plt.show()
        filename = removesuffix(root_filename, ".root")
        filename = removesuffix(filename, "__depth")
        # filepath = join_path([root_dirpath, run_profile["year"]])

        save_csv("{}__{}__ls__numevent".format(run_profile["id"], filename), num_event_df,
                 filepath=TH1RootObj.data_export_dirpath, index=False, ignore_format=False)
        return

    # TH3RootObj = TH3RootHandler(
    #     run_profile=run_profile)

    if root_loader == "ROOT":
        TH1RootObj = TH3RootHandler(
        run_profile=run_profile)
    elif root_loader == "uproot":
        TH1RootObj = TH3UpRootHandler(
        run_profile=run_profile)


    TH3RootObj.load_TFiledata(filename=root_filename, dirpath=root_dirpath)
    print(TH3RootObj.TFileObj, TH3RootObj.filepath)
    if "depth" not in root_filename:
        # "Hcal4DQMAnalyzer/hist3D" for hist data ,
        TH3RootObj.extract_TH(tree_hist_tag=tree_hist_tag)
        # TH3RootObj.plot_TH_root()
        TH3RootObj.convert_TH_to_numpy()
        if isplot:
            TH3RootObj.plot_TH_np(title=run_profile["id"])
        # TH3RootObj.plot_TH_2d(index_axis="y", column_axis="z", target_idx=1, figsize=(8, 20))
        filename = removesuffix(root_filename, ".root")
        TH3RootObj.prepare_dataset(suffix=filename)
    else:
        for depth in range(1, depth_size+1):
            TH3RootObj.extract_TH(
                tree_hist_tag="{}_depth{}".format(tree_hist_tag, depth))
            # TH3RootObj.plot_TH_root()
            TH3RootObj.convert_TH_to_numpy()
            if isplot:
                TH3RootObj.plot_TH_np(title=run_profile["id"])
            # TH3RootObj.plot_TH_2d(index_axis="y", column_axis="z", target_idx=1, figsize=(8, 20))
            filename = removesuffix(root_filename, ".root")
            filename = "{}{}".format(filename, depth)
            TH3RootObj.prepare_dataset(suffix=filename)


def rootfile_loader(filename=None, filepath=None, root_loader="uproot"):
    # filepath = "../../data/rootfiles/DIGIOCCUPANCY/Run355456__hehb_cut20fc__depth.root"
    # root_filename = "output__depth.root.root"

    if filepath is None:
        # filepath = r"../../data/rootfiles/DIGIOCCUPANCY/" + filename
        filepath = main_path + "/data/rootfiles/DIGIOCCUPANCY/" + filename
    
    print(filepath)
    # TH3RootObj = TH3RootHandler()
    # TH3RootObj.load_TFiledata(filepath=filepath)
    # TH3RootObj.extract_TH(tree_hist_tag="HcalDigiAnalyzer/hist3D")

    if root_loader == "ROOT":
        TH3RootObj = TH3RootHandler()
    elif root_loader == "uproot":
        TH3RootObj = TH3UpRootHandler()

    TH3RootObj.load_TFiledata(filepath=filepath)
    # TH3RootObj_ = copy.deepcopy(TH3RootObj)

    # with depth
    # DQM TH3 histogram to numpy conversion
    tree_hist_tag = "Hcal4DQMAnalyzer/hist3D"
    TH3Obj_np = root_th3_with_lumi_to_numpy_convert(TH3RootObj,
                                                    filepath=filepath, tree_hist_tag=tree_hist_tag, root_loader=root_loader, th_dim=3)
    meta_data = TH3RootObj.get_meta_data()

    # evttree/LumiSec
    # TH3RootObj = TH3RootHandler()
    # TH3RootObj.load_TFiledata(filepath=filepath)
    tree_hist_tag = "Hcal4DQMAnalyzer/evttree/LumiSec"  # for event data
    num_event_df = root_th3_with_lumi_to_numpy_convert(TH3RootObj,
                                                       filepath=filepath, tree_hist_tag=tree_hist_tag, root_loader=root_loader, th_dim=3)

    # evttree/RunNum
    tree_hist_tag = "Hcal4DQMAnalyzer/evttree/RunNum"  # for event data
    run_num_df = root_th3_with_lumi_to_numpy_convert(TH3RootObj,
                                                       filepath=filepath, tree_hist_tag=tree_hist_tag, root_loader=root_loader, th_dim=3)

    num_event_df["RunId"] = run_num_df.iloc[:, 0]

    return TH3Obj_np, num_event_df, meta_data

if __name__ == '__main__':

    # root_filepath = join_path([data_path, dataset]) + "QIE11DigiOccupancy.root"
    root_dirpath = r"C:\Users\mulugetawa\Documents\CERN\InductionProject\DESMOD_HCAL\data\HCAL_ONLINE_DQM\DIGI_OCCUPANCY_CUT"
    root_filename = "QIE11DigiOccupancy.root"

    TH3RootObj = TH3RootHandler()

    TH3RootObj.load_TFiledata(filename=root_filename, dirpath=root_dirpath)
    print(TH3RootObj.TFileObj, TH3RootObj.filepath)

    TH3RootObj.extract_TH(tree_hist_tag="HcalDigiAnalyzer/hist3D")

    TH3RootObj.plot_TH_root()

    TH3RootObj.convert_TH_to_numpy()

    # print(TH3RootObj.TH3Obj_np.shape)

    TH3RootObj.plot_TH_np()
    TH3RootObj.plot_TH_2d(index_axis="y", column_axis="z",
                          target_idx=1, figsize=(8, 20))

    TH3RootObj.prepare_dataset()


# in ROOT
#  change working dir
# gSystem -> cd("C:/Users/mulugetawa/Documents/CERN/Data/HCAL_ONLINE_DQM/ZeroBias/2018/")
# use the command to combine root files
# hadd output.root input1.root  input2.root

# gSystem -> cd("C:/Users/mulugetawa/Documents/CERN/Data/HCAL_ONLINE_DQM/ZeroBias/2018/Run325170/")
# gSystem->Exec("hadd output.root output_1.root output_2.root")

# gSystem -> cd("C:/Users/mulugetawa/Documents/CERN/Data/HCAL_ONLINE_DQM/ZeroBias/2018/org/Hcal4DQMAnalyzer_Run323978/211007_084112/0000/")
# gSystem->Exec("hadd output.root output_1.root output_2.root output_3.root")

# gSystem -> cd("C:/Users/mulugetawa/Documents/CERN/Data/HCAL_ONLINE_DQM/ZeroBias/2018/org/Hcal4DQMAnalyzer_Run325170/211007_084207/0000/")
# gSystem->Exec("hadd output.root output_1.root output_2.root")


# gSystem -> cd("C:/Users/mulugetawa/Documents/CERN/Data/HCAL_ONLINE_DQM/ZeroBias/2018/org/Hcal4DQMAnalyzer_Run323978/211123_172432/0000/")
# gSystem->Exec("hadd output__depth.root output_1.root output_2.root output_3.root")

# gSystem -> cd("C:/Users/mulugetawa/Documents/CERN/Data/HCAL_ONLINE_DQM/ZeroBias/2018/org/Hcal4DQMAnalyzer_Run325170/211123_172530/0000/")
# gSystem->Exec("hadd output__depth.root output_1.root output_1-1.root output_2.root output_3.root output_4.root output_5.root output_6.root")
