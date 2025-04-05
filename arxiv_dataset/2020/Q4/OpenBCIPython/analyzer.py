from __future__ import print_function

import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from lib.dtw import dtw

matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('legend', fontsize=20)
# manager = plt.get_current_fig_manager()
# manager.resize(*manager.window.maxsize())

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics.pairwise import manhattan_distances
from preprocessing.preprocessing import PreProcessor
from preprocessing.ssa import SingularSpectrumAnalysis


class SignalAnalyzer():
    def __init__(self, activity_type, project_path, dataset_location):
        self.raw_data = pd.read_csv(dataset_location)
        self.config_file = project_path + "/config/config.json"
        self.raw_data = self.raw_data.ix[:, 0:13].dropna()
        self.raw_channel_data = self.raw_data.ix[:, 2:7]
        self.raw_kinect_angle_data = self.raw_data.ix[:, 10:13]
        self.channel_length = self.raw_channel_data.shape[1]
        self.kinect_angle_length = 3
        self.angle_names = ["wrist", "elbow", "shoulder"]
        self.signal_types = ["noise_signal", "noise_reduced_signal", "feature_vector"]
        self.raw_channel_data_set = []
        self.output_buffer = []
        self.activity_type = activity_type
        self.project_path = project_path
        self.dataset_location = dataset_location
        self.channels_names = ["ch1", "ch2", "ch3", "ch4", "ch5"]
        with open(self.config_file) as config:
            self.config = json.load(config)
            self.config["train_dir_abs_location"] = self.project_path + "/build/dataset/train"

    def nomalize_signal(self, input_signal):
        mean = np.mean(input_signal, axis=0)
        input_signal -= mean
        return input_signal / np.std(input_signal, axis=0)

    def reconstructed_channel_data(self):
        for i in range(0, self.channel_length):
            self.raw_channel_data_set.append(self.nomalize_signal(self.raw_channel_data.ix[:, i]))
        for i in range(0, self.channel_length):
            preprocessor = PreProcessor(i, self.raw_channel_data_set, self.output_buffer, self.config)
            preprocessor.processor(i, activity_type=activity_type)

    def reconstructed_kinect_signals(self):
        kinect_angles = []
        for j in range(0, self.kinect_angle_length):
            nomalize_signal = self.nomalize_signal(self.raw_kinect_angle_data.ix[:, j])
            reconstructed_signal = SingularSpectrumAnalysis(nomalize_signal,
                                                            int(self.config["window_size"])) \
                .execute(int(self.config["number_of_principle_component"]))
            max_value = reconstructed_signal.max(axis=0)
            min_value = reconstructed_signal.min(axis=0)
            mapping = interp1d([min_value, max_value], [0, 180])
            kinect_angles.append(mapping(np.array(reconstructed_signal)))
        with open(
                                        project_path + "/build/dataset/train/result/reconstructed_" + activity_type + "_kinect__angles_.csv",
                'w') as f:
            np.savetxt(f, np.transpose(np.array(kinect_angles)), delimiter=',', fmt='%.18e')

    def append_channel_data(self):
        for i in range(0, len(self.signal_types)):
            signal_type = self.signal_types[i]
            noise_signals = []
            for i in range(0, self.channel_length):
                processed_signal = pd.read_csv(str(self.config["train_dir_abs_location"]) + "/" + str(i) + "_" +
                                               activity_type + "_" + signal_type + ".csv")
                noise_signals.append(np.array(processed_signal.ix[:, 0]))
            with open(str(self.config[
                              "train_dir_abs_location"]) + "/result/" + activity_type + "_" + signal_type + "s" + ".csv",
                      'w') as f:
                np.savetxt(f, np.transpose(np.array(noise_signals)), delimiter=',', fmt='%.18e')

    def plot_signals(self, is_save=False, start=0, end=0, fsamp=1, is_raw=False, is_compare=False):
        matplotlib.rc('xtick', labelsize=10)
        matplotlib.rc('ytick', labelsize=10)
        matplotlib.rc('axes', titlesize=15)
        matplotlib.rc('legend', fontsize=15)
        if is_raw:
            raw_channels_data = pd.read_csv(self.dataset_location).ix[:, 2:7].dropna()
        else:
            raw_channels_data = pd.read_csv(self.config["train_dir_abs_location"]
                                            + "/result/"+self.activity_type+"_feature_vectors.csv").dropna()
        noise_reducer_signal_data = pd.read_csv(self.config["train_dir_abs_location"]
                                        + "/result/"+self.activity_type+"_noise_reduced_signals.csv").dropna()
        self.save_channels = PdfPages('channels_'+self.activity_type+'_reconstructed.pdf')
        graph_legend = []
        handle_as = []
        labels_as = []
        num_ch = len(self.channels_names)
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(hspace=.5)

        index = 1
        num_types = 1

        if is_compare:
            num_types = 2
        for h in range(0, num_ch):
            # preprocessor = PreProcessor(h, None, None, self.config)
            ax = plt.subplot(num_ch*num_types, num_types, index)
            if (end == 0):
                end = raw_channels_data.ix[:, h].shape[0] - 1
            x = np.arange(start, end, 1)
            input_signal = raw_channels_data.ix[:, h][start * fsamp:end * fsamp]
            noise_reduced_signal = noise_reducer_signal_data.ix[:, h][start * fsamp:end * fsamp]

            l1 = ax.plot(noise_reduced_signal, linewidth=1.0, label="raw signal")
            graph_legend.append(l1)

            index+=1
            if is_compare:
                ax = plt.subplot(num_ch * num_types, num_types, index)
                l2 = ax.plot(input_signal, linewidth=1.0, label="svd signal")
                graph_legend.append(l2)
                index += 1

            # with open("input.csv", 'w') as f:
            #     np.savetxt(f, input_signal, delimiter=',', fmt='%.18e')

            # noise_reducer_signal = preprocessor.apply_noise_reducer_filer(input_signal)
            # l2 = ax.plot(x, noise_reducer_signal, linewidth=3.0, label="noise_reducer_signal")
            # graph_legend.append(l2)

            # normalize_signal = preprocessor.nomalize_signal(noise_reducer_signal)
            # l3 = ax.plot(x, normalize_signal, linewidth=1.0, label="normalize_signal")
            # graph_legend.append(l3)

            # reconstructed_signal = SingularSpectrumAnalysis(noise_reducer_signal, self.config["window_size"], False).execute(1)
            # l4 = ax.plot(x,reconstructed_signal, linewidth=1.0, label='reconstructed signal with SSA')
            # graph_legend.append(l4)

            handles, labels = ax.get_legend_handles_labels()
            handle_as.append(handles)
            labels_as.append(labels)
            plt.title(self.channels_names[h])
            # leg = plt.legend(handles=handles, labels=labels)

        fig.legend(handles=handle_as[0], labels=labels_as[0])
        fig.text(0.5, 0.04, 'position', ha='center', fontsize=10)
        fig.text(0.04, 0.5, 'angle(0-180)', va='center', rotation='vertical', fontsize=10)
        fig.tight_layout()
        if is_save:
            self.save_channels.savefig(bbox_inches='tight')
            self.save_channels.close()
        else:

            plt.show()

    def plot_kinect_angles(self, is_save=False, start=0, end=0, fsamp=1, is_raw=False):
        if is_raw==True:
            kinect_angle_data = pd.read_csv(self.dataset_location).ix[:, 10:13].dropna()
        else:
            kinect_angle_data = pd.read_csv(self.config["train_dir_abs_location"]
                                                + "/result/reconstructed_"+self.activity_type+"_kinect__angles_.csv").dropna()
        graph_legend = []
        handle_as = []
        labels_as = []
        self.save_kinect_anagle = PdfPages(''+self.activity_type+'_kinect_angles_reconstructed.pdf')
        num_ch = 3

        fig = plt.figure(figsize=(15, 10))
        fig.subplots_adjust(hspace=.5)
        for h in range(0, num_ch):
            ax = plt.subplot(num_ch, 1, h + 1)
            if (end == 0):
                end = kinect_angle_data.ix[:, h].shape[0] - 1

            input_signal = kinect_angle_data.ix[:, h][start * fsamp:end * fsamp]
            x = np.arange(start, end, 1)
            l1 = ax.plot(x, input_signal, linewidth=1.0, label="raw signal")
            graph_legend.append(l1)

            # nomalize_signal = self.nomalize_signal(input_signal)

            # max_value = reconstructed_signal.max(axis=0)
            # min_value = reconstructed_signal.min(axis=0)
            # mapping = interp1d([min_value, max_value], [0, 180])
            # reconstructed_signal= mapping(np.array(reconstructed_signal))

            handles, labels = ax.get_legend_handles_labels()
            handle_as.append(handles)
            labels_as.append(labels)
            plt.title(self.angle_names[h])
            # leg = plt.legend(handles=handles, labels=labels)

        fig.legend(handles=handle_as[0], labels=labels_as[0])
        fig.text(0.5, 0.04, 'position', ha='center', fontsize=20)
        fig.text(0.04, 0.5, 'angle(0-180)', va='center', rotation='vertical', fontsize=20)
        if is_save:
            self.save_kinect_anagle.savefig(bbox_inches='tight')
            self.save_kinect_anagle.close()
        else:
            plt.show()

    def apply_dwt(self, nomalized_signal, start, end, pattern_start_at, pattern_end_at, is_apply_dwt, channel_number=1):
        if(is_apply_dwt):
            pattern = np.array(nomalized_signal.ix[:, channel_number][pattern_start_at:pattern_end_at])
            result = []
            possion = []
            final_result = []
            size = pattern_end_at - pattern_start_at
            counter = start
            for i in range(0, int(np.floor((end-start)/5))):
            # for i in range(0, 3):
                y = np.array(nomalized_signal.ix[:, channel_number][counter:counter + size]).tolist()
                possion.append(counter)
                counter += 5
                dist, cost, acc, path = dtw(pattern, y, manhattan_distances)
                print (dist)
                result.append(dist)
            final_result.append(result)
            final_result.append(possion)

            with open(self.config["train_dir_abs_location"] + "/result/"+self.activity_type+"_dwt_result.csv", 'w') as f:
                np.savetxt(f, np.transpose(np.array(final_result)), delimiter=',', fmt='%.18e')
            return result, possion
        else:
            dwt_result = pd.read_csv(self.config["train_dir_abs_location"]
                                            + "/result/"+self.activity_type+"_dwt_result.csv").dropna()
            return dwt_result.ix[:,0], dwt_result.ix[:,1]

    def plot_kinect_angles_with_activity_signals(self, start=0, end=0, fsamp=1, is_raw=False):
        if is_raw:
            channels_data = self.nomalize_signal(pd.read_csv(self.dataset_location).ix[:, 2:7].dropna())
            kinect_angle_data = pd.read_csv(self.dataset_location).ix[:, 10:13].dropna()
        else:
            channels_data = pd.read_csv(self.config["train_dir_abs_location"]
                                            + "/result/"+self.activity_type+"_feature_vectors.csv").dropna()
            kinect_angle_data = pd.read_csv(self.config["train_dir_abs_location"]
                                            + "/result/reconstructed_"+self.activity_type+"_kinect__angles_.csv").dropna()

        graph_legend = []
        handle_as = []
        labels_as = []

        fig = plt.figure(figsize=(15, 10))
        fig.subplots_adjust(hspace=.5)
        if end==0:
            end= kinect_angle_data.ix[:, 0].shape[0] - 1

        x = np.arange(start, end, 1)
        for i in range(0, 5):
            ax = plt.subplot(810 + i + 1)
            l1 = ax.plot(channels_data.ix[:, i][start:end], linewidth=1.0, label="Processed signal with SSA")
            graph_legend.append(l1)
            handles, labels = ax.get_legend_handles_labels()
            handle_as.append(handles)
            labels_as.append(labels)
            plt.title(self.channels_names[i])


        for j in range(0, 3):
            ax = plt.subplot(815 + 1 + j)
            l1 = ax.plot(x, kinect_angle_data.ix[:, j][start:end], linewidth=1.0, label="Processed signal with SSA")
            graph_legend.append(l1)
            handles, labels = ax.get_legend_handles_labels()
            handle_as.append(handles)
            labels_as.append(labels)
            plt.title(self.channels_names[j])

        fig.legend(handles=handle_as[0], labels=labels_as[0])
        fig.text(0.5, 0.04, 'position', ha='center', fontsize=10)
        fig.text(0.04, 0.5, 'angle(0-180)', va='center', rotation='vertical', fontsize=10)
        plt.show()

    def plot_detected_pattern(self, start=0, end=0, fsamp=1, is_raw=False, pattern_start_at=0, pattern_end_at=200, is_apply_dwt=False, channel_number=1):
        if is_raw:
            channels_data = pd.read_csv(self.dataset_location).ix[:, 2:7].dropna()
            kinect_angle_data = pd.read_csv(self.dataset_location).ix[:, 10:13].dropna()
        else:
            channels_data = pd.read_csv(self.config["train_dir_abs_location"]
                                            + "/result/"+self.activity_type+"_feature_vectors.csv").dropna()
            kinect_angle_data = pd.read_csv(self.config["train_dir_abs_location"]
                                            + "/result/reconstructed_"+self.activity_type+"_kinect__angles_.csv").dropna()

        nomalized_signal = self.nomalize_signal(kinect_angle_data)
        # mapping = interp1d([-1,1],[0,180])
        if end==0:
            end = nomalized_signal.shape[0] - 1

        distance, possion = self.apply_dwt(nomalized_signal, start, end, pattern_start_at, pattern_end_at, is_apply_dwt, channel_number)
        _, mintab = self.lowest_point_detect(distance, .3)
        if len(mintab)==0:
            print ("No patterns were detected...")
            return

        indices = possion[np.array(mintab[:, 0], dtype=int)]

        graph_legend = []
        handle_as = []
        labels_as = []

        fig = plt.figure(figsize=(15, 10))
        fig.subplots_adjust(hspace=.5)
        x = np.arange(start, end, 1)
        for i in range(0, 5):
            ax = plt.subplot(810 + i + 1)
            l1 = ax.plot(x, self.nomalize_signal(channels_data.ix[:, i][start:end]), linewidth=1.0,
                         label="Processed signal with SSA")
            graph_legend.append(l1)
            handles, labels = ax.get_legend_handles_labels()
            handle_as.append(handles)
            labels_as.append(labels)
            plt.title(self.channels_names[i])
            for i in indices:
                plt.plot([i, i], [2,1], '-r')

        for j in range(0, 3):
            ax = plt.subplot(815 + 1 + j)
            l1 = ax.plot(x, self.nomalize_signal(kinect_angle_data.ix[:, j][start:end]), linewidth=1.0,
                         label="Processed signal with SSA")
            graph_legend.append(l1)
            handles, labels = ax.get_legend_handles_labels()
            handle_as.append(handles)
            labels_as.append(labels)
            plt.title(self.channels_names[j])
            for i in indices:
                plt.plot([i, i], [2,1], '-r')

        fig.legend(handles=handle_as[0], labels=labels_as[0])
        fig.text(0.5, 0.04, 'position', ha='center', fontsize=10)
        fig.text(0.04, 0.5, 'angle(0-180)', va='center', rotation='vertical', fontsize=10)
        plt.show()

    def lowest_point_detect(self, v, delta, x=None):
        maxtab = []
        mintab = []
        if x is None:
            x = np.arange(len(v))
        v = np.asarray(v)
        if len(v) != len(x):
            sys.exit('Input vectors v and x must have same length')
        if not np.isscalar(delta):
            sys.exit('Input argument delta must be a scalar')
        if delta <= 0:
            sys.exit('Input argument delta must be positive')
        mn, mx = np.Inf, -np.Inf
        mnpos, mxpos = np.NaN, np.NaN
        lookformax = True
        for i in np.arange(len(v)):
            this = v[i]
            if this > mx:
                mx = this
                mxpos = x[i]
            if this < mn:
                mn = this
                mnpos = x[i]
            if lookformax:
                if this < mx - delta:
                    maxtab.append((mxpos, mx))
                    mn = this
                    mnpos = x[i]
                    lookformax = False
            else:
                if this > mn + delta:
                    mintab.append((mnpos, mn))
                    mx = this
                    mxpos = x[i]
                    lookformax = True
        return np.array(maxtab), np.array(mintab)

    def execute(self, is_init=False):
        start = 0
        end = 0
        if is_init:
            self.reconstructed_channel_data()
            self.reconstructed_kinect_signals()
            self.append_channel_data()
        # self.plot_kinect_angles(start=start, end=end, is_raw=False)
        # self.plot_signals(start=start, end=end, is_raw=True)
        self.plot_detected_pattern(pattern_start_at=4400, pattern_end_at=5000, start=start, end=end, is_apply_dwt=True, channel_number=1)
        # self.plot_detected_pattern(pattern_start_at=3710, pattern_end_at=3830, start=start, end=end, is_apply_dwt=True, channel_number=1)
        #self.plot_kinect_angles_with_activity_signals(start, end, is_raw=False)

project_path = "/home/runge/openbci/git/OpenBCI_Python"
dataset_location = project_path+ "/build/dataset2017-5-5_23-55-32new_straight_up_filttered.csv"
activity_type = "straight_up"

# dataset_location = project_path + "/build/dataset2017-5-5_23-55-32new_bycept_filttered.csv"
# activity_type = "bycept"


signal_analyzer = SignalAnalyzer(activity_type, project_path, dataset_location)
signal_analyzer.execute()
