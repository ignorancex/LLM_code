# ///////////// Copyright 2023 Vishnu Boddeti. All rights reserved. /////////////
# //
# //   Project     : Capacity of Generative Face Models
# //   File        : plotting.py
# //   Description : Plotting functions for capacity estimation
# //
# //   Created On: 08/20/2023
# //   Created By: Gautam Sreekumar
# ////////////////////////////////////////////////////////////////////////////

import os
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy import interpolate

from utils import get_parameters
from constants import *

import config
args = config.parse_args()

plt.rcParams.update({"text.usetex": True,
                     "font.family": "Times",
                     "font.size": 16})

def plot_capacity(face_model, ref_dataset, dataset):
    _, far, _, threshold = get_parameters(face_model, ref_dataset)
    fname = os.path.join(
        args.output_path,
        'pkl_files',
        dataset,        
        f'{face_model}_{ref_dataset}_capacity.pkl'
    )
    data = pd.read_pickle(fname)

    k_ = list(data.keys())
    for k in k_:
        if not isinstance(data[k], pd.Series):
            data[k] = pd.Series((_ for _ in data[k]))

    capacity = data["capacity"]

    # plot capacity as a function of delta in log-scale
    fig, ax = plt.subplots(1, 1)

    for i in range(len(far)):
        plt.plot([threshold[i], threshold[i]], [min(capacity), max(capacity)], '--', linewidth=2)

    plt.legend(["FAR of {:.2f}\%".format(val) for val in far], loc='upper left')
    plt.plot(data["cos_delta"], data["capacity"], "-s",
             lw=2, color=(0.5, 0, 0))

    ax.set_xlabel(r'Cosine Similarity Threshold ($\cos\delta$)')
    ax.set_ylabel('Capacity')
    ax.set_yscale('log')
    ax.set_title('Capacity for ' + DATASET_LABELS[DATASETS.index(dataset)])
    ax.grid(True)

    temp = os.path.join(
        args.output_path,
        'plots_dir',
        dataset
    )
    if os.path.exists(temp) is False:
        os.makedirs(temp)
    fname = os.path.join(temp, f'{face_model}_{ref_dataset}_capacity')
    fig.savefig(fname+".png", dpi=300, bbox_inches='tight')
    if SAVE_PDF:
        fig.savefig(fname+".pdf", bbox_inches='tight')
    plt.close()
    plt.clf()

def plot_joint_capacity():
    for _, ref_dataset in enumerate(REF_DATASETS):
        for _, face_model in enumerate(FACE_MODELS):
            _, far, _, threshold = get_parameters(face_model, ref_dataset)
            fig, ax = plt.subplots(1, len(DATASETS), figsize=(5*len(DATASETS), 5))
            for d_id, dataset in enumerate(DATASETS):
                fname = os.path.join(
                    args.output_path,
                    'pkl_files',
                    dataset,
                    f'{face_model}_{ref_dataset}_capacity.pkl'
                )
                data = pd.read_pickle(fname)

                k_ = list(data.keys())
                for k in k_:
                    if not isinstance(data[k], pd.Series):
                        data[k] = pd.Series((_ for _ in data[k]))

                capacity = data["capacity"]

                for i in range(len(far)):
                    ax[d_id].plot([threshold[i], threshold[i]], [min(capacity), max(capacity)],
                                  '--', linewidth=2)

                ax[d_id].plot(data["cos_delta"], data["capacity"], "-s",
                              lw=2, color=(0.5, 0, 0))
                ax[d_id].set_yscale('log')
                ax[d_id].set_title(DATASET_LABELS[d_id])
                ax[d_id].xaxis.label.set_fontsize(20)
                ax[d_id].grid(True)

            fig.legend(["FAR of {:.2f}\%".format(val) for val in far],
                       loc='upper center', ncols=3,
                       bbox_to_anchor=(0.5, 1.1),
                       prop={'size': 21})

            ax[0].set_ylabel('Capacity')
            ax[0].yaxis.label.set_fontsize(20)
            fig.supxlabel(r'Cosine Similarity Threshold ($\cos\delta$)')

            temp = os.path.join(
                args.output_path,
                'plots_dir'
            )
            if os.path.exists(temp) is False:
                os.makedirs(temp)
            fname = os.path.join(temp,f'{face_model}_{ref_dataset}_joint_capacity')
            fig.savefig(fname+".png", dpi=300, bbox_inches='tight')
            if SAVE_PDF:
                fig.savefig(fname+".pdf", bbox_inches='tight')
            plt.close()
            plt.clf()

def plot_capacity_gender(face_model, ref_dataset, dataset):
    _, far, _, threshold = get_parameters(face_model, ref_dataset)
    fname = os.path.join(
        args.output_path,
        'pkl_files',
        dataset,
        f'{face_model}_{ref_dataset}_capacity_gender.pkl'
    )
    data = pd.read_pickle(fname)

    k_ = list(data.keys())
    for k in k_:
        if not isinstance(data[k], pd.Series):
            data[k] = pd.Series((_ for _ in data[k]))

    capacity = data["capacity"]

    # plot capacity as a function of delta in log-scale
    fig, ax = plt.subplots(1, 1)

    for i in range(len(far)):
        plt.plot([threshold[i], threshold[i]], [min(capacity), max(capacity)], '--', linewidth=2)
    plt.legend(["FAR of {:.2f}\%".format(val) for val in far], loc='upper left')

    cols = plt.cm.tab10(np.array([3, 4]))
    cols = ((116/255., 31/255., 146/255.), (15/255., 142/255., 4/255.))
    m_plot, = plt.plot(data["cos_delta"][data["gender"] == "male"], data["capacity"][data["gender"] == "male"],
             "-s", lw=2, color=cols[0])
    f_plot, = plt.plot(data["cos_delta"][data["gender"] == "female"], data["capacity"][data["gender"] == "female"],
             "-s", lw=2, color=cols[1])

    plt.gca().add_artist(plt.legend([m_plot, f_plot], ["Male", "Female"], loc='lower right'))
    ax.set_xlabel(r'Cosine Similarity Threshold ($\cos\delta$)')
    ax.set_ylabel('Capacity')
    ax.set_yscale('log')
    ax.set_title('Capacity for ' + DATASET_LABELS[DATASETS.index(dataset)] + ' w.r.t. gender')
    ax.grid(True)

    temp = os.path.join(
        args.output_path,
        'plots_dir',
        dataset
    )
    if os.path.exists(temp) is False:
        os.makedirs(temp)
    fname = os.path.join(temp, f'{face_model}_{ref_dataset}_capacity_gender')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    if SAVE_PDF:
        fig.savefig(fname+".pdf", bbox_inches='tight')
    plt.close()
    plt.clf()

def plot_joint_capacity_gender():
    cols = ((116/255., 31/255., 146/255.), (15/255., 142/255., 4/255.))

    for _, ref_dataset in enumerate(REF_DATASETS):
        for _, face_model in enumerate(FACE_MODELS):
            _, far, _, threshold = get_parameters(face_model, ref_dataset)
            fig, ax = plt.subplots(1, len(DATASETS), figsize=(5*len(DATASETS), 5))
            for d_id, dataset in enumerate(DATASETS):
                fname = os.path.join(
                    args.output_path,
                    'pkl_files',
                    dataset,
                    f'{face_model}_{ref_dataset}_capacity_gender.pkl'
                )
                data = pd.read_pickle(fname)

                k_ = list(data.keys())
                for k in k_:
                    if not isinstance(data[k], pd.Series):
                        data[k] = pd.Series((_ for _ in data[k]))

                capacity = data["capacity"]

                far_handles = []
                for i in range(len(far)):
                    _, = ax[d_id].plot([threshold[i], threshold[i]], [min(capacity), max(capacity)],
                                  '--', linewidth=2)
                    far_handles.append(_)

                m_plot, = ax[d_id].plot(data["cos_delta"][data["gender"] == "male"],
                                        data["capacity"][data["gender"] == "male"],
                                        "-s", lw=2, color=cols[0])
                f_plot, = ax[d_id].plot(data["cos_delta"][data["gender"] == "female"],
                                        data["capacity"][data["gender"] == "female"],
                                        "-s", lw=2, color=cols[1])
                ax[d_id].set_yscale('log')
                ax[d_id].set_title(DATASET_LABELS[d_id])
                ax[d_id].xaxis.label.set_fontsize(20)
                ax[d_id].grid(True)

            fig.supxlabel(r'Cosine Similarity Threshold ($\cos\delta$)')
            fig.legend([f_plot, m_plot],
                       ["Male", "Female"],
                       loc='upper left', ncols=2,
                       bbox_to_anchor=(0.2, 1.1),
                       prop={'size': 21})

            far_labels = [r"FAR={:.2f}\%".format(val) for val in far]
            plt.gca().add_artist(plt.legend(far_handles, far_labels, loc='upper right', ncols=3,
                                     bbox_to_anchor=(0.7, 1.3), prop={"size": 21}))

            ax[0].set_ylabel('Capacity')
            ax[0].yaxis.label.set_fontsize(20)
            
            temp = os.path.join(
                args.output_path,
                'plots_dir'
            )
            if os.path.exists(temp) is False:
                os.makedirs(temp)
            
            fname = os.path.join(temp, f'{face_model}_{ref_dataset}_joint_capacity_gender')
            fig.savefig(fname+".png", dpi=300, bbox_inches='tight')
            if SAVE_PDF:
                fig.savefig(fname+".pdf", bbox_inches='tight')
            plt.close()
            plt.clf()


def plot_capacity_age(face_model, ref_dataset, dataset):
    _, far, _, threshold = get_parameters(face_model, ref_dataset)
    fname = os.path.join(
        args.output_path,
        'pkl_files',
        dataset,
        f'{face_model}_{ref_dataset}_capacity_age.pkl'
    )
    data = pd.read_pickle(fname)

    k_ = list(data.keys())
    for k in k_:
        if not isinstance(data[k], pd.Series):
            data[k] = pd.Series((_ for _ in data[k]))

    capacity = data["capacity"]

    # plot capacity as a function of delta in log-scale
    fig, ax = plt.subplots(1, 1)
    for i in range(len(far)):
        plt.plot([threshold[i], threshold[i]], [min(capacity), max(capacity)], '--', linewidth=2)    

    colors = plt.cm.rainbow(np.linspace(0, 1, len(data["age"].unique())))
    age_plots = []
    for i, a in enumerate(data["age"].unique()):
        mask = data["age"] == a
        a_plot, = plt.plot(data["cos_delta"][mask], data["capacity"][mask],
                 "-s", lw=2, color=colors[i])
        age_plots.append(a_plot)

    legend1 = plt.legend(["FAR of {:.2f}\%".format(val) for val in far],
                         loc='lower right')
    # plt.legend(age_plots, [f"Age {_}" for _ in data["age"].unique()], loc='upper left', ncols=2)
    plt.legend(age_plots, [f"Age {_}" for _ in data["age"].unique()], loc='upper left')
    plt.gca().add_artist(legend1)

    ax.set_xlabel(r'Cosine Similarity Threshold ($\cos\delta$)')
    ax.set_ylabel('Capacity')
    ax.set_yscale('log')
    ax.set_title('Capacity for ' + DATASET_LABELS[DATASETS.index(dataset)] + ' w.r.t. age')
    ax.grid(True)

    temp = os.path.join(
        args.output_path,
        'plots_dir',
        dataset
    )
    if os.path.exists(temp) is False:
        os.makedirs(temp)
    
    fname = os.path.join(temp, f'{face_model}_{ref_dataset}_capacity_age')
    fig.savefig(fname+".png", dpi=300, bbox_inches='tight')
    if SAVE_PDF:
        fig.savefig(fname+".pdf", bbox_inches='tight')
    plt.close()
    plt.clf()

def plot_joint_capacity_age():
    colors = plt.cm.rainbow(np.linspace(0, 1, 8))
    age_plots = []
    for _, ref_dataset in enumerate(REF_DATASETS):
        for _, face_model in enumerate(FACE_MODELS):
            _, far, _, threshold = get_parameters(face_model, ref_dataset)
            fig, ax = plt.subplots(1, len(DATASETS), figsize=(5*len(DATASETS), 5))
            for d_id, dataset in enumerate(DATASETS):
                
                if dataset.lower() != "dcface":
                    fname = os.path.join(
                        args.output_path,
                        'pkl_files',
                        dataset,
                        f'{face_model}_{ref_dataset}_capacity_age.pkl'
                    )
                    data = pd.read_pickle(fname)

                    k_ = list(data.keys())
                    for k in k_:
                        if not isinstance(data[k], pd.Series):
                            data[k] = pd.Series((_ for _ in data[k]))

                    capacity = data["capacity"]

                    far_handles = []
                    for i in range(len(far)):
                        _, = ax[d_id].plot([threshold[i], threshold[i]], [min(capacity), max(capacity)],
                                    '--', linewidth=2)
                        far_handles.append(_)

                    temp_age_plots = []
                    for i, a in enumerate(data["age"].unique()):
                        mask = data["age"] == a
                        a_plot, = ax[d_id].plot(data["cos_delta"][mask], data["capacity"][mask],
                                "-s", lw=2, color=colors[i])
                        temp_age_plots.append(a_plot)

                    if len(temp_age_plots) > len(age_plots):
                        age_plots = temp_age_plots.copy()

                    ax[d_id].set_yscale('log')
                    ax[d_id].set_title(DATASET_LABELS[d_id])
                    ax[d_id].xaxis.label.set_fontsize(20)
                    ax[d_id].grid(True)

            fig.supxlabel(r'Cosine Similarity Threshold ($\cos\delta$)')
            fig.legend(age_plots, [f"Age {int(_)-10}-{_}" for _ in data["age"].unique()],
                       loc='upper left', ncols=4,
                       bbox_to_anchor=(0.12, 1.16),
                       prop={'size': 21})

            far_labels = [r"FAR={:.2f}\%".format(val) for val in far]
            plt.gca().add_artist(plt.legend(far_handles, far_labels, loc='upper right', ncols=3,
                                     bbox_to_anchor=(0.84, 1.3), prop={"size": 21}))

            ax[0].set_ylabel('Capacity')
            ax[0].yaxis.label.set_fontsize(20)
            
            temp = os.path.join(
                args.output_path,
                'plots_dir'
            )
            if os.path.exists(temp) is False:
                os.makedirs(temp)
            
            fname = os.path.join(temp, f'{face_model}_{ref_dataset}_joint_capacity_age')
            fig.savefig(fname+".png", dpi=300, bbox_inches='tight')
            if SAVE_PDF:
                fig.savefig(fname+".pdf", bbox_inches='tight')
            plt.close()
            plt.clf()

def plot_all_results(face_model, ref_dataset, dataset):
    plot_capacity(face_model, ref_dataset, dataset)
    plot_capacity_gender(face_model, ref_dataset, dataset)
    if dataset != "dcface":
        plot_capacity_age(face_model, ref_dataset, dataset)

def get_all_tabular_values(dataset, ref_dataset, face_model):
    fname = os.path.join(
        args.output_path,
        'pkl_files',
        dataset,
        f'{face_model}_{ref_dataset}_capacity.pkl'
    )
    data = pd.read_pickle(fname)

    k_ = list(data.keys())
    for k in k_:
        if not isinstance(data[k], pd.Series):
            data[k] = pd.Series((_ for _ in data[k]))

    capacity = data["capacity"]
    cos_delta = data["cos_delta"]

    return cos_delta, capacity

def plot_tabular_data(tabular_data):
    # Plot the effect of varying face model and reference dataset for a
    # given FAR value and generative model

    for d_id, dataset in enumerate(DATASETS):
        for f_id, far in enumerate(FARS):
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

            im = ax.imshow(tabular_data[:, :, d_id, f_id], cmap="viridis",
                           norm=mpl.colors.LogNorm())
            ax.set_xticks(np.arange(len(FACE_MODELS)), FACE_MODEL_LABELS)
            ax.set_yticks(np.arange(len(REF_DATASETS)), REF_DATASET_LABELS)
            ax.grid(False)

            fig.colorbar(im, ax=ax)
            temp = os.path.join(
                args.output_path,
                'ablations_dir',
                dataset
            )
            if os.path.exists(temp) is False:
                os.makedirs(temp)
            fname = os.path.join(temp, f'facemodel_reference_{far}')
            fig.savefig(fname+".png", dpi=300, bbox_inches='tight')
            if SAVE_PDF:
                fig.savefig(fname+".pdf", bbox_inches='tight')
            plt.close()
            plt.clf()

    fig, ax = plt.subplots(1, len(DATASETS), figsize=(4*len(DATASETS), 4))

    vmin = np.min(tabular_data[..., 0])
    vmax = np.max(tabular_data[..., 0])
    for d_id, dataset in enumerate(DATASETS):
        far = 0.1
        f_id = 0

        im = ax[d_id].imshow(tabular_data[:, :, d_id, f_id], cmap="viridis",
                             norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))

        ax[d_id].set_xticks(np.arange(len(FACE_MODELS)), FACE_MODEL_LABELS, rotation=70)
        ax[d_id].set_yticks(np.arange(len(REF_DATASETS)), REF_DATASET_LABELS)
        ax[d_id].grid(False)
        ax[d_id].set_title(f"{DATASET_LABELS[d_id]}")

    cax = fig.add_axes((0.9, 0.1, 0.03, 0.8))
    fig.colorbar(im, cax=cax)
    
    temp = os.path.join(
        args.output_path,
        'ablations_dir',
        dataset
    )
    if os.path.exists(temp) is False:
        os.makedirs(temp)
    fname = os.path.join(temp, f'facemodel_overall_{far}')
    fig.savefig(fname+".png", dpi=300, bbox_inches='tight')
    if SAVE_PDF:
        fig.savefig(fname+".pdf", bbox_inches='tight')
    plt.close()
    plt.clf()

def plot_tabular_data_at_cos_similarity():
    fig, ax = plt.subplots(1, len(DATASETS), figsize=(4*len(DATASETS), 4))

    cos_value = 0.75 # interpolate capacity at cos_value
    tabular_data = np.zeros((len(DATASETS), len(REF_DATASETS), len(FACE_MODELS)))

    for d_id, dataset in enumerate(DATASETS):
        for r_id, ref_dataset in enumerate(REF_DATASETS):
            for f_id, face_model in enumerate(FACE_MODELS):
                cos_delta_, capacity_ = get_all_tabular_values(dataset, ref_dataset, face_model)
                fn = interpolate.interp1d(cos_delta_, capacity_, kind='slinear')
                cap_ = fn(cos_value)
                tabular_data[d_id, r_id, f_id] = cap_
    
    vmin = np.min(tabular_data)
    vmax = np.max(tabular_data)

    for d_id, dataset in enumerate(DATASETS):
        im = ax[d_id].imshow(tabular_data[d_id], cmap="viridis",
                             norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
        ax[d_id].set_xticks(np.arange(len(FACE_MODELS)), FACE_MODEL_LABELS, rotation=70)
        ax[d_id].set_yticks(np.arange(len(REF_DATASETS)), REF_DATASET_LABELS)
        ax[d_id].grid(False)
        ax[d_id].set_title(f"{DATASET_LABELS[d_id]}")

    cax = fig.add_axes((0.9, 0.1, 0.03, 0.8))
    fig.colorbar(im, cax=cax)
    
    temp = os.path.join(
        args.output_path,
        'ablations_dir',
        dataset
    )
    if os.path.exists(temp) is False:
        os.makedirs(temp)
    fname = os.path.join(temp, f'facemodel_overall_at_cos_sim_{cos_value}')
    fig.savefig(fname+".png", dpi=300, bbox_inches='tight')
    if SAVE_PDF:
        fig.savefig(fname+".pdf", bbox_inches='tight')
    plt.close()
    plt.clf()

def plot_same_facemodel_capacity():
    marker_list = ["s", "^", "o", "v", "+", "D"]
    col_list = sns.color_palette("colorblind", len(DATASETS))

    print(len(DATASETS), len(REF_DATASETS), len(FACE_MODELS), len(marker_list), len(col_list))

    for _, ref_dataset in enumerate(REF_DATASETS):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        for f_id, face_model in enumerate(FACE_MODELS):
            handles = []
            _, far, _, threshold = get_parameters(face_model, ref_dataset)
            max_capacity_1 = -1
            max_capacity_2 = -1
            min_capacity_1 = np.inf
            min_capacity_2 = np.inf
            # Zoom into FAR @ 0.1%
            axins1 = axs[f_id].inset_axes([0.02, 0.58, 0.4, 0.4])
            # Zoom into FAR @ 10%
            axins2 = axs[f_id].inset_axes([0.58, 0.02, 0.4, 0.4])
            for d_id, dataset in enumerate(DATASETS):
                fname = os.path.join(
                    args.output_path,
                    'pkl_files',
                    dataset,
                    f'{face_model}_{ref_dataset}_capacity.pkl'
                )
                data = pd.read_pickle(fname)
                _, = axs[f_id].plot(data["cos_delta"], data["capacity"], marker=marker_list[d_id],
                               lw=2, color=col_list[d_id])
                axins1.plot(data["cos_delta"], data["capacity"], marker=marker_list[d_id],
                        lw=2, color=col_list[d_id])
                axins2.plot(data["cos_delta"], data["capacity"], marker=marker_list[d_id],
                        lw=2, color=col_list[d_id])

                # Calculate min and max capacity for this data
                fn = interpolate.interp1d(data["cos_delta"], data["capacity"], kind='cubic')
                max_capacity_1 = max(max_capacity_1, fn(threshold[0]))
                max_capacity_2 = max(max_capacity_2, fn(threshold[2]))
                min_capacity_1 = min(min_capacity_1, fn(threshold[0]))
                min_capacity_2 = min(min_capacity_2, fn(threshold[2]))
                handles.append(_)

            axs[f_id].set_yscale('log')
            cap_max = max_capacity_1*1.2
            cap_min = min_capacity_1/1.2
            cap_range = np.log(cap_max) - np.log(cap_min)
            x_range = axs[f_id].get_xlim()[1] - axs[f_id].get_xlim()[0]
            y_range = np.log(axs[f_id].get_ylim()[1]) - np.log(axs[f_id].get_ylim()[0])
            w = cap_range * x_range / y_range
            axins1.set_ylim(cap_min, cap_max)
            axins1.set_xlim(threshold[0]-w/2., threshold[0]+w/2.)
            axins1.set_title(r"FAR @ 0.1\%", y=0, pad=-12,
                             fontdict={"fontsize": 12})
            axins1.set_xticks([])
            axins1.set_yticks([])

            cap_max = max_capacity_2*1.2
            cap_min = min_capacity_2/1.2
            cap_range = np.log(cap_max) - np.log(cap_min)
            w = cap_range * x_range / y_range
            axins2.set_ylim(cap_min, cap_max)
            axins2.set_xlim(threshold[2]-w/2., threshold[2]+w/2.)
            axins2.set_title(r"FAR @ 10\%",
                             fontdict={"fontsize": 12})
            axins2.set_xticks([])
            axins2.set_yticks([])

            axs[f_id].indicate_inset_zoom(axins1, edgecolor="black")
            axs[f_id].indicate_inset_zoom(axins2, edgecolor="black")

            axs[f_id].set_title(f"{FACE_MODEL_LABELS[f_id]}")

        fig.legend(handles, DATASET_LABELS, loc='upper center',
                   ncol=5, bbox_to_anchor=(0.5, 1.05))
        axs[0].set_ylabel(r"Capacity")
        axs[0].set_xlabel(r"Cosine similarity threshold")
        axs[1].set_xlabel(r"Cosine similarity threshold")

        temp = os.path.join(
            args.output_path,
            'ablations_dir',
            dataset
        )
        if os.path.exists(temp) is False:
            os.makedirs(temp)
        fname = os.path.join(temp, f'same_facemodel_capacity_{ref_dataset}')
        fig.savefig(fname+".png", dpi=300, bbox_inches='tight')
        if SAVE_PDF:
            fig.savefig(fname+".pdf", bbox_inches='tight')
        plt.close()
        plt.clf()

    for _, ref_dataset in enumerate(REF_DATASETS):
        for f_id, face_model in enumerate(FACE_MODELS):
            handles = []
            _, far, _, threshold = get_parameters(face_model, ref_dataset)
            max_capacity_1 = -1
            max_capacity_2 = -1
            min_capacity_1 = np.inf
            min_capacity_2 = np.inf

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            
            for d_id, dataset in enumerate(DATASETS):
                fname = os.path.join(
                    args.output_path,
                    'pkl_files',
                    dataset,
                    f'{face_model}_{ref_dataset}_capacity.pkl'
                )
                data = pd.read_pickle(fname)
                _, = ax.plot(data["cos_delta"], data["capacity"], marker=marker_list[d_id],
                               lw=2, color=col_list[d_id])
                handles.append(_)

            far_handles = []
            for i in range(len(far)):
                _, = ax.plot([threshold[i], threshold[i]], ax.get_ylim(), "--", linewidth=2)
                far_handles.append(_)

            ax.set_yscale('log')
            ax.set_xlabel(r'Cosine Similarity Threshold ($\cos\delta$)')
            ax.set_ylabel('Capacity')

            fig.legend(handles, DATASET_LABELS, loc="upper left",
                       bbox_to_anchor=[0.11, 0.9],
                       ncol=1, alignment="center")

            far_labels = [r"FAR={:.2f}\%".format(val) for val in far]
            ax.add_artist(plt.legend(far_handles, far_labels, loc='lower right', ncols=1))

            temp = os.path.join(
                args.output_path,
                'ablations_dir',
                dataset
            )
            if os.path.exists(temp) is False:
                os.makedirs(temp)
            fname = os.path.join(temp, f'same_facemodel_capacity_{ref_dataset}_{face_model}')
            fig.savefig(fname+".png", dpi=300, bbox_inches='tight')
            if SAVE_PDF:
                fig.savefig(fname+".pdf", bbox_inches='tight')
            plt.close()
            plt.clf()

def plot_same_dataset_capacity():
    marker_list = ["s", "^", "o"]
    col_list = plt.cm.tab10(np.array([0, 1, 2]))

    for _, ref_dataset in enumerate(REF_DATASETS):
        for d_id, dataset in enumerate(DATASETS):
            fig, ax = plt.subplots(1, 1)
            handles = []
            for f_id, face_model in enumerate(FACE_MODELS):
                fname = os.path.join(
                    args.output_path,
                    'pkl_files',
                    dataset,
                    f'{face_model}_{ref_dataset}_capacity.pkl'
                )
                data = pd.read_pickle(fname)
                _, = ax.plot(data["cos_delta"], data["capacity"], marker=marker_list[f_id],
                               lw=2, color=col_list[f_id])
                handles.append(_)

            ax.set_title(f"{DATASET_LABELS[d_id]}")
            ax.set_xlabel(r'Cosine Similarity Threshold ($\cos\delta$)')
            ax.set_ylabel('Capacity')
            ax.set_yscale('log')
            ax.grid()

            fig.legend(handles, FACE_MODEL_LABELS, loc='upper left',
                       ncol=1, bbox_to_anchor=(0.3, 0.8))
            fig.tight_layout()

            temp = os.path.join(
                args.output_path,
                'ablations_dir',
                dataset
            )
            if os.path.exists(temp) is False:
                os.makedirs(temp)
            fname = os.path.join(temp, f'same_dataset_capacity_{ref_dataset}')
            fig.savefig(fname+".png", dpi=300, bbox_inches='tight')
            if SAVE_PDF:
                fig.savefig(fname+".pdf", bbox_inches='tight')
            plt.close()
            plt.clf()