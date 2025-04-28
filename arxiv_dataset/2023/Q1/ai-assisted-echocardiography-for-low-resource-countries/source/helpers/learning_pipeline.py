import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_class_distribution(dataset_obj, label_id):
    count_class_dict = {
        'BKGR': 0,
        "4CV": 0
    }

    for clip_index_i in range(len(dataset_obj)):
        data_idx = dataset_obj[clip_index_i]
        label_id_idx = data_idx[1]
        label = label_id[label_id_idx]
        count_class_dict[label] += 1
        # count_class_dict[label]+= 1* number_of_frames_per_segment_in_a_clip

    return count_class_dict


def plot_from_dict(dict_obj, plot_title, **kwargs):
    return sns.barplot(data=pd.DataFrame.from_dict([dict_obj]).melt(),
                       x="variable", y="value", hue="variable", **kwargs).set_title(plot_title)


def creating_pair_of_clips(dataset, label_id):
    number_of_clips = len(dataset)
    clips = []
    for clip_index in range(int(number_of_clips)):
        data_idx = dataset[clip_index]
        data_clip_idx = data_idx[0]
        label_clip_idx = data_idx[1]
        clip_frame_clip_idx = data_idx[2]
        n_available_frames_clip_idx = data_idx[3]
        print(
            f' CLIP:{clip_index:02d} of {label_id[label_clip_idx]} label for {data_clip_idx.size()} TOTAL_FRAMES: {n_available_frames_clip_idx} from clip_frame_clip_idx {clip_frame_clip_idx}')
        clips.append([data_clip_idx, label_clip_idx, clip_index, clip_frame_clip_idx, n_available_frames_clip_idx])

    return (clips)


def pair_clips_labels(clips):
    pair_clips_labels_ = []
    number_of_clips = len(clips)
    for clip_index_i_A in range(int(number_of_clips / 2)):
        clip_index_i_B = int(number_of_clips / 2) + clip_index_i_A
        print(f' pair_clips_labels[{clip_index_i_A}]-- BKRG:{clip_index_i_A}, 4CV:{clip_index_i_B}')
        data_clip_i_A = clips[clip_index_i_A][0]
        label_i_A = clips[clip_index_i_A][1]
        clip_i_A = clips[clip_index_i_A][2]
        number_of_frames_A = clips[clip_index_i_A][4]
        data_clip_i_B = clips[clip_index_i_B][0]
        label_i_B = clips[clip_index_i_B][1]
        clip_i_B = clips[clip_index_i_B][2]
        number_of_frames_B = clips[clip_index_i_B][4]
        pair_clips_labels_.append(
            [data_clip_i_A, label_i_A, clip_i_A, number_of_frames_A, data_clip_i_B, label_i_B, clip_i_B,
             number_of_frames_B])

    return (pair_clips_labels_)


def animate_clips(pair_clips_labels, label_id, NUMBER_OF_FRAMES_PER_SEGMENT_IN_A_CLIP,
                  interval_between_frames_in_milliseconds):
    # print(f' CLIP: for {label_id[pair_clips_labels[1]]} ')
    fig = plt.figure()
    pair_of_clip_index_i_frames = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    data_clip_tensor_A = pair_clips_labels[0]
    label_clip_A = pair_clips_labels[1]
    clip_i_A = pair_clips_labels[2]
    number_of_frames_A = pair_clips_labels[3]
    data_clip_tensor_B = pair_clips_labels[4]
    label_clip_B = pair_clips_labels[5]
    clip_i_B = pair_clips_labels[6]
    number_of_frames_B = pair_clips_labels[7]

    ax1.title.set_text(f' CLIP: {clip_i_A:02d}--\
    {label_id[label_clip_A]} with {number_of_frames_A} of \
    {NUMBER_OF_FRAMES_PER_SEGMENT_IN_A_CLIP} frames [for Subject ]')
    ax2.title.set_text(f' CLIP: {clip_i_B:02d}--\
    {label_id[label_clip_B]} with {number_of_frames_B} of \
    {NUMBER_OF_FRAMES_PER_SEGMENT_IN_A_CLIP} frames [for Subject ]')
    for frames_idx in range(data_clip_tensor_A[:, :, ...].size()[1]):
        imA = ax1.imshow(data_clip_tensor_A[:, frames_idx, ...].cpu().detach().numpy().transpose(1, 2, 0),
                         cmap=plt.get_cmap('gray'))
        imB = ax2.imshow(data_clip_tensor_B[:, frames_idx, ...].cpu().detach().numpy().transpose(1, 2, 0),
                         cmap=plt.get_cmap('gray'))
        pair_of_clip_index_i_frames.append([imA, imB])
    fig.tight_layout()
    # return fig, pair_of_clip_index_i_frames

    anim = animation.ArtistAnimation(fig, pair_of_clip_index_i_frames, interval=interval_between_frames_in_milliseconds,
                                     blit=True, repeat_delay=1000)
    return anim


def json2DataFrame(PATH, FILENAME, TYPE_str, FRAMES_PER_SEGMENT_IN_A_CLIP, BatchClips, LR, STR_VARIABLE_NAME,
                   RUN_VERSION):
    FULL_PATH_FILENAME = os.path.join(PATH, FILENAME)
    pd_read_json = pd.read_json(
        FULL_PATH_FILENAME,
        orient='columns',
        typ='series')

    Filtered_JSON_stats = pd_read_json[TYPE_str]
    DFDICT = pd.DataFrame.from_dict(Filtered_JSON_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})
    DFDICT.insert(1, 'LR', 'lr' + str(LR), True)
    DFDICT.insert(2, 'N_BatchClips', 'bc' + str(BatchClips), True)
    DFDICT.insert(3, 'FRXClips', str(FRAMES_PER_SEGMENT_IN_A_CLIP), True)
    DFDICT.rename(columns={"variable": "datatype"}, inplace=True)
    DFDICT.rename(columns={"value": STR_VARIABLE_NAME}, inplace=True)
    DFDICT.insert(6, 'RUN_NN', str(RUN_VERSION), True)
    return DFDICT


def jsonPARAMS2DataFrame(PATH, FILENAME):
    FULL_PATH_FILENAME = os.path.join(PATH, FILENAME)
    pd_read_json = pd.read_json(
        FULL_PATH_FILENAME,
        orient='columns',
        typ='series')
    return pd_read_json
