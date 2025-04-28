import json
import os
import random

import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm

from source.helpers.various import timer_func_decorator, msec_to_timestamp, to_grayscale, ToImageTensor, \
    cropped_frame, masks_us_image

# constants
S2MS = 1000  # second to millisecond


class EchoClassesDataset(torch.utils.data.Dataset):
    """
    EchoClassesDataset Class to load video and json labels using torch.utils.data.

    Arguments:
        echodataset_path(str): Main path of videos and json files
        temporal_data_path (str): Path for temporal data files
        participant_videos_list (srt):  Lists of video files
        participant_path_json_list (srt): List of json files
        crop_bounds_for_us_image (Tuple) - Crop bounds, a dictionary in format ('start_x':w0, 'start_y':h0, 'width':w, 'height':h).
        pretransform_im_size (Tuple) - 'width':w, 'height':h
        pretransform (torch.Transform): a transform, e.g. normalization, etc, that is done determninistically and before augmentation (Default = None)
        transform (torch.Transform): a transform, e.g. for data augmentation, normalization, etc (Default = None)
        number_of_frames_per_segment_in_a_clip (int): duration of the clips, in number of frames
        device: It could be torch.device('cpu') or torch.device('cuda:0') etc
        use_tmp_storage: It is set to true, saves clips into a tmp folder to speed up training.
        max_background_duration_in_secs: to limit their max length to 10 seconds. background clips might be long (60 seconds sampled at 30fps),

    """

    def __init__(
            self,
            echodataset_path: str,
            temporal_data_path: str,
            participant_videos_list: str,
            participant_path_json_list: str,
            crop_bounds_for_us_image=None,
            pretransform_im_size=None,
            pretransform=None,
            transform=None,
            number_of_frames_per_segment_in_a_clip: int = 20,
            sliding_window_length_in_percentage_of_frames_per_segment: float = 0.0,
            device=torch.device('cpu'),
            max_background_duration_in_secs: int = 10,
            use_tmp_storage=False
    ):
        self.echodataset_path = echodataset_path
        self.participant_videos_list = participant_videos_list
        self.participant_path_json_list = participant_path_json_list
        self.crop_bounds_for_us_image = crop_bounds_for_us_image
        self.transform = transform
        self.pretransform_im_size = pretransform_im_size
        self.pretransform = pretransform
        self.number_of_frames_per_segment_in_a_clip = number_of_frames_per_segment_in_a_clip
        self.sliding_window_length_in_percentage_of_frames_per_segment = sliding_window_length_in_percentage_of_frames_per_segment
        self.device = device
        self.use_tmp_storage = use_tmp_storage
        self.temporal_data_path = temporal_data_path + 'echovideodatafiles_FRAMESPERCLIP{}_PIXELSIZE_{}W{}H'.format(
            self.number_of_frames_per_segment_in_a_clip, self.pretransform_im_size[0], self.pretransform_im_size[1])
        self.max_background_duration_in_secs = max_background_duration_in_secs

        videolist = os.path.join(echodataset_path, participant_videos_list)
        annotationlist = os.path.join(echodataset_path, participant_path_json_list)

        self.video_filenames = [self.echodataset_path + os.sep + line.strip() for line in open(videolist)]
        self.annotation_filenames = [self.echodataset_path + os.sep + line.strip() for line in open(annotationlist)]

        self.BACKGROUND_LABEL = 0
        self.FOURCV_LABEL = 1
        self.MAX_BACKGROUND_DURATION_IN_MS = max_background_duration_in_secs * S2MS

        # read the json files to see where the labeled parts start and end.
        # As we read them, we will create a list of clips, called self.idx_to_clip
        self.idx_to_clip = []
        # Each entry of this list has the following information:
        #   [video_id, clip_id_within_video, start_time_ms, end_time_ms, label]
        # where the information means:
        #   video_id: integer with the index of the video as accessed in the self.video_filenames list
        #   clip_id_within_video: integer with the number of the clip within the video, if there is more than one
        #   start_time_ms and end_time_ms are self explanatory
        #   label: 0 (backgorund) or 1 (4 chamber)

        # also count how many samples there are from each class, to be able to balance later
        self.class_count = [0, 0]
        for video_id, json_filename_i in enumerate(self.annotation_filenames):
            with open(json_filename_i, "r") as json_file:
                json_data = json.load(json_file)
            # check that the annotation we need is encoded in the metadata
            if len(json_data['metadata']) == 0:
                print('[ERROR] [EchoClassesDataset.__init__()] Error reading {} (empty). Removing from list'.format(
                    json_filename_i))
                continue

            # get the video length. This will help define the segments outside the label as backgorund
            video_name = self.video_filenames[video_id]
            cap = cv.VideoCapture(video_name)
            if cap.isOpened() == False:
                print('[ERROR] [EchoClassesDataset.__init__()] Unable to read video ' + video_name)
                exit(-1)
            fps = cap.get(cv.CAP_PROP_FPS)
            frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            video_duration_i = frame_count / fps * S2MS

            # Print video features
            print(f'  ')
            print(f'  ')
            print(f'  ')
            print(f'  VIDEO_FEATURES')
            print(f'    video_name={video_name}')
            print(f'    Frame_height={frame_height}, frame_width={frame_width} fps={fps} nframes={frame_count} ')
            print(f'  ')
            print(f'  ')
            cap.release()

            # Now read the segments in the json file
            nclips_in_video = 0
            end_time_ms = 0
            for seg_i, segment in enumerate(json_data['metadata']):
                timestamps_of_labels = json_data['metadata'][segment]['z']
                start_time_ms = timestamps_of_labels[0] * S2MS
                # if for the first clip the start time is not 0, then there is a background clip before
                # for the clips that are not the first one,
                # if the start time is greater than the end time of the previous clip,
                # it means there is a background clip in between
                if start_time_ms > end_time_ms:
                    # limiting the duration of background clips with MAX_BACKGROUND_DURATION_IN_MS
                    number_of_frames_per_segment_in_a_clip = start_time_ms - end_time_ms
                    if number_of_frames_per_segment_in_a_clip > self.MAX_BACKGROUND_DURATION_IN_MS:
                        excess = number_of_frames_per_segment_in_a_clip - self.MAX_BACKGROUND_DURATION_IN_MS
                        background_clip = [video_id, nclips_in_video, end_time_ms + excess / 2,
                                           start_time_ms - excess / 2, self.BACKGROUND_LABEL]
                    else:
                        background_clip = [video_id, nclips_in_video, end_time_ms, start_time_ms, self.BACKGROUND_LABEL]
                    self.idx_to_clip.append(background_clip)
                    self.class_count[self.BACKGROUND_LABEL] += 1
                    nclips_in_video += 1
                end_time_ms = timestamps_of_labels[1] * S2MS
                # segments are always labeled as FOURCV, background is what remains
                entry = [video_id, nclips_in_video, start_time_ms, end_time_ms, self.FOURCV_LABEL]
                self.idx_to_clip.append(entry)
                self.class_count[self.FOURCV_LABEL] += 1
                nclips_in_video += 1
            # Last, if the end_time_ms of the last clip is earlier than the end of the video, then
            # there is a last background clip to be added
            if end_time_ms < video_duration_i:
                # background clips can be very long, so lets take, form the given interval, the central part
                # up to MAX_BACKGROUND_DURATION_IN_MS msec
                number_of_frames_per_segment_in_a_clip = video_duration_i - end_time_ms
                if number_of_frames_per_segment_in_a_clip > self.MAX_BACKGROUND_DURATION_IN_MS:
                    excess = number_of_frames_per_segment_in_a_clip - self.MAX_BACKGROUND_DURATION_IN_MS
                    background_clip = [video_id, nclips_in_video, end_time_ms + excess / 2,
                                       video_duration_i - excess / 2,
                                       self.BACKGROUND_LABEL]
                else:
                    background_clip = [video_id, nclips_in_video, end_time_ms, video_duration_i, self.BACKGROUND_LABEL]
                self.idx_to_clip.append(background_clip)
                self.class_count[self.BACKGROUND_LABEL] += 1
                nclips_in_video += 1

        # most likely there will be many more samples of BACKGROUND than FOURCV,
        # so we want to balance the dataset by randomly removing background samples until they
        # are the same amount
        if self.class_count[self.BACKGROUND_LABEL] > self.class_count[self.FOURCV_LABEL]:

            # extract the submatrix where the label is this
            idx_to_clip_i = [[] for i in range(len(self.class_count))]
            for sample in self.idx_to_clip:
                l = sample[-1]
                idx_to_clip_i[l].append(sample)

            idx_to_clip_smaller = []
            # now randomly permute the background list
            background_sublist = random.sample(idx_to_clip_i[self.BACKGROUND_LABEL],
                                               self.class_count[self.FOURCV_LABEL])
            self.class_count[self.BACKGROUND_LABEL] = len(background_sublist)
            idx_to_clip_smaller.append(background_sublist)
            idx_to_clip_smaller.append(idx_to_clip_i[self.FOURCV_LABEL])
            self.idx_to_clip = [item for sublist in idx_to_clip_smaller for item in sublist]

        self.samples_count = len(self.idx_to_clip)

    def __len__(self):
        return self.samples_count

    def __getitem__(self, index: int):
        """
        Arguments:
            index (int): index position of the clip (not the video) to return the data

        Returns
           video_data clip (tensor): video data clip with background or fourchamberview, for file 'index',
        """

        video_idx = self.idx_to_clip[index][0]
        video_name = self.video_filenames[video_idx]
        clip_earliest_start_ms = self.idx_to_clip[index][2]
        clip_latest_end_ms = self.idx_to_clip[index][3]
        clip_label = self.idx_to_clip[index][4]
        video_file_name_id = os.path.splitext(os.path.basename(video_name))[0][10:15]
        prefix = self.participant_videos_list.split('.')[0].split('_')[-1]
        save_filename = self.temporal_data_path + '/videoID_{:02d}_{}_label_{:02d}.pth'.format(index, video_file_name_id, clip_label)
        if self.use_tmp_storage is True and os.path.isfile(save_filename) is True:
            video_data = torch.load(save_filename)
        else:
            cap = cv.VideoCapture(video_name)
            if cap.isOpened() == False:
                print('[ERROR] [EchoClassesDataset.__getitem__()] Unable to read video ' + video_name)
                exit(-1)

            # extract the frames of the clip
            frames_torch = []

            cap.set(cv.CAP_PROP_POS_MSEC, clip_earliest_start_ms)
            while True:
                success, frame = cap.read()
                if not success and len(frames_torch) < 1:
                    print('[ERROR] [EchoClassesDataset.__getitem__()] Video {} has less than 1 frame, skipping'.format(
                        video_name))
                    exit(-1)
                    # the video has finished
                    break

                if frame is None:
                    # no frame here! video has finished
                    break

                msec = cap.get(cv.CAP_PROP_POS_MSEC)

                if msec > clip_latest_end_ms:
                    break

                gray_frame = to_grayscale(frame)
                masked_gray_frame = masks_us_image(gray_frame)
                if self.crop_bounds_for_us_image is not None:
                    cropped_masked_gray_frame = cropped_frame(masked_gray_frame, self.crop_bounds_for_us_image)

                if self.pretransform is not None:
                    frame_torch = self.pretransform(cropped_masked_gray_frame).squeeze()
                else:
                    # image to tensor is covered by a pretransform (which we may always use) but leaving this here just in case.
                    frame_torch = ToImageTensor(cropped_masked_gray_frame).squeeze()

                frames_torch.append(frame_torch)
            cap.release()
            # make a  tensor of the clip
            video_data = torch.stack(frames_torch)

            if self.use_tmp_storage is True and os.path.isfile(save_filename) is False:
                if not os.path.isdir(self.temporal_data_path):
                    print(
                        '[INFO] [EchoClassesDataset.__getitem__()] - computing clips and saving to temporary folder {}'.format(
                            self.temporal_data_path))
                    os.makedirs(self.temporal_data_path, exist_ok=False)
                torch.save(video_data, save_filename)

        # Now extract a random segment of number_of_frames_per_segment_in_a_clip from the video data.
        # This could be done perhaps using data augmentation
        # clip_duration_ms = self.number_of_frames_per_segment_in_a_clip / self.video_framerate * S2MS
        n_available_frames = video_data.shape[0]
        if n_available_frames < self.number_of_frames_per_segment_in_a_clip:
            clip_frame0 = 0
            clip_frame1 = n_available_frames
        else:
            # randomly pick a window segment within the clip
            latest_start_frame = np.max((n_available_frames - self.number_of_frames_per_segment_in_a_clip, 0))
            earliest_end_frame = self.number_of_frames_per_segment_in_a_clip
            clip_frame0 = int(np.random.uniform(np.min((latest_start_frame, earliest_end_frame)),
                                                np.max((latest_start_frame, earliest_end_frame))))
            clip_frame1 = np.min((clip_frame0 + self.number_of_frames_per_segment_in_a_clip, n_available_frames))
        clip_data = video_data[clip_frame0:clip_frame1, ...]

        if clip_data.shape[0] == 0:
            # we should never get here
            print('[ERROR] [EchoClassesDataset.__get_item__()] Empty video in {}'.format(index))
            exit(-1)

        if clip_data.shape[0] < self.number_of_frames_per_segment_in_a_clip:
            # if < 30frames, add the last one over, as if the user had frozen
            extra_frames = [clip_data[-1, ...].unsqueeze(0)] * (
                    self.number_of_frames_per_segment_in_a_clip - clip_data.shape[0])
            clip_data = torch.cat([clip_data] + extra_frames, dim=0)

        if self.transform is not None:
            clip_data_transformed = []
            for f in range(clip_data.shape[0]):
                if f == 0:
                    # save the state of the transform, so that all frames are augmented in the same way
                    state = torch.get_rng_state()
                else:
                    torch.set_rng_state(state)
                clip_data_transformed_f = self.transform(clip_data[f, ...]).squeeze()
                clip_data_transformed.append(clip_data_transformed_f)
            clip_data = torch.stack(clip_data_transformed)

        ## Sliding window for of random segment
        slided_window_segment = []
        if self.sliding_window_length_in_percentage_of_frames_per_segment > 0.0 and self.sliding_window_length_in_percentage_of_frames_per_segment < 1.0:
            sliding_window_length_in_frame_length = int(
                self.number_of_frames_per_segment_in_a_clip * self.sliding_window_length_in_percentage_of_frames_per_segment)
            # print( f' sliding_window_length: {sliding_window_length_in_frame_length}' )
            for i in range(0, self.number_of_frames_per_segment_in_a_clip):
                max_window_length_i = i + (sliding_window_length_in_frame_length - 1)
                if max_window_length_i < self.number_of_frames_per_segment_in_a_clip:
                    # print('window_idx',i, max_window_length_i)
                    window_i = clip_data[i: max_window_length_i + 1]
                    slided_window_segment.append(window_i)
            clip_data = torch.cat(slided_window_segment, dim=0)

        clip_data = clip_data.to(self.device)  # Tensor.Size([Fi,H,W])
        clip_data = clip_data.unsqueeze(0)  # add channel data Tensor.Size([C,Fi,H,W])
        # clip_data = clip_data.permute(1, 0, 2, 3) # Permute Tensor.Size([Fi,C,H,W])

        return clip_data, clip_label, clip_frame0, n_available_frames


class ViewVideoDataset(torch.utils.data.Dataset):
    """
    ViewVideoDataset Class for loading video using torch.utils.data.
    """

    def __init__(
            self,
            participants_videos_path: str,
            crop_bounds=None,
            transform=None
    ):
        """
        Arguments

        participant_videos_path (srt):  the folder where there input files are.
        crop_bounds - Crop bounds, a tuple in format (w0, h0, w, h).
        transform (torch.Transform): a transform, e.g. for data augmentation, normalization, etc (Default = None)
        """

        self.participants_videos_path = participants_videos_path
        self.crop_bounds = crop_bounds
        self.transform = transform

        self.video_filenames = []

        for Participant_i in enumerate(sorted(os.listdir(self.participants_videos_path))):
            part_i_path = self.participants_videos_path + '/' + Participant_i[1]
            for T_days_i in enumerate(sorted(os.listdir(part_i_path))):
                days_i_path = part_i_path + '/' + T_days_i[1]
                for video_file_name_i in sorted(os.listdir(days_i_path)):
                    path_video_file_name_i = days_i_path + '/' + video_file_name_i
                    if path_video_file_name_i.endswith('.mp4'):
                        self.video_filenames += [path_video_file_name_i]

    def __len__(self):
        return len(self.video_filenames)

    def __getitem__(self, video_index: int):
        """
        Arguments:
            video_index (int): video_index position to return the data

        Returns:
            video_data clip (tensor): vide data clip with the 4ch view, for file 'video_index',
        """

        video_name = self.video_filenames[video_index]
        print(video_name)

        cap = cv.VideoCapture(video_name)
        if cap.isOpened() == False:
            print('[ERROR] [ViewVideoDataset.__getitem__()] Unable to open video ' + video_name)
            exit(-1)

        # Get parameters of input video
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(np.ceil(cap.get(cv.CAP_PROP_FPS)))
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        # Print video features
        print(f'  ')
        print(f'  ')
        print(f'  ')
        print(f'  VIDEO_FEATURES')
        print(f'    video_name={video_name}')
        print(f'    Frame_height={frame_height}, frame_width={frame_width} fps={fps} nframes={frame_count} ')
        print(f'  ')
        print(f'  ')

        start_frame_number = 4000
        end_frame_number = 4403
        total_number_of_frames = end_frame_number - start_frame_number

        if start_frame_number >= end_frame_number:
            raise Exception("start frame number must be less than end frame number")

        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame_number)

        frames_torch = []

        pbar = tqdm(total=total_number_of_frames - 1)
        while cap.isOpened():
            success, image_frame_3ch_i = cap.read()

            if not success and len(frames_torch) < 1:
                print(
                    '[ERROR] [VideoDataset.__getitem__()] Video {} has less than 1 frame, skipping'.format(video_name))
                exit(-1)
                break

            if image_frame_3ch_i is None:
                # no frame here! video is finished
                break

            if cap.get(cv.CAP_PROP_POS_FRAMES) >= end_frame_number:
                break

            # frame_msec = cap.get(cv.CAP_PROP_POS_MSEC)
            # current_frame_timestamp = msec_to_timestamp(frame_msec)

            frame_gray = to_grayscale(image_frame_3ch_i)
            masked_frame = masks_us_image(frame_gray)
            cropped_image_frame_ = cropped_frame(masked_frame, self.crop_bounds)

            # cv.imshow('window', cropped_image_frame_)
            # cv.waitKey()

            frame_torch = ToImageTensor(cropped_image_frame_)
            frames_torch.append(frame_torch.detach())

            pbar.update(1)

        pbar.close()
        cap.release()

        video_data = torch.stack(frames_torch)  # "Fi,C,H,W"

        if self.transform is not None:
            video_data = self.transform(video_data)

        return video_data


class EchoVideoDataset(torch.utils.data.Dataset):
    """
    EchoVideoDataset Class for Loading Video using torch.utils.data.
    """

    def __init__(
            self,
            participant_videos_path: str,
            participant_path_json_files: str,
            transform=None
    ):
        """
        Arguments

        participant_videos_path (srt):  the folder where there input files are.
        participant_path_json_files (str): text file with the names of json annotation files, in the same order as the video_list.
        transform (torch.Transform): a transform, e.g. for data augmentation, normalization, etc (Default = None)
        """

        self.participant_videos_path = participant_videos_path
        self.participant_path_json_files = participant_path_json_files
        self.transform = transform

        self.video_filenames = []
        for T_days_i in enumerate(sorted(os.listdir(self.participant_videos_path))):
            days_i_path = self.participant_videos_path + '/' + T_days_i[1]
            for video_file_name_i in sorted(os.listdir(days_i_path)):
                path_video_file_name_i = days_i_path + '/' + video_file_name_i
                if path_video_file_name_i.endswith('.mp4'):
                    self.video_filenames += [path_video_file_name_i]

        self.annotation_filenames = []
        for json_i in enumerate(sorted(os.listdir(self.participant_path_json_files))):
            self.annotation_filenames += [self.participant_path_json_files + '/' + json_i[1]]

    def __len__(self):
        return len(self.video_filenames)

    @timer_func_decorator
    def __getitem__(self, video_index: int):
        """
        Arguments:
            video_index (int): video_index position to return the data

        Returns:
           video_data clip (tensor): vide data clip with the 4ch view, for file 'video_index',
        """

        image_frame_index = 0
        video_name = self.video_filenames[video_index]
        jsonfile_name = self.annotation_filenames[video_index]

        cap = cv.VideoCapture(video_name)
        if cap.isOpened() == False:
            print('[ERROR] [ViewVideoDataset.__getitem__()] Unable to open video ' + video_name)
            exit(-1)

        # Get parameters of input video
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(np.ceil(cap.get(cv.CAP_PROP_FPS)))
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        # Print video features
        print(f'  ')
        print(f'  ')
        print(f'  video_name={video_name}')
        print(f'  Frame_height={frame_height}, frame_width={frame_width} fps={fps} nframes={frame_count} ')
        print(f'  jsonfile_name={jsonfile_name}')

        ## Extracting timestams in json files for labelled of four chamber views (4CV)
        start_label_timestamps_ms = []
        end_label_timestamps_ms = []
        with open(jsonfile_name, "r") as json_file:
            json_data = json.load(json_file)
            if len(json_data['metadata']) == 0:  ## Check if the metadata is empty
                id_clip_to_extract = 0
                print(f'')
                print(f'  No 4CV labels for {json_file_i}')
                print(f'')
                # break
            else:
                for key in json_data['metadata']:
                    timestamps_of_labels = json_data['metadata'][key]['z']
                    start_label_ms = timestamps_of_labels[0] * S2MS
                    end_label_ms = timestamps_of_labels[1] * S2MS
                    start_label_timestamps_ms.append(start_label_ms)
                    end_label_timestamps_ms.append(end_label_ms)

        number_of_labelled_clips = int(len(start_label_timestamps_ms))
        print(f'  number_of_labelled_clips={number_of_labelled_clips}')
        print(f'  ')
        print(f'  ')

        frames_torch = []
        pbar = tqdm(total=frame_count)
        while True:
            success, image_frame_3ch_i = cap.read()
            if (success == True):

                frame_msec = cap.get(cv.CAP_PROP_POS_MSEC)
                current_frame_timestamp = msec_to_timestamp(frame_msec)
                frame_gray = to_grayscale(image_frame_3ch_i)
                frame_torch = ToImageTensor(frame_gray)

                #### PLAYGROUND
                ## cap.set(cv.CAP_PROP_POS_MSEC, start_label_timestamps_ms[id_clip_to_extract])

                # ## condition for  minute_label
                for clips_i in range(0, number_of_labelled_clips):
                    if (current_frame_timestamp[0] >= int(msec_to_timestamp(start_label_timestamps_ms[clips_i])[0])) & (
                            current_frame_timestamp[0] <= int(msec_to_timestamp(end_label_timestamps_ms[clips_i])[0])):
                        # condition for second label
                        if (current_frame_timestamp[1] >= int(
                                msec_to_timestamp(start_label_timestamps_ms[clips_i])[1])) & (
                                current_frame_timestamp[1] <= int(
                            msec_to_timestamp(end_label_timestamps_ms[clips_i])[1])):
                            # print(
                            #     f'  clip {clips_i}; image_frame_index {image_frame_index}, frame_msec {frame_msec}, current_frame_timestamp {current_frame_timestamp}')
                            frames_torch.append(frame_torch)

                pbar.update(1)
                image_frame_index += 1

            else:
                break

        pbar.close()
        cap.release()

        video_data = torch.stack(frames_torch)  # "Fi,C,H,W"
        # video_data = video_data.squeeze() # "Fi,H,W" for one channel

        if self.transform is not None:
            video_data = self.transform(video_data)

        return video_data
