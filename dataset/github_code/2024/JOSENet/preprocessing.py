import os
from natsort import natsorted
import cv2



def create_segments_and_labels (paths, args):
    segments = {'train': [], 'val': [] }
    labels = {'train': [], 'val': [] }
    old_root = ""
    train_val = ['train', 'val']
    events = ['Fight', 'NonFight']
    for tv in train_val:
        for event_type in events:

            label = 1 if (event_type == 'Fight') else 0

            for root, dirs, files in os.walk(paths.jpg_frames+"/"+tv+"/"+event_type):
                new_root = root
                for file_name in natsorted(files):
                    path_to_image = os.path.join(root, file_name)
                    
                    # add to the list which should has length args.clip_frames
                    if (old_root == new_root):
                        new_segment.append(path_to_image)

                        if (len(new_segment) == args.clip_frames):
                            segments[tv].append(new_segment)
                            labels[tv].append(label)
                            new_segment = []
                    # add the a new list a new images
                    else:
                        new_segment = []
                        new_segment.append(path_to_image)
                        
                        old_root = new_root

    return segments, labels


def create_window_segments_and_labels (paths, args):
    segments = {'train': [], 'val': [] }
    labels = {'train': [], 'val': [] }
    old_root = ""
    train_val = ['train', 'val']
    events = ['Fight', 'NonFight']
    window_shift = 0
    for tv in train_val:
        for event_type in events:
            label = 1 if (event_type == 'Fight') else 0

            # Loop over all the folders with videos frames
            for root, dirs, files in os.walk(paths.jpg_frames+"/"+tv+"/"+event_type):
                new_root = root
                # Iterate among the frames of that video
                idx = 0
                files = natsorted(files)
                while (idx < len(files)):
                    if (idx % args.interval == 0):
                        file_name = files[idx]
                        path_to_image = os.path.join(root, file_name)
                        # add to the list which should has length args.clip_frames
                        # if we are still inside the same video...
                        if (old_root == new_root):
                            new_segment.append(path_to_image)
                            # If the length of the segment is over, just append it to the bigger list
                            if (len(new_segment) == args.clip_frames):
                                segments[tv].append(new_segment)
                                labels[tv].append(label)
                                new_segment = []
                                idx = idx - window_shift
                        # Otherwise, generate a new_segment and that frame append to it
                        else:
                            new_segment = []
                            new_segment.append(path_to_image)

                            old_root = new_root
                    idx += 1

    return segments, labels

def create_window_segments_and_labels_UCFCrime (paths, args):
    segments = {'train': [], 'val': [] }
    old_root = ""
    train_val = ['train', 'val']
    window_shift = 0
    for tv in train_val:
        # Loop over all the folders with videos frames
        for root, dirs, files in os.walk(paths.jpg_frames+"/"+tv):
            new_root = root
            # Iterate among the frames of that video
            idx = 0
            files = natsorted(files)
            while (idx < len(files)):
                if (idx % args.interval == 0):
                    file_name = files[idx]
                    path_to_image = os.path.join(root, file_name)

                    # add to the list which should has length args.clip_frames
                    # if we are still inside the same video...
                    if (old_root == new_root):
                        new_segment.append(path_to_image)

                        # If the length of the segment is over, just append it to the bigger list
                        if (len(new_segment) == args.clip_frames):
                            segments[tv].append(new_segment)
                            new_segment = []
                            idx = idx - window_shift
                    # Otherwise, generate a new_segment and that frame append to it
                    else:
                        new_segment = []
                        new_segment.append(path_to_image)

                        old_root = new_root

                idx += 1
    return segments


def generate_flow (prev, next):
    # Convert to gray scale
    prvs = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Capture another frame and convert to gray scale
    next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

    # Optical flow is now calculated
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow
