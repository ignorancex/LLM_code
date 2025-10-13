import random
import matplotlib.pyplot as plt
import numpy as np
import pdb
import torch
import torch.nn.functional as F


def valid_crop_resize(data_numpy,valid_frame_num,p_interval,window):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    #crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1-p) * valid_size/2)
        data = data_numpy[:, begin+bias:end-bias, :, :]# center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)),64), valid_size)# constraint cropped_length lower bound as 64
        bias = np.random.randint(0,valid_size-cropped_length+1)
        data = data_numpy[:, begin+bias:begin+bias+cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data,dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data

def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]

def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def _rot(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros),dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim = 1)  # T,3,3

    ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 1)

    rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 1)

    rot = rz.matmul(ry).matmul(rx)
    return rot


def random_rot(data_numpy, theta=0.3):
    """
    data_numpy: C,T,V,M
    """
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V*M)  # T,3,V*M
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot, ] * T, dim=0)
    rot = _rot(rot)  # T,3,3
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()
    return data_torch


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


def random_occlusion(data_numpy, p=0.5, min_occlusion_length=10, max_occlusion_length=50):
    """
    Apply random occlusion to skeleton data.

    Parameters:
    - data_numpy: numpy array of shape (C, T, V, M)
    - p: probability of applying occlusion
    - min_occlusion_length: minimum length of occlusion sequences
    - max_occlusion_length: maximum length of occlusion sequences

    Returns:
    - data_numpy: augmented data with occlusion applied
    """
    C, T, V, M = data_numpy.shape

    if np.random.rand() > p:
        return data_numpy

    # Loop over each person individually
    for m in range(M):
        current_frame = 0
        while current_frame < T:
            remaining_frames = T - current_frame
            if remaining_frames < min_occlusion_length:
                break
            max_possible_length = min(max_occlusion_length, remaining_frames)
            occlusion_length = np.random.randint(min_occlusion_length, max_possible_length + 1)

            # Randomly decide the type of occlusion
            occlusion_type = np.random.choice(['joint', 'frame'])

            if occlusion_type == 'joint':
                # Apply joint occlusion
                for frame_idx in range(current_frame, current_frame + occlusion_length):
                    num_joints_to_occlude = np.random.randint(1, V + 1)
                    joints_to_occlude = np.random.choice(V, num_joints_to_occlude, replace=False)
                    data_numpy[:, frame_idx, joints_to_occlude, m] = 0  # Set occluded joints to zero
            elif occlusion_type == 'frame':
                # Apply frame occlusion
                data_numpy[:, current_frame:current_frame + occlusion_length, :, m] = 0  # Set entire frames to zero

            current_frame += occlusion_length

            # Optionally, introduce skips between occlusion sequences
            if current_frame < T:
                remaining_frames = T - current_frame
                skip_length = np.random.randint(1, min(10, remaining_frames + 1))
                current_frame += skip_length
    return data_numpy


def random_joint_occlusion(data_numpy, p=0.1, min_occlusion_length=2, max_occlusion_length=10):
    """
    Apply random joint occlusion to skeleton data.

    Parameters:
    - data_numpy: numpy array of shape (C, T, V, M)
    - p: probability of applying joint occlusion
    - min_occlusion_length: minimum length of occlusion sequences
    - max_occlusion_length: maximum length of occlusion sequences

    Returns:
    - data_numpy: augmented data with joint occlusion applied
    """
    C, T, V, M = data_numpy.shape

    if np.random.rand() > p:
        return data_numpy

    # Loop over each person individually
    for m in range(M):
        current_frame = 0
        while current_frame < T:
            remaining_frames = T - current_frame
            if remaining_frames < min_occlusion_length:
                break

            max_possible_length = min(max_occlusion_length, remaining_frames)
            occlusion_length = np.random.randint(min_occlusion_length, max_possible_length + 1)

            # Apply joint occlusion
            for frame_idx in range(current_frame, current_frame + occlusion_length):
                num_joints_to_occlude = np.random.randint(1, V + 1)
                joints_to_occlude = np.random.choice(V, num_joints_to_occlude, replace=False)
                data_numpy[:, frame_idx, joints_to_occlude, m] = 0  # Set occluded joints to zero

            current_frame += occlusion_length

            # Optionally, introduce skips between occlusion sequences
            if current_frame < T:
                remaining_frames = T - current_frame
                skip_length = np.random.randint(1, min(10, remaining_frames + 1))
                current_frame += skip_length
    return data_numpy

# -----------------------------------------------------------------------
# Body part occlusion: Use for testing  
# -----------------------------------------------------------------------

def body_part_occlusion(data_numpy, body_parts=['Left Arm'], p=0.1, min_occlusion_length=2, max_occlusion_length=10):
    """
    Apply joint occlusion to specific body parts in skeleton data.
    Parameters:
    - data_numpy: numpy array of shape (C, T, V, M)
    - body_parts: list of body parts to occlude (e.g., ['Left Arm', 'Right Arm'])
    - p: probability of applying joint occlusion
    - min_occlusion_length: minimum length of occlusion sequences
    - max_occlusion_length: maximum length of occlusion sequences
    Returns:
    - data_numpy: augmented data with joint occlusion applied to specified body parts
    """
    C, T, V, M = data_numpy.shape
    if np.random.rand() > p:
        return data_numpy
    
    # Map body parts to joint indices
    body_part_joints = {
    'Left Arm': [5, 6, 7, 8, 22, 23],  # Left shoulder, elbow, wrist, hand, hand tip, thumb
    'Right Arm': [9, 10, 11, 12, 24, 25],  # Right shoulder, elbow, wrist, hand, hand tip, thumb
    'Two Hands': [8, 22, 23, 12, 24, 25],  # Left and right hands, hand tips, thumbs
    'Two Legs': [13, 14, 15, 16, 17, 18, 19, 20],  # Left and right hips, knees, ankles, feet
    'Trunk': [1, 2, 3, 4, 21]  # Spine base, mid, shoulder, neck, head
    }

    # Get the list of joints to occlude based on selected body parts
    joints_to_occlude = []
    for part in body_parts:
        joints_to_occlude.extend(body_part_joints.get(part, []))
    # Convert joint indices to zero-based indexing
    joints_to_occlude = [j - 1 for j in joints_to_occlude]
    # Ensure joints_to_occlude is not empty
    if not joints_to_occlude:
        return data_numpy
    # Loop over each person individually
    for m in range(M):
        current_frame = 0
        while current_frame < T:
            remaining_frames = T - current_frame
            if remaining_frames < min_occlusion_length:
                break
            max_possible_length = min(max_occlusion_length, remaining_frames)
            occlusion_length = np.random.randint(min_occlusion_length, max_possible_length + 1)
            # Apply joint occlusion to specified body parts
            for frame_idx in range(current_frame, current_frame + occlusion_length):
                data_numpy[:, frame_idx, joints_to_occlude, m] = 0  # Set occluded joints to zero
            current_frame += occlusion_length
            # Optionally, introduce skips between occlusion sequences
            if current_frame < T:
                remaining_frames = T - current_frame
                skip_length = np.random.randint(1, min(10, remaining_frames + 1))
                current_frame += skip_length
    return data_numpy



def random_frame_occlusion(data_numpy, p=0.1, min_occlusion_length=10, max_occlusion_length=50):
    """
    Apply random frame occlusion to skeleton data.

    Parameters:
    - data_numpy: numpy array of shape (C, T, V, M)
    - p: probability of applying frame occlusion
    - min_occlusion_length: minimum length of occlusion sequences
    - max_occlusion_length: maximum length of occlusion sequences

    Returns:
    - data_numpy: augmented data with frame occlusion applied
    """
    C, T, V, M = data_numpy.shape

    if np.random.rand() > p:
        return data_numpy

    # Loop over each person individually
    for m in range(M):
        current_frame = 0
        while current_frame < T:
            remaining_frames = T - current_frame
            if remaining_frames < min_occlusion_length:
                break

            max_possible_length = min(max_occlusion_length, remaining_frames)
            occlusion_length = np.random.randint(min_occlusion_length, max_possible_length + 1)

            # Apply frame occlusion
            data_numpy[:, current_frame:current_frame + occlusion_length, :, m] = 0  # Set entire frames to zero

            current_frame += occlusion_length

            # Optionally, introduce skips between occlusion sequences
            if current_frame < T:
                remaining_frames = T - current_frame
                skip_length = np.random.randint(1, min(10, remaining_frames + 1))
                current_frame += skip_length
    return data_numpy


# -----------------------------------------------------------------------
# Fixed occlusion length: Use for testing 
# -----------------------------------------------------------------------

def fixed_frame_occlusion(data_numpy, p=0.1, occlusion_lengths=[10, 20, 30, 40, 50], occlusion_length=10):
    """
    Apply frame occlusion to skeleton data with fixed occlusion lengths.

    Parameters:
    - data_numpy: numpy array of shape (C, T, V, M)
    - p: probability of applying frame occlusion
    - occlusion_lengths: list of possible occlusion lengths to choose from
    - occlusion_length: fixed occlusion length to use (overrides occlusion_lengths if provided)

    Returns:
    - data_numpy: augmented data with frame occlusion applied
    """
    C, T, V, M = data_numpy.shape
    if np.random.rand() > p:
        return data_numpy
    # If occlusion_length is specified, use it; otherwise, randomly select from occlusion_lengths
    if occlusion_length is not None:
        occlusion_length_values = [occlusion_length]
    else:
        occlusion_length_values = occlusion_lengths
    # Loop over each person individually
    for m in range(M):
        current_frame = 0
        while current_frame < T:
            remaining_frames = T - current_frame
            if remaining_frames < min(occlusion_length_values):
                break
            # Select occlusion length
            occlusion_length_value = random.choice(occlusion_length_values)
            occlusion_length_value = min(occlusion_length_value, remaining_frames)
            # Apply frame occlusion
            data_numpy[:, current_frame:current_frame + occlusion_length_value, :, m] = 0  # Set entire frames to zero
            current_frame += occlusion_length_value
            # Optionally, introduce skips between occlusion sequences
            if current_frame < T:
                remaining_frames = T - current_frame
                skip_length = np.random.randint(1, min(10, remaining_frames + 1))
                current_frame += skip_length
    return data_numpy


def random_jitter(data_numpy, sigma=0.1, probability=0.1):
    """
    probability=0.1
    sigma=0.05 next
    Apply jittering augmentation to skeleton data by adding Gaussian noise to random frames for each person.

    Parameters:
    - data_numpy: numpy array of shape (C, T, V, M)
    - sigma: Standard deviation of Gaussian noise
    - probability: Probability of applying noise to each frame per person

    Returns:
    - data_numpy: Augmented data with jittering applied
    """
    C, T, V, M = data_numpy.shape
    augmented_data = np.copy(data_numpy)
    # Loop over each person individually
    for m in range(M):
        for t in range(T):
            if np.random.rand() < probability:
                # Apply noise to all joints at frame t for person m
                noise = np.random.normal(0, sigma, (C, V))
                augmented_data[:, t, :, m] += noise
    return augmented_data