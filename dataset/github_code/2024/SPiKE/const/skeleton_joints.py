"""Num joint coordinates (15 joints x 3 coordinates)"""
NUM_COORD_JOINTS = 45

"""
Textual representations of joint indices
"""
joint_indices = {
    0: "Head",
    1: "Neck",
    2: "R Shoulder",
    3: "L Shoulder",
    4: "R Elbow",
    5: "L Elbow",
    6: "R Hand",
    7: "L Hand",
    8: "Torso",
    9: "R Hip",
    10: "L Hip",
    11: "R Knee",
    12: "L Knee",
    13: "R Foot",
    14: "L Foot"
}

"""
Connections between joints to form a skeleton.
(idx1, idx, limb colour)
"""
joint_connections = [
    (14, 12, "#FF69B4"),  # L foot -- L knee (bright pink)
    (12, 10, "#FF6347"),  # L knee -- L Hip (tomato red)
    (13, 11, "#32CD32"),  # R foot -- R knee (lime green)
    (11, 9, "#FFD700"),   # R knee -- R Hip (gold)
    (10, 8, "#6495ED"),   # L hip -- Torso (cornflower blue)
    (9, 8, "#FF00FF"),    # R hip -- Torso (magenta)
    (8, 1, "#00FFFF"),    # Torso -- Neck (cyan)
    (1, 0, "#ADFF2F"),    # Neck -- Head (green yellow)
    (7, 5, "#FFA500"),    # L Hand -- L Elbow (orange)
    (5, 3, "#808080"),    # L Elbow -- L Shoulder (gray)
    (3, 1, "#708090"),    # L Shoulder -- Neck (slate gray)
    (6, 4, "#808000"),    # R Hand -- R Elbow (olive)
    (4, 2, "#FFD700"),    # R Elbow -- R Shoulder (gold again, for consistency with the hip)
    (2, 1, "#4B0082"),    # R Shoulder -- Neck (indigo)
]

"""
Indices of pairs of joint to exchange with each other when flipping sides. Format: (left, right)
"""
side_idx = [
    (14, 13),  # foot
    (12, 11),  # knee
    (10, 9),  # hip
    (7, 6),  # hand
    (5, 4),  # elbow
    (3, 2),  # shoulder
]

"""
Exchange positions of left-side joints for right-side joints and vice-versa
"""
def flip_joint_sides(joints):
    """
    Flip the positions of left-side joints for right-side joints and vice-versa.

    Args:
        joints (torch.Tensor): Tensor containing joint positions.

    Returns:
        torch.Tensor: Tensor with flipped joint positions.
    """
    flipped_joints = joints.clone()
    for l_idx, r_idx in side_idx:
        flipped_joints[l_idx, :], flipped_joints[r_idx, :] = joints[r_idx, :].clone(), joints[l_idx, :].clone()
    return flipped_joints
