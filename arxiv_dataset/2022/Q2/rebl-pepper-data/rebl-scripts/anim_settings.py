import numpy as np

# A list with the 8 categories (no Pos/Tir category):
all_categories = ['Pos/Exc', 'Neu/Cal', 'Neu/Tir', 'Neu/Exc', 'Neg/Cal', 'Neg/Tir', 'Neg/Exc', 'Pos/Cal']

# Constants from the dataset
n_joints = 17  # 17 joints
n_categories = 8

# StandInit posture without the time dimension
standInit = [-0.2, 0.0, -0.0399996, -0.0, -0.0099995, -0.5225337, -1.2282506, 0.6, 1.5596846, 0.1427103, -0.0004972,
             0.5225337, 1.2282506, 0.6, 1.5596846, -0.1427103, 0.0004972]

# Normalized standInit posture with *naoqi* limits
standInit_norm = [0.37715381, 0.5, 0.48074111, 0.5, 0.49028934, 0.66922499, 0.20554991, 0.60416665, 0.87390519,
                  0.08625503, 0.4998637, 0.33077501, 0.79445009, 0.60416665, 0.87390519, 0.91374497, 0.5001363]

# Decoded from (0,0,0) latent mean. In rads
standInit_dec = [-0.235, 0.018, -0.072, -0.004, 0.006, -0.696, -1.096, 0.482,
                 1.463, 0.145, -0.215, 0.67, 1.121, 0.481, 1.484, -0.15,
                 0.153]

joints_names = ['HeadPitch', 'HeadYaw', 'HipPitch', 'HipRoll', 'KneePitch', 'LElbowRoll', 'LElbowYaw', 'LHand',
                'LShoulderPitch', 'LShoulderRoll', 'LWristYaw', 'RElbowRoll', 'RElbowYaw', 'RHand', 'RShoulderPitch',
                'RShoulderRoll', 'RWristYaw']

# Dataframe columns for the output of the sampling
out_cols = joints_names + ['time']

# Min-max ranges of joints
joints_minmax = np.array([[-0.7068583369255066, 0.6370452046394348],  # 'HeadPitch' 0
                          [-2.0856685638427734, 2.0856685638427734],  # 'HeadYaw'  0
                          [-1.0384708642959595, 1.0384708642959595],  # 'HipPitch'  0
                          [-0.514872133731842, 0.514872133731842],  # 'HipRoll'  0
                          [-0.514872133731842, 0.514872133731842],  # 'KneePitch'  0
                          [-1.5620696544647217, -0.008726646192371845],  # 'LElbowRoll'  -1
                          [-2.0856685638427734, 2.0856685638427734],  # 'LElbowYaw' 0
                          [0.019999999552965164, 0.9800000190734863],  # 'LHand    0.1
                          [-2.0856685638427734, 2.0856685638427734],  # 'LShoulderPitch' 0
                          [0.008726646192371845, 1.5620696544647217],  # 'LShoulderRoll'  1
                          [-1.8238691091537476, 1.8238691091537476],  # 'LWristYaw'     0
                          [0.008726646192371845, 1.5620696544647217],  # 'RElbowRoll'   1
                          [-2.0856685638427734, 2.0856685638427734],  # 'RElbowYaw'   0
                          [0.019999999552965164, 0.9800000190734863],  # 'RHand'    0.5
                          [-2.0856685638427734, 2.0856685638427734],  # 'RShoulderPitch'   0
                          [-1.5620696544647217, -0.008726646192371845],  # 'RShoulderRoll'  -1
                          [-1.8238691091537476, 1.8238691091537476]])  # 'RWristYaw'   0

# Timestamps lag cannot be smaller than these limits: Calculated as abs(max-min)/maxVelocity per joint
maxVelocity_lims = {'LShoulderPitch': 0.5683034521145831, 'LShoulderRoll': 0.1683374199825035,
                    'LHand': 0.07643312001973787, 'LElbowYaw': 0.5683034521145831, 'LWristYaw': 0.20983935340878543,
                    'KneePitch': 0.3511173243852647, 'HipRoll': 0.45356705168743694,
                    'RShoulderRoll': 0.1683374199825035, 'RWristYaw': 0.20983935340878543,
                    'HeadYaw': 0.5683034521145831, 'RShoulderPitch': 0.5683034521145831,
                    'RElbowYaw': 0.5683034521145831, 'RElbowRoll': 0.1683374199825035, 'HeadPitch': 0.14564024409779705,
                    'RHand': 0.07643312001973787, 'LElbowRoll': 0.1683374199825035, 'HipPitch': 0.708185756103779}

maxVelocity = maxVelocity_lims['HipPitch']

# Used for the generated animation in sampling.py as the first timestamp: the max maxVelocity_lims
start_time = [maxVelocity + 0.05]

# # Get maxVelocity limits
# for joint in joints_names:
#     min_angle = motion._getFullLimits(joint)[0][0]
#     max_angle = motion._getFullLimits(joint)[0][1]
#     maxVelocity = motion._getFullLimits(joint)[0][2]
#
#     maxVelocity_lims[joint] = abs(max_angle - min_angle) / maxVelocity

leds_names = {
    'R0R': 'Face/Led/Red/Right/0Deg/Actuator/Value',
    'R0G': 'Face/Led/Green/Right/0Deg/Actuator/Value',
    'R0B': 'Face/Led/Blue/Right/0Deg/Actuator/Value',

    'L0R': 'Face/Led/Red/Left/0Deg/Actuator/Value',
    'L0G': 'Face/Led/Green/Left/0Deg/Actuator/Value',
    'L0B': 'Face/Led/Blue/Left/0Deg/Actuator/Value',

    'R1R': 'Face/Led/Red/Right/45Deg/Actuator/Value',
    'R1G': 'Face/Led/Green/Right/45Deg/Actuator/Value',
    'R1B': 'Face/Led/Blue/Right/45Deg/Actuator/Value',

    'L1R': 'Face/Led/Red/Left/45Deg/Actuator/Value',
    'L1G': 'Face/Led/Green/Left/45Deg/Actuator/Value',
    'L1B': 'Face/Led/Blue/Left/45Deg/Actuator/Value',

    'R2R': 'Face/Led/Red/Right/90Deg/Actuator/Value',
    'R2G': 'Face/Led/Green/Right/90Deg/Actuator/Value',
    'R2B': 'Face/Led/Blue/Right/90Deg/Actuator/Value',

    'L2R': 'Face/Led/Red/Left/90Deg/Actuator/Value',
    'L2G': 'Face/Led/Green/Left/90Deg/Actuator/Value',
    'L2B': 'Face/Led/Blue/Left/90Deg/Actuator/Value',

    'R3R': 'Face/Led/Red/Right/135Deg/Actuator/Value',
    'R3G': 'Face/Led/Green/Right/135Deg/Actuator/Value',
    'R3B': 'Face/Led/Blue/Right/135Deg/Actuator/Value',

    'L3R': 'Face/Led/Red/Left/135Deg/Actuator/Value',
    'L3G': 'Face/Led/Green/Left/135Deg/Actuator/Value',
    'L3B': 'Face/Led/Blue/Left/135Deg/Actuator/Value',

    'R4R': 'Face/Led/Red/Right/180Deg/Actuator/Value',
    'R4G': 'Face/Led/Green/Right/180Deg/Actuator/Value',
    'R4B': 'Face/Led/Blue/Right/180Deg/Actuator/Value',

    'L4R': 'Face/Led/Red/Left/180Deg/Actuator/Value',
    'L4G': 'Face/Led/Green/Left/180Deg/Actuator/Value',
    'L4B': 'Face/Led/Blue/Left/180Deg/Actuator/Value',

    'R5R': 'Face/Led/Red/Right/225Deg/Actuator/Value',
    'R5G': 'Face/Led/Green/Right/225Deg/Actuator/Value',
    'R5B': 'Face/Led/Blue/Right/225Deg/Actuator/Value',

    'L5R': 'Face/Led/Red/Left/225Deg/Actuator/Value',
    'L5G': 'Face/Led/Green/Left/225Deg/Actuator/Value',
    'L5B': 'Face/Led/Blue/Left/225Deg/Actuator/Value',

    'R6R': 'Face/Led/Red/Right/270Deg/Actuator/Value',
    'R6G': 'Face/Led/Green/Right/270Deg/Actuator/Value',
    'R6B': 'Face/Led/Blue/Right/270Deg/Actuator/Value',

    'L6R': 'Face/Led/Red/Left/270Deg/Actuator/Value',
    'L6G': 'Face/Led/Green/Left/270Deg/Actuator/Value',
    'L6B': 'Face/Led/Blue/Left/270Deg/Actuator/Value',

    'R7R': 'Face/Led/Red/Right/315Deg/Actuator/Value',
    'R7G': 'Face/Led/Green/Right/315Deg/Actuator/Value',
    'R7B': 'Face/Led/Blue/Right/315Deg/Actuator/Value',

    'L7R': 'Face/Led/Red/Left/315Deg/Actuator/Value',
    'L7G': 'Face/Led/Green/Left/315Deg/Actuator/Value',
    'L7B': 'Face/Led/Blue/Left/315Deg/Actuator/Value',
}

leds_keys = ['R0R', 'R0G', 'R0B',
             'L0R', 'L0G', 'L0B',
             'R1R', 'R1G', 'R1B',
             'L1R', 'L1G', 'L1B',
             'R2R', 'R2G', 'R2B',
             'L2R', 'L2G', 'L2B',
             'R3R', 'R3G', 'R3B',
             'L3R', 'L3G', 'L3B',
             'R4R', 'R4G', 'R4B',
             'L4R', 'L4G', 'L4B',
             'R5R', 'R5G', 'R5B',
             'L5R', 'L5G', 'L5B',
             'R6R', 'R6G', 'R6B',
             'L6R', 'L6G', 'L6B',
             'R7R', 'R7G', 'R7B',
             'L7R', 'L7G', 'L7B']

leds_short = ['RightFaceLed1',
              'LeftFaceLed1',
              'RightFaceLed2',
              'LeftFaceLed2',
              'RightFaceLed3',
              'LeftFaceLed3',
              'RightFaceLed4',
              'LeftFaceLed4',
              'RightFaceLed5',
              'LeftFaceLed5',
              'RightFaceLed6',
              'LeftFaceLed6',
              'RightFaceLed7',
              'LeftFaceLed7',
              'RightFaceLed8',
              'LeftFaceLed8'
              ]
