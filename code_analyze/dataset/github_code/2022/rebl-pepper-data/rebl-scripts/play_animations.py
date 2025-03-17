import os
import time

import pandas as pd

from anim_settings import *
from anim_utils import *
from anim_dict import gen_anim_list
from anim_dict import orig_anim_list

"""
Loads generated or original animation dataframes and displays them on the Choregraphe simulator (no LEDs) 
or the physical robot. The naoqi function that runs the animations does not use any interpolation. 

CAUTION: Please note that this code has only been tested in specific setups that might differ from yours significantly.
Thus, it might cause abrupt movements on a physical robot. It is provided as an example, but you might have to 
tune some parameters related to speed or velocities to achieve a smooth result. Check the TODO comments to find
which parts you might have to change or tune.   

"""


def play(motion_ses, leds_ses, id_anim, anim_dir):
    anim_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if anim_dir == 'generated_animations':
        name_anim = gen_anim_list[id_anim]
        print("Generated animation: " + name_anim)
        anim_dir_path = os.path.join(anim_dir_path, anim_dir, name_anim)
        df = pd.read_csv(anim_dir_path, index_col=0, skipinitialspace=True)
        # Freeze the KneePitch. Due to safety hardware controls, it does not react according to the learned values.
        df['KneePitch'] = -1.49414

    elif anim_dir == 'original_animations':
        name_anim = orig_anim_list[id_anim]
        print("Original animation: " + name_anim)
        anim_dir_path = os.path.join(anim_dir_path, anim_dir, 'data_raw.csv')
        df = pd.read_csv(anim_dir_path, index_col=0, skipinitialspace=True)
        df = df.loc[df['id'] == name_anim, :].reset_index(drop=True)
        df.drop('id', axis=1, inplace=True)

    else:
        print("anim_dir must be either 'generated_animations' or 'original_animations' ")
        sys.exit(1)

    df = df.astype('float')

    motion_ses.setStiffnesses("WholeBody", 1)

    angle_list = df.loc[0, joints_names].tolist()
    # TODO: Last parameter (maxSpeedFraction) might need tuning
    motion_ses.angleInterpolationWithSpeed(joints_names, angle_list, 0.2)

    # Play an animation from a dataframe
    for i in range(df.shape[0]):

        angle_list = df.loc[i, joints_names].tolist()
        leds_list = df.loc[i, leds_keys].tolist()
        # Creates a list with sublists of three elements (RGB values for each LED)
        rgb_chunks = [leds_list[x * 3:(x + 1) * 3] for x in range((len(leds_list) + 3 - 1) // 3)]

        # 16 LEDs (names in anim_settings.leds_short list ), with a 3 RGBs each
        for l in range(16):
            leds_ses.fadeRGB(leds_short[l], rgb_chunks[l][0], rgb_chunks[l][1], rgb_chunks[l][2], 0., _async=True)

        # Motion
        # TODO: Last parameter (maxSpeedFraction) might need tuning
        motion_ses.setAngles(joints_names, angle_list, 1)
        time.sleep(0.015)  # TODO: Time delay between postures might need tuning

    posture_ses.goToPosture("StandInit", 1)  # TODO: Last parameter (maxSpeedFraction) might need tuning

    time.sleep(1)
    return None


if __name__ == "__main__":
    # TODO: Change anim_dir and id_anim_range accordingly
    anim_dir = 'original_animations'  # 'generated_animations' or 'original_animations'
    # Play (all or selected) generated or original animations sequentially. Check anim_dict.gen_anim_list (generated)
    # or anim_dict.orig_anim_list (original) for selecting the ids.
    id_anim_range = range(1, 37)  # TODO: Change the range accordingly

    # Connect to the robot
    naoqi_session = naoqi_connect(str_ip='127.0.0.1', n_port=46321)  # TODO: Change ip and port
    motion_ses, aplayer_ses, posture_ses, leds_ses, alife_ses = load_modules(naoqi_session)
    init_rest(leds_ses, alife_ses, posture_ses)

    # Play the animations
    for id_anim in id_anim_range:
        play(motion_ses, leds_ses, id_anim, anim_dir)
