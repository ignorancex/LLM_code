from config.globals import *
from util import *


def speed_matching_correction(ai_action, down_speed_raw, down_occupancy_raw, pre_speed_limit_raw):
    """
    Given an action from the MARL-based policy, this function returns the modified action after going through the
    speed-matching safety guard
    :param ai_action: The discrete integer value generated by the MARL-based policy
    :param down_speed_raw: One of the input states to the MARL-based policy, i.e., the raw downstream traffic speed
    :param down_occupancy_raw: One of the input states to the MARL-based policy, i.e., the raw downstream traffic
    occupancy
    :param pre_speed_limit_raw: One of the input states to the MARL-based policy, i.e., the raw selected
    speed limit from one downstream VSL agent
    :return: sm_corrected: The modified action after going through the speed-matching safety guard
    """
    sm_corrected = None
    if ai_action == 0:
        if down_speed_raw <= 35:
            pass
        elif 35 < down_speed_raw <= 40:
            sm_corrected = 1
        elif 40 < down_speed_raw <= 50:
            if pre_speed_limit_raw == 30:
                sm_corrected = speed_to_action(pre_speed_limit_raw + step_down_value)
            else:
                sm_corrected = 2
        elif 50 < down_speed_raw <= 60:
            if pre_speed_limit_raw == 30 or pre_speed_limit_raw == 40:
                sm_corrected = speed_to_action(pre_speed_limit_raw + step_down_value)
            else:
                sm_corrected = 3
        else:
            if pre_speed_limit_raw == 30 or pre_speed_limit_raw == 40 or pre_speed_limit_raw == 50:
                sm_corrected = speed_to_action(pre_speed_limit_raw + step_down_value)
            else:
                sm_corrected = 4
    elif ai_action == 4:
        if down_speed_raw <= 35 and down_occupancy_raw >= 12:
            sm_corrected = 0
        elif 35 < down_speed_raw <= 40 and down_occupancy_raw >= 12:
            sm_corrected = 1
        elif 40 < down_speed_raw <= 50 and down_occupancy_raw >= 12:
            sm_corrected = 2
        elif 50 < down_speed_raw <= 60 and down_occupancy_raw >= 12:
            sm_corrected = 3
        else:
            pass

    return sm_corrected


def max_speed_limit_correction(input_speed_limit, mm, mm_to_maxspeedlimit):
    """
    Given a speed limit, this function returns the modified speed limit after going through the
    max-speed-limit-correction safety guard
    :param input_speed_limit: The input speed limit to this function, should be an integer
    :param mm: The mile marker of the vsl controller
    :param mm_to_maxspeedlimit: A dictionary mapping from the milemarker of VSL controllers to its maximum speed limit
    :return max_corrected: The modified speed limit after going through the max speed limit correction safety guard
    """
    if input_speed_limit > mm_to_maxspeedlimit[mm]:
        max_corrected = mm_to_maxspeedlimit[mm]
    else:
        max_corrected = input_speed_limit
    return max_corrected


def bounce_correction(speed_limit_list):
    """
    Given a sequence of speed limits, this function returns the modified speed limit after going through the debounce
    safety guard
    :param speed_limit_list: a list of speed limits
    :return: a list of speed limits without bounce of order 1
    """
    for i in range(len(speed_limit_list)-2):
        speed_limit_window = speed_limit_list[i: i+3]
        if (speed_limit_window[1] > speed_limit_window[0]) and (speed_limit_window[1] > speed_limit_window[-1]):
            speed_limit_list[i+1] = max(speed_limit_window[0], speed_limit_window[-1])
    return speed_limit_list

