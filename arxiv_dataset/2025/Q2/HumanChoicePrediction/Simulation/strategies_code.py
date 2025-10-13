import numpy as np

################################
# CONSTS
################################

REVIEWS = 0
BOT_ACTION = 1
USER_DECISION = 2

################################
# FUNCTIONS
################################

def user_score(previous_rounds):
                        return sum([(r[REVIEWS].mean()-8)*r[USER_DECISION] for r in previous_rounds])

def bot_score(previous_rounds):
                            return sum([r[USER_DECISION] for r in previous_rounds])

def play_mean(reviews):
        tmp = reviews - reviews.mean()
        tmp = abs(tmp)
        return reviews[np.argmin(tmp)]

def play_median(reviews):
        return sorted(reviews)[3]


################################
# STRATEGEIES CODES
################################

def strategy_0(reviews, previous_rounds):
    """#0 play max
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)"""
    return reviews.max()

def strategy_1(reviews, previous_rounds):
    """#1 play min
    (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)"""
    return reviews.min()

def strategy_2(reviews, previous_rounds):
    """#2 play mean
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)"""
    return play_mean(reviews)

def strategy_3(reviews, previous_rounds):
    """#3 Is hotel score >= 8? if True, [play max]. else, [play min]
    (-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1)"""
    if reviews.mean() >= 8:
        return reviews.max()
    else:
        return reviews.min()

def strategy_4(reviews, previous_rounds):
    """#4 Is hotel score >= 8? if True, [play min]. else, [play max]
    (1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1)"""
    if reviews.mean() >= 8:
        return reviews.min()
    else:
        return reviews.max()

def strategy_5(reviews, previous_rounds):
    """#5 Is hotel score >= 8? if True, [play max]. else, [play mean]
    (0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1)"""
    if reviews.mean() >= 8:
        return reviews.max()
    else:
        return play_mean(reviews)

def strategy_6(reviews, previous_rounds):
    """#6 Is hotel score >= 8? if True, [play min]. else, [play mean]
    (0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1)"""
    if reviews.mean() >= 8:
        return reviews.min()
    else:
        return play_mean(reviews)

def strategy_7(reviews, previous_rounds):
    """#7 Is hotel score >= 8? if True, [play mean]. else, [play max]
    (1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0)"""
    if reviews.mean() >= 8:
        return play_mean(reviews)
    else:
        return reviews.max()

def strategy_8(reviews, previous_rounds):
    """#8 Is hotel score >= 8? if True, [play mean]. else, [play min]
    (-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0)"""
    if reviews.mean() >= 8:
        return play_mean(reviews)
    else:
        return reviews.min()

def strategy_9(reviews, previous_rounds):
    """#9 Is user earn more than bot? if True, [play max]. else, [play min]
    (-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.max()
    else:
        return reviews.min()

def strategy_10(reviews, previous_rounds):
    """#10 Is user earn more than bot? if True, [play min]. else, [play max]
    (1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.min()
    else:
        return reviews.max()

def strategy_11(reviews, previous_rounds):
    """#11 Is user earn more than bot? if True, [play max]. else, [play mean]
    (0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.max()
    else:
        return play_mean(reviews)

def strategy_12(reviews, previous_rounds):
    """#12 Is user earn more than bot? if True, [play min]. else, [play mean]
    (0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.min()
    else:
        return play_mean(reviews)

def strategy_13(reviews, previous_rounds):
    """#13 Is user earn more than bot? if True, [play mean]. else, [play max]
    (1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return play_mean(reviews)
    else:
        return reviews.max()

def strategy_14(reviews, previous_rounds):
    """#14 Is user earn more than bot? if True, [play mean]. else, [play min]
    (-1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return play_mean(reviews)
    else:
        return reviews.min()

def strategy_15(reviews, previous_rounds):
    """#15 Is hotel was chosen in last round? if True, [play max]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        return reviews.min()

def strategy_16(reviews, previous_rounds):
    """#16 Is hotel was chosen in last round? if True, [play min]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        return reviews.max()

def strategy_17(reviews, previous_rounds):
    """#17 Is hotel was chosen in last round? if True, [play max]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        return play_mean(reviews)

def strategy_18(reviews, previous_rounds):
    """#18 Is hotel was chosen in last round? if True, [play min]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        return play_mean(reviews)

def strategy_19(reviews, previous_rounds):
    """#19 Is hotel was chosen in last round? if True, [play mean]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        return reviews.max()

def strategy_20(reviews, previous_rounds):
    """#20 Is hotel was chosen in last round? if True, [play mean]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        return reviews.min()

def strategy_21(reviews, previous_rounds):
    """#21 Is hotel score in the last round >= 8? if True, [play max]. else, [play min]
    (-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return reviews.max()
    else:
        return reviews.min()

def strategy_22(reviews, previous_rounds):
    """#22 Is hotel score in the last round >= 8? if True, [play min]. else, [play max]
    (1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return reviews.min()
    else:
        return reviews.max()

def strategy_23(reviews, previous_rounds):
    """#23 Is hotel score in the last round >= 8? if True, [play max]. else, [play mean]
    (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return reviews.max()
    else:
        return play_mean(reviews)

def strategy_24(reviews, previous_rounds):
    """#24 Is hotel score in the last round >= 8? if True, [play min]. else, [play mean]
    (0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return reviews.min()
    else:
        return play_mean(reviews)

def strategy_25(reviews, previous_rounds):
    """#25 Is hotel score in the last round >= 8? if True, [play mean]. else, [play max]
    (1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return play_mean(reviews)
    else:
        return reviews.max()

def strategy_26(reviews, previous_rounds):
    """#26 Is hotel score in the last round >= 8? if True, [play mean]. else, [play min]
    (-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return play_mean(reviews)
    else:
        return reviews.min()

def strategy_27(reviews, previous_rounds):
    """#27 Is user earn more than bot? if True, [play max]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_28(reviews, previous_rounds):
    """#28 Is user earn more than bot? if True, [play max]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_29(reviews, previous_rounds):
    """#29 Is user earn more than bot? if True, [play min]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_30(reviews, previous_rounds):
    """#30 Is user earn more than bot? if True, [play min]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_31(reviews, previous_rounds):
    """#31 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [play max]
    (1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return reviews.max()

def strategy_32(reviews, previous_rounds):
    """#32 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [play min]
    (-1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return reviews.min()

def strategy_33(reviews, previous_rounds):
    """#33 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [play max]
    (1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return reviews.max()

def strategy_34(reviews, previous_rounds):
    """#34 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [play min]
    (-1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return reviews.min()

def strategy_35(reviews, previous_rounds):
    """#35 Is user earn more than bot? if True, [play max]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_36(reviews, previous_rounds):
    """#36 Is user earn more than bot? if True, [play max]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 1, 1, 1, 1, 0, 0, -1, -1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_37(reviews, previous_rounds):
    """#37 Is user earn more than bot? if True, [play max]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_38(reviews, previous_rounds):
    """#38 Is user earn more than bot? if True, [play max]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, 1, 1, 1, 1, -1, -1, 0, 0, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_39(reviews, previous_rounds):
    """#39 Is user earn more than bot? if True, [play min]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, -1, -1, -1, -1, 0, 0, 1, 1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_40(reviews, previous_rounds):
    """#40 Is user earn more than bot? if True, [play min]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_41(reviews, previous_rounds):
    """#41 Is user earn more than bot? if True, [play min]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, -1, -1, -1, -1, 1, 1, 0, 0, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_42(reviews, previous_rounds):
    """#42 Is user earn more than bot? if True, [play min]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_43(reviews, previous_rounds):
    """#43 Is user earn more than bot? if True, [play mean]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_44(reviews, previous_rounds):
    """#44 Is user earn more than bot? if True, [play mean]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_45(reviews, previous_rounds):
    """#45 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [play mean]
    (0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, -1, -1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return play_mean(reviews)

def strategy_46(reviews, previous_rounds):
    """#46 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [play max]
    (1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return reviews.max()

def strategy_47(reviews, previous_rounds):
    """#47 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [play min]
    (-1, -1, -1, -1, 0, 0, 1, 1, -1, -1, -1, -1, 0, 0, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return reviews.min()

def strategy_48(reviews, previous_rounds):
    """#48 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [play mean]
    (0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, 1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return play_mean(reviews)

def strategy_49(reviews, previous_rounds):
    """#49 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [play max]
    (1, 1, 1, 1, 0, 0, -1, -1, 1, 1, 1, 1, 0, 0, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return reviews.max()

def strategy_50(reviews, previous_rounds):
    """#50 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [play min]
    (-1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return reviews.min()

def strategy_51(reviews, previous_rounds):
    """#51 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [play max]
    (1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return reviews.max()

def strategy_52(reviews, previous_rounds):
    """#52 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [play min]
    (-1, -1, -1, -1, 1, 1, 0, 0, -1, -1, -1, -1, 1, 1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return reviews.min()

def strategy_53(reviews, previous_rounds):
    """#53 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [play max]
    (1, 1, 1, 1, -1, -1, 0, 0, 1, 1, 1, 1, -1, -1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return reviews.max()

def strategy_54(reviews, previous_rounds):
    """#54 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [play min]
    (-1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return reviews.min()

def strategy_55(reviews, previous_rounds):
    """#55 Is hotel score >= 8? if True, [play max]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 1, 1, -1, -1, 1, 1, 0, 0, 1, 1, -1, -1, 1, 1)"""
    if reviews.mean() >= 8:
        return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_56(reviews, previous_rounds):
    """#56 Is hotel score >= 8? if True, [play max]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, 1, 1, 0, 0, 1, 1, -1, -1, 1, 1, 0, 0, 1, 1)"""
    if reviews.mean() >= 8:
        return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_57(reviews, previous_rounds):
    """#57 Is hotel score >= 8? if True, [play min]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, -1, -1, 1, 1, -1, -1, 0, 0, -1, -1, 1, 1, -1, -1)"""
    if reviews.mean() >= 8:
        return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_58(reviews, previous_rounds):
    """#58 Is hotel score >= 8? if True, [play min]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, -1, -1, 0, 0, -1, -1, 1, 1, -1, -1, 0, 0, -1, -1)"""
    if reviews.mean() >= 8:
        return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_59(reviews, previous_rounds):
    """#59 Is hotel score >= 8? if True, [play mean]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, 0, 0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 0, 0)"""
    if reviews.mean() >= 8:
        return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_60(reviews, previous_rounds):
    """#60 Is hotel score >= 8? if True, [play mean]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 0, 0, -1, -1, 0, 0)"""
    if reviews.mean() >= 8:
        return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_61(reviews, previous_rounds):
    """#61 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [play mean]
    (0, 0, -1, -1, 0, 0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        return play_mean(reviews)

def strategy_62(reviews, previous_rounds):
    """#62 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [play min]
    (-1, -1, 0, 0, -1, -1, 1, 1, -1, -1, 0, 0, -1, -1, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return reviews.min()

def strategy_63(reviews, previous_rounds):
    """#63 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [play mean]
    (0, 0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 0, 0, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        return play_mean(reviews)

def strategy_64(reviews, previous_rounds):
    """#64 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [play max]
    (1, 1, 0, 0, 1, 1, -1, -1, 1, 1, 0, 0, 1, 1, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return reviews.max()

def strategy_65(reviews, previous_rounds):
    """#65 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [play min]
    (-1, -1, 1, 1, -1, -1, 0, 0, -1, -1, 1, 1, -1, -1, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return reviews.min()

def strategy_66(reviews, previous_rounds):
    """#66 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [play max]
    (1, 1, -1, -1, 1, 1, 0, 0, 1, 1, -1, -1, 1, 1, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return reviews.max()

def strategy_67(reviews, previous_rounds):
    """#67 Is user earn more than bot? if True, [play mean]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_68(reviews, previous_rounds):
    """#68 Is user earn more than bot? if True, [play mean]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_69(reviews, previous_rounds):
    """#69 Is user earn more than bot? if True, [play mean]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_70(reviews, previous_rounds):
    """#70 Is user earn more than bot? if True, [play mean]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_71(reviews, previous_rounds):
    """#71 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return play_mean(reviews)

def strategy_72(reviews, previous_rounds):
    """#72 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [play mean]
    (0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return play_mean(reviews)

def strategy_73(reviews, previous_rounds):
    """#73 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [play mean]
    (0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return play_mean(reviews)

def strategy_74(reviews, previous_rounds):
    """#74 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [play mean]
    (0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return play_mean(reviews)

def strategy_75(reviews, previous_rounds):
    """#75 Is hotel was chosen in last round? if True, [play max]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_76(reviews, previous_rounds):
    """#76 Is hotel was chosen in last round? if True, [play max]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_77(reviews, previous_rounds):
    """#77 Is hotel was chosen in last round? if True, [play min]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_78(reviews, previous_rounds):
    """#78 Is hotel was chosen in last round? if True, [play min]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_79(reviews, previous_rounds):
    """#79 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return reviews.max()

def strategy_80(reviews, previous_rounds):
    """#80 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return reviews.min()

def strategy_81(reviews, previous_rounds):
    """#81 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return reviews.max()

def strategy_82(reviews, previous_rounds):
    """#82 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return reviews.min()

def strategy_83(reviews, previous_rounds):
    """#83 Is hotel was chosen in last round? if True, [play max]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_84(reviews, previous_rounds):
    """#84 Is hotel was chosen in last round? if True, [play max]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_85(reviews, previous_rounds):
    """#85 Is hotel was chosen in last round? if True, [play max]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_86(reviews, previous_rounds):
    """#86 Is hotel was chosen in last round? if True, [play max]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_87(reviews, previous_rounds):
    """#87 Is hotel was chosen in last round? if True, [play min]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_88(reviews, previous_rounds):
    """#88 Is hotel was chosen in last round? if True, [play min]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_89(reviews, previous_rounds):
    """#89 Is hotel was chosen in last round? if True, [play min]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_90(reviews, previous_rounds):
    """#90 Is hotel was chosen in last round? if True, [play min]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_91(reviews, previous_rounds):
    """#91 Is hotel was chosen in last round? if True, [play mean]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_92(reviews, previous_rounds):
    """#92 Is hotel was chosen in last round? if True, [play mean]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_93(reviews, previous_rounds):
    """#93 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, -1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return play_mean(reviews)

def strategy_94(reviews, previous_rounds):
    """#94 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return reviews.max()

def strategy_95(reviews, previous_rounds):
    """#95 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 1, 1, 0, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return reviews.min()

def strategy_96(reviews, previous_rounds):
    """#96 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return play_mean(reviews)

def strategy_97(reviews, previous_rounds):
    """#97 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, 0, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return reviews.max()

def strategy_98(reviews, previous_rounds):
    """#98 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return reviews.min()

def strategy_99(reviews, previous_rounds):
    """#99 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return reviews.max()

def strategy_100(reviews, previous_rounds):
    """#100 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 0, 0, 1, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return reviews.min()

def strategy_101(reviews, previous_rounds):
    """#101 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 0, 0, -1, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return reviews.max()

def strategy_102(reviews, previous_rounds):
    """#102 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return reviews.min()

def strategy_103(reviews, previous_rounds):
    """#103 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [play mean]
    (0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 1, 1, 0, 0, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return play_mean(reviews)

def strategy_104(reviews, previous_rounds):
    """#104 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [play min]
    (-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 1, 1, -1, -1, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return reviews.min()

def strategy_105(reviews, previous_rounds):
    """#105 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [play mean]
    (0, 0, 1, 1, 0, 0, 1, 1, 0, 0, -1, -1, 0, 0, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return play_mean(reviews)

def strategy_106(reviews, previous_rounds):
    """#106 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [play max]
    (1, 1, 0, 0, 1, 1, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return reviews.max()

def strategy_107(reviews, previous_rounds):
    """#107 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [play min]
    (-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 0, 0, -1, -1, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return reviews.min()

def strategy_108(reviews, previous_rounds):
    """#108 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [play max]
    (1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 0, 0, 1, 1, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return reviews.max()

def strategy_109(reviews, previous_rounds):
    """#109 Is hotel score >= 8? if True, [play max]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 1, 1, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1)"""
    if reviews.mean() >= 8:
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_110(reviews, previous_rounds):
    """#110 Is hotel score >= 8? if True, [play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, 1, 1, -1, -1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1)"""
    if reviews.mean() >= 8:
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_111(reviews, previous_rounds):
    """#111 Is hotel score >= 8? if True, [play min]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, -1, -1, 0, 0, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1)"""
    if reviews.mean() >= 8:
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_112(reviews, previous_rounds):
    """#112 Is hotel score >= 8? if True, [play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, -1, -1, 1, 1, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1)"""
    if reviews.mean() >= 8:
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_113(reviews, previous_rounds):
    """#113 Is hotel score >= 8? if True, [play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, 0, 0, -1, -1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0)"""
    if reviews.mean() >= 8:
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_114(reviews, previous_rounds):
    """#114 Is hotel score >= 8? if True, [play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 0, 0, 1, 1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0)"""
    if reviews.mean() >= 8:
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_115(reviews, previous_rounds):
    """#115 Is hotel was chosen in last round? if True, [play mean]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_116(reviews, previous_rounds):
    """#116 Is hotel was chosen in last round? if True, [play mean]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_117(reviews, previous_rounds):
    """#117 Is hotel was chosen in last round? if True, [play mean]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_118(reviews, previous_rounds):
    """#118 Is hotel was chosen in last round? if True, [play mean]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_119(reviews, previous_rounds):
    """#119 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return play_mean(reviews)

def strategy_120(reviews, previous_rounds):
    """#120 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return play_mean(reviews)

def strategy_121(reviews, previous_rounds):
    """#121 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return play_mean(reviews)

def strategy_122(reviews, previous_rounds):
    """#122 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return play_mean(reviews)

def strategy_123(reviews, previous_rounds):
    """#123 Is hotel score >= 8? if True, [play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1)"""
    if reviews.mean() >= 8:
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_124(reviews, previous_rounds):
    """#124 Is hotel score >= 8? if True, [play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1)"""
    if reviews.mean() >= 8:
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_125(reviews, previous_rounds):
    """#125 Is hotel score >= 8? if True, [play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1)"""
    if reviews.mean() >= 8:
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_126(reviews, previous_rounds):
    """#126 Is hotel score >= 8? if True, [play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1)"""
    if reviews.mean() >= 8:
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_127(reviews, previous_rounds):
    """#127 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [play max]
    (1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return reviews.max()

def strategy_128(reviews, previous_rounds):
    """#128 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [play min]
    (-1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return reviews.min()

def strategy_129(reviews, previous_rounds):
    """#129 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [play max]
    (1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return reviews.max()

def strategy_130(reviews, previous_rounds):
    """#130 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [play min]
    (-1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return reviews.min()

def strategy_131(reviews, previous_rounds):
    """#131 Is hotel score >= 8? if True, [play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1)"""
    if reviews.mean() >= 8:
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_132(reviews, previous_rounds):
    """#132 Is hotel score >= 8? if True, [play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 1, 1, 0, -1, 1, 1, 0, -1, 1, 1, 0, -1, 1, 1)"""
    if reviews.mean() >= 8:
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_133(reviews, previous_rounds):
    """#133 Is hotel score >= 8? if True, [play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1)"""
    if reviews.mean() >= 8:
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_134(reviews, previous_rounds):
    """#134 Is hotel score >= 8? if True, [play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, 1, 1, -1, 0, 1, 1, -1, 0, 1, 1, -1, 0, 1, 1)"""
    if reviews.mean() >= 8:
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_135(reviews, previous_rounds):
    """#135 Is hotel score >= 8? if True, [play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, -1, -1, 0, 1, -1, -1, 0, 1, -1, -1, 0, 1, -1, -1)"""
    if reviews.mean() >= 8:
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_136(reviews, previous_rounds):
    """#136 Is hotel score >= 8? if True, [play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1)"""
    if reviews.mean() >= 8:
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_137(reviews, previous_rounds):
    """#137 Is hotel score >= 8? if True, [play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, -1, -1, 1, 0, -1, -1, 1, 0, -1, -1, 1, 0, -1, -1)"""
    if reviews.mean() >= 8:
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_138(reviews, previous_rounds):
    """#138 Is hotel score >= 8? if True, [play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1)"""
    if reviews.mean() >= 8:
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_139(reviews, previous_rounds):
    """#139 Is hotel score >= 8? if True, [play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0)"""
    if reviews.mean() >= 8:
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_140(reviews, previous_rounds):
    """#140 Is hotel score >= 8? if True, [play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0)"""
    if reviews.mean() >= 8:
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_141(reviews, previous_rounds):
    """#141 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [play mean]
    (0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return play_mean(reviews)

def strategy_142(reviews, previous_rounds):
    """#142 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [play max]
    (1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return reviews.max()

def strategy_143(reviews, previous_rounds):
    """#143 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [play min]
    (-1, -1, 0, 1, -1, -1, 0, 1, -1, -1, 0, 1, -1, -1, 0, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return reviews.min()

def strategy_144(reviews, previous_rounds):
    """#144 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [play mean]
    (0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return play_mean(reviews)

def strategy_145(reviews, previous_rounds):
    """#145 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [play max]
    (1, 1, 0, -1, 1, 1, 0, -1, 1, 1, 0, -1, 1, 1, 0, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return reviews.max()

def strategy_146(reviews, previous_rounds):
    """#146 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [play min]
    (-1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return reviews.min()

def strategy_147(reviews, previous_rounds):
    """#147 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [play max]
    (1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return reviews.max()

def strategy_148(reviews, previous_rounds):
    """#148 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [play min]
    (-1, -1, 1, 0, -1, -1, 1, 0, -1, -1, 1, 0, -1, -1, 1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return reviews.min()

def strategy_149(reviews, previous_rounds):
    """#149 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [play max]
    (1, 1, -1, 0, 1, 1, -1, 0, 1, 1, -1, 0, 1, 1, -1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return reviews.max()

def strategy_150(reviews, previous_rounds):
    """#150 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [play min]
    (-1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return reviews.min()

def strategy_151(reviews, previous_rounds):
    """#151 Is hotel score in the last round >= 8? if True, [play max]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 1, -1, 1, 0, 1, -1, 1, 0, 1, -1, 1, 0, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_152(reviews, previous_rounds):
    """#152 Is hotel score in the last round >= 8? if True, [play max]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, 1, 0, 1, -1, 1, 0, 1, -1, 1, 0, 1, -1, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_153(reviews, previous_rounds):
    """#153 Is hotel score in the last round >= 8? if True, [play min]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, -1, 1, -1, 0, -1, 1, -1, 0, -1, 1, -1, 0, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_154(reviews, previous_rounds):
    """#154 Is hotel score in the last round >= 8? if True, [play min]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, -1, 0, -1, 1, -1, 0, -1, 1, -1, 0, -1, 1, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_155(reviews, previous_rounds):
    """#155 Is hotel score in the last round >= 8? if True, [play mean]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_156(reviews, previous_rounds):
    """#156 Is hotel score in the last round >= 8? if True, [play mean]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_157(reviews, previous_rounds):
    """#157 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [play mean]
    (0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return play_mean(reviews)

def strategy_158(reviews, previous_rounds):
    """#158 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [play min]
    (-1, 0, -1, 1, -1, 0, -1, 1, -1, 0, -1, 1, -1, 0, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return reviews.min()

def strategy_159(reviews, previous_rounds):
    """#159 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [play mean]
    (0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return play_mean(reviews)

def strategy_160(reviews, previous_rounds):
    """#160 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [play max]
    (1, 0, 1, -1, 1, 0, 1, -1, 1, 0, 1, -1, 1, 0, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return reviews.max()

def strategy_161(reviews, previous_rounds):
    """#161 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [play min]
    (-1, 1, -1, 0, -1, 1, -1, 0, -1, 1, -1, 0, -1, 1, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return reviews.min()

def strategy_162(reviews, previous_rounds):
    """#162 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [play max]
    (1, -1, 1, 0, 1, -1, 1, 0, 1, -1, 1, 0, 1, -1, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return reviews.max()

def strategy_163(reviews, previous_rounds):
    """#163 Is hotel score >= 8? if True, [play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)"""
    if reviews.mean() >= 8:
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_164(reviews, previous_rounds):
    """#164 Is hotel score >= 8? if True, [play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0)"""
    if reviews.mean() >= 8:
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_165(reviews, previous_rounds):
    """#165 Is hotel score >= 8? if True, [play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)"""
    if reviews.mean() >= 8:
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_166(reviews, previous_rounds):
    """#166 Is hotel score >= 8? if True, [play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0)"""
    if reviews.mean() >= 8:
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_167(reviews, previous_rounds):
    """#167 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [play mean]
    (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return play_mean(reviews)

def strategy_168(reviews, previous_rounds):
    """#168 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [play mean]
    (0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return play_mean(reviews)

def strategy_169(reviews, previous_rounds):
    """#169 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [play mean]
    (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return play_mean(reviews)

def strategy_170(reviews, previous_rounds):
    """#170 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [play mean]
    (0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return play_mean(reviews)

def strategy_171(reviews, previous_rounds):
    """#171 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        return reviews.max()

def strategy_172(reviews, previous_rounds):
    """#172 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        return reviews.min()

def strategy_173(reviews, previous_rounds):
    """#173 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        return reviews.max()

def strategy_174(reviews, previous_rounds):
    """#174 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        return reviews.min()

def strategy_175(reviews, previous_rounds):
    """#175 Is hotel was chosen in last round? if True, [play max]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_176(reviews, previous_rounds):
    """#176 Is hotel was chosen in last round? if True, [play max]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_177(reviews, previous_rounds):
    """#177 Is hotel was chosen in last round? if True, [play min]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_178(reviews, previous_rounds):
    """#178 Is hotel was chosen in last round? if True, [play min]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_179(reviews, previous_rounds):
    """#179 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        return play_mean(reviews)

def strategy_180(reviews, previous_rounds):
    """#180 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return reviews.max()

def strategy_181(reviews, previous_rounds):
    """#181 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return reviews.min()

def strategy_182(reviews, previous_rounds):
    """#182 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        return play_mean(reviews)

def strategy_183(reviews, previous_rounds):
    """#183 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return reviews.max()

def strategy_184(reviews, previous_rounds):
    """#184 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return reviews.min()

def strategy_185(reviews, previous_rounds):
    """#185 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return reviews.max()

def strategy_186(reviews, previous_rounds):
    """#186 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return reviews.min()

def strategy_187(reviews, previous_rounds):
    """#187 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return reviews.max()

def strategy_188(reviews, previous_rounds):
    """#188 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return reviews.min()

def strategy_189(reviews, previous_rounds):
    """#189 Is hotel was chosen in last round? if True, [play max]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_190(reviews, previous_rounds):
    """#190 Is hotel was chosen in last round? if True, [play max]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_191(reviews, previous_rounds):
    """#191 Is hotel was chosen in last round? if True, [play max]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_192(reviews, previous_rounds):
    """#192 Is hotel was chosen in last round? if True, [play max]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_193(reviews, previous_rounds):
    """#193 Is hotel was chosen in last round? if True, [play min]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_194(reviews, previous_rounds):
    """#194 Is hotel was chosen in last round? if True, [play min]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_195(reviews, previous_rounds):
    """#195 Is hotel was chosen in last round? if True, [play min]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_196(reviews, previous_rounds):
    """#196 Is hotel was chosen in last round? if True, [play min]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_197(reviews, previous_rounds):
    """#197 Is hotel was chosen in last round? if True, [play mean]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_198(reviews, previous_rounds):
    """#198 Is hotel was chosen in last round? if True, [play mean]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_199(reviews, previous_rounds):
    """#199 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [play mean]
    (0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return play_mean(reviews)

def strategy_200(reviews, previous_rounds):
    """#200 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [play min]
    (-1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return reviews.min()

def strategy_201(reviews, previous_rounds):
    """#201 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [play mean]
    (0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return play_mean(reviews)

def strategy_202(reviews, previous_rounds):
    """#202 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [play max]
    (1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return reviews.max()

def strategy_203(reviews, previous_rounds):
    """#203 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [play min]
    (-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return reviews.min()

def strategy_204(reviews, previous_rounds):
    """#204 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [play max]
    (1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return reviews.max()

def strategy_205(reviews, previous_rounds):
    """#205 Is user earn more than bot? if True, [play max]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_206(reviews, previous_rounds):
    """#206 Is user earn more than bot? if True, [play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_207(reviews, previous_rounds):
    """#207 Is user earn more than bot? if True, [play min]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_208(reviews, previous_rounds):
    """#208 Is user earn more than bot? if True, [play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_209(reviews, previous_rounds):
    """#209 Is user earn more than bot? if True, [play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_210(reviews, previous_rounds):
    """#210 Is user earn more than bot? if True, [play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_211(reviews, previous_rounds):
    """#211 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return play_mean(reviews)

def strategy_212(reviews, previous_rounds):
    """#212 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return play_mean(reviews)

def strategy_213(reviews, previous_rounds):
    """#213 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return play_mean(reviews)

def strategy_214(reviews, previous_rounds):
    """#214 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return play_mean(reviews)

def strategy_215(reviews, previous_rounds):
    """#215 Is hotel was chosen in last round? if True, [play mean]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_216(reviews, previous_rounds):
    """#216 Is hotel was chosen in last round? if True, [play mean]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_217(reviews, previous_rounds):
    """#217 Is hotel was chosen in last round? if True, [play mean]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_218(reviews, previous_rounds):
    """#218 Is hotel was chosen in last round? if True, [play mean]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_219(reviews, previous_rounds):
    """#219 Is user earn more than bot? if True, [play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_220(reviews, previous_rounds):
    """#220 Is user earn more than bot? if True, [play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_221(reviews, previous_rounds):
    """#221 Is user earn more than bot? if True, [play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_222(reviews, previous_rounds):
    """#222 Is user earn more than bot? if True, [play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_223(reviews, previous_rounds):
    """#223 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [play max]
    (1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return reviews.max()

def strategy_224(reviews, previous_rounds):
    """#224 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [play min]
    (-1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return reviews.min()

def strategy_225(reviews, previous_rounds):
    """#225 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [play max]
    (1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return reviews.max()

def strategy_226(reviews, previous_rounds):
    """#226 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [play min]
    (-1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return reviews.min()

def strategy_227(reviews, previous_rounds):
    """#227 Is user earn more than bot? if True, [play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_228(reviews, previous_rounds):
    """#228 Is user earn more than bot? if True, [play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 1, 1, 1, 1, 0, -1, 0, -1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_229(reviews, previous_rounds):
    """#229 Is user earn more than bot? if True, [play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_230(reviews, previous_rounds):
    """#230 Is user earn more than bot? if True, [play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, 1, 1, 1, 1, -1, 0, -1, 0, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_231(reviews, previous_rounds):
    """#231 Is user earn more than bot? if True, [play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, -1, -1, -1, -1, 0, 1, 0, 1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_232(reviews, previous_rounds):
    """#232 Is user earn more than bot? if True, [play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_233(reviews, previous_rounds):
    """#233 Is user earn more than bot? if True, [play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, -1, -1, -1, -1, 1, 0, 1, 0, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_234(reviews, previous_rounds):
    """#234 Is user earn more than bot? if True, [play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_235(reviews, previous_rounds):
    """#235 Is user earn more than bot? if True, [play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, 0, 0, 0, 0, -1, 1, -1, 1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_236(reviews, previous_rounds):
    """#236 Is user earn more than bot? if True, [play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_237(reviews, previous_rounds):
    """#237 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [play mean]
    (0, 0, 0, 0, -1, 1, -1, 1, 0, 0, 0, 0, -1, 1, -1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return play_mean(reviews)

def strategy_238(reviews, previous_rounds):
    """#238 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [play max]
    (1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return reviews.max()

def strategy_239(reviews, previous_rounds):
    """#239 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [play min]
    (-1, -1, -1, -1, 0, 1, 0, 1, -1, -1, -1, -1, 0, 1, 0, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return reviews.min()

def strategy_240(reviews, previous_rounds):
    """#240 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [play mean]
    (0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return play_mean(reviews)

def strategy_241(reviews, previous_rounds):
    """#241 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [play max]
    (1, 1, 1, 1, 0, -1, 0, -1, 1, 1, 1, 1, 0, -1, 0, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return reviews.max()

def strategy_242(reviews, previous_rounds):
    """#242 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [play min]
    (-1, -1, -1, -1, 0, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return reviews.min()

def strategy_243(reviews, previous_rounds):
    """#243 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [play max]
    (1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return reviews.max()

def strategy_244(reviews, previous_rounds):
    """#244 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [play min]
    (-1, -1, -1, -1, 1, 0, 1, 0, -1, -1, -1, -1, 1, 0, 1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return reviews.min()

def strategy_245(reviews, previous_rounds):
    """#245 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [play max]
    (1, 1, 1, 1, -1, 0, -1, 0, 1, 1, 1, 1, -1, 0, -1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return reviews.max()

def strategy_246(reviews, previous_rounds):
    """#246 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [play min]
    (-1, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return reviews.min()

def strategy_247(reviews, previous_rounds):
    """#247 Is hotel score in the last round >= 8? if True, [play max]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 1, 0, 1, -1, 1, -1, 1, 0, 1, 0, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_248(reviews, previous_rounds):
    """#248 Is hotel score in the last round >= 8? if True, [play max]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, 1, -1, 1, 0, 1, 0, 1, -1, 1, -1, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_249(reviews, previous_rounds):
    """#249 Is hotel score in the last round >= 8? if True, [play min]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, -1, 0, -1, 1, -1, 1, -1, 0, -1, 0, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_250(reviews, previous_rounds):
    """#250 Is hotel score in the last round >= 8? if True, [play min]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, -1, 1, -1, 0, -1, 0, -1, 1, -1, 1, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_251(reviews, previous_rounds):
    """#251 Is hotel score in the last round >= 8? if True, [play mean]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, 0, -1, 0, 1, 0, 1, 0, -1, 0, -1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_252(reviews, previous_rounds):
    """#252 Is hotel score in the last round >= 8? if True, [play mean]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 0, 1, 0, -1, 0, -1, 0, 1, 0, 1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_253(reviews, previous_rounds):
    """#253 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [play mean]
    (0, -1, 0, -1, 0, 1, 0, 1, 0, -1, 0, -1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        return play_mean(reviews)

def strategy_254(reviews, previous_rounds):
    """#254 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [play min]
    (-1, 0, -1, 0, -1, 1, -1, 1, -1, 0, -1, 0, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return reviews.min()

def strategy_255(reviews, previous_rounds):
    """#255 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [play mean]
    (0, 1, 0, 1, 0, -1, 0, -1, 0, 1, 0, 1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        return play_mean(reviews)

def strategy_256(reviews, previous_rounds):
    """#256 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [play max]
    (1, 0, 1, 0, 1, -1, 1, -1, 1, 0, 1, 0, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return reviews.max()

def strategy_257(reviews, previous_rounds):
    """#257 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [play min]
    (-1, 1, -1, 1, -1, 0, -1, 0, -1, 1, -1, 1, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return reviews.min()

def strategy_258(reviews, previous_rounds):
    """#258 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [play max]
    (1, -1, 1, -1, 1, 0, 1, 0, 1, -1, 1, -1, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return reviews.max()

def strategy_259(reviews, previous_rounds):
    """#259 Is user earn more than bot? if True, [play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_260(reviews, previous_rounds):
    """#260 Is user earn more than bot? if True, [play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_261(reviews, previous_rounds):
    """#261 Is user earn more than bot? if True, [play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_262(reviews, previous_rounds):
    """#262 Is user earn more than bot? if True, [play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_263(reviews, previous_rounds):
    """#263 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [play mean]
    (0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return play_mean(reviews)

def strategy_264(reviews, previous_rounds):
    """#264 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [play mean]
    (0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return play_mean(reviews)

def strategy_265(reviews, previous_rounds):
    """#265 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [play mean]
    (0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return play_mean(reviews)

def strategy_266(reviews, previous_rounds):
    """#266 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [play mean]
    (0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return play_mean(reviews)

def strategy_267(reviews, previous_rounds):
    """#267 Is hotel was chosen in last round? if True, [play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_268(reviews, previous_rounds):
    """#268 Is hotel was chosen in last round? if True, [play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_269(reviews, previous_rounds):
    """#269 Is hotel was chosen in last round? if True, [play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_270(reviews, previous_rounds):
    """#270 Is hotel was chosen in last round? if True, [play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_271(reviews, previous_rounds):
    """#271 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return reviews.max()

def strategy_272(reviews, previous_rounds):
    """#272 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return reviews.min()

def strategy_273(reviews, previous_rounds):
    """#273 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return reviews.max()

def strategy_274(reviews, previous_rounds):
    """#274 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return reviews.min()

def strategy_275(reviews, previous_rounds):
    """#275 Is hotel was chosen in last round? if True, [play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_276(reviews, previous_rounds):
    """#276 Is hotel was chosen in last round? if True, [play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, -1, 0, -1, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_277(reviews, previous_rounds):
    """#277 Is hotel was chosen in last round? if True, [play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_278(reviews, previous_rounds):
    """#278 Is hotel was chosen in last round? if True, [play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, 0, -1, 0, 1, 1, 1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_279(reviews, previous_rounds):
    """#279 Is hotel was chosen in last round? if True, [play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_280(reviews, previous_rounds):
    """#280 Is hotel was chosen in last round? if True, [play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_281(reviews, previous_rounds):
    """#281 Is hotel was chosen in last round? if True, [play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 0, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_282(reviews, previous_rounds):
    """#282 Is hotel was chosen in last round? if True, [play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_283(reviews, previous_rounds):
    """#283 Is hotel was chosen in last round? if True, [play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_284(reviews, previous_rounds):
    """#284 Is hotel was chosen in last round? if True, [play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_285(reviews, previous_rounds):
    """#285 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, -1, 1, -1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return play_mean(reviews)

def strategy_286(reviews, previous_rounds):
    """#286 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return reviews.max()

def strategy_287(reviews, previous_rounds):
    """#287 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 0, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return reviews.min()

def strategy_288(reviews, previous_rounds):
    """#288 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return play_mean(reviews)

def strategy_289(reviews, previous_rounds):
    """#289 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, 0, -1, 0, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return reviews.max()

def strategy_290(reviews, previous_rounds):
    """#290 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return reviews.min()

def strategy_291(reviews, previous_rounds):
    """#291 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return reviews.max()

def strategy_292(reviews, previous_rounds):
    """#292 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return reviews.min()

def strategy_293(reviews, previous_rounds):
    """#293 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [play max]
    (1, 1, 1, 1, 1, 1, 1, 1, -1, 0, -1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return reviews.max()

def strategy_294(reviews, previous_rounds):
    """#294 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [play min]
    (-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return reviews.min()

def strategy_295(reviews, previous_rounds):
    """#295 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [play mean]
    (0, -1, 0, -1, 0, -1, 0, -1, 0, 1, 0, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        return play_mean(reviews)

def strategy_296(reviews, previous_rounds):
    """#296 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [play min]
    (-1, 0, -1, 0, -1, 0, -1, 0, -1, 1, -1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return reviews.min()

def strategy_297(reviews, previous_rounds):
    """#297 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [play mean]
    (0, 1, 0, 1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        return play_mean(reviews)

def strategy_298(reviews, previous_rounds):
    """#298 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [play max]
    (1, 0, 1, 0, 1, 0, 1, 0, 1, -1, 1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return reviews.max()

def strategy_299(reviews, previous_rounds):
    """#299 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [play min]
    (-1, 1, -1, 1, -1, 1, -1, 1, -1, 0, -1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return reviews.min()

def strategy_300(reviews, previous_rounds):
    """#300 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [play max]
    (1, -1, 1, -1, 1, -1, 1, -1, 1, 0, 1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return reviews.max()

def strategy_301(reviews, previous_rounds):
    """#301 Is hotel score in the last round >= 8? if True, [play max]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 1, 0, 1, 0, 1, 0, 1, -1, 1, -1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_302(reviews, previous_rounds):
    """#302 Is hotel score in the last round >= 8? if True, [play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, 1, -1, 1, -1, 1, -1, 1, 0, 1, 0, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_303(reviews, previous_rounds):
    """#303 Is hotel score in the last round >= 8? if True, [play min]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, -1, 0, -1, 0, -1, 0, -1, 1, -1, 1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_304(reviews, previous_rounds):
    """#304 Is hotel score in the last round >= 8? if True, [play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, -1, 1, -1, 1, -1, 1, -1, 0, -1, 0, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_305(reviews, previous_rounds):
    """#305 Is hotel score in the last round >= 8? if True, [play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, 0, -1, 0, -1, 0, -1, 0, 1, 0, 1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_306(reviews, previous_rounds):
    """#306 Is hotel score in the last round >= 8? if True, [play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 0, 1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_307(reviews, previous_rounds):
    """#307 Is hotel was chosen in last round? if True, [play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_308(reviews, previous_rounds):
    """#308 Is hotel was chosen in last round? if True, [play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_309(reviews, previous_rounds):
    """#309 Is hotel was chosen in last round? if True, [play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_310(reviews, previous_rounds):
    """#310 Is hotel was chosen in last round? if True, [play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_311(reviews, previous_rounds):
    """#311 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        return play_mean(reviews)

def strategy_312(reviews, previous_rounds):
    """#312 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        return play_mean(reviews)

def strategy_313(reviews, previous_rounds):
    """#313 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        return play_mean(reviews)

def strategy_314(reviews, previous_rounds):
    """#314 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [play mean]
    (0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        return play_mean(reviews)

def strategy_315(reviews, previous_rounds):
    """#315 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_316(reviews, previous_rounds):
    """#316 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_317(reviews, previous_rounds):
    """#317 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_318(reviews, previous_rounds):
    """#318 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_319(reviews, previous_rounds):
    """#319 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_320(reviews, previous_rounds):
    """#320 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_321(reviews, previous_rounds):
    """#321 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_322(reviews, previous_rounds):
    """#322 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_323(reviews, previous_rounds):
    """#323 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_324(reviews, previous_rounds):
    """#324 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_325(reviews, previous_rounds):
    """#325 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_326(reviews, previous_rounds):
    """#326 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_327(reviews, previous_rounds):
    """#327 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_328(reviews, previous_rounds):
    """#328 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_329(reviews, previous_rounds):
    """#329 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_330(reviews, previous_rounds):
    """#330 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_331(reviews, previous_rounds):
    """#331 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_332(reviews, previous_rounds):
    """#332 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_333(reviews, previous_rounds):
    """#333 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_334(reviews, previous_rounds):
    """#334 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_335(reviews, previous_rounds):
    """#335 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_336(reviews, previous_rounds):
    """#336 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_337(reviews, previous_rounds):
    """#337 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_338(reviews, previous_rounds):
    """#338 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_339(reviews, previous_rounds):
    """#339 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_340(reviews, previous_rounds):
    """#340 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_341(reviews, previous_rounds):
    """#341 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_342(reviews, previous_rounds):
    """#342 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_343(reviews, previous_rounds):
    """#343 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_344(reviews, previous_rounds):
    """#344 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_345(reviews, previous_rounds):
    """#345 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_346(reviews, previous_rounds):
    """#346 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_347(reviews, previous_rounds):
    """#347 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_348(reviews, previous_rounds):
    """#348 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_349(reviews, previous_rounds):
    """#349 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_350(reviews, previous_rounds):
    """#350 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_351(reviews, previous_rounds):
    """#351 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_352(reviews, previous_rounds):
    """#352 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_353(reviews, previous_rounds):
    """#353 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_354(reviews, previous_rounds):
    """#354 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_355(reviews, previous_rounds):
    """#355 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_356(reviews, previous_rounds):
    """#356 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_357(reviews, previous_rounds):
    """#357 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, 0, 0, -1, -1, 1, 1, -1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_358(reviews, previous_rounds):
    """#358 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, 0, 0, -1, -1, 1, 1, -1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_359(reviews, previous_rounds):
    """#359 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_360(reviews, previous_rounds):
    """#360 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 1, 1, 0, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_361(reviews, previous_rounds):
    """#361 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_362(reviews, previous_rounds):
    """#362 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_363(reviews, previous_rounds):
    """#363 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_364(reviews, previous_rounds):
    """#364 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_365(reviews, previous_rounds):
    """#365 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, 1, 1, 0, 0, -1, -1, 0, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_366(reviews, previous_rounds):
    """#366 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_367(reviews, previous_rounds):
    """#367 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_368(reviews, previous_rounds):
    """#368 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 0, 0, 1, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_369(reviews, previous_rounds):
    """#369 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 0, 0, -1, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_370(reviews, previous_rounds):
    """#370 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_371(reviews, previous_rounds):
    """#371 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, -1, -1, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_372(reviews, previous_rounds):
    """#372 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_373(reviews, previous_rounds):
    """#373 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, -1, -1, -1, -1, 1, 1, 0, 0, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_374(reviews, previous_rounds):
    """#374 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_375(reviews, previous_rounds):
    """#375 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, 0, 0, 0, 0, -1, -1, 1, 1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_376(reviews, previous_rounds):
    """#376 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_377(reviews, previous_rounds):
    """#377 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_378(reviews, previous_rounds):
    """#378 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_379(reviews, previous_rounds):
    """#379 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_380(reviews, previous_rounds):
    """#380 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, 1, 1, 1, 1, -1, -1, 0, 0, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_381(reviews, previous_rounds):
    """#381 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, 0, 0, 0, 0, -1, -1, 1, 1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_382(reviews, previous_rounds):
    """#382 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 0, 0, 0, 0, 1, 1, -1, -1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_383(reviews, previous_rounds):
    """#383 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_384(reviews, previous_rounds):
    """#384 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_385(reviews, previous_rounds):
    """#385 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_386(reviews, previous_rounds):
    """#386 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_387(reviews, previous_rounds):
    """#387 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 0, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_388(reviews, previous_rounds):
    """#388 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_389(reviews, previous_rounds):
    """#389 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 1, -1, -1, 1, 1, 0, 0, 0, 0, -1, -1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_390(reviews, previous_rounds):
    """#390 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, -1, -1, -1, 1, 1, 0, 0, 0, 0, -1, -1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_391(reviews, previous_rounds):
    """#391 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_392(reviews, previous_rounds):
    """#392 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 1, 0, 0, 1, 1, -1, -1, -1, -1, 0, 0, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_393(reviews, previous_rounds):
    """#393 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_394(reviews, previous_rounds):
    """#394 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 0, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_395(reviews, previous_rounds):
    """#395 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 1, 1, 1, -1, -1, 0, 0, 0, 0, 1, 1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_396(reviews, previous_rounds):
    """#396 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, -1, 1, 1, -1, -1, 0, 0, 0, 0, 1, 1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_397(reviews, previous_rounds):
    """#397 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, -1, 0, 0, -1, -1, 1, 1, 1, 1, 0, 0, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_398(reviews, previous_rounds):
    """#398 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_399(reviews, previous_rounds):
    """#399 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_400(reviews, previous_rounds):
    """#400 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, 1, 1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_401(reviews, previous_rounds):
    """#401 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, -1, -1, -1, 0, 0, 1, 1, 1, 1, -1, -1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_402(reviews, previous_rounds):
    """#402 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_403(reviews, previous_rounds):
    """#403 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, -1, -1, 1, 1, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_404(reviews, previous_rounds):
    """#404 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 1, 1, -1, -1, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_405(reviews, previous_rounds):
    """#405 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, -1, -1, 0, 0, -1, -1, 1, 1, 1, 1, 0, 0, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_406(reviews, previous_rounds):
    """#406 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, -1, -1, -1, -1, 1, 1, 0, 0, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_407(reviews, previous_rounds):
    """#407 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, 0, 0, 1, 1, 0, 0, -1, -1, 1, 1, 1, 1, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_408(reviews, previous_rounds):
    """#408 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 1, 1, -1, -1, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_409(reviews, previous_rounds):
    """#409 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, 1, 1, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_410(reviews, previous_rounds):
    """#410 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 1, 1, -1, -1, 1, 1, 0, 0, -1, -1, -1, -1, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_411(reviews, previous_rounds):
    """#411 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, 1, 1, 1, 1, -1, -1, 0, 0, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_412(reviews, previous_rounds):
    """#412 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, 1, 1, 0, 0, 1, 1, -1, -1, -1, -1, 0, 0, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_413(reviews, previous_rounds):
    """#413 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, 0, 0, 1, 1, 0, 0, -1, -1, -1, -1, 1, 1, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_414(reviews, previous_rounds):
    """#414 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 0, 0, -1, -1, 0, 0, 1, 1, -1, -1, -1, -1, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_415(reviews, previous_rounds):
    """#415 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 0, 0, 1, 1, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_416(reviews, previous_rounds):
    """#416 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 0, 0, -1, -1, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_417(reviews, previous_rounds):
    """#417 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 0, 0, 1, 1, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_418(reviews, previous_rounds):
    """#418 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 0, 0, -1, -1, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_419(reviews, previous_rounds):
    """#419 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, -1, -1, 0, 0, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_420(reviews, previous_rounds):
    """#420 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_421(reviews, previous_rounds):
    """#421 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, -1, -1, 1, 1, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_422(reviews, previous_rounds):
    """#422 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, -1, -1, -1, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_423(reviews, previous_rounds):
    """#423 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, 0, 0, -1, -1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_424(reviews, previous_rounds):
    """#424 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 0, 0, 1, 1, 1, 1, -1, -1, 0, 0, -1, -1, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_425(reviews, previous_rounds):
    """#425 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_426(reviews, previous_rounds):
    """#426 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 1, 1, 0, 0, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_427(reviews, previous_rounds):
    """#427 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 1, 1, 1, -1, -1, 0, 0, 1, 1, 0, 0, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_428(reviews, previous_rounds):
    """#428 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, 1, 1, -1, -1, -1, -1, 0, 0, 1, 1, 0, 0, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_429(reviews, previous_rounds):
    """#429 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, 0, 0, -1, -1, -1, -1, 1, 1, 0, 0, 1, 1, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_430(reviews, previous_rounds):
    """#430 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 0, 0, 1, 1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_431(reviews, previous_rounds):
    """#431 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_432(reviews, previous_rounds):
    """#432 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 1, 1, 1, 0, 0, -1, -1, 1, 1, -1, -1, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_433(reviews, previous_rounds):
    """#433 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, -1, -1, -1, 0, 0, 1, 1, -1, -1, 1, 1, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_434(reviews, previous_rounds):
    """#434 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_435(reviews, previous_rounds):
    """#435 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_436(reviews, previous_rounds):
    """#436 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_437(reviews, previous_rounds):
    """#437 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_438(reviews, previous_rounds):
    """#438 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_439(reviews, previous_rounds):
    """#439 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_440(reviews, previous_rounds):
    """#440 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_441(reviews, previous_rounds):
    """#441 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_442(reviews, previous_rounds):
    """#442 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_443(reviews, previous_rounds):
    """#443 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_444(reviews, previous_rounds):
    """#444 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_445(reviews, previous_rounds):
    """#445 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_446(reviews, previous_rounds):
    """#446 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_447(reviews, previous_rounds):
    """#447 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_448(reviews, previous_rounds):
    """#448 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_449(reviews, previous_rounds):
    """#449 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_450(reviews, previous_rounds):
    """#450 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_451(reviews, previous_rounds):
    """#451 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_452(reviews, previous_rounds):
    """#452 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 1, 1, 0, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_453(reviews, previous_rounds):
    """#453 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_454(reviews, previous_rounds):
    """#454 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_455(reviews, previous_rounds):
    """#455 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, 1, 1, 0, 0, -1, -1, 0, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_456(reviews, previous_rounds):
    """#456 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_457(reviews, previous_rounds):
    """#457 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_458(reviews, previous_rounds):
    """#458 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_459(reviews, previous_rounds):
    """#459 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_460(reviews, previous_rounds):
    """#460 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 0, 0, 1, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_461(reviews, previous_rounds):
    """#461 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_462(reviews, previous_rounds):
    """#462 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_463(reviews, previous_rounds):
    """#463 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, 1, 1, -1, -1, 0, 0, -1, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_464(reviews, previous_rounds):
    """#464 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_465(reviews, previous_rounds):
    """#465 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_466(reviews, previous_rounds):
    """#466 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_467(reviews, previous_rounds):
    """#467 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_468(reviews, previous_rounds):
    """#468 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_469(reviews, previous_rounds):
    """#469 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_470(reviews, previous_rounds):
    """#470 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_471(reviews, previous_rounds):
    """#471 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_472(reviews, previous_rounds):
    """#472 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_473(reviews, previous_rounds):
    """#473 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_474(reviews, previous_rounds):
    """#474 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_475(reviews, previous_rounds):
    """#475 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_476(reviews, previous_rounds):
    """#476 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 1, 1, 1, 1, 0, 0, -1, -1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_477(reviews, previous_rounds):
    """#477 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_478(reviews, previous_rounds):
    """#478 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, 1, 1, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_479(reviews, previous_rounds):
    """#479 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, -1, -1, -1, -1, 0, 0, 1, 1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_480(reviews, previous_rounds):
    """#480 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_481(reviews, previous_rounds):
    """#481 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, -1, -1, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_482(reviews, previous_rounds):
    """#482 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_483(reviews, previous_rounds):
    """#483 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_484(reviews, previous_rounds):
    """#484 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 0, 0, 0, 1, 1, -1, -1, -1, -1, 0, 0, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_485(reviews, previous_rounds):
    """#485 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_486(reviews, previous_rounds):
    """#486 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_487(reviews, previous_rounds):
    """#487 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 1, 1, 0, 0, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_488(reviews, previous_rounds):
    """#488 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_489(reviews, previous_rounds):
    """#489 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_490(reviews, previous_rounds):
    """#490 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_491(reviews, previous_rounds):
    """#491 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_492(reviews, previous_rounds):
    """#492 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 0, 1, 1, 0, 0, -1, -1, -1, -1, 1, 1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_493(reviews, previous_rounds):
    """#493 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_494(reviews, previous_rounds):
    """#494 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_495(reviews, previous_rounds):
    """#495 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 0, -1, -1, 0, 0, 1, 1, 1, 1, -1, -1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_496(reviews, previous_rounds):
    """#496 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_497(reviews, previous_rounds):
    """#497 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_498(reviews, previous_rounds):
    """#498 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_499(reviews, previous_rounds):
    """#499 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_500(reviews, previous_rounds):
    """#500 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_501(reviews, previous_rounds):
    """#501 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_502(reviews, previous_rounds):
    """#502 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_503(reviews, previous_rounds):
    """#503 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, 0, 0, 0, 0, -1, -1, 1, 1, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_504(reviews, previous_rounds):
    """#504 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_505(reviews, previous_rounds):
    """#505 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_506(reviews, previous_rounds):
    """#506 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_507(reviews, previous_rounds):
    """#507 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_508(reviews, previous_rounds):
    """#508 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 1, 1, -1, -1, 1, 1, 0, 0, 0, 0, -1, -1, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_509(reviews, previous_rounds):
    """#509 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_510(reviews, previous_rounds):
    """#510 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, 1, 1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_511(reviews, previous_rounds):
    """#511 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, -1, -1, 1, 1, -1, -1, 0, 0, 0, 0, 1, 1, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_512(reviews, previous_rounds):
    """#512 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_513(reviews, previous_rounds):
    """#513 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, -1, -1, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_514(reviews, previous_rounds):
    """#514 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_515(reviews, previous_rounds):
    """#515 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_516(reviews, previous_rounds):
    """#516 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, -1, -1, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_517(reviews, previous_rounds):
    """#517 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_518(reviews, previous_rounds):
    """#518 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_519(reviews, previous_rounds):
    """#519 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 1, 1, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_520(reviews, previous_rounds):
    """#520 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_521(reviews, previous_rounds):
    """#521 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_522(reviews, previous_rounds):
    """#522 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_523(reviews, previous_rounds):
    """#523 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_524(reviews, previous_rounds):
    """#524 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 1, 1, 0, 0, 0, 0, -1, -1, 1, 1, -1, -1, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_525(reviews, previous_rounds):
    """#525 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_526(reviews, previous_rounds):
    """#526 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, 1, 1, -1, -1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_527(reviews, previous_rounds):
    """#527 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, -1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_528(reviews, previous_rounds):
    """#528 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_529(reviews, previous_rounds):
    """#529 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, -1, -1, 1, 1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_530(reviews, previous_rounds):
    """#530 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_531(reviews, previous_rounds):
    """#531 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_532(reviews, previous_rounds):
    """#532 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_533(reviews, previous_rounds):
    """#533 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_534(reviews, previous_rounds):
    """#534 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_535(reviews, previous_rounds):
    """#535 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_536(reviews, previous_rounds):
    """#536 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_537(reviews, previous_rounds):
    """#537 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_538(reviews, previous_rounds):
    """#538 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_539(reviews, previous_rounds):
    """#539 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_540(reviews, previous_rounds):
    """#540 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_541(reviews, previous_rounds):
    """#541 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_542(reviews, previous_rounds):
    """#542 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_543(reviews, previous_rounds):
    """#543 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_544(reviews, previous_rounds):
    """#544 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_545(reviews, previous_rounds):
    """#545 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_546(reviews, previous_rounds):
    """#546 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_547(reviews, previous_rounds):
    """#547 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_548(reviews, previous_rounds):
    """#548 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_549(reviews, previous_rounds):
    """#549 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_550(reviews, previous_rounds):
    """#550 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_551(reviews, previous_rounds):
    """#551 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_552(reviews, previous_rounds):
    """#552 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_553(reviews, previous_rounds):
    """#553 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_554(reviews, previous_rounds):
    """#554 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_555(reviews, previous_rounds):
    """#555 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, -1, -1, 1, 1, 0, 1, 0, 1, -1, -1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_556(reviews, previous_rounds):
    """#556 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, -1, -1, 1, 1, 0, -1, 0, -1, -1, -1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_557(reviews, previous_rounds):
    """#557 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, -1, -1, 1, 1, 1, 0, 1, 0, -1, -1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_558(reviews, previous_rounds):
    """#558 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, -1, 1, 1, -1, 0, -1, 0, -1, -1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_559(reviews, previous_rounds):
    """#559 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, 0, 0, 1, 1, -1, 1, -1, 1, 0, 0, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_560(reviews, previous_rounds):
    """#560 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 0, 0, 1, 1, 1, -1, 1, -1, 0, 0, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_561(reviews, previous_rounds):
    """#561 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 1, 1, -1, -1, 0, 1, 0, 1, 1, 1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_562(reviews, previous_rounds):
    """#562 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 1, 1, -1, -1, 0, -1, 0, -1, 1, 1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_563(reviews, previous_rounds):
    """#563 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 1, -1, -1, 1, 0, 1, 0, 1, 1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_564(reviews, previous_rounds):
    """#564 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, 1, 1, -1, -1, -1, 0, -1, 0, 1, 1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_565(reviews, previous_rounds):
    """#565 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, 0, 0, -1, -1, -1, 1, -1, 1, 0, 0, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_566(reviews, previous_rounds):
    """#566 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 0, 0, -1, -1, 1, -1, 1, -1, 0, 0, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_567(reviews, previous_rounds):
    """#567 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, 1, 1, 0, 0, -1, 1, -1, 1, 1, 1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_568(reviews, previous_rounds):
    """#568 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, 1, 0, 0, 1, -1, 1, -1, 1, 1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_569(reviews, previous_rounds):
    """#569 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, -1, 0, 0, -1, 1, -1, 1, -1, -1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_570(reviews, previous_rounds):
    """#570 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, -1, -1, 0, 0, 1, -1, 1, -1, -1, -1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_571(reviews, previous_rounds):
    """#571 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, -1, 1, -1, 1, 0, 0, 1, 1, -1, 1, -1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_572(reviews, previous_rounds):
    """#572 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, -1, 1, -1, 1, 0, 0, -1, -1, -1, 1, -1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_573(reviews, previous_rounds):
    """#573 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, -1, 1, -1, 1, 1, 1, 0, 0, -1, 1, -1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_574(reviews, previous_rounds):
    """#574 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, 1, -1, 1, -1, -1, 0, 0, -1, 1, -1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_575(reviews, previous_rounds):
    """#575 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, 0, 1, 0, 1, -1, -1, 1, 1, 0, 1, 0, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_576(reviews, previous_rounds):
    """#576 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 0, 1, 0, 1, 1, 1, -1, -1, 0, 1, 0, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_577(reviews, previous_rounds):
    """#577 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 1, -1, 1, -1, 0, 0, 1, 1, 1, -1, 1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_578(reviews, previous_rounds):
    """#578 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 1, -1, 1, -1, 0, 0, -1, -1, 1, -1, 1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_579(reviews, previous_rounds):
    """#579 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, -1, 1, -1, 1, 1, 0, 0, 1, -1, 1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_580(reviews, previous_rounds):
    """#580 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, 1, -1, 1, -1, -1, -1, 0, 0, 1, -1, 1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_581(reviews, previous_rounds):
    """#581 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, 0, -1, 0, -1, -1, -1, 1, 1, 0, -1, 0, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_582(reviews, previous_rounds):
    """#582 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 0, -1, 0, -1, 1, 1, -1, -1, 0, -1, 0, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_583(reviews, previous_rounds):
    """#583 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, 1, 0, 1, 0, -1, -1, 1, 1, 1, 0, 1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_584(reviews, previous_rounds):
    """#584 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 0, 1, 0, 1, 1, -1, -1, 1, 0, 1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_585(reviews, previous_rounds):
    """#585 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, 0, -1, 0, -1, -1, 1, 1, -1, 0, -1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_586(reviews, previous_rounds):
    """#586 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, -1, 0, -1, 0, 1, 1, -1, -1, -1, 0, -1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_587(reviews, previous_rounds):
    """#587 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, -1, -1, 0, 1, 1, 1, 0, 1, -1, -1, 0, 1, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_588(reviews, previous_rounds):
    """#588 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, -1, -1, 0, -1, 1, 1, 0, -1, -1, -1, 0, -1, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_589(reviews, previous_rounds):
    """#589 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, -1, -1, 1, 0, 1, 1, 1, 0, -1, -1, 1, 0, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_590(reviews, previous_rounds):
    """#590 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, -1, -1, 0, 1, 1, -1, 0, -1, -1, -1, 0, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_591(reviews, previous_rounds):
    """#591 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, 0, 0, -1, 1, 1, 1, -1, 1, 0, 0, -1, 1, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_592(reviews, previous_rounds):
    """#592 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 0, 0, 1, -1, 1, 1, 1, -1, 0, 0, 1, -1, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_593(reviews, previous_rounds):
    """#593 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 1, 1, 0, 1, -1, -1, 0, 1, 1, 1, 0, 1, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_594(reviews, previous_rounds):
    """#594 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 1, 1, 0, -1, -1, -1, 0, -1, 1, 1, 0, -1, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_595(reviews, previous_rounds):
    """#595 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 1, 1, 0, -1, -1, 1, 0, 1, 1, 1, 0, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_596(reviews, previous_rounds):
    """#596 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, 1, 1, -1, 0, -1, -1, -1, 0, 1, 1, -1, 0, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_597(reviews, previous_rounds):
    """#597 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, 0, 0, -1, 1, -1, -1, -1, 1, 0, 0, -1, 1, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_598(reviews, previous_rounds):
    """#598 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 0, 0, 1, -1, -1, -1, 1, -1, 0, 0, 1, -1, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_599(reviews, previous_rounds):
    """#599 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, 1, 1, -1, 1, 0, 0, -1, 1, 1, 1, -1, 1, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_600(reviews, previous_rounds):
    """#600 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, 1, 1, -1, 0, 0, 1, -1, 1, 1, 1, -1, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_601(reviews, previous_rounds):
    """#601 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, -1, -1, 1, 0, 0, -1, 1, -1, -1, -1, 1, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_602(reviews, previous_rounds):
    """#602 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, -1, -1, 1, -1, 0, 0, 1, -1, -1, -1, 1, -1, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_603(reviews, previous_rounds):
    """#603 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, -1, 1, 1, 1, -1, 1, 0, 0, -1, 1, 1, 1, -1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_604(reviews, previous_rounds):
    """#604 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, -1, 1, -1, -1, -1, 1, 0, 0, -1, 1, -1, -1, -1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_605(reviews, previous_rounds):
    """#605 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, -1, 1, 0, 0, -1, 1, 1, 1, -1, 1, 0, 0, -1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_606(reviews, previous_rounds):
    """#606 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, 1, 0, 0, -1, 1, -1, -1, -1, 1, 0, 0, -1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_607(reviews, previous_rounds):
    """#607 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, 0, 1, 1, 1, 0, 1, -1, -1, 0, 1, 1, 1, 0, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_608(reviews, previous_rounds):
    """#608 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 0, 1, -1, -1, 0, 1, 1, 1, 0, 1, -1, -1, 0, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_609(reviews, previous_rounds):
    """#609 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 1, -1, 1, 1, 1, -1, 0, 0, 1, -1, 1, 1, 1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_610(reviews, previous_rounds):
    """#610 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 1, -1, -1, -1, 1, -1, 0, 0, 1, -1, -1, -1, 1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_611(reviews, previous_rounds):
    """#611 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, -1, 0, 0, 1, -1, 1, 1, 1, -1, 0, 0, 1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_612(reviews, previous_rounds):
    """#612 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, 1, -1, 0, 0, 1, -1, -1, -1, 1, -1, 0, 0, 1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_613(reviews, previous_rounds):
    """#613 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, 0, -1, 1, 1, 0, -1, -1, -1, 0, -1, 1, 1, 0, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_614(reviews, previous_rounds):
    """#614 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 0, -1, -1, -1, 0, -1, 1, 1, 0, -1, -1, -1, 0, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_615(reviews, previous_rounds):
    """#615 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, 1, 0, 1, 1, 1, 0, -1, -1, 1, 0, 1, 1, 1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_616(reviews, previous_rounds):
    """#616 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 0, -1, -1, 1, 0, 1, 1, 1, 0, -1, -1, 1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_617(reviews, previous_rounds):
    """#617 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, 0, 1, 1, -1, 0, -1, -1, -1, 0, 1, 1, -1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_618(reviews, previous_rounds):
    """#618 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, -1, 0, -1, -1, -1, 0, 1, 1, -1, 0, -1, -1, -1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_619(reviews, previous_rounds):
    """#619 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, -1, 1, -1, 0, 1, 1, 1, 0, -1, 1, -1, 0, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_620(reviews, previous_rounds):
    """#620 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, -1, -1, -1, 0, 1, -1, 1, 0, -1, -1, -1, 0, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_621(reviews, previous_rounds):
    """#621 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, -1, 0, -1, 1, 1, 0, 1, 1, -1, 0, -1, 1, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_622(reviews, previous_rounds):
    """#622 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, -1, -1, 1, 0, 1, -1, -1, 0, -1, -1, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_623(reviews, previous_rounds):
    """#623 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, 0, 1, 0, -1, 1, 1, 1, -1, 0, 1, 0, -1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_624(reviews, previous_rounds):
    """#624 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 0, -1, 0, 1, 1, -1, 1, 1, 0, -1, 0, 1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_625(reviews, previous_rounds):
    """#625 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 1, 1, 1, 0, -1, 1, -1, 0, 1, 1, 1, 0, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_626(reviews, previous_rounds):
    """#626 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 1, -1, 1, 0, -1, -1, -1, 0, 1, -1, 1, 0, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_627(reviews, previous_rounds):
    """#627 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 1, 1, -1, 0, -1, 1, 1, 0, 1, 1, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_628(reviews, previous_rounds):
    """#628 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, 1, 0, 1, -1, -1, 0, -1, -1, 1, 0, 1, -1, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_629(reviews, previous_rounds):
    """#629 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, 0, 1, 0, -1, -1, 1, -1, -1, 0, 1, 0, -1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_630(reviews, previous_rounds):
    """#630 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 0, -1, 0, 1, -1, -1, -1, 1, 0, -1, 0, 1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_631(reviews, previous_rounds):
    """#631 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, 1, 1, 1, -1, 0, 1, 0, -1, 1, 1, 1, -1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_632(reviews, previous_rounds):
    """#632 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, 1, 1, 0, -1, 0, 1, 1, -1, 1, 1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_633(reviews, previous_rounds):
    """#633 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, -1, -1, 0, 1, 0, -1, -1, 1, -1, -1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_634(reviews, previous_rounds):
    """#634 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, -1, -1, -1, 1, 0, -1, 0, 1, -1, -1, -1, 1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_635(reviews, previous_rounds):
    """#635 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, -1, 0, 1, 1, -1, 1, 1, 0, -1, 0, 1, 1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_636(reviews, previous_rounds):
    """#636 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, -1, 0, 1, -1, -1, -1, 1, 0, -1, 0, 1, -1, -1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_637(reviews, previous_rounds):
    """#637 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, -1, 1, 1, 0, -1, 0, 1, 1, -1, 1, 1, 0, -1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_638(reviews, previous_rounds):
    """#638 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, 1, 0, -1, 0, 1, -1, -1, -1, 1, 0, -1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_639(reviews, previous_rounds):
    """#639 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, 0, -1, 1, 1, 0, 1, 1, -1, 0, -1, 1, 1, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_640(reviews, previous_rounds):
    """#640 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 0, 1, 1, -1, 0, -1, 1, 1, 0, 1, 1, -1, 0, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_641(reviews, previous_rounds):
    """#641 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 1, 0, -1, 1, 1, 1, -1, 0, 1, 0, -1, 1, 1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_642(reviews, previous_rounds):
    """#642 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 1, 0, -1, -1, 1, -1, -1, 0, 1, 0, -1, -1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_643(reviews, previous_rounds):
    """#643 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, -1, 0, 1, 0, -1, 1, 1, 1, -1, 0, 1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_644(reviews, previous_rounds):
    """#644 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, 1, -1, -1, 0, 1, 0, -1, -1, 1, -1, -1, 0, 1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_645(reviews, previous_rounds):
    """#645 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, 0, -1, -1, 1, 0, 1, -1, -1, 0, -1, -1, 1, 0, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_646(reviews, previous_rounds):
    """#646 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 0, 1, -1, -1, 0, -1, -1, 1, 0, 1, -1, -1, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_647(reviews, previous_rounds):
    """#647 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, 1, -1, 0, 1, 1, 1, 0, -1, 1, -1, 0, 1, 1, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_648(reviews, previous_rounds):
    """#648 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 0, -1, 1, -1, 0, 1, 1, 1, 0, -1, 1, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_649(reviews, previous_rounds):
    """#649 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, 0, 1, -1, 1, 0, -1, -1, -1, 0, 1, -1, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_650(reviews, previous_rounds):
    """#650 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, -1, 1, 0, -1, -1, -1, 0, 1, -1, 1, 0, -1, -1, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_651(reviews, previous_rounds):
    """#651 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_652(reviews, previous_rounds):
    """#652 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, 0, 1, 1, 0, -1, 0, -1, 0, 0, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_653(reviews, previous_rounds):
    """#653 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_654(reviews, previous_rounds):
    """#654 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, 0, 0, 1, 1, -1, 0, -1, 0, 0, 0, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_655(reviews, previous_rounds):
    """#655 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 0, -1, -1, 0, 1, 0, 1, 0, 0, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_656(reviews, previous_rounds):
    """#656 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, 0, -1, -1, 0, -1, 0, -1, 0, 0, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_657(reviews, previous_rounds):
    """#657 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 0, 0, -1, -1, 1, 0, 1, 0, 0, 0, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_658(reviews, previous_rounds):
    """#658 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, 0, 0, -1, -1, -1, 0, -1, 0, 0, 0, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_659(reviews, previous_rounds):
    """#659 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_660(reviews, previous_rounds):
    """#660 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 1, 1, 0, 0, 0, -1, 0, -1, 1, 1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_661(reviews, previous_rounds):
    """#661 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_662(reviews, previous_rounds):
    """#662 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, 1, 1, 0, 0, -1, 0, -1, 0, 1, 1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_663(reviews, previous_rounds):
    """#663 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, -1, -1, 0, 0, 0, 1, 0, 1, -1, -1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_664(reviews, previous_rounds):
    """#664 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, -1, -1, 0, 0, 0, -1, 0, -1, -1, -1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_665(reviews, previous_rounds):
    """#665 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, -1, -1, 0, 0, 1, 0, 1, 0, -1, -1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_666(reviews, previous_rounds):
    """#666 Is user earn more than bot? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, -1, 0, 0, -1, 0, -1, 0, -1, -1, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_667(reviews, previous_rounds):
    """#667 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_668(reviews, previous_rounds):
    """#668 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 1, 0, 1, 0, 0, -1, -1, 0, 1, 0, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_669(reviews, previous_rounds):
    """#669 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_670(reviews, previous_rounds):
    """#670 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, 0, 1, 0, 1, -1, -1, 0, 0, 0, 1, 0, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_671(reviews, previous_rounds):
    """#671 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, -1, 0, -1, 0, 0, 1, 1, 0, -1, 0, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_672(reviews, previous_rounds):
    """#672 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, -1, 0, -1, 0, 0, -1, -1, 0, -1, 0, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_673(reviews, previous_rounds):
    """#673 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 0, -1, 0, -1, 1, 1, 0, 0, 0, -1, 0, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_674(reviews, previous_rounds):
    """#674 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, 0, -1, 0, -1, -1, -1, 0, 0, 0, -1, 0, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_675(reviews, previous_rounds):
    """#675 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_676(reviews, previous_rounds):
    """#676 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 1, 0, 1, 0, 0, 0, -1, -1, 1, 0, 1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_677(reviews, previous_rounds):
    """#677 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_678(reviews, previous_rounds):
    """#678 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, 1, 0, 1, 0, -1, -1, 0, 0, 1, 0, 1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_679(reviews, previous_rounds):
    """#679 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, -1, 0, -1, 0, 0, 0, 1, 1, -1, 0, -1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_680(reviews, previous_rounds):
    """#680 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, -1, 0, -1, 0, 0, 0, -1, -1, -1, 0, -1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_681(reviews, previous_rounds):
    """#681 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, -1, 0, -1, 0, 1, 1, 0, 0, -1, 0, -1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_682(reviews, previous_rounds):
    """#682 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, 0, -1, 0, -1, -1, 0, 0, -1, 0, -1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_683(reviews, previous_rounds):
    """#683 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_684(reviews, previous_rounds):
    """#684 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, 0, 0, -1, 1, 1, 0, -1, 0, 0, 0, -1, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_685(reviews, previous_rounds):
    """#685 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_686(reviews, previous_rounds):
    """#686 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, 0, 0, -1, 0, 1, 1, -1, 0, 0, 0, -1, 0, 1, 1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_687(reviews, previous_rounds):
    """#687 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 0, 0, 1, -1, -1, 0, 1, 0, 0, 0, 1, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_688(reviews, previous_rounds):
    """#688 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, 0, 0, -1, -1, -1, 0, -1, 0, 0, 0, -1, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_689(reviews, previous_rounds):
    """#689 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 0, 0, 1, 0, -1, -1, 1, 0, 0, 0, 1, 0, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_690(reviews, previous_rounds):
    """#690 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, 0, 0, -1, 0, -1, -1, -1, 0, 0, 0, -1, 0, -1, -1)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_691(reviews, previous_rounds):
    """#691 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_692(reviews, previous_rounds):
    """#692 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 1, 1, 0, -1, 0, 0, 0, -1, 1, 1, 0, -1, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_693(reviews, previous_rounds):
    """#693 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_694(reviews, previous_rounds):
    """#694 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, 1, 1, -1, 0, 0, 0, -1, 0, 1, 1, -1, 0, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_695(reviews, previous_rounds):
    """#695 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, -1, -1, 0, 1, 0, 0, 0, 1, -1, -1, 0, 1, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_696(reviews, previous_rounds):
    """#696 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, -1, -1, 0, -1, 0, 0, 0, -1, -1, -1, 0, -1, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_697(reviews, previous_rounds):
    """#697 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, -1, -1, 1, 0, 0, 0, 1, 0, -1, -1, 1, 0, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_698(reviews, previous_rounds):
    """#698 Is hotel score >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, -1, -1, 0, 0, 0, -1, 0, -1, -1, -1, 0, 0, 0)"""
    if reviews.mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_699(reviews, previous_rounds):
    """#699 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_700(reviews, previous_rounds):
    """#700 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 1, -1, -1, 0, 1, 0, 0, 0, 1, -1, -1, 0, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_701(reviews, previous_rounds):
    """#701 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_702(reviews, previous_rounds):
    """#702 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, 0, 1, 0, 0, 0, 1, -1, -1, 0, 1, 0, 0, 0, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_703(reviews, previous_rounds):
    """#703 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, -1, 1, 1, 0, -1, 0, 0, 0, -1, 1, 1, 0, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_704(reviews, previous_rounds):
    """#704 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, -1, -1, -1, 0, -1, 0, 0, 0, -1, -1, -1, 0, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_705(reviews, previous_rounds):
    """#705 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 0, -1, 0, 0, 0, -1, 1, 1, 0, -1, 0, 0, 0, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_706(reviews, previous_rounds):
    """#706 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, 0, -1, 0, 0, 0, -1, -1, -1, 0, -1, 0, 0, 0, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_707(reviews, previous_rounds):
    """#707 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_708(reviews, previous_rounds):
    """#708 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 1, 0, -1, -1, 1, 0, 0, 0, 1, 0, -1, -1, 1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_709(reviews, previous_rounds):
    """#709 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_710(reviews, previous_rounds):
    """#710 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, 1, 0, 0, 0, 1, 0, -1, -1, 1, 0, 0, 0, 1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_711(reviews, previous_rounds):
    """#711 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, -1, 0, 1, 1, -1, 0, 0, 0, -1, 0, 1, 1, -1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_712(reviews, previous_rounds):
    """#712 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, -1, 0, -1, -1, -1, 0, 0, 0, -1, 0, -1, -1, -1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_713(reviews, previous_rounds):
    """#713 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, -1, 0, 0, 0, -1, 0, 1, 1, -1, 0, 0, 0, -1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_714(reviews, previous_rounds):
    """#714 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, 0, 0, 0, -1, 0, -1, -1, -1, 0, 0, 0, -1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_715(reviews, previous_rounds):
    """#715 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_716(reviews, previous_rounds):
    """#716 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, 0, 0, 1, -1, 1, 0, 0, -1, 0, 0, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_717(reviews, previous_rounds):
    """#717 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_718(reviews, previous_rounds):
    """#718 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, 0, 0, 0, -1, 1, 0, 1, -1, 0, 0, 0, -1, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_719(reviews, previous_rounds):
    """#719 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 0, 0, -1, 1, -1, 0, 0, 1, 0, 0, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_720(reviews, previous_rounds):
    """#720 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, 0, 0, -1, -1, -1, 0, 0, -1, 0, 0, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_721(reviews, previous_rounds):
    """#721 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 0, 0, 0, 1, -1, 0, -1, 1, 0, 0, 0, 1, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_722(reviews, previous_rounds):
    """#722 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, 0, 0, 0, -1, -1, 0, -1, -1, 0, 0, 0, -1, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_723(reviews, previous_rounds):
    """#723 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_724(reviews, previous_rounds):
    """#724 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 1, -1, 1, 0, 0, -1, 0, 0, 1, -1, 1, 0, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_725(reviews, previous_rounds):
    """#725 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_726(reviews, previous_rounds):
    """#726 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, 1, 0, 1, -1, 0, 0, 0, -1, 1, 0, 1, -1, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_727(reviews, previous_rounds):
    """#727 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, -1, 1, -1, 0, 0, 1, 0, 0, -1, 1, -1, 0, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_728(reviews, previous_rounds):
    """#728 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, -1, -1, -1, 0, 0, -1, 0, 0, -1, -1, -1, 0, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_729(reviews, previous_rounds):
    """#729 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, -1, 0, -1, 1, 0, 0, 0, 1, -1, 0, -1, 1, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_730(reviews, previous_rounds):
    """#730 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, -1, -1, 0, 0, 0, -1, -1, 0, -1, -1, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_731(reviews, previous_rounds):
    """#731 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_732(reviews, previous_rounds):
    """#732 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 1, -1, 0, -1, 1, 0, 0, 0, 1, -1, 0, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_733(reviews, previous_rounds):
    """#733 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_734(reviews, previous_rounds):
    """#734 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, 0, -1, 1, 0, 0, 0, 1, -1, 0, -1, 1, 0, 0, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_735(reviews, previous_rounds):
    """#735 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, -1, 1, 0, 1, -1, 0, 0, 0, -1, 1, 0, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_736(reviews, previous_rounds):
    """#736 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, -1, -1, 0, -1, -1, 0, 0, 0, -1, -1, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_737(reviews, previous_rounds):
    """#737 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 0, 1, -1, 0, 0, 0, -1, 1, 0, 1, -1, 0, 0, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_738(reviews, previous_rounds):
    """#738 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, 0, -1, -1, 0, 0, 0, -1, -1, 0, -1, -1, 0, 0, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_739(reviews, previous_rounds):
    """#739 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_740(reviews, previous_rounds):
    """#740 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 1, 0, 0, -1, 1, -1, 0, 0, 1, 0, 0, -1, 1, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_741(reviews, previous_rounds):
    """#741 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_742(reviews, previous_rounds):
    """#742 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, 1, -1, 0, 0, 1, 0, 0, -1, 1, -1, 0, 0, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_743(reviews, previous_rounds):
    """#743 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, -1, 0, 0, 1, -1, 1, 0, 0, -1, 0, 0, 1, -1, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_744(reviews, previous_rounds):
    """#744 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, -1, 0, 0, -1, -1, -1, 0, 0, -1, 0, 0, -1, -1, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_745(reviews, previous_rounds):
    """#745 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, -1, 1, 0, 0, -1, 0, 0, 1, -1, 1, 0, 0, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_746(reviews, previous_rounds):
    """#746 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, 0, 0, -1, 0, 0, -1, -1, -1, 0, 0, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_747(reviews, previous_rounds):
    """#747 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_748(reviews, previous_rounds):
    """#748 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_749(reviews, previous_rounds):
    """#749 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_750(reviews, previous_rounds):
    """#750 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_751(reviews, previous_rounds):
    """#751 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_752(reviews, previous_rounds):
    """#752 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_753(reviews, previous_rounds):
    """#753 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_754(reviews, previous_rounds):
    """#754 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_755(reviews, previous_rounds):
    """#755 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_756(reviews, previous_rounds):
    """#756 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_757(reviews, previous_rounds):
    """#757 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_758(reviews, previous_rounds):
    """#758 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_759(reviews, previous_rounds):
    """#759 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_760(reviews, previous_rounds):
    """#760 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_761(reviews, previous_rounds):
    """#761 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_762(reviews, previous_rounds):
    """#762 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_763(reviews, previous_rounds):
    """#763 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_764(reviews, previous_rounds):
    """#764 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_765(reviews, previous_rounds):
    """#765 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_766(reviews, previous_rounds):
    """#766 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_767(reviews, previous_rounds):
    """#767 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_768(reviews, previous_rounds):
    """#768 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_769(reviews, previous_rounds):
    """#769 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_770(reviews, previous_rounds):
    """#770 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_771(reviews, previous_rounds):
    """#771 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 1, 0, 1, -1, -1, 1, 1, -1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_772(reviews, previous_rounds):
    """#772 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, -1, 0, -1, -1, -1, 1, 1, -1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_773(reviews, previous_rounds):
    """#773 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 0, 1, 0, -1, -1, 1, 1, -1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_774(reviews, previous_rounds):
    """#774 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, 0, -1, 0, -1, -1, 1, 1, -1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_775(reviews, previous_rounds):
    """#775 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, 1, -1, 1, 0, 0, 1, 1, 0, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_776(reviews, previous_rounds):
    """#776 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 1, 1, 0, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_777(reviews, previous_rounds):
    """#777 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 1, 0, 1, 1, 1, -1, -1, 1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_778(reviews, previous_rounds):
    """#778 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, -1, 0, -1, 1, 1, -1, -1, 1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_779(reviews, previous_rounds):
    """#779 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 0, 1, 0, 1, 1, -1, -1, 1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_780(reviews, previous_rounds):
    """#780 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, 0, -1, 0, 1, 1, -1, -1, 1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_781(reviews, previous_rounds):
    """#781 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, 1, -1, 1, 0, 0, -1, -1, 0, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_782(reviews, previous_rounds):
    """#782 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, -1, 1, -1, 0, 0, -1, -1, 0, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_783(reviews, previous_rounds):
    """#783 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 0, 0, 1, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_784(reviews, previous_rounds):
    """#784 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 0, 0, 1, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_785(reviews, previous_rounds):
    """#785 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 0, 0, -1, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_786(reviews, previous_rounds):
    """#786 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 0, 0, -1, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_787(reviews, previous_rounds):
    """#787 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_788(reviews, previous_rounds):
    """#788 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_789(reviews, previous_rounds):
    """#789 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 0, 0, -1, 1, -1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_790(reviews, previous_rounds):
    """#790 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, 0, 0, -1, 1, -1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_791(reviews, previous_rounds):
    """#791 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_792(reviews, previous_rounds):
    """#792 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, -1, -1, 0, 1, 0, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_793(reviews, previous_rounds):
    """#793 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_794(reviews, previous_rounds):
    """#794 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_795(reviews, previous_rounds):
    """#795 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_796(reviews, previous_rounds):
    """#796 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_797(reviews, previous_rounds):
    """#797 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, 1, 1, 0, -1, 0, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_798(reviews, previous_rounds):
    """#798 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_799(reviews, previous_rounds):
    """#799 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_800(reviews, previous_rounds):
    """#800 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, -1, -1, 1, 0, 1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_801(reviews, previous_rounds):
    """#801 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, 1, -1, -1, 1, 1, -1, 0, -1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_802(reviews, previous_rounds):
    """#802 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, -1, 1, 1, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_803(reviews, previous_rounds):
    """#803 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, -1, -1, 0, 1, -1, -1, 0, 1, 1, 1, 0, 1, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_804(reviews, previous_rounds):
    """#804 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, -1, -1, 0, -1, -1, -1, 0, -1, 1, 1, 0, -1, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_805(reviews, previous_rounds):
    """#805 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, -1, -1, 1, 0, -1, -1, 1, 0, 1, 1, 1, 0, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_806(reviews, previous_rounds):
    """#806 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 1, 1, -1, 0, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_807(reviews, previous_rounds):
    """#807 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 1, 1, -1, 1, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_808(reviews, previous_rounds):
    """#808 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 1, 1, 1, -1, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_809(reviews, previous_rounds):
    """#809 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 1, 1, 0, 1, 1, 1, 0, 1, -1, -1, 0, 1, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_810(reviews, previous_rounds):
    """#810 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 1, 1, 0, -1, 1, 1, 0, -1, -1, -1, 0, -1, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_811(reviews, previous_rounds):
    """#811 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 1, 1, 0, 1, 1, 1, 0, -1, -1, 1, 0, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_812(reviews, previous_rounds):
    """#812 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, 1, 1, -1, 0, 1, 1, -1, 0, -1, -1, -1, 0, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_813(reviews, previous_rounds):
    """#813 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, 0, 0, -1, 1, 0, 0, -1, 1, -1, -1, -1, 1, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_814(reviews, previous_rounds):
    """#814 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, -1, 1, -1, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_815(reviews, previous_rounds):
    """#815 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 0, 0, -1, 1, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_816(reviews, previous_rounds):
    """#816 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 0, 0, 1, -1, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_817(reviews, previous_rounds):
    """#817 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 0, 0, -1, 1, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_818(reviews, previous_rounds):
    """#818 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 0, 0, 1, -1, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_819(reviews, previous_rounds):
    """#819 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, -1, 1, 0, 0, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_820(reviews, previous_rounds):
    """#820 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, -1, 1, 0, 0, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_821(reviews, previous_rounds):
    """#821 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, -1, 1, 1, 1, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_822(reviews, previous_rounds):
    """#822 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, 1, -1, -1, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_823(reviews, previous_rounds):
    """#823 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, 0, 1, -1, -1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_824(reviews, previous_rounds):
    """#824 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 0, 1, 1, 1, 0, 1, -1, -1, 0, 1, -1, -1, 0, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_825(reviews, previous_rounds):
    """#825 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 1, -1, 0, 0, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_826(reviews, previous_rounds):
    """#826 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 1, -1, 0, 0, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_827(reviews, previous_rounds):
    """#827 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, -1, 1, 1, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_828(reviews, previous_rounds):
    """#828 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, 1, -1, -1, -1, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_829(reviews, previous_rounds):
    """#829 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, 0, -1, -1, -1, 0, -1, 1, 1, 0, -1, 1, 1, 0, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_830(reviews, previous_rounds):
    """#830 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 0, -1, 1, 1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_831(reviews, previous_rounds):
    """#831 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, 1, 0, -1, -1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_832(reviews, previous_rounds):
    """#832 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 0, 1, 1, 1, 0, -1, -1, 1, 0, -1, -1, 1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_833(reviews, previous_rounds):
    """#833 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, 0, -1, -1, -1, 0, 1, 1, -1, 0, 1, 1, -1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_834(reviews, previous_rounds):
    """#834 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, -1, 0, 1, 1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_835(reviews, previous_rounds):
    """#835 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, -1, 1, -1, 0, -1, 1, -1, 0, 1, 1, 1, 0, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_836(reviews, previous_rounds):
    """#836 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, -1, -1, -1, 0, -1, -1, -1, 0, 1, -1, 1, 0, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_837(reviews, previous_rounds):
    """#837 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, -1, 0, -1, 1, -1, 0, -1, 1, 1, 0, 1, 1, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_838(reviews, previous_rounds):
    """#838 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, -1, -1, -1, 0, -1, -1, 1, 0, 1, -1, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_839(reviews, previous_rounds):
    """#839 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, 0, 1, 0, -1, 0, 1, 0, -1, 1, 1, 1, -1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_840(reviews, previous_rounds):
    """#840 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 0, -1, 0, 1, 0, -1, 0, 1, 1, -1, 1, 1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_841(reviews, previous_rounds):
    """#841 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 1, 1, 1, 0, 1, 1, 1, 0, -1, 1, -1, 0, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_842(reviews, previous_rounds):
    """#842 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 1, -1, 1, 0, 1, -1, 1, 0, -1, -1, -1, 0, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_843(reviews, previous_rounds):
    """#843 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 1, 1, 1, 0, 1, 1, -1, 0, -1, 1, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_844(reviews, previous_rounds):
    """#844 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, 1, 0, 1, -1, 1, 0, 1, -1, -1, 0, -1, -1, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_845(reviews, previous_rounds):
    """#845 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, 0, 1, 0, -1, 0, 1, 0, -1, -1, 1, -1, -1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_846(reviews, previous_rounds):
    """#846 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 0, -1, 0, 1, 0, -1, 0, 1, -1, -1, -1, 1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_847(reviews, previous_rounds):
    """#847 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, 1, 1, 1, -1, 1, 1, 1, -1, 0, 1, 0, -1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_848(reviews, previous_rounds):
    """#848 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, 1, -1, 1, 1, 1, -1, 1, 1, 0, -1, 0, 1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_849(reviews, previous_rounds):
    """#849 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play min]
    (-1, -1, 1, -1, -1, -1, 1, -1, -1, 0, 1, 0, -1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_850(reviews, previous_rounds):
    """#850 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play max]
    (1, -1, -1, -1, 1, -1, -1, -1, 1, 0, -1, 0, 1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_851(reviews, previous_rounds):
    """#851 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, -1, 0, 1, 0, -1, 0, 1, 1, -1, 1, 1, 1, -1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_852(reviews, previous_rounds):
    """#852 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, -1, 0, 1, 0, -1, 0, 1, -1, -1, -1, 1, -1, -1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_853(reviews, previous_rounds):
    """#853 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, -1, 1, 1, 1, -1, 1, 1, 0, -1, 0, 1, 0, -1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_854(reviews, previous_rounds):
    """#854 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, 1, -1, -1, -1, 1, 0, -1, 0, 1, 0, -1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_855(reviews, previous_rounds):
    """#855 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, 0, -1, 1, -1, 0, -1, 1, 1, 0, 1, 1, 1, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_856(reviews, previous_rounds):
    """#856 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 0, 1, 1, 1, 0, 1, 1, -1, 0, -1, 1, -1, 0, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_857(reviews, previous_rounds):
    """#857 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 1, 0, -1, 0, 1, 0, -1, 1, 1, 1, -1, 1, 1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_858(reviews, previous_rounds):
    """#858 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 1, 0, -1, 0, 1, 0, -1, -1, 1, -1, -1, -1, 1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_859(reviews, previous_rounds):
    """#859 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, -1, 1, 1, 1, -1, 0, 1, 0, -1, 0, 1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_860(reviews, previous_rounds):
    """#860 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, 1, -1, -1, -1, 1, -1, -1, 0, 1, 0, -1, 0, 1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_861(reviews, previous_rounds):
    """#861 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, 0, -1, -1, -1, 0, -1, -1, 1, 0, 1, -1, 1, 0, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_862(reviews, previous_rounds):
    """#862 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 0, 1, -1, 1, 0, 1, -1, -1, 0, -1, -1, -1, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_863(reviews, previous_rounds):
    """#863 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, 1, -1, 0, -1, 1, -1, 0, 1, 1, 1, 0, 1, 1, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_864(reviews, previous_rounds):
    """#864 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 0, 1, 1, 1, 0, -1, 1, -1, 0, -1, 1, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_865(reviews, previous_rounds):
    """#865 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, 0, -1, -1, -1, 0, 1, -1, 1, 0, 1, -1, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_866(reviews, previous_rounds):
    """#866 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, -1, 1, 0, 1, -1, 1, 0, -1, -1, -1, 0, -1, -1, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_867(reviews, previous_rounds):
    """#867 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_868(reviews, previous_rounds):
    """#868 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 1, 1, 0, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_869(reviews, previous_rounds):
    """#869 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_870(reviews, previous_rounds):
    """#870 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 1, 1, 0, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_871(reviews, previous_rounds):
    """#871 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 1, 0, 1, 0, 0, -1, -1, 0, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_872(reviews, previous_rounds):
    """#872 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, -1, 0, -1, 0, 0, -1, -1, 0, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_873(reviews, previous_rounds):
    """#873 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 0, 1, 0, 0, 0, -1, -1, 0, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_874(reviews, previous_rounds):
    """#874 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, 0, -1, 0, 0, 0, -1, -1, 0, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_875(reviews, previous_rounds):
    """#875 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_876(reviews, previous_rounds):
    """#876 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, -1, 0, -1, 1, 1, 0, 0, 1, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_877(reviews, previous_rounds):
    """#877 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_878(reviews, previous_rounds):
    """#878 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, 0, -1, 0, 1, 1, 0, 0, 1, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_879(reviews, previous_rounds):
    """#879 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 1, 0, 1, -1, -1, 0, 0, -1, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_880(reviews, previous_rounds):
    """#880 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, -1, 0, -1, -1, -1, 0, 0, -1, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_881(reviews, previous_rounds):
    """#881 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 0, 1, 0, -1, -1, 0, 0, -1, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_882(reviews, previous_rounds):
    """#882 Is hotel was chosen in last round? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, 0, -1, 0, -1, -1, 0, 0, -1, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_883(reviews, previous_rounds):
    """#883 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_884(reviews, previous_rounds):
    """#884 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, -1, -1, 0, 1, 0, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_885(reviews, previous_rounds):
    """#885 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_886(reviews, previous_rounds):
    """#886 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_887(reviews, previous_rounds):
    """#887 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 1, 1, 0, -1, 0, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_888(reviews, previous_rounds):
    """#888 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_889(reviews, previous_rounds):
    """#889 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 0, 0, 0, -1, 0, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_890(reviews, previous_rounds):
    """#890 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, 0, 0, 0, -1, 0, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_891(reviews, previous_rounds):
    """#891 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_892(reviews, previous_rounds):
    """#892 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, -1, -1, 1, 0, 1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_893(reviews, previous_rounds):
    """#893 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_894(reviews, previous_rounds):
    """#894 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_895(reviews, previous_rounds):
    """#895 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 1, 0, 0, 1, 1, -1, 0, -1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_896(reviews, previous_rounds):
    """#896 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, -1, 0, 0, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_897(reviews, previous_rounds):
    """#897 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 0, 1, 1, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_898(reviews, previous_rounds):
    """#898 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, 0, -1, -1, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_899(reviews, previous_rounds):
    """#899 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_900(reviews, previous_rounds):
    """#900 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 1, 1, 0, -1, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_901(reviews, previous_rounds):
    """#901 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_902(reviews, previous_rounds):
    """#902 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 1, 1, -1, 0, 1, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_903(reviews, previous_rounds):
    """#903 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -1, 0, 1, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_904(reviews, previous_rounds):
    """#904 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, 0, 0, -1, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_905(reviews, previous_rounds):
    """#905 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, -1, -1, 1, 0, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_906(reviews, previous_rounds):
    """#906 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, 0, 0, -1, 0, 0, 0, -1, 0, -1, -1, -1, 0, -1, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_907(reviews, previous_rounds):
    """#907 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_908(reviews, previous_rounds):
    """#908 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 1, 1, 0, -1, 1, 1, 0, -1, 0, 0, 0, -1, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_909(reviews, previous_rounds):
    """#909 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_910(reviews, previous_rounds):
    """#910 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, 1, 1, -1, 0, 1, 1, -1, 0, 0, 0, -1, 0, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_911(reviews, previous_rounds):
    """#911 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, -1, -1, 0, 1, -1, -1, 0, 1, 0, 0, 0, 1, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_912(reviews, previous_rounds):
    """#912 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, 0, 0, -1, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_913(reviews, previous_rounds):
    """#913 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, -1, -1, 1, 0, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_914(reviews, previous_rounds):
    """#914 Is hotel score >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_915(reviews, previous_rounds):
    """#915 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_916(reviews, previous_rounds):
    """#916 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 1, 0, 0, 0, 1, -1, -1, 0, 1, -1, -1, 0, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_917(reviews, previous_rounds):
    """#917 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_918(reviews, previous_rounds):
    """#918 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, 0, 1, -1, -1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_919(reviews, previous_rounds):
    """#919 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, -1, 0, 0, 0, -1, 1, 1, 0, -1, 1, 1, 0, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_920(reviews, previous_rounds):
    """#920 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, -1, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_921(reviews, previous_rounds):
    """#921 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 0, -1, 1, 1, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_922(reviews, previous_rounds):
    """#922 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, 0, -1, -1, -1, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_923(reviews, previous_rounds):
    """#923 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_924(reviews, previous_rounds):
    """#924 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 1, 0, 0, 0, 1, 0, -1, -1, 1, 0, -1, -1, 1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_925(reviews, previous_rounds):
    """#925 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_926(reviews, previous_rounds):
    """#926 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, 1, 0, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_927(reviews, previous_rounds):
    """#927 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, -1, 0, 0, 0, -1, 0, 1, 1, -1, 0, 1, 1, -1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_928(reviews, previous_rounds):
    """#928 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, -1, 0, 0, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_929(reviews, previous_rounds):
    """#929 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, -1, 0, 1, 1, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_930(reviews, previous_rounds):
    """#930 Is hotel score >= 8? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, 0, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0)"""
    if reviews.mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_931(reviews, previous_rounds):
    """#931 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_932(reviews, previous_rounds):
    """#932 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, 0, 0, 0, -1, 0, 0, 1, -1, 1, 0, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_933(reviews, previous_rounds):
    """#933 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_934(reviews, previous_rounds):
    """#934 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, 0, 0, 0, -1, 0, 0, 0, -1, 1, 0, 1, -1, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_935(reviews, previous_rounds):
    """#935 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 1, -1, 0, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_936(reviews, previous_rounds):
    """#936 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 0, -1, 0, 0, 0, -1, 0, 0, -1, -1, -1, 0, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_937(reviews, previous_rounds):
    """#937 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 0, 0, 0, 1, 0, 0, 0, 1, -1, 0, -1, 1, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_938(reviews, previous_rounds):
    """#938 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, 0, 0, 0, -1, 0, 0, 0, -1, -1, 0, -1, -1, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_939(reviews, previous_rounds):
    """#939 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_940(reviews, previous_rounds):
    """#940 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, 1, -1, 1, 0, 1, -1, 1, 0, 0, -1, 0, 0, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_941(reviews, previous_rounds):
    """#941 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_942(reviews, previous_rounds):
    """#942 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, 1, 0, 1, -1, 1, 0, 1, -1, 0, 0, 0, -1, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_943(reviews, previous_rounds):
    """#943 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play max. else, play mean]
    (0, -1, 1, -1, 0, -1, 1, -1, 0, 0, 1, 0, 0, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_944(reviews, previous_rounds):
    """#944 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play min. else, play mean]
    (0, -1, -1, -1, 0, -1, -1, -1, 0, 0, -1, 0, 0, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_945(reviews, previous_rounds):
    """#945 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play max]
    (1, -1, 0, -1, 1, -1, 0, -1, 1, 0, 0, 0, 1, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_946(reviews, previous_rounds):
    """#946 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score >= 8? if True, play mean. else, play min]
    (-1, -1, 0, -1, -1, -1, 0, -1, -1, 0, 0, 0, -1, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_947(reviews, previous_rounds):
    """#947 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_948(reviews, previous_rounds):
    """#948 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 1, 0, 0, 0, 1, -1, 0, -1, 1, -1, 0, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_949(reviews, previous_rounds):
    """#949 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_950(reviews, previous_rounds):
    """#950 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, 0, -1, 1, -1, 0, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_951(reviews, previous_rounds):
    """#951 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, -1, 0, 0, 0, -1, 1, 0, 1, -1, 1, 0, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_952(reviews, previous_rounds):
    """#952 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, -1, 0, 0, 0, -1, -1, 0, -1, -1, -1, 0, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_953(reviews, previous_rounds):
    """#953 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 0, 1, -1, 1, 0, 1, -1, 0, 0, 0, -1, 0, 0, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_954(reviews, previous_rounds):
    """#954 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, 0, -1, -1, -1, 0, -1, -1, 0, 0, 0, -1, 0, 0, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_955(reviews, previous_rounds):
    """#955 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_956(reviews, previous_rounds):
    """#956 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 1, 0, 0, 0, 1, 0, 0, -1, 1, -1, 0, -1, 1, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_957(reviews, previous_rounds):
    """#957 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_958(reviews, previous_rounds):
    """#958 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, 1, -1, 0, -1, 1, -1, 0, 0, 1, 0, 0, 0, 1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_959(reviews, previous_rounds):
    """#959 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, -1, 0, 0, 0, -1, 0, 0, 1, -1, 1, 0, 1, -1, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_960(reviews, previous_rounds):
    """#960 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, -1, 0, 0, 0, -1, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_961(reviews, previous_rounds):
    """#961 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, -1, 1, 0, 1, -1, 1, 0, 0, -1, 0, 0, 0, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_962(reviews, previous_rounds):
    """#962 Is hotel score in the last round >= 8? if True, [Is hotel score >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, 0, -1, -1, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if reviews.mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_963(reviews, previous_rounds):
    """#963 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_964(reviews, previous_rounds):
    """#964 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_965(reviews, previous_rounds):
    """#965 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_966(reviews, previous_rounds):
    """#966 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_967(reviews, previous_rounds):
    """#967 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_968(reviews, previous_rounds):
    """#968 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_969(reviews, previous_rounds):
    """#969 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_970(reviews, previous_rounds):
    """#970 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_971(reviews, previous_rounds):
    """#971 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_972(reviews, previous_rounds):
    """#972 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_973(reviews, previous_rounds):
    """#973 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_974(reviews, previous_rounds):
    """#974 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_975(reviews, previous_rounds):
    """#975 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_976(reviews, previous_rounds):
    """#976 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_977(reviews, previous_rounds):
    """#977 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_978(reviews, previous_rounds):
    """#978 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_979(reviews, previous_rounds):
    """#979 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_980(reviews, previous_rounds):
    """#980 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_981(reviews, previous_rounds):
    """#981 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_982(reviews, previous_rounds):
    """#982 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_983(reviews, previous_rounds):
    """#983 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_984(reviews, previous_rounds):
    """#984 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_985(reviews, previous_rounds):
    """#985 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_986(reviews, previous_rounds):
    """#986 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_987(reviews, previous_rounds):
    """#987 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 1, 0, 1, -1, -1, -1, -1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_988(reviews, previous_rounds):
    """#988 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_989(reviews, previous_rounds):
    """#989 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 0, 1, 0, -1, -1, -1, -1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_990(reviews, previous_rounds):
    """#990 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_991(reviews, previous_rounds):
    """#991 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_992(reviews, previous_rounds):
    """#992 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_993(reviews, previous_rounds):
    """#993 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_994(reviews, previous_rounds):
    """#994 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, -1, 0, -1, 1, 1, 1, 1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_995(reviews, previous_rounds):
    """#995 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_996(reviews, previous_rounds):
    """#996 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, 0, -1, 0, 1, 1, 1, 1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_997(reviews, previous_rounds):
    """#997 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_998(reviews, previous_rounds):
    """#998 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_999(reviews, previous_rounds):
    """#999 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_1000(reviews, previous_rounds):
    """#1000 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_1001(reviews, previous_rounds):
    """#1001 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_1002(reviews, previous_rounds):
    """#1002 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_1003(reviews, previous_rounds):
    """#1003 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1004(reviews, previous_rounds):
    """#1004 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1005(reviews, previous_rounds):
    """#1005 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, 0, 0, -1, 1, -1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1006(reviews, previous_rounds):
    """#1006 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, 0, 0, -1, 1, -1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1007(reviews, previous_rounds):
    """#1007 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_1008(reviews, previous_rounds):
    """#1008 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, -1, -1, 0, 1, 0, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_1009(reviews, previous_rounds):
    """#1009 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1010(reviews, previous_rounds):
    """#1010 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1011(reviews, previous_rounds):
    """#1011 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1012(reviews, previous_rounds):
    """#1012 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1013(reviews, previous_rounds):
    """#1013 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, 1, 1, 0, -1, 0, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_1014(reviews, previous_rounds):
    """#1014 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_1015(reviews, previous_rounds):
    """#1015 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_1016(reviews, previous_rounds):
    """#1016 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, -1, -1, 1, 0, 1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_1017(reviews, previous_rounds):
    """#1017 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 1, 1, 1, -1, 0, -1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_1018(reviews, previous_rounds):
    """#1018 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_1019(reviews, previous_rounds):
    """#1019 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, -1, -1, -1, -1, 0, 1, 0, 1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1020(reviews, previous_rounds):
    """#1020 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1021(reviews, previous_rounds):
    """#1021 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, -1, -1, -1, -1, 1, 0, 1, 0, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1022(reviews, previous_rounds):
    """#1022 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1023(reviews, previous_rounds):
    """#1023 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, 0, 0, 0, 0, -1, 1, -1, 1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_1024(reviews, previous_rounds):
    """#1024 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_1025(reviews, previous_rounds):
    """#1025 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1026(reviews, previous_rounds):
    """#1026 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 1, 1, 1, 1, 0, -1, 0, -1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1027(reviews, previous_rounds):
    """#1027 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1028(reviews, previous_rounds):
    """#1028 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, 1, 1, 1, 1, -1, 0, -1, 0, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1029(reviews, previous_rounds):
    """#1029 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, 0, 0, 0, 0, -1, 1, -1, 1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_1030(reviews, previous_rounds):
    """#1030 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_1031(reviews, previous_rounds):
    """#1031 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_1032(reviews, previous_rounds):
    """#1032 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_1033(reviews, previous_rounds):
    """#1033 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()

def strategy_1034(reviews, previous_rounds):
    """#1034 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play max]
    (1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()

def strategy_1035(reviews, previous_rounds):
    """#1035 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 0, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1036(reviews, previous_rounds):
    """#1036 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1037(reviews, previous_rounds):
    """#1037 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 1, -1, 1, -1, 1, 0, 0, 0, 0, -1, 1, -1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1038(reviews, previous_rounds):
    """#1038 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, -1, -1, 1, -1, 1, 0, 0, 0, 0, -1, 1, -1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1039(reviews, previous_rounds):
    """#1039 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, -1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_1040(reviews, previous_rounds):
    """#1040 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 1, 0, 1, 0, 1, -1, -1, -1, -1, 0, 1, 0, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_1041(reviews, previous_rounds):
    """#1041 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1042(reviews, previous_rounds):
    """#1042 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 0, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1043(reviews, previous_rounds):
    """#1043 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 1, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1044(reviews, previous_rounds):
    """#1044 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1045(reviews, previous_rounds):
    """#1045 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, -1, 0, -1, 0, -1, 1, 1, 1, 1, 0, -1, 0, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_1046(reviews, previous_rounds):
    """#1046 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 1, 0, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_1047(reviews, previous_rounds):
    """#1047 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_1048(reviews, previous_rounds):
    """#1048 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 1, 1, 0, 1, 0, -1, -1, -1, -1, 1, 0, 1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_1049(reviews, previous_rounds):
    """#1049 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, -1, -1, 0, -1, 0, 1, 1, 1, 1, -1, 0, -1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_1050(reviews, previous_rounds):
    """#1050 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 1, -1, 0, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_1051(reviews, previous_rounds):
    """#1051 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, -1, 0, -1, 1, -1, 1, -1, 0, 1, 0, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1052(reviews, previous_rounds):
    """#1052 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, -1, 0, -1, -1, -1, -1, -1, 0, 1, 0, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1053(reviews, previous_rounds):
    """#1053 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, -1, 1, -1, 0, -1, 0, -1, 1, 1, 1, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1054(reviews, previous_rounds):
    """#1054 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, -1, 0, -1, -1, 1, -1, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1055(reviews, previous_rounds):
    """#1055 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, 0, -1, 0, 1, 0, 1, 0, -1, 1, -1, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_1056(reviews, previous_rounds):
    """#1056 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 0, 1, 0, -1, 0, -1, 0, 1, 1, 1, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_1057(reviews, previous_rounds):
    """#1057 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 1, 0, 1, 1, 1, 1, 1, 0, -1, 0, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1058(reviews, previous_rounds):
    """#1058 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 1, 0, 1, -1, 1, -1, 1, 0, -1, 0, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1059(reviews, previous_rounds):
    """#1059 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 1, 0, 1, 1, -1, 1, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1060(reviews, previous_rounds):
    """#1060 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, 1, -1, 1, 0, 1, 0, 1, -1, -1, -1, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1061(reviews, previous_rounds):
    """#1061 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, 0, -1, 0, 1, 0, 1, 0, -1, -1, -1, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_1062(reviews, previous_rounds):
    """#1062 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 0, 1, 0, -1, 0, -1, 0, 1, -1, 1, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_1063(reviews, previous_rounds):
    """#1063 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, 1, -1, 1, 1, 1, 1, 1, -1, 0, -1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_1064(reviews, previous_rounds):
    """#1064 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, 1, 1, 1, -1, 1, -1, 1, 1, 0, 1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_1065(reviews, previous_rounds):
    """#1065 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play max. else, play min]
    (-1, -1, -1, -1, 1, -1, 1, -1, -1, 0, -1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()

def strategy_1066(reviews, previous_rounds):
    """#1066 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play min. else, play max]
    (1, -1, 1, -1, -1, -1, -1, -1, 1, 0, 1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()

def strategy_1067(reviews, previous_rounds):
    """#1067 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, -1, 0, -1, 0, 1, 0, 1, 1, -1, 1, -1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1068(reviews, previous_rounds):
    """#1068 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, 1, 0, 1, -1, -1, -1, -1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1069(reviews, previous_rounds):
    """#1069 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, -1, 1, -1, 1, 1, 1, 1, 0, -1, 0, -1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1070(reviews, previous_rounds):
    """#1070 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, -1, -1, 1, -1, 1, 0, -1, 0, -1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1071(reviews, previous_rounds):
    """#1071 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, 0, -1, 0, -1, 1, -1, 1, 1, 0, 1, 0, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_1072(reviews, previous_rounds):
    """#1072 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 0, 1, 0, 1, 1, 1, 1, -1, 0, -1, 0, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_1073(reviews, previous_rounds):
    """#1073 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, -1, 0, -1, 1, 1, 1, 1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1074(reviews, previous_rounds):
    """#1074 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 1, 0, 1, 0, -1, 0, -1, -1, 1, -1, 1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1075(reviews, previous_rounds):
    """#1075 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 1, 1, -1, 1, -1, 0, 1, 0, 1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1076(reviews, previous_rounds):
    """#1076 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, 1, -1, 1, -1, -1, -1, -1, 0, 1, 0, 1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1077(reviews, previous_rounds):
    """#1077 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, 0, -1, 0, -1, -1, -1, -1, 1, 0, 1, 0, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_1078(reviews, previous_rounds):
    """#1078 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 0, 1, 0, 1, -1, 1, -1, -1, 0, -1, 0, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_1079(reviews, previous_rounds):
    """#1079 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, 1, -1, 1, -1, 0, -1, 0, 1, 1, 1, 1, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_1080(reviews, previous_rounds):
    """#1080 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, 1, 1, 1, 1, 0, 1, 0, -1, 1, -1, 1, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_1081(reviews, previous_rounds):
    """#1081 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play min]
    (-1, -1, -1, -1, -1, 0, -1, 0, 1, -1, 1, -1, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return reviews.min()

def strategy_1082(reviews, previous_rounds):
    """#1082 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play max]
    (1, -1, 1, -1, 1, 0, 1, 0, -1, -1, -1, -1, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return reviews.max()

def strategy_1083(reviews, previous_rounds):
    """#1083 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1084(reviews, previous_rounds):
    """#1084 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1085(reviews, previous_rounds):
    """#1085 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1086(reviews, previous_rounds):
    """#1086 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1087(reviews, previous_rounds):
    """#1087 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1088(reviews, previous_rounds):
    """#1088 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1089(reviews, previous_rounds):
    """#1089 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1090(reviews, previous_rounds):
    """#1090 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1091(reviews, previous_rounds):
    """#1091 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1092(reviews, previous_rounds):
    """#1092 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, -1, 0, -1, 1, 1, 1, 1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1093(reviews, previous_rounds):
    """#1093 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1094(reviews, previous_rounds):
    """#1094 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, 0, -1, 0, 1, 1, 1, 1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1095(reviews, previous_rounds):
    """#1095 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 1, 0, 1, -1, -1, -1, -1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1096(reviews, previous_rounds):
    """#1096 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1097(reviews, previous_rounds):
    """#1097 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 0, 1, 0, -1, -1, -1, -1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1098(reviews, previous_rounds):
    """#1098 Is hotel was chosen in last round? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1099(reviews, previous_rounds):
    """#1099 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1100(reviews, previous_rounds):
    """#1100 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, -1, -1, 0, 1, 0, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1101(reviews, previous_rounds):
    """#1101 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1102(reviews, previous_rounds):
    """#1102 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1103(reviews, previous_rounds):
    """#1103 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, 1, 1, 0, -1, 0, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1104(reviews, previous_rounds):
    """#1104 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1105(reviews, previous_rounds):
    """#1105 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1106(reviews, previous_rounds):
    """#1106 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1107(reviews, previous_rounds):
    """#1107 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1108(reviews, previous_rounds):
    """#1108 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, -1, -1, 1, 0, 1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1109(reviews, previous_rounds):
    """#1109 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1110(reviews, previous_rounds):
    """#1110 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1111(reviews, previous_rounds):
    """#1111 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 1, 1, 1, -1, 0, -1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1112(reviews, previous_rounds):
    """#1112 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1113(reviews, previous_rounds):
    """#1113 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1114(reviews, previous_rounds):
    """#1114 Is hotel was chosen in last round? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1115(reviews, previous_rounds):
    """#1115 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1116(reviews, previous_rounds):
    """#1116 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1117(reviews, previous_rounds):
    """#1117 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1118(reviews, previous_rounds):
    """#1118 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1, 0, 1, 1, 1, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1119(reviews, previous_rounds):
    """#1119 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1120(reviews, previous_rounds):
    """#1120 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1121(reviews, previous_rounds):
    """#1121 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1122(reviews, previous_rounds):
    """#1122 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1, 0, -1, -1, -1, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1123(reviews, previous_rounds):
    """#1123 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1124(reviews, previous_rounds):
    """#1124 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, 1, 1, 1, 1, 0, -1, 0, -1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1125(reviews, previous_rounds):
    """#1125 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1126(reviews, previous_rounds):
    """#1126 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, 1, 1, 1, 1, -1, 0, -1, 0, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1127(reviews, previous_rounds):
    """#1127 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play max. else, play mean]
    (0, 1, 0, 1, -1, -1, -1, -1, 0, 1, 0, 1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1128(reviews, previous_rounds):
    """#1128 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play min. else, play mean]
    (0, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1129(reviews, previous_rounds):
    """#1129 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play max]
    (1, 0, 1, 0, -1, -1, -1, -1, 1, 0, 1, 0, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1130(reviews, previous_rounds):
    """#1130 Is user earn more than bot? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is hotel score in the last round >= 8? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0, 0, 0, 0, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1131(reviews, previous_rounds):
    """#1131 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1132(reviews, previous_rounds):
    """#1132 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 0, 0, 1, 0, 1, -1, -1, -1, -1, 0, 1, 0, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1133(reviews, previous_rounds):
    """#1133 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1134(reviews, previous_rounds):
    """#1134 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1135(reviews, previous_rounds):
    """#1135 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 0, 0, -1, 0, -1, 1, 1, 1, 1, 0, -1, 0, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1136(reviews, previous_rounds):
    """#1136 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 0, 0, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1137(reviews, previous_rounds):
    """#1137 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1138(reviews, previous_rounds):
    """#1138 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1139(reviews, previous_rounds):
    """#1139 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1140(reviews, previous_rounds):
    """#1140 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 0, 1, 0, 1, 0, -1, -1, -1, -1, 1, 0, 1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1141(reviews, previous_rounds):
    """#1141 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1142(reviews, previous_rounds):
    """#1142 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, -1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1143(reviews, previous_rounds):
    """#1143 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 0, -1, 0, -1, 0, 1, 1, 1, 1, -1, 0, -1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1144(reviews, previous_rounds):
    """#1144 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1145(reviews, previous_rounds):
    """#1145 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 1, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1146(reviews, previous_rounds):
    """#1146 Is user earn more than bot? if True, [Is hotel score in the last round >= 8? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, -1, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1, 0)"""
    if user_score(previous_rounds) >= bot_score(previous_rounds):
        if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1147(reviews, previous_rounds):
    """#1147 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1148(reviews, previous_rounds):
    """#1148 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, 0, -1, 0, 0, 1, 0, 1, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1149(reviews, previous_rounds):
    """#1149 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1150(reviews, previous_rounds):
    """#1150 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play max. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, 0, -1, 0, 0, 0, 0, 0, -1, 1, -1, 1, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1151(reviews, previous_rounds):
    """#1151 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 0, 0, 0, 1, 0, 1, 0, 0, -1, 0, -1, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1152(reviews, previous_rounds):
    """#1152 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 0, 0, 0, -1, 0, -1, 0, 0, -1, 0, -1, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1153(reviews, previous_rounds):
    """#1153 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 0, 1, 0, 0, 0, 0, 0, 1, -1, 1, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1154(reviews, previous_rounds):
    """#1154 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play min. else, play mean]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, 0, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1155(reviews, previous_rounds):
    """#1155 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1156(reviews, previous_rounds):
    """#1156 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, 1, 0, 1, -1, 1, -1, 1, 0, 0, 0, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1157(reviews, previous_rounds):
    """#1157 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1158(reviews, previous_rounds):
    """#1158 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play max]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, 1, -1, 1, 0, 1, 0, 1, -1, 0, -1, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1159(reviews, previous_rounds):
    """#1159 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play max. else, play mean]
    (0, -1, 0, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1160(reviews, previous_rounds):
    """#1160 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play min. else, play mean]
    (0, -1, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1161(reviews, previous_rounds):
    """#1161 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play max]
    (1, -1, 1, -1, 0, -1, 0, -1, 1, 0, 1, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1162(reviews, previous_rounds):
    """#1162 Is hotel score in the last round >= 8? if True, [Is hotel was chosen in last round? if True, play mean. else, play min]. else, [Is user earn more than bot? if True, play mean. else, play min]
    (-1, -1, -1, -1, 0, -1, 0, -1, -1, 0, -1, 0, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1163(reviews, previous_rounds):
    """#1163 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1164(reviews, previous_rounds):
    """#1164 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 0, 0, 1, 0, 1, -1, 0, -1, 0, -1, 1, -1, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1165(reviews, previous_rounds):
    """#1165 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1166(reviews, previous_rounds):
    """#1166 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play max. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, 1, -1, 1, 0, 0, 0, 0, 0, 1, 0, 1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.max()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1167(reviews, previous_rounds):
    """#1167 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 0, 0, 0, 0, -1, 0, -1, 1, 0, 1, 0, 1, -1, 1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1168(reviews, previous_rounds):
    """#1168 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 0, 0, 0, 0, -1, 0, -1, -1, 0, -1, 0, -1, -1, -1, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1169(reviews, previous_rounds):
    """#1169 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 0, 1, 0, 1, -1, 1, -1, 0, 0, 0, 0, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1170(reviews, previous_rounds):
    """#1170 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play min. else, play mean]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, 0, -1, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, -1, 0, -1)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return reviews.min()
        else:
            return play_mean(reviews)
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1171(reviews, previous_rounds):
    """#1171 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1172(reviews, previous_rounds):
    """#1172 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, 1, 0, 1, 0, 0, 0, 0, -1, 1, -1, 1, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1173(reviews, previous_rounds):
    """#1173 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1174(reviews, previous_rounds):
    """#1174 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play max]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, 1, -1, 1, -1, 0, -1, 0, 0, 1, 0, 1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.max()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

def strategy_1175(reviews, previous_rounds):
    """#1175 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play max. else, play mean]
    (0, -1, 0, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, 0, 1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.max()
        else:
            return play_mean(reviews)

def strategy_1176(reviews, previous_rounds):
    """#1176 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play min. else, play mean]
    (0, -1, 0, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, -1, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return reviews.min()
        else:
            return play_mean(reviews)

def strategy_1177(reviews, previous_rounds):
    """#1177 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play max]
    (1, -1, 1, -1, 1, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.max()

def strategy_1178(reviews, previous_rounds):
    """#1178 Is hotel score in the last round >= 8? if True, [Is user earn more than bot? if True, play mean. else, play min]. else, [Is hotel was chosen in last round? if True, play mean. else, play min]
    (-1, -1, -1, -1, -1, 0, -1, 0, 0, -1, 0, -1, 0, 0, 0, 0)"""
    if len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8:
        if user_score(previous_rounds) >= bot_score(previous_rounds):
            return play_mean(reviews)
        else:
            return reviews.min()
    else:
        if len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True:
            return play_mean(reviews)
        else:
            return reviews.min()

