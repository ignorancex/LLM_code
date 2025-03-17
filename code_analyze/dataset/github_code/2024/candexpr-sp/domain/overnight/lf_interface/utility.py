
def from_12_to_24(hour, am_or_pm):
    # if not (1 <= hour <= 12):
    #     breakpoint()
    assert 1 <= hour <= 12
    assert am_or_pm in ['am', 'pm']

    if am_or_pm == 'am':
        offset = 0
    else:
        assert am_or_pm == 'pm'
        offset = 12

    if hour == 12:
        return 0 + offset
    else:
        return hour + offset


def from_24_to_12(hour):
    assert 0 <= hour <= 23

    if hour < 12:
        am_or_pm = 'am'
        offset = 0
    else:
        am_or_pm = 'pm'
        offset = -12

    if hour == 0 or hour == 12:
        new_hour = 12
    else:
        new_hour = hour + offset

    return new_hour, am_or_pm


_MONTHS = (
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december'
)
_MONTH_TO_NUM = dict([month, idx + 1] for idx, month in enumerate(_MONTHS))


def num_to_month(num):
    return _MONTHS[num - 1]


def month_to_num(month):
    return _MONTH_TO_NUM[month]
