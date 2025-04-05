import numpy as np
import torch

def duration_event_common_format(durations, events, risk = 1, risk_exclude = False):
    if risk_exclude:
        return np.array([(True if e != risk else False, max(d, 0) ) for e, d in zip(events, durations)], dtype = [('e', bool), ('t', float)])
    else:
        return np.array([(True if e == risk else False, max(d, 0) ) for e, d in zip(events, durations)], dtype = [('e', bool), ('t', float)])

def pad_col(input, where='end'):
    """Addes a column of `val` at the start of end of `input`."""
    if len(input.shape) != 2:

        pad = torch.zeros_like(input[:, :,:1])
    else:
        pad = torch.zeros_like(input[:, :1])

    if where == 'end':
        return torch.cat([input, pad], dim=-1)
    elif where == 'start':
        return torch.cat([pad, input], dim=-1)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")

def index_at_time(time, time_points, eps = 1e-7):
    """Returns the index of `time` in `time_points`."""
    maxtime = time.max(axis=-1)
    assert (maxtime >= time).all(), "time should be less than maxtime"
    return (time_points >= time-eps).argmax(axis=-1)

if __name__ == '__main__':
    print(index_at_time(1, np.array([[0,1,2,3,4,5]])))