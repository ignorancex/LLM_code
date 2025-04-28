import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd
import seaborn as sns
from .dataset_helpers import c2l_BNCI2014008, l2c_BNCI2014008, relabel_events_BNCI2014008
from mne.decoding import Vectorizer
from moabb.datasets import BNCI2014008
from moabb.paradigms import P300
dataset = BNCI2014008()
# mne version has to be 1.4 (pip install mne==1.4)
def relabel_events_BNCI(ev, n_stim, n_repetition):
    """Reformat labels from BNCI dataset for cumulative evaluation

    In BNCI, events are grouped by pair: target/nontarget labels and
    position labels (column 1 to 6 or row 1 to 6). Those event pair share the
    same time sample and are not ordered.

    Original event label are 1 for non target, 2 for target, 3 to 8 for column
    1 to 6 and 9 to 14 for line 1 to 6

    Output events are encoded with 4+ digits: thousands encode the trial number,
    tens/hundreds indicate the position and unit indicate target status:
    1010 is trial 1/col 1/non target,
    2041 is trial 2/col 4/target,
    4080 is trial 4/line 3/non target,
    35111 is trial 35/line 6/target
    """
    i, t_len, n_ev = 0, n_stim * n_repetition * 2, len(ev)

    new_ev = []
    while i < n_ev:
        tgt = pos = None

        if ev[i, 2] == 1:
            # non target event is first
            tgt, t = 0, ev[i, 0]
        elif ev[i, 2] == 2:
            # target event is first
            tgt, t = 1, ev[i, 0]
        else:
            # position event is first
            pos, t = ev[i, 2] - 2, ev[i, 0]

        i += 1
        trial = (i // t_len) + 1
        if t != ev[i, 0]:
            raise ValueError("event time differs within pair")
        if ev[i, 2] == 1:
            tgt = 0
        elif ev[i, 2] == 2:
            tgt = 1
        else:
            pos = ev[i, 2] - 2
        new_ev.append([t, 0, trial * 1000 + pos * 10 + tgt])
        i += 1
    new_ev = np.array(new_ev)

    event_id = {}
    for trial_idx in range(trial):
        tc = (trial_idx + 1) * 1000
        for pos_idx in range(12):
            pc = (pos_idx + 1) * 10
            if pos_idx < 6:
                event_id[f"trial{trial_idx + 1}/col{pos_idx + 1}/nontarget"] = tc + pc
                event_id[f"trial{trial_idx + 1}/col{pos_idx + 1}/target"] = tc + pc + 1
            else:
                event_id[f"trial{trial_idx + 1}/row{pos_idx % 6 + 1}/nontarget"] = (
                    tc + pc
                )
                event_id[f"trial{trial_idx + 1}/row{pos_idx % 6 + 1}/target"] = (
                    tc + pc + 1
                )

    return new_ev, event_id

def get_epochs(subj, tmin, tmax, baseline, fmin, fmax):
    """get epoch from BNCI 2014-008 subject"""
    data = dataset.get_data(subjects=[subj])
    # raw = data[subj]["session_0"]["run_0"]
    raw = data[subj]['0']['0']
    raw_ev = mne.find_events(raw)
    ev, event_id = relabel_events_BNCI2014008(raw_ev)

    raw = raw.filter(fmin, fmax, method="iir")
    picks = mne.pick_types(raw.info, eeg=True, stim=False)
    ep = mne.Epochs(
        raw,
        ev,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        proj=False,
        baseline=baseline,
        preload=True,
        verbose=False,
        picks=picks,
        on_missing="ignore",
    )
    return ep

def process_trial(trial, sfreq = None):
    eegmat = trial.get_data() if sfreq is None else trial.resample(sfreq).get_data()
    n_flash, n_channel, n_time = eegmat.shape
    # process flash type and code
    flash_info = trial.events[:, 2].astype(str)
    flash_code = np.array([x[-3:-1] for x in flash_info]).astype(int)
    flash_type = np.array([x[-1] for x in flash_info]).astype(int)
    # order EEG mat
    eegmat = np.take_along_axis(
        eegmat.reshape([-1, 12, n_channel, n_time]),
        flash_code.reshape([-1, 12]).argsort(axis=1)[:, :, None, None],
        axis=1
    ).reshape([-1, 2, 6, n_channel, n_time])
    # order flash type
    y = np.zeros(eegmat.shape[:3])
    target_code = np.unique(flash_code[flash_type==1])
    assert len(target_code) == 2
    y[:, 0, target_code[0]-1] = 1
    y[:, 1, target_code[1]-7] = 1
    return eegmat, y

def get_processed_data(subject_id, 
                       tmin=-0.3, tmax=1.0, baseline = (-0.3, 0.0), fmin = 0.1, fmax = 24,
                       sfreq = None):
    ep = get_epochs(subject_id, tmin, tmax, baseline, fmin, fmax)
    n_trials = len(set([t.split("/")[0] for t in list(ep.event_id.keys())]))
    processed = [process_trial(ep[f"trial{trial_num}"], sfreq) for trial_num in range(1, n_trials+1)]
    eegmat = np.stack([x[0] for x in processed])
    y = np.stack([x[1] for x in processed])
    return eegmat, y