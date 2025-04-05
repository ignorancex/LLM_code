#!/usr/bin/python3.7
"""Prints interesting scalars from tf eval log files.

Usage:
chmod ugo+x ./tensorflow_gan/examples/print_tf_log.py
gsutil ls gs://ilya-test-eu/experiments/imagenet128_expt/eval_eval/* | ./tensorflow_gan/examples/print_tf_log.py
"""

import fileinput

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_tensorflow_log(path, scores={}, score_names=[]):
    """Print every event that has 'eval' in  csv format to the screen.
    """

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 0,
        'images': 0,
        'scalars': 1000,
        'histograms': 0
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    tags = event_acc.Tags()
    
    event_tags = tags['scalars'] if 'scalars' in tags else []
    event_tags = [et for et in event_tags if 'eval' in et]
    
    for event_tag in event_tags:
        score_name = event_tag.split('/')[-1]
        if score_name not in score_names:
            score_names.append(score_name)
            
        events = event_acc.Scalars(event_tag)
        for event in events:
            if event.step not in scores:
                scores[event.step] = {}
            scores[event.step][score_name] = event.value

    return scores, score_names

def print_tensorflow_log(scores, score_names):

    steps = sorted(scores.keys())

    print(','.join(['step'] + score_names))
    for step in steps:
        row = [( '{:.4f}'.format(scores[step][name]) if name in scores[step] else '') for name in score_names]
        print(','.join(['{:06d}'.format(step)] + row))

if __name__ == '__main__':
    scores = {}
    score_names = []
    with fileinput.input() as f_input:
        for i, line in enumerate(f_input):
            if i == 0:
                print('Skipping first line: {}',format(line.strip()))
            else:
                print('Getting scores for: {}',format(line.strip()))
                scores, score_names = get_tensorflow_log(line.strip(), scores, score_names)
    print('Printing scores:')
    print_tensorflow_log(scores, score_names)
