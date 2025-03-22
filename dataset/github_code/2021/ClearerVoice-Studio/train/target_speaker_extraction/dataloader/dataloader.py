import numpy as np

def dataloader_wrapper(args, partition):
    if args.network_reference.cue == 'lip':
        from .dataset_lip import get_dataloader_lip as get_dataloader
    elif args.network_reference.cue == 'gesture':
        from .dataset_gesture import get_dataloader_gesture as get_dataloader
    elif args.network_reference.cue == 'eeg':
        from .dataset_eeg import get_dataloader_eeg as get_dataloader
    elif args.network_reference.cue == 'speech':
        from .dataset_speech import get_dataloader_speech as get_dataloader
    else:
        raise NameError('Wrong reference for dataloader selection')
    return get_dataloader(args, partition)
    






















