import copy
from omegaconf import OmegaConf

import torch
import torch.nn as nn

from dsf.utils.model_utils import Model
from dsf.utils.engine import train_model

from tasks.load_dataset import load_dataset


config_paths = [
    # './configs/ptb_word/gru_1L256.yml',
    # './configs/ptb_word/gru_1L1024.yml',
    # './configs/ptb_word/gru_3L256.yml',
    # './configs/ptb_word/gru_3L1024.yml',

    # './configs/wikitext103_word/gru_1L512.yml',
    # './configs/wikitext103_word/gru_3L512.yml',
    # './configs/wikitext103_word/gru_6L512.yml',
    './configs/wikitext103_word/gru_12L512.yml',

    # './configs/wikitext103_word/gru_3L512_seqlen1024.yml',

    # './configs/wikitext103_word/gru_3L128.yml',
    # './configs/wikitext103_word/gru_3L256.yml',
    # './configs/wikitext103_word/gru_3L1024.yml',

    # './configs/wikitext103_word/rnn_3L512.yml',
    # './configs/wikitext103_word/lstm_3L512.yml',

]



def main(config_path):
    """ Run experiments for the given configuration file. """

    # Define available state transition methods 
    state_transitions = [
        'BPTT',
        'FT_BPTT',
        'diagonal'
        ]
    measures = []

    # prepare training
    base_config = OmegaConf.load(config_path)


    # Load the PTB dataset
    train_loader, val_loader, task_info = load_dataset(base_config)
    for k, v in task_info.items():
        base_config.dataset[k] = v


    # Generate configurations
    configs = []

    # DFARNN
    for method in state_transitions:

        if method == "SSM":
            # Generate one config per measure
            for measure in measures:
                config = copy.deepcopy(base_config)
                config.model.use_dfa = True
                config.model.state_transition = method
                config.model.measure = measure
                configs.append(config)
        elif method in ["FT_BPTT", "diagonal"]:
            # Generate config without measure for non-SSM methods
            config = copy.deepcopy(base_config)
            config.model.use_dfa = True
            config.model.state_transition = method
            config.model.measure = None
            configs.append(config)
        elif method == "BPTT":
            # Standard backpropagation
            config = copy.deepcopy(base_config)
            config.model.use_dfa = False
            config.model.state_transition = method
            config.model.measure = None
            configs.append(config)


    for config in configs:
        print(f"Training on {config.model.state_transition}, with {config.model.measure} measure initialization")

        
        model = Model(config)

        print(model)
        print("Model has ", sum(p.numel() for p in model.parameters()), " parameters")

        if config.dataset.task_type == 'language_modeling':
            criterion_1 = nn.CrossEntropyLoss(ignore_index=config.dataset.pad_token_id, reduction='sum')
            criterion = lambda preds, Y: criterion_1(preds.view(-1, config.dataset.vocab_size), Y.view(-1))
        elif config.dataset.task_type == 'sequence_classification':
            criterion = nn.CrossEntropyLoss(reduction='sum')
        else:
            raise NotImplementedError
        train_model(model=model, config=config, train_loader=train_loader, val_loader=val_loader, criterion=criterion)


if __name__ == "__main__":
    for config_path in config_paths:
        main(config_path)
