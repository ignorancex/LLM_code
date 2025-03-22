import os

import torch
from torch.utils.data import TensorDataset
from .utils import (compute_metrics, convert_examples_to_features, output_modes,
                   processors)


def build_CodeSearchNetDataset(cfg, data_dir, tokenizer, task='codesearch', ttype='train'):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    if ttype == 'train':
        file_name = cfg['train_file'].split('.')[0]
    elif ttype == 'dev':
        file_name = cfg['dev_file'].split('.')[0]
    elif ttype == 'test':
        file_name = cfg['test_file'].split('.')[0]
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        ttype,
        file_name,
        'MSTAGI',
        str(cfg['max_seq_length']),
        str(task)))

    try:
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if ttype == 'test':
            examples, instances = processor.get_test_examples(data_dir, cfg['test_file'])
    except:
        print("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if ttype == 'train':
            examples = processor.get_train_examples(data_dir, cfg['train_file'])
        elif ttype == 'dev':
            examples = processor.get_dev_examples(data_dir, cfg['dev_file'])
        elif ttype == 'test':
            examples, instances = processor.get_test_examples(data_dir, cfg['test_file'])

        features = convert_examples_to_features(examples, label_list, cfg['max_seq_length'], tokenizer, output_mode,
                                                cls_token_at_end=False,
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                cls_token_segment_id=1,
                                                pad_on_left=False,
                                                # pad on the left for xlnet
                                                pad_token_segment_id=0)
        print("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if (ttype == 'test'):
        return dataset, instances
    else:
        return dataset
    