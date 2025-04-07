import os
import clip
import pickle
import torch
import argparse
import numpy as np
from torchvision import datasets

# utility
from utils.directory import load_config
from trees.tree_utils import load_tree, load_hie_distance
# textual representation / image prompts
from models.textual_representation import zeroshot_classifier, class_textual_representations
from models.textual_representation import TEMPLATES, merge_template_names
from models.textual_representation import textual_to_raw_embedding, merging_textual_representations
# inference
from models.flat_inference import flat_classification
from models.vcd_inference import vcd_classification
from models.hie_inference import hie_classification
from models.clip_utils import custom_clip_transform
# eval
from models.evaluation import evaluation



DATASETS = [
    "food-101", "ucf-101", "cub-200", "sun-324", "imagenet",
]


# save predictions
PRED_HOME_DIR = './results'


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-paths-config", type=str, default="./data_paths.yml", help="Path to data paths yaml file")
    parser.add_argument("--dataset", type=str, default="imagenet", choices=DATASETS)
    parser.add_argument("--prompt", type=str, default="clip", choices=TEMPLATES)
    parser.add_argument("--merging-prompts", nargs='+', default=[], help='input a list of to be merged language prompts')
    parser.add_argument("--inference", type=str, default='flat', choices=['flat', 'vcd', 'hie'])
    parser.add_argument("--hie-lambda", type=float, default=None, help='logits fusing hyperparameter of HIE methods')
    parser.add_argument("--partition", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", type=int, default=0, choices=[0, 1], 
                        help='set to 1 to overwrite existing embedding files')
    opts = parser.parse_args()

    
    opts.save_dir = PRED_HOME_DIR + f'-{opts.partition}/{opts.dataset}'
    if not os.path.exists(opts.save_dir): os.makedirs(opts.save_dir)
    print(f'{opts.prompt} prompt, {opts.dataset}-{opts.partition} dataset, {opts.inference} inference')
    # print('-----------------------------------------------------')
    

    # CLIP model
    model, _ = clip.load("ViT-L/14@336px")
    print("model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("input resolution:", model.visual.input_resolution)
    print("context length:", model.context_length)
    print("vocab size:", model.vocab_size)

    # dataset & loader
    opts.data_paths = load_config(opts.data_paths_config)
    opts.dataset_path = opts.data_paths[opts.dataset]
    dataset_dir = os.path.join(opts.dataset_path, opts.partition)
    eval_dataset = datasets.ImageFolder(
        root=dataset_dir, 
        transform=custom_clip_transform(model.visual.input_resolution, None, None),
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, 
        num_workers=opts.workers,
        batch_size=opts.batch_size, 
        drop_last=False,
        shuffle=False,
        pin_memory=False,
    )

    classnames = eval_dataset.classes
    print(f"number of classes: {len(classnames)}")
    print(f"peak classnames[:5] : {classnames[:5]}")

    # load tree & hierarchical distance
    dataset_tree = load_tree(opts.dataset, './trees')
    hdist = load_hie_distance(opts.dataset, './trees')
    tree_height = dataset_tree.height() - 1
    print(f'tree height: {tree_height}, # of leaves: {len(dataset_tree.leaves())}')

    # load textual representations / image prompts
    if opts.prompt != 'merge':
        img_prompts = class_textual_representations(classnames, opts)
    else:
        img_prompts = merging_textual_representations(opts.merging_prompts, classnames, opts)

    
    # pass image prompts to text encoder for the respective raw embeddings
    embed_dir = f'./raw_embedding/{opts.dataset}'
    if not os.path.exists(embed_dir): os.makedirs(embed_dir)

    if opts.prompt == 'merge':
        merge_name = merge_template_names(opts.merging_prompts)
        opts.raw_embed_path = os.path.join(embed_dir, f'{opts.inference}-inf-{opts.prompt}-{merge_name}-prompts.pkl')
    else:
        opts.raw_embed_path = os.path.join(embed_dir, f'{opts.inference}-inf-{opts.prompt}-prompts.pkl')

    if os.path.exists(opts.raw_embed_path) and opts.overwrite != 1:
        with open(opts.raw_embed_path, 'rb') as f:
            class_raw_embedding = pickle.load(f)
        print(f'loading raw embeddings from {opts.raw_embed_path}')
    else:
        class_raw_embedding = textual_to_raw_embedding(model, img_prompts, opts)
        with open(opts.raw_embed_path, 'wb') as f:
            pickle.dump(class_raw_embedding, f)
        print(f'saving raw embeddings at {opts.raw_embed_path}')

    # acquire zero-shot classifier weights
    cls_weights = zeroshot_classifier(class_raw_embedding, classnames, dataset_tree, opts)
    

    # inference
    if opts.inference == 'flat':
        gt, pred, pred_crm = flat_classification(opts, model, eval_loader, cls_weights, hdist)

    elif opts.inference == 'vcd':
        gt, pred, pred_crm = vcd_classification(opts, model, eval_loader, cls_weights, hdist)
    
    elif opts.inference == 'hie':
        gt, pred, pred_crm = hie_classification(opts, model, eval_loader, cls_weights, hdist)
    
    else: raise NotImplementedError(f'opts.inference {opts.inference} not implemented')
    
    # evaluation
    evaluation(eval_loader, hdist, gt, pred, pred_crm)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')

# EOF