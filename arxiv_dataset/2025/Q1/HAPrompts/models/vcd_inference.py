"""
isolated vcd inference function
ensemble over log-probability (logit) space
"""

import os
import torch
import pickle
import numpy as np

from tqdm import tqdm 
from models.textual_representation import merge_template_names, name_preprocessing




def get_crm_cost_tensor(hdist, classes):
    """
        hdist: hierarchical distance dictionary
        classes: class names that are key words for hdist
    """
    num_classes = len(classes)
    cost = [[0 for i in range(num_classes)] for j in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            cost[i][j] = hdist[(classes[i], classes[j])]
    cost = torch.tensor(cost, dtype=torch.half)
    return cost


# vcd inference, not aggregating class textual representations into one embedding per class
# aggregating class textual representations in its log-probability space
def vcd_classification(opts, clip_model, dataloader, zeroshot_weights, hdist):

    assert hdist is not None
    
    merged_template_name = None
    if opts.prompt != 'merge':
        save_file_name = \
            f'{opts.inference}-inf-{opts.prompt}-prompt.pkl' \
            
    else:
        merged_template_name = merge_template_names(opts.merging_prompts)
        save_file_name = \
            f'{opts.inference}-inf-{opts.prompt}-{merged_template_name}-prompt.pkl' \
            
    save_path = os.path.join(opts.save_dir, save_file_name)

    if os.path.exists(save_path) and opts.overwrite != 1:
        with open(save_path, 'rb') as f:
            eval_res = pickle.load(f)
        print(f'loading inference results at {save_path}')
        return eval_res['gt'], eval_res['pred'], eval_res['pred_crm']

    
    classnames = dataloader.dataset.class_to_idx
    classnames = name_preprocessing(classnames)
    n_classses = len(classnames)
    cost = get_crm_cost_tensor(hdist, classnames).cuda()

    prediction_method = []
    prediction_crm = []
    gt = []

    # inference
    with torch.no_grad():
        for _, (images, target) in enumerate(tqdm(dataloader)):
            images = images.cuda()

            # visual encoding
            img_features = clip_model.encode_image(images) # -> (batch_size, embed_dim)

            # (batch_size, embed_dim) / (batch_size, 1) -> (batch_size, embed_dim)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            img_description_similarity = [None] * n_classses
            for i, (k, v) in enumerate(zeroshot_weights.items()):
                # img_feature -> (batch_size, embed_dim)
                # v -> (# descriptor, embed_dim)
                # dot_product_matrix -> (batch_size, # descriptor)
                dot_product_matrix = img_features @ v.T
                # img_description_similarity -> [(batch_size,),...,(batch_size,)]
                img_description_similarity[i] = dot_product_matrix.mean(dim=1)
            
            # create tensor of similarity means -> [batch_size, n_classes]
            vcd_logits = 100.0 * torch.stack(img_description_similarity, dim=1)

            # vcd softmax
            vcd_softmaxes = torch.nn.functional.softmax(vcd_logits, dim=1)

            # compute negative risk of crm
            negative_risk = -1.0 * vcd_softmaxes @ cost

            # collect inference results
            gt.append(target.numpy())
            pred = torch.argmax(vcd_logits, dim=1)
            pred_crm = torch.argmax(negative_risk, dim=1)
            
            prediction_method.append(pred)
            prediction_crm.append(pred_crm)

    # convert to ndarray
    gt = np.concatenate(gt)
    prediction_method = torch.cat(prediction_method, dim=0)
    prediction_method = prediction_method.detach().cpu().numpy()
    prediction_crm = torch.cat(prediction_crm, dim=0)
    prediction_crm = prediction_crm.detach().cpu().numpy()
    

    print(f'gt.shape: {gt.shape}')
    print(f'prediction_method.shape: {prediction_method.shape}')
    print(f'prediction_crm.shape: {prediction_crm.shape}')

    with open(save_path, 'wb') as f:
        pickle.dump({'gt': gt, 'pred': prediction_method, 'pred_crm': prediction_crm}, f)
    print(f'saving inference results at {save_path}')

    return gt, prediction_method, prediction_crm