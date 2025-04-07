"""
isolated hiecomp inference function
score fusion of HIE method
"""

import os
import torch
import pickle
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm 
from models.textual_representation import name_preprocessing




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


def hie_classification(opts, clip_model, dataloader, zeroshot_weights, hdist):

    # --- Hie hyperparameters --- #
    eta = 0
    lamda = opts.hie_lambda
    # --- Hie hyperparameters --- #

    assert hdist is not None
    save_path = os.path.join(opts.save_dir, f'{opts.inference}-{opts.hie_lambda:.1f}-lambda-inf-{opts.prompt}-prompt.pkl')

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
    

    with torch.no_grad():
        for _, (images, target) in enumerate(tqdm(dataloader)):
            images = images.cuda()
            

            # visual encoding
            img_features = clip_model.encode_image(images) # -> (batch_size, embed_dim)

            # (batch_size, embed_dim) / (batch_size, 1) -> (batch_size, embed_dim)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            # --- tree inference 
            img_tree_similarity_cumulative = [None] * n_classses
            for i, (k, v) in enumerate(zeroshot_weights.items()):

                dot_product_matrix_base = (img_features @ v[0].T).mean(dim=1)
                if len(v) > 1:
                    score = torch.stack([(img_features @ f.T).mean(dim=1) for f in v[1:]], dim=1)
                    diffs = score[:, 1:] - score[:, :-1]
                    padded_diffs = F.pad(diffs, (1, 0, 0, 0), value=1)
                    mask = padded_diffs > eta

                    first_false = (mask == False).cumsum(dim=1) >= 1

                    # set values to False after the first false in each row
                    mask[first_false] = False

                    dot_product_matrix_comp = \
                        (score * mask).sum(dim=1) / mask.sum(dim=1)
                    img_tree_similarity_cumulative[i] = \
                          lamda * dot_product_matrix_base + \
                          (1-lamda) * dot_product_matrix_comp
                else:
                    img_tree_similarity_cumulative[i] = dot_product_matrix_base

            # create tensor of similarity means
            hiecomp_logits = 100.0 * torch.stack(img_tree_similarity_cumulative, dim=1)

            # hiecomp softmax
            hiecomp_softmaxes = F.softmax(hiecomp_logits, dim=1)

            # compute negative risk of crm
            negative_risk = -1.0 * hiecomp_softmaxes @ cost

            # collect inference results
            gt.append(target.numpy())
            pred = torch.argmax(hiecomp_logits, dim=1)
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