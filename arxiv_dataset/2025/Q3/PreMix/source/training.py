import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from functools import partial
import tqdm
import copy

from source.dataset_utils import collate_features

class Training(object):
    def __init__(
        self,
        model: nn.Module,
        pool_dataset: torch.utils.data.Dataset,
        batch_size: Optional[int] = 1,
    ):
        self.model = model
        self.pool_dataset = pool_dataset
        self.batch_size = batch_size
        self.collate_fn = partial(collate_features, label_type="int")
        
    def predict(model, pool_dataset, collate_fn, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        loader = torch.utils.data.DataLoader(
            pool_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        p = torch.zeros([len(pool_dataset), pool_dataset.num_classes])

        with tqdm.tqdm(
            loader,
            desc=(f"Predict"),
            unit=" slide",
            ncols=80,
            unit_scale=batch_size,
            leave=True,
        ) as t:

            with torch.no_grad():

                for i, batch in enumerate(t):

                    idx, slide_id, features, mask, labels = batch

                    features, mask, labels = features.to(device, non_blocking=True), mask.to(device, non_blocking=True), labels.to(
                            device, non_blocking=True
                    )

                    embeds, zero_embeds, logits = model(x=features, mask=mask) # WSI features

                    pred = logits.max(1)[1]
                    p[idx] = pred
            
            return p.cpu()
    
    def predict_prob(model, pool_dataset, collate_fn, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
    
        loader = torch.utils.data.DataLoader(
            pool_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        probs = torch.zeros([len(pool_dataset), pool_dataset.num_classes])

        with tqdm.tqdm(
            loader,
            desc=(f"Predict Prob"),
            unit=" slide",
            ncols=80,
            unit_scale=batch_size,
            leave=True,
        ) as t:

            with torch.no_grad():

                for i, batch in enumerate(t):

                    idx, slide_id, features, mask, labels = batch

                    features, mask, labels = features.to(device, non_blocking=True), mask.to(device, non_blocking=True), labels.to(
                        device, non_blocking=True
                    ) 

                    embeds, zero_embeds, logits = model(x=features, mask=mask) # WSI features

                    prob = F.softmax(logits, dim=1)
                    probs[idx] = prob.cpu()
            
            return probs

    def predict_prob_embed(model, pool_dataset, collate_fn, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
    
        loader = torch.utils.data.DataLoader(
            pool_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        probs = torch.zeros([len(pool_dataset), pool_dataset.num_classes])
        embeddings = torch.zeros([len(pool_dataset), 192])

        with tqdm.tqdm(
            loader,
            desc=(f"Predict Prob Embedding"),
            unit=" slide",
            ncols=80,
            unit_scale=batch_size,
            leave=True,
        ) as t:

            with torch.no_grad():

                for i, batch in enumerate(t):

                    idx, slide_id, features, mask, labels = batch

                    features, mask, labels = features.to(device, non_blocking=True), mask.to(device, non_blocking=True), labels.to(
                        device, non_blocking=True
                    ) 
                
                    embeds, zero_embeds, logits = model(x=features, mask=mask) # WSI features

                    prob = F.softmax(logits, dim=1)
                    probs[idx] = prob.cpu()
                    embeddings[idx] = embeds.cpu()
            
            return probs, embeddings

    def predict_all_representations(model, pool_dataset, collate_fn, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
    
        loader = torch.utils.data.DataLoader(
            pool_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        probs = torch.zeros([len(pool_dataset), pool_dataset.num_classes])
        embeddings = torch.zeros([len(pool_dataset), 192])

        with tqdm.tqdm(
            loader,
            desc=(f"Predict All Representations"),
            unit=" slide",
            ncols=80,
            unit_scale=batch_size,
            leave=True,
        ) as t:

            with torch.no_grad():

                for i, batch in enumerate(t):
                        
                    idx, slide_id, features, mask, labels = batch

                    features, mask, labels = features.to(device, non_blocking=True), mask.to(device, non_blocking=True), labels.to(
                        device, non_blocking=True
                    )  

                    print(mask)
                    embeds, zero_embeds, logits = model(x=features, mask=mask) # WSI features

                    prob = F.softmax(logits, dim=1)
                    probs[idx] = prob.cpu()
                    embeddings[idx] = embeds.cpu()
            
            return probs, embeddings

    def predict_all_representations(model, pool_dataset, collate_fn, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        loader = torch.utils.data.DataLoader(
            pool_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        probs = torch.zeros([len(pool_dataset), pool_dataset.num_classes])
        embeddings = torch.zeros([len(pool_dataset), 192])

        with tqdm.tqdm(
            loader,
            desc=(f"Predict All Representations"),
            unit=" slide",
            ncols=80,
            unit_scale=batch_size,
            leave=True,
        ) as t:

            with torch.no_grad():

                for i, batch in enumerate(t):

                    idx, slide_id, features, mask, labels = batch

                    features, mask, labels = features.to(device, non_blocking=True), mask.to(device, non_blocking=True), labels.to(
                        device, non_blocking=True
                    ) 

                    embeds, zero_embeds, logits = model(x=features, mask=mask) # WSI features
                    
                    prob = F.softmax(logits, dim=1)
                    probs[idx] = prob.cpu()
                    embeddings[idx] = embeds.cpu()
            
            return probs, embeddings

    def predict_prob_dropout(model, pool_dataset, collate_fn, batch_size, n_drop):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.train() # to activate dropout?
    
        loader = torch.utils.data.DataLoader(
            pool_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        probs = torch.zeros([len(pool_dataset), pool_dataset.num_classes])

        with tqdm.tqdm(
            loader,
            desc=(f"Predict Prob Dropout"),
            unit=" slide",
            ncols=80,
            unit_scale=batch_size,
            leave=True,
        ) as t:

            for n in range(n_drop):
                print("n_drop {}/{}".format(n + 1, n_drop))

                with torch.no_grad():

                    for i, batch in enumerate(t):

                        idx, slide_id, features, mask, labels = batch

                        features, mask, labels = features.to(device, non_blocking=True), mask.to(device, non_blocking=True), labels.to(
                            device, non_blocking=True
                        ) 

                        embeds, zero_embeds, logits = model(x=features, mask=mask) # WSI features

                        prob = F.softmax(logits, dim=1)
                        probs[idx] += prob.cpu()
            probs /= n_drop

            return probs

    def predict_prob_dropout_split(model, pool_dataset, collate_fn, batch_size, n_drop):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.train() # to activate dropout?
    
        loader = torch.utils.data.DataLoader(
            pool_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        probs = torch.zeros([n_drop, len(pool_dataset), pool_dataset.num_classes])

        with tqdm.tqdm(
            loader,
            desc=(f"Predict Prob Dropout Split"),
            unit=" slide",
            ncols=80,
            unit_scale=batch_size,
            leave=True,
        ) as t:

            for n in range(n_drop):
                print("n_drop {}/{}".format(n + 1, n_drop))

                with torch.no_grad():

                    for i, batch in enumerate(t):

                        idx, slide_id, features, mask, labels = batch

                        features, mask, labels = features.to(device, non_blocking=True), mask.to(device, non_blocking=True), labels.to(
                            device, non_blocking=True
                        )  

                        embeds, zero_embeds, logits = model(x=features, mask=mask) # WSI features

                        probs[n][idx] += F.softmax(logits, dim=1).cpu()

            return probs
    
    def predict_prob_embed_dropout_split(model, pool_dataset, collate_fn, batch_size, n_drop):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.train() # to activate dropout?
    
        loader = torch.utils.data.DataLoader(
            pool_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        probs = torch.zeros([n_drop, len(pool_dataset), pool_dataset.num_classes])
        embeddings = torch.zeros([n_drop, len(pool_dataset), 192])

        with tqdm.tqdm(
            loader,
            desc=(f"Predict Prob embeddings Dropout Split"),
            unit=" slide",
            ncols=80,
            unit_scale=batch_size,
            leave=True,
        ) as t:

            for n in range(n_drop):
                print("n_drop {}/{}".format(n + 1, n_drop))

                with torch.no_grad():

                    for i, batch in enumerate(t):

                        idx, slide_id, features, mask, labels = batch

                        features, mask, labels = features.to(device, non_blocking=True), mask.to(device, non_blocking=True), labels.to(
                            device, non_blocking=True
                        )   

                        embeds, zero_embeds, logits = model(x=features, mask=mask) # WSI features

                        probs[n][idx] += F.softmax(logits, dim=1).cpu()
                        embeddings[n][idx] = embeds.cpu()

            return probs, embeddings
    
    def get_embedding(model, pool_dataset, collate_fn, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
    
        loader = torch.utils.data.DataLoader(
            pool_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        embedding = torch.zeros([len(pool_dataset), 192])

        with tqdm.tqdm(
            loader,
            desc=(f"Get Embedding"),
            unit=" slide",
            ncols=80,
            unit_scale=batch_size,
            leave=True,
        ) as t:

            with torch.no_grad():

                for i, batch in enumerate(t):

                    idx, slide_id, features, mask, labels = batch

                    features, mask, labels = features.to(device, non_blocking=True), mask.to(device, non_blocking=True), labels.to(
                        device, non_blocking=True
                    )  

                    embeds, zero_embeds, logits = model(x=features, mask=mask) # WSI features

                    embedding[idx] = embeds.cpu()

            return embedding

    def get_grad_embedding(model, pool_dataset, collate_fn, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        nLab = pool_dataset.num_classes
        embedding = np.zeros([len(pool_dataset), 192 * nLab])

        loader = torch.utils.data.DataLoader(
            pool_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        with tqdm.tqdm(
            loader,
            desc=(f"Get Grad Embedding"),
            unit=" slide",
            ncols=80,
            unit_scale=batch_size,
            leave=True,
        ) as t:

            with torch.no_grad():

                for i, batch in enumerate(t):

                    idx, slide_id, features, mask, labels = batch

                    features, mask, labels = features.to(device, non_blocking=True), mask.to(device, non_blocking=True), labels.to(
                        device, non_blocking=True
                    )  

                    embeds, zero_embeds, logits = model(x=features, mask=mask) # WSI features

                    embeddings = embeds.data.cpu().numpy()
                    batchProbs = F.softmax(logits, dim=1).data.cpu().numpy()
                    maxInds = np.argmax(batchProbs, 1)
                    for j in range(len(labels)):
                        for c in range(nLab):
                            if c == maxInds[j]:
                                embedding[idx[j]][192 * c: 192 * (c + 1)] = copy.deepcopy(embeddings[j]) * (1 - batchProbs[j][c])
                            else:
                                embedding[idx[j]][192 * c: 192 * (c + 1)] = copy.deepcopy(embeddings[j]) * (-1 * batchProbs[j][c])

                return torch.Tensor(embedding)