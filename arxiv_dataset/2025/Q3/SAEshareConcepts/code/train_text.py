from utils import concatenate_run, get_dataloader_flowerstext, get_dataloader_cocotext
import torch
from tqdm import tqdm
from rich import print
from sparse_autoencoders import TopKSAE
from argparse import ArgumentParser
import wrappedmodels
import os

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = ArgumentParser()
    parser.add_argument('--model', default='ClipText')
    parser.add_argument('--clip_model_name', default='ViT-L/14-quickgelu')
    parser.add_argument('--siglip2_model_name', default='google/siglip2-large-patch16-384')
    parser.add_argument('--bert_model_name', default='bert-large-uncased')
    parser.add_argument('--deberta_model_name', default="microsoft/deberta-large")
    parser.add_argument('--layers', type=int, nargs='+', default=[-1])
    parser.add_argument('--top_k', type=int, default=32)
    parser.add_argument('--expansion_factor', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--runname', default="last_run")
    args = parser.parse_args()

    d = vars(args).copy()
    del d['layers']
    wrapped = getattr(wrappedmodels, args.model)(layers=args.layers,**d).to(device)

    print(f"Number of Layers : {len(wrapped.layers_to_record)}")

    sae_list = [
        TopKSAE(d_vit=wrapped.d_vit(layer), expansion_factor=args.expansion_factor, top_k=args.top_k).to(device)
    for layer in args.layers]

    optim_list = [
        torch.optim.Adam(sae.parameters(), lr=args.lr) for sae in sae_list
    ]

    dl_train = get_dataloader_cocotext(batch_size=args.batch_size,shuffle=True)

    print(f"--- {args.runname} ---")
    print(vars(args))
    

    for batch in tqdm(dl_train, desc='Training'):
        toks = wrapped.transform(batch).to(device)
        if isinstance(toks, dict):
            b = {k: v.to(device) for k, v in toks.items()}
        else: b = toks
        _ = wrapped(b)
        for layer, sae, optimizer in zip(args.layers, sae_list, optim_list):
            layer_output = wrapped.activations[layer].to(device)
            optimizer.zero_grad()
            sae_out = sae(layer_output)
            loss = sae.loss(layer_output, sae_out)
            loss.backward()
            
            optimizer.step()


    dl_record = get_dataloader_cocotext(batch_size=args.batch_size,shuffle=False)

    _b_counter = 0

    if not os.path.isdir("sae_features"): os.mkdir("sae_features")
    if not os.path.isdir(f"sae_features/{args.runname}"): os.mkdir(f"sae_features/{args.runname}")
    for l in args.layers:
        if not os.path.isdir(f"sae_features/{args.runname}/layer{str(l).zfill(2)}"):
            os.mkdir(f"sae_features/{args.runname}/layer{str(l).zfill(2)}")

    for batch in tqdm(dl_record, desc='Recording'):
        toks = wrapped.transform(batch).to(device)
        if isinstance(toks, dict):
            b = {k: v.to(device) for k, v in toks.items()}
        else: b = toks
        _ = wrapped(b)
        for layer, sae in zip(args.layers, sae_list):
            layer_output = wrapped.activations[layer].to(device)
            feats = sae.get_cls_features(layer_output)
            torch.save(feats.to_sparse(), f"sae_features/{args.runname}/layer{str(layer).zfill(2)}/{_b_counter}.pt")
        _b_counter += 1

    concatenate_run(args.runname)