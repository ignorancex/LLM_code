from utils import get_dataloader_cocoimage, get_dataloader_flowersimage, concatenate_run
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
    parser.add_argument('--model', default='ClipVision')
    parser.add_argument('--clip_model_name', default='ViT-L/14-quickgelu')
    parser.add_argument('--dino_model_name', default='dinov2_vitl14')
    parser.add_argument('--siglip2_model_name', default='google/siglip2-large-patch16-384')
    parser.add_argument('--vit_model_name', default="google/vit-large-patch16-384")
    parser.add_argument('--layers', type=int, nargs='+', default=[-1])
    parser.add_argument('--top_k', type=int, default=32)
    parser.add_argument('--expansion_factor', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--runname', default="last_run")
    args = parser.parse_args()

    d = vars(args).copy()
    del d['layers']

    wm = wrappedmodels.SigLIP2Vision([-1])
    print("\nAAA"*50)


    wrapped = getattr(wrappedmodels, args.model)(layers=args.layers,**d).to(device)

    print(f"Number of Layers : {len(wrapped.layers_to_record)}")

    sae_list = [
        TopKSAE(d_vit=wrapped.d_vit(layer), expansion_factor=args.expansion_factor, top_k=args.top_k).to(device)
    for layer in args.layers]

    optim_list = [
        torch.optim.Adam(sae.parameters(), lr=args.lr) for sae in sae_list
    ]

    dl_train = get_dataloader_cocoimage(wrapped.transform, batch_size=args.batch_size, shuffle=True)


    print(f"--- {args.runname} ---")
    print(vars(args))

    for batch in tqdm(dl_train, desc='Training'):
        b1 = batch.to(device)
        _ = wrapped(b1)
        for layer, sae, optimizer in zip(args.layers, sae_list, optim_list):
            layer_output = wrapped.activations[layer].to(device)
            if len(layer_output.shape) == 4: layer_output = torch.flatten(layer_output, start_dim=-2)

            optimizer.zero_grad()
            sae_out = sae(layer_output)
            loss = sae.loss(layer_output, sae_out)
            loss.backward()
            
            optimizer.step()

    dl_record = get_dataloader_cocoimage(wrapped.transform, batch_size=args.batch_size, shuffle=False) # Shuffle false to keep same recording order across runs

    _b_counter = 0

    # Saving features to disk
    if not os.path.isdir("sae_features"): os.mkdir("sae_features")
    if not os.path.isdir(f"sae_features/{args.runname}"): os.mkdir(f"sae_features/{args.runname}")
    for l in args.layers:
        if not os.path.isdir(f"sae_features/{args.runname}/layer{str(l).zfill(2)}"):
            os.mkdir(f"sae_features/{args.runname}/layer{str(l).zfill(2)}")



    for batch in tqdm(dl_record, desc='Recording'):

        b1 = batch.to(device)
        _ = wrapped(b1)
        for layer, sae in zip(args.layers, sae_list):
            layer_output = wrapped.activations[layer].to(device)
            if len(layer_output.shape) == 4: layer_output = torch.flatten(layer_output, start_dim=-2)

            feats = sae.get_cls_features(layer_output)
            torch.save(feats.to_sparse(), f"sae_features/{args.runname}/layer{str(layer).zfill(2)}/{_b_counter}.pt")
        _b_counter += 1

    concatenate_run(args.runname)