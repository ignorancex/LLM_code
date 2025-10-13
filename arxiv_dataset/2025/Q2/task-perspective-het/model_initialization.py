import clip
from models import HugeModel, HugeModelCBAM
import torch
from argparse import ArgumentParser
import os

# Define argument parser
def parse_input():

    parser = ArgumentParser()
    
    parser.add_argument('--task', type=str)
    parser.add_argument('--init_model_dir', type=str, default='./initial_models/')
    parser.add_argument('--use_cbam', type=bool, default=False)
    
    return parser.parse_args()

args = parse_input()
clip_model, _ = clip.load("ViT-B/32", device=torch.device("cpu"))
if args.use_cbam:
    init_model = HugeModelCBAM(clip_model, args.task)
else:
    init_model = HugeModel(clip_model, args.task)
torch.save(init_model.state_dict(), os.path.join(args.init_model_dir, f"init_model_{args.task}{'_cbam' if args.use_cbam else None}.pth"))
