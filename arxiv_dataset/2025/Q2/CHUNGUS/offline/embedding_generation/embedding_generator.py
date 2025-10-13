import os
import tqdm
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataset import WRIZZDataset
from saver import Save
from models import get_model
from torch.utils.data import DataLoader


def extract_embeddings(network, dataloader, device, resolution):
    """ Extract the embeddings 
    
    :param network: network to use
    :param dataloader: dataloader to predict on
    :param device: device to use (must be cuda device for FeatUp)
    :param resolution: resolution for inference
    """

    embeddings = []
    labels = []
    image_files = []
    cls_tokens = []
    
    network.eval()
    network.to(device)
    with torch.no_grad():
        for item in tqdm.tqdm(dataloader):
            data = item['data']
            B = data[0].shape[0]
            imageA, imageB, targets = data[0].to(device), data[1].to(device), data[2].to(device)
            raw_predictions = network(torch.stack([imageA, imageB], dim=1).flatten(0,1)) # [B,2,3,nH,nW] -> [B*2,3,nH,nW]
            cls_token = raw_predictions['cls_token'].unflatten(0, (B,2)) # [B,2,384]
            features = F.interpolate(raw_predictions['features'], size=resolution, mode='bilinear').unflatten(0, (B,2)) # [B,2,384,H,W]
            features = features.permute((0,1,3,4,2)) # [B,2,H,W,384]
            pa = torch.stack([features[b,targets[b,:,0],targets[b,:,2],targets[b,:,1]] for b in range(B)], dim=0) # [B,A,384]
            pb = torch.stack([features[b,targets[b,:,3],targets[b,:,5],targets[b,:,4]] for b in range(B)], dim=0) # [B,A,384]
            
            embeddings.append(torch.stack([pa, pb], dim=2).detach().cpu()) # [B,A,2,384]
            cls_tokens.append(cls_token.detach().cpu())
            labels.append(targets[:,:,-1].detach().cpu()) # [B,A]
            image_files += list(zip(item['imageA'], item['imageB']))
    
    embeddings = torch.concatenate(embeddings, dim=0).numpy()
    labels = torch.concatenate(labels, dim=0).numpy()
    cls_tokens = torch.concatenate(cls_tokens, dim=0).numpy()

    print("Finished embedding generation")
    print("embeddings shape: {}".format(embeddings.shape))
    print("labels shape: {}".format(labels.shape))
    print("cls_tokens shape: {}".format(cls_tokens.shape))
    
    return {
        'embeddings': embeddings,
        'labels': labels,
        'cls_tokens': cls_tokens,
        'image_files': image_files # tuples of form (imageA, imageB)
    }


def main(settings):
    """ Main program """

    # Create save
    save = Save(Path(settings['save_folder']), should_exist=False)
    save.write_settings(settings)
    
    # Set seed
    seed = settings['seed']
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Setup
    device = torch.device('cuda')
    folder_path = Path(settings['data_folder'])
    csv_file = Path(settings['data_csv'])
    in_memory = True if settings['in_memory'] else False
    resolution = (settings['res_y'], settings['res_x'])
    
    # Create dataloader
    dataset = WRIZZDataset(folder_path, csv_file, resolution,
                           augmentation=None,
                           in_memory=in_memory,
                           normalize=True,
                           skip_intra=settings['skip_intra'],
                           skip_cross=settings['skip_cross'])
    dataloader = DataLoader(dataset, batch_size=settings['batch_size'], num_workers=settings['num_workers'])
    
    # Create network
    network = get_model(settings['model'])
    network.to(device)
    
    # Process
    results = {'settings': settings, 'data': [extract_embeddings(network, dataloader, device, resolution)]}
        
    # Save
    np.save(save.save_file, results)
    print("Finished")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # seed
    parser.add_argument("--seed", type=int, default=123)
    
    # data
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--save_folder", type=str, required=True)
    
    # resolution
    parser.add_argument("--res_x", type=int, default=224)
    parser.add_argument("--res_y", type=int, default=224)
    
    # data loading
    parser.add_argument("--batch_size", type=int, default=2) # each batch actually has 2 images (so batch_size=2 => 4 images in batch)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--in_memory", default=False, action='store_true')
    # settings for more specific data selection
    parser.add_argument("--skip_intra", default=False, action='store_true')
    parser.add_argument("--skip_cross", default=False, action='store_true')
    
    # model
    parser.add_argument("--model", type=str, default="dinov2")
    
    args = parser.parse_args()
    main(vars(args)) # pass args as dictionary
