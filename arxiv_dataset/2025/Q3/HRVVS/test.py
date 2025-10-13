from dataset.dataset import Vein
from models import build_model
from utils.cfg import parse_args
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from utils.misc import compute_metrix
from torchvision.utils import save_image
import os


args = parse_args()
torch.cuda.set_device(args.gpu_device)
test_dataset = Vein(args, mode='test')
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
model = build_model(args)
test_model_path = 'logs/vein/20250728_225525_stop-0/best_vvs_in_vein.pth'
model.load_state_dict(torch.load(test_model_path, map_location='cpu')['model'], strict=False)
output_dir = os.path.join('vis_new', args.dataset)
os.makedirs(output_dir, exist_ok=True)


with torch.no_grad():
    model.eval()
    pbar = tqdm(test_loader, desc='Test', unit='pack', total=len(test_loader))
    m_iou, m_dice, m_s_score, m_f_score, m_e_score, m_mae = 0, 0, 0, 0, 0, 0
    num_img= 0
    
    for pack in pbar:
        torch.cuda.empty_cache()
        images, masks, vid = pack['images'].cuda(), pack['masks'].cuda(), pack['video_id']
        for i in range(len(images)):
            image = images[i].unsqueeze(0)
            mask = masks[i].unsqueeze(0)
            pred = model(image, vid[i])
            iou, dice, s_score, f_score, e_score, mae = compute_metrix(pred, mask, activation=torch.sigmoid)
            # if dice > 0.8:
                # save_image(pred, os.path.join(output_dir, f'{names[i]}.png'))
                # save_image(mask, os.path.join(output_dir, f'{names[i]}_gt.png'))
            num_img += 1
            [m_iou, m_dice, m_s_score, m_f_score, m_e_score, m_mae] = [m + n for m, n in zip([m_iou, m_dice, m_s_score, m_f_score, m_e_score, m_mae], [iou, dice, s_score, f_score, e_score, mae])]
        
        pbar.set_postfix({'dice': '{:.3f}'.format(dice)})
        pbar.update()
    
    m_iou /= num_img
    m_dice /= num_img
    m_s_score /= num_img
    m_f_score /= num_img
    m_e_score /= num_img
    m_mae /= num_img

    print(f'IoU: {m_iou}, Dice: {m_dice}, S-measure: {m_s_score}, F-measure: {m_f_score}, E-measure: {m_e_score}, MAE: {m_mae}')