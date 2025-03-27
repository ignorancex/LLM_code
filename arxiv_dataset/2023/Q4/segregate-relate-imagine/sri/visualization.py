import torch
import torchvision
import numpy as np
from torchvision.utils import make_grid

def colour_seg_masks(masks, palette='15'):
    colours = {
        "palette":
        [[204,78,51],
        [82,180,78],
        [125,102,215],
        [151,177,49],
        [193,91,184],
        [217,149,40],
        [121,125,196],
        [182,136,63],
        [76,170,212],
        [208,67,130],
        [74,170,134],
        [202,86,102],
        [117,145,74],
        [187,109,153],
        [188,120,88]]
    }
    # NOTE: Maps negative (ignore) labels to black
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)
    assert masks.dim() == 4
    assert masks.shape[1] == 1
    #colours = json.load(open(f'utils/colour_palette{palette}.json'))
    img_r = torch.zeros_like(masks)
    img_g = torch.zeros_like(masks)
    img_b = torch.zeros_like(masks)
    for c_idx in range(masks.max().item() + 1):
        c_map = masks == c_idx
        if c_map.any():
            img_r[c_map] = colours['palette'][c_idx][0]
            img_g[c_map] = colours['palette'][c_idx][1]
            img_b[c_map] = colours['palette'][c_idx][2]
    return torch.cat([img_r, img_g, img_b], dim=1)


def visualise_outputs(model, vis_batch, writer, mode, iter_idx):
    
    model.eval()
    
    # Only visualise for eight images
    # Forward pass
    vis_input = vis_batch[:8]
    if next(model.parameters()).is_cuda:
        vis_input = vis_input.cuda()
    with torch.no_grad():
        output, losses, stats, att_stats, comp_stats = model(vis_input)
    if len(output) == 2:
        output, ordered_output = output
        stats, sri_stats = stats
        writer.add_image(mode+'_ordered_recon', make_grid(ordered_output), iter_idx)
    else:
        output = output[0]
        stats = stats[0]
    # Input and recon
    writer.add_image(mode+'_input', make_grid(vis_batch[:8]), iter_idx)
    writer.add_image(mode+'_recon', make_grid(output), iter_idx)
    
    # Instance segmentations
    # if 'instances' in vis_batch:
    #     grid = make_grid(colour_seg_masks(vis_batch['instances'][:8]))
    #     writer.add_image(mode+'_instances_gt', grid, iter_idx)
    # Segmentation predictions
    for field in ['log_m_k', 'log_m_r_k']:
        if field in stats:
            log_masks = stats[field]
        else:
            continue
        ins_seg = torch.argmax(torch.cat(log_masks, 1), 1, True)
        grid = make_grid(colour_seg_masks(ins_seg))
        if field == 'log_m_k':
            writer.add_image(mode+'_instances', grid, iter_idx)
        elif field == 'log_m_r_k':
            writer.add_image(mode+'_instances_r', grid, iter_idx)
    # Decomposition
    for key in ['mx_r_k', 'x_r_k', 'log_m_k', 'log_m_r_k']:
        if key not in stats:
            continue
        for step, val in enumerate(stats[key]):
            if 'log' in key:
                val = val.exp()
            writer.add_image(f'{mode}_{key}/k{step}', make_grid(val), iter_idx)
    
    # Generation
    try:
        output, stats = model.module.sample(batch_size=8, K_steps=model.module.K_steps, temp=1)
        writer.add_image('samples', make_grid(output), iter_idx)
        for key in ['x_k', 'log_m_k', 'mx_k']:
            if key not in stats:
                continue
            for step, val in enumerate(stats[key]):
                if 'log' in key:
                    val = val.exp()
                writer.add_image(f'gen_{key}/k{step}', make_grid(val),
                                    iter_idx)
        
        _, sri_comp_stats = comp_stats
        imagination_recon, _, _ = \
                model.module.genesisv2.decode_latents(sri_comp_stats.z_imagination,
                                    the_decoder=model.module.decoder)
        writer.add_image('gen_imagine', make_grid(imagination_recon[:8]), iter_idx)
    except NotImplementedError:
        print("Sampling not implemented for this model.")
    
    model.train()

def get_mask_plot_colors(nr_colors):
    colours = np.array(
           [[198,55,210],
            [82,216,69],
            [71,143,210],
            [75,214,163],
            [208,125,70],
            [91,60,210],
            [183,214,70],
            [129,215,70],
            [207,62,109]], dtype=np.float32)

    colors = np.reshape(colours / 255., (colours.shape[0],3))[:nr_colors,:]
    return colors


def create_mask_image(masks):
    """
    masks is [K,1,H,W]
    """
    masks = masks.squeeze(1)
    color_conv = get_mask_plot_colors(masks.shape[0])
    color_mask = torch.einsum('hwk,kd->hwd', masks.permute(1,2,0), torch.from_numpy(color_conv).to(masks.device))
    color_mask = torch.clamp(color_mask, 0.0, 1.0)
    return color_mask.permute(2,0,1)  # [3,H,W]