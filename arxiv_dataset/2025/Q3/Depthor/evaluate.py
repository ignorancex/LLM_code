import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import cv2
from tqdm import tqdm
from src.utils.model_io import load_weights
from src.dataloader.real import REAL
from src.dataloader.nyu import NYUV2
from src.dataloader.zju import ZJU
from src.dataloader.hypersim import HYPERSIM
from src.utils.utils import RunningAverageDict, compute_errors
from src.config import args


def predict_tta(model, input_data, args):
    # flops, params = profile(model, inputs=(input_data,))
    # print('imgencoder FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('imgencoder Params = ' + str(params / 1000 ** 2) + 'M')

    _, pred = model(input_data)
    pred = pred.cpu().numpy()
    pred = np.clip(pred, args.min_depth, args.max_depth)
    return torch.Tensor(pred)


def eval(model, test_loader, args, device):
    if args.save_dir is not None and not os.path.exists(f'{args.save_dir}'):
        os.system(f'mkdir -p {args.save_dir}')
    metrics = RunningAverageDict()

    with torch.no_grad():
        model.eval()
        for index, batch in enumerate(tqdm(test_loader)):

            gt = batch['depth']
            for key in batch:
                if hasattr(batch[key], 'to'):
                    batch[key] = batch[key].to(device)
            input_data = batch

            final = predict_tta(model, input_data, args)
            final = final.squeeze().cpu().numpy()

            impath = f"{batch['image_path'][0]}"
            im_subfolder = batch['image_folder'][0]
            vis_folder = f'{im_subfolder}'

            # if not os.path.exists(f'{args.save_dir}/{vis_folder}'):
            #     os.system(f'mkdir -p {args.save_dir}/{vis_folder}')
            #
            # if args.save_pred:
            #     colormap = cv2.applyColorMap(cv2.convertScaleAbs(final, alpha=70), cv2.COLORMAP_JET)
            #     if args.dataset == 'nyu':
            #         jpg_path = os.path.join(args.save_dir, f"{impath}")  # for nyu
            #     elif args.dataset == 'hypersim':
            #         jpg_path = os.path.join(args.save_dir, vis_folder, f"{impath}")  # for whole hypersim
            #     elif args.dataset == 'zju':
            #         jpg_path = os.path.join(args.save_dir, vis_folder, f"{impath}")  # for zju
            #     elif args.dataset == 'arkit':
            #         jpg_path = os.path.join(args.save_dir, vis_folder, f"{impath}")
            #     cv2.imwrite(jpg_path, colormap)

            # if args.save_error_map:
            #     error_map = np.abs(gt.squeeze().cpu().numpy() - final)
            #     viz = colorize(torch.from_numpy(error_map).unsqueeze(0), vmin=0, vmax=1.2, cmap='jet')
            #     Image.fromarray(viz).save(os.path.join(args.save_dir, vis_folder, f"{impath}_error.png"))
            #
            # if args.save_rgb:
            #     img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
            #     img = img.squeeze().permute([1, 2, 0]).cpu().numpy()
            #     img = (img * 255).astype(np.uint8)
            #     rgb = img.copy()
            #     Image.fromarray(rgb).save(os.path.join(args.save_dir, vis_folder, f"{impath}_rgb.png"))
            #
            # if args.save_for_demo:
            #     pred = (final * 1000).astype('uint16')
            #     pred_path = os.path.join(args.save_dir, vis_folder, f"{impath}_demo.png")
            #     Image.fromarray(pred).save(pred_path)

            gt = gt.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt > args.min_depth, gt < args.max_depth)
            if valid_mask.any():
                metrics.update(compute_errors(gt[valid_mask], final[valid_mask], gt, final, args.max_depth_eval))
                # print(compute_errors(gt[valid_mask], final[valid_mask], gt, final, args.max_depth_eval))
    metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
    print(f"Metrics: {metrics}")


if __name__ == '__main__':
    device = torch.device('cuda')
    if args.dataset == 'nyu':
        test_loader = NYUV2(args, 'online_eval').data
    elif args.dataset == 'real':
        test_loader = REAL(args, 'online_eval').data
    elif args.dataset == 'hypersim':
        test_loader = HYPERSIM(args, 'online_eval').data
    elif args.dataset == 'zju':
        test_loader = ZJU(args, 'online_eval').data

    ###################### eval cformer ###########################
    # from src.models.cformer.model.completionformer import Depthor
    # model = Depthor().to(device)
    # model = load_weights(model, args.weight_path)
    # model.set_extra_param(device=device)
    # eval(model, test_loader, args, device)

    ###################### eval mde ###########################
    # from src.models.depth_anything_v2_metric.depthor import Depthor
    # model = Depthor().to(device)
    # model.set_extra_param(device=device)
    # eval(model, test_loader, args, device)

    # from src.models.depth_anything_v2.depthor import Depthor
    # model = Depthor().to(device)
    # model.set_extra_param(device=device)
    # eval(model, test_loader, args, device)

    ##################### eval depthor ###########################
    from src.models.depthor import Depthor
    model = Depthor(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm='linear').to(device)
    model = load_weights(model, args.weight_path)
    model.set_extra_param(device=device)
    eval(model, test_loader, args, device)

    ##################### eval depthor-s ###########################
    # from src.models.depthor_s import Depthor
    # model = Depthor(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm='linear').to(device)
    # model = load_weights(model, args.weight_path)
    # model.set_extra_param(device=device)
    # eval(model, test_loader, args, device)

    ###################### eval penet #############################
    # from src.models.penet.penet import Depthor
    # model = Depthor(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm).to(device)
    # model = load_weights(model, args.weight_path)
    # model.set_extra_param(device=device)
    # model = model.eval()
    # eval(model, test_loader, args, device)
