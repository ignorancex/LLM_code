try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import os
import torch
from torchmetrics.functional.regression import pearson_corrcoef
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, ssim
from utils.depth_utils import estimate_depth
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, AblateParams
from lpipsPyTorch import lpips
import csv
from sd_guidance import StableDiffusionGuidance
from warp import inverse_warp

def process_log(model_path, opt=None, abla=None, text=None, refresh=False):
    if(refresh):
        with open(os.path.join(model_path, "train_log.txt"), 'w') as cfg_log_f:
            for arg in vars(opt):
                write_text = f'{str(arg)}: {str(getattr(opt, arg))}'
                cfg_log_f.write(write_text)
                cfg_log_f.write('\n')
            for arg in vars(abla):
                write_text = f'{str(arg)}: {str(getattr(abla, arg))}'
                cfg_log_f.write(write_text)
                cfg_log_f.write('\n')
    else:
        if(text == None): return
        with open(os.path.join(model_path, "train_log.txt"), 'a+') as cfg_log_f:
            cfg_log_f.write(str(text))
            cfg_log_f.write('\n')

def record_training(args, init=True, iter_num=None, psnr_num=None, ssim_num=None, lpips_num=None):
    csv_path = os.path.join(args.model_path, 'record.csv')
    if(init):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iter', 'psnr', 'ssim', 'lpips'])
    else:
        assert psnr_num != None and ssim_num != None and lpips_num != None
        with open(csv_path, 'a+', newline='') as f:
            writer = csv.writer(f)
            row_data = [iter_num, psnr_num, ssim_num, lpips_num]
            writer.writerow(row_data)
               
def training(dataset, opt, pipe, abla, args):
    blip_rst_dir = os.path.join(dataset.source_path, 'blip_rst.txt')
    with open(blip_rst_dir, 'r') as f:
        read_blip_rst = f.readline()
        f.close()
    random_select_info = read_blip_rst.split(':')[0]
    blip_rst = read_blip_rst.split(':')[-1]
    print(random_select_info, blip_rst)
    if args.add_sd_guidance or args.add_warp_sds_guidance or args.add_warp_sds_guidance_2 or args.add_sds_guidance:
        sd_guidance = StableDiffusionGuidance(blip_rst=blip_rst, use_lora=False, use_sd15=(args.add_warp_sds_guidance_2 or args.add_sds_guidance), guidance_scale=args.guidance_scale)
        sd_guidance.configure()
        
    process_log(args.model_path, opt=opt, abla=abla, text=None, refresh=True)
    record_training(args, init=True)
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    viewpoint_stack, pseudo_stack = None, None
    ema_loss_for_log = 0.0
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        if (iteration - 1) == debug_from:
            pipe.debug = True
        if iteration % args.sh_interval == 0:
            gaussians.oneupSHdegree()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 =  l1_loss_mask(image, gt_image)
        loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))
        
        rendered_depth = render_pkg["depth"][0]
        midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda()
        rendered_depth = rendered_depth.reshape(-1, 1)
        midas_depth = midas_depth.reshape(-1, 1)
        depth_loss = min(
                        (1 - pearson_corrcoef( - midas_depth, rendered_depth)),
                        (1 - pearson_corrcoef(1 / (midas_depth + 200.), rendered_depth))
        )
        loss += args.depth_weight * depth_loss
        if iteration > args.end_sample_pseudo:
            args.depth_weight = min(0.001, args.depth_weight)
        if iteration % args.sample_pseudo_interval == 0 and iteration > args.start_sample_pseudo and iteration < args.end_sample_pseudo:
            if not pseudo_stack:
                pseudo_stack, closest_cam_stack = scene.getPseudoCamerasWithClosestViews()
                pseudo_stack = pseudo_stack.copy()
                closest_cam_stack = closest_cam_stack.copy()
            randint_idx = randint(0, len(pseudo_stack) - 1)
            pseudo_cam, closest_cam_1 = pseudo_stack.pop(randint_idx), closest_cam_stack.pop(randint_idx)
            render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
            rendered_img_pseudo = render_pkg_pseudo["render"]
            rendered_depth_pseudo = render_pkg_pseudo["depth"][0]
            midas_depth_pseudo = estimate_depth(rendered_img_pseudo, mode='train')
            closest_image_1 = closest_cam_1.original_image.cuda()
            render_pkg_1 = render(closest_cam_1, gaussians, pipe, background)
            closest_depth_1 = render_pkg_1["depth"]
            loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
            warp_rst_1 = inverse_warp(closest_image_1, closest_depth_1.detach(), rendered_depth_pseudo.unsqueeze(0).detach(), closest_cam_1.extrinsic_matrix, pseudo_cam.extrinsic_matrix, closest_cam_1.intrinsic_matrix)
            if(args.add_pixel_guidance and iteration > args.pixel_guidance_start_iter):
                warped_masked_strict_image = warp_rst_1["warped_img"] * (warp_rst_1["mask_warp"] & warp_rst_1["mask_depth_strict"])
                pseudo_masked_strict_image = rendered_img_pseudo * (warp_rst_1["mask_warp"] & warp_rst_1["mask_depth_strict"])
                Ll1_masked_pseudo =  l1_loss_mask(pseudo_masked_strict_image, warped_masked_strict_image.detach())
                loss += args.pixel_pseudo_weight * Ll1_masked_pseudo
            if((args.add_warp_sds_guidance or args.add_warp_sds_guidance_2 or args.add_sds_guidance) and iteration > args.warp_sds_guidance_start_iter):
                sd_mask_1 = warp_rst_1["mask_inv"].unsqueeze(0).unsqueeze(0)
                sd_img_1 = warp_rst_1["warped_img"].unsqueeze(0)
                sd_mask_1_512 = torch.nn.functional.interpolate(sd_mask_1.float(), size=(512, 512), mode='bilinear', align_corners=False)
                sd_mask_1_512 = (sd_mask_1_512 > 0.5).float()
                sd_img_1_512 = torch.nn.functional.interpolate(sd_img_1, size=(512, 512), mode='bilinear', align_corners=False)
                rendered_img_pseudo_BCHW = rendered_img_pseudo.unsqueeze(0)
                rendered_img_pseudo_512 = torch.nn.functional.interpolate(rendered_img_pseudo_BCHW, size=(512, 512), mode='bilinear', align_corners=False)
                if args.use_lora:
                    print('no implementation!')
                    exit(0)
                if args.use_lora_2:
                    print('no implementation!')
                    exit(0)
                if args.add_warp_sds_guidance and (not args.add_warp_sds_guidance_2):        
                    loss_warp_sds_1 = sd_guidance.cal_warp_sds_grad(
                        image=sd_img_1_512,
                        mask_image=sd_mask_1_512,
                        rendered_image=rendered_img_pseudo_512, 
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale
                    )
                    loss += loss_scale * args.warp_sds_pseudo_weight * loss_warp_sds_1
                if args.add_warp_sds_guidance_2 and args.add_warp_sds_guidance:        
                    loss_warp_sds_1, loss_warp_sds_2 = sd_guidance.cal_warp_sds_grad_2_2(
                        image=sd_img_1_512,
                        mask_image=sd_mask_1_512,
                        rendered_image=rendered_img_pseudo_512, 
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale
                    )
                    loss += loss_scale * args.warp_sds_pseudo_weight * loss_warp_sds_1
                    loss += loss_scale * args.warp_sds_pseudo_weight_2 * loss_warp_sds_2
                if args.add_sds_guidance:
                    loss_sds_1 = sd_guidance.cal_sds_grad(
                        image=sd_img_1_512,
                        mask_image=sd_mask_1_512,
                        rendered_image=rendered_img_pseudo_512, 
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale
                    )
                    loss += loss_scale * args.sds_pseudo_weight * loss_sds_1
            rendered_depth_pseudo = rendered_depth_pseudo.reshape(-1, 1)
            midas_depth_pseudo = midas_depth_pseudo.reshape(-1, 1)
            depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo, -midas_depth_pseudo)).mean()
            if torch.isnan(depth_loss_pseudo).sum() == 0:
                loss += loss_scale * args.depth_pseudo_weight * depth_loss_pseudo
        loss.backward()

        if(iteration % args.sample_pseudo_interval == 0 and iteration > args.start_sample_pseudo and iteration < args.end_sample_pseudo):
            loss_dict = {
                "l1_loss": Ll1.item(), 
                "depth_loss": depth_loss.item(), 
                "depth_loss_pseudo": depth_loss_pseudo.item(), 
            }
            training_step_report(tb_writer, iteration, **loss_dict)
            if args.add_pixel_guidance and iteration > args.pixel_guidance_start_iter:
                loss_dict = {
                    "Ll1_masked_pseudo": Ll1_masked_pseudo.item(),
                }
                training_step_report(tb_writer, iteration, **loss_dict)
            if args.add_warp_sds_guidance and iteration > args.warp_sds_guidance_start_iter:
                loss_dict = {
                    "loss_warp_sds_1": loss_warp_sds_1.item(),
                }
                training_step_report(tb_writer, iteration, **loss_dict)
            if args.add_warp_sds_guidance_2 and iteration > args.warp_sds_guidance_start_iter:
                loss_dict = {
                    "loss_warp_sds_2": loss_warp_sds_2.item(),
                }
                training_step_report(tb_writer, iteration, **loss_dict)
            if args.add_sds_guidance and iteration > args.warp_sds_guidance_start_iter:
                loss_dict = {
                    "loss_sds_1": loss_sds_1.item(),
                }
                training_step_report(tb_writer, iteration, **loss_dict)
        
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, render, (pipe, background), args)

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            if  iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if args.origin:
                        size_threshold = None
                        gaussians.densify_and_prune_origin(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration)
                    else:
                        size_threshold = None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration)
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
            gaussians.update_learning_rate(iteration)
            if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                    iteration > args.start_sample_pseudo:
                if args.origin:
                    gaussians.reset_opacity_origin(args.reset_param)
                else:
                    gaussians.reset_opacity(args.reset_param)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_step_report(tb_writer, iteration, **loss_dict):
    if tb_writer:
        for k in loss_dict.keys():
            add_scalar_name = f'view_all_loss_training/{str(k)}'
            tb_writer.add_scalar(add_scalar_name, loss_dict[k], iteration)

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs, args):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                if(config['name'] == 'test'):
                    l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                    for idx, viewpoint in enumerate(config['cameras']):
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if tb_writer and (idx < 8):
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        l1_test += l1_loss(image, gt_image).mean().double()
                        _mask = None
                        _psnr = psnr(image, gt_image, _mask).mean().double()
                        _ssim = ssim(image, gt_image, _mask).mean().double()
                        _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                        psnr_test += _psnr
                        ssim_test += _ssim
                        lpips_test += _lpips
                        
                    psnr_test /= len(config['cameras'])
                    ssim_test /= len(config['cameras'])
                    lpips_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])
                    print_text = "\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                        iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test)
                    print(print_text)
                    process_log(args.model_path, text=print_text)
                    record_training(args, init=False, iter_num=iteration, psnr_num=psnr_test.item(), ssim_num=ssim_test.item(), lpips_num=lpips_test.item())
                    if tb_writer:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    ap = AblateParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_00, 10_00, 15_00, 20_00, 30_00, 40_00, 50_00, 60_00, 70_00, 80_00, 90_00, 100_00])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20_00, 50_00, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[20_00, 50_00, 10_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--train_bg", action="store_true")
    parser.add_argument('--my_debug', action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(args.test_iterations)
    print("Optimizing " + args.model_path)
    args.test_iterations = [idx for idx in range(500, args.iterations + 1, 500)]
    if(args.my_debug):
        pycode = f'rm -rf {args.model_path}'
        print(pycode)
        os.system(pycode)
        os.makedirs(args.model_path, exist_ok=True)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), ap.extract(args), args)
    print("\nTraining complete.")