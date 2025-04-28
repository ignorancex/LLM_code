import time
import torch
import utils
import torchvision

def evaluate(model, policy_net, base_alpha, base_beta, criterion, data_loader, args, device, print_freq=100, log_suffix=""):
    model.eval()
    policy_net.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            #new_alpha, new_beta, base_alpha, base_beta = policy_net(image, base_alpha, base_beta)
            _dist = torch.distributions.Beta(torch.exp(base_alpha), torch.exp(base_beta))
            _mask = _dist.sample()
            _mask_in = _mask.reshape(-1, 1, args.grid, args.grid)
            if args.grid != 224:
                _mask_in = torch.nn.functional.interpolate(_mask_in, (args.train_crop_size, args.train_crop_size))
            _mask_in = torchvision.transforms.functional.gaussian_blur(_mask_in, kernel_size=args.kernel, sigma=args.sigma)
                
            output = model(image * _mask_in)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg
