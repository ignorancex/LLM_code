import time
import torch
import utils
import torchvision

def train_one_epoch(model, policy_net, base_alpha, base_beta, criterion, optimizer, pol_optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    policy_net.train()    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # action
            new_alpha, new_beta, base_alpha, base_beta = policy_net(image, base_alpha, base_beta)
            _dist = torch.distributions.Beta(torch.exp(new_alpha), torch.exp(new_beta))
            _mask = _dist.sample()
            _mask_in = _mask.reshape(-1, 1, args.grid, args.grid)
            if args.grid != 224:
                _mask_in = torch.nn.functional.interpolate(_mask_in, (args.train_crop_size, args.train_crop_size))
            _mask_in = torchvision.transforms.functional.gaussian_blur(_mask_in, kernel_size=args.kernel, sigma=args.sigma)

            # envoriment   
            output = model(image * _mask_in)

            # objective
            logp = _dist.log_prob(_mask)
            loss = torch.mean(torch.log(torch.sum(logp.exp(), 1)) * criterion(output, target))

        optimizer.zero_grad()
        pol_optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                scaler.unscale_(pol_optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                nn.utils.clip_grad_norm_(policy_net.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.step(pol_optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                nn.utils.clip_grad_norm_(policy_net.parameters(), args.clip_grad_norm)
            optimizer.step()
            pol_optimizer.step()
            
        base_alpha, base_beta = base_alpha.detach(), base_beta.detach()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        
    return base_alpha, base_beta        
