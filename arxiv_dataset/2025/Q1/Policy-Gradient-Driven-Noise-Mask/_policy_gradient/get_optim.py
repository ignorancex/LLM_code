import torch

def get_optim(model, policy_net, args):
    criterion = torch.nn.CrossEntropyLoss()
    
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
        pol_optimizer = torch.optim.SGD(
            policy_net.parameters(),
            lr=args.lr / 10.0,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )         
        
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        pol_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pol_optimizer, T_max=args.epochs, eta_min=args.lr_min)

    return criterion, optimizer, pol_optimizer, main_lr_scheduler, pol_lr_scheduler
    
