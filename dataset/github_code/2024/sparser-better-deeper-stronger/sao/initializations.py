from sao.init_calls import *


def get_initializer(model, args):
    print(f"=> Initializing model with {args.weight_init}")
    if args.weight_init == "eco":
        model = ECO_Init(
            model,
            method=args.pruning_method,
            gain=args.gain,
            sparsity=args.sparsity,
            degree=args.degree,
            activation=args.activation,
            in_channels_0=args.in_channels_0,
            num_classes=args.num_classes,
        )
    elif args.weight_init == "delta-eco":
        model = Delta_ECO_Init(
            model,
            method=args.pruning_method,
            gain=args.gain,
            sparsity=args.sparsity,
            degree=args.degree,
            activation=args.activation,
            in_channels_0=args.in_channels_0,
            num_classes=args.num_classes,
        )
    elif args.weight_init == "delta":
        model = Delta_Init(
            model,
            method=args.pruning_method,
            gain=args.gain,
            sparsity=args.sparsity,
            degree=args.degree,
            activation=args.activation,
            in_channels_0=args.in_channels_0,
            num_classes=args.num_classes,
        )
    elif args.weight_init == "kaiming-normal":
        model = Kaiming_Init(model, args)

    return model
