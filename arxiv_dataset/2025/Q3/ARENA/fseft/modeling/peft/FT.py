

def set_peft(model, args):
    # Freeze params
    freeze_params(model, args)


def set_training_mode(model, args):
    model.train()


def set_bb(args):
    return []


def freeze_params(model, args):
    print("UnFreezing weights... ", end="\n")
    for name, param in model.named_parameters():
        param.requires_grad = True
