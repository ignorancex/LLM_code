from .MLP import TwoNN
from .CNN import ShallowCNN
from .GRU import ShallowGRU


def create_model(args):
    """
    Create model
    :param args:
    :return:
    """
    shape_in = args.shape_in
    shape_out = args.shape_out

    if args.model == '2nn':
        model = TwoNN(shape_in=shape_in, shape_out=shape_out)
    elif args.model == 'cnn':
        model = ShallowCNN(shape_in=shape_in, shape_out=shape_out)
    elif args.model == 'gru':
        model = ShallowGRU(embedding=args.embed.weights, shape_out=shape_out)
    else:
        raise NotImplementedError('Unknown model. ')

    model.to(args.device)

    return model
