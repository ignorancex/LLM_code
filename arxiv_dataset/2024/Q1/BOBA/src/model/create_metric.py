import torch


def create_metric(name='acc'):
    """
    metric function can be any function with scalar output.
    """
    if name == 'acc':  # accuracy for multi-class
        return lambda logits, target: logits.argmax(dim=1).eq(target).float().mean()
    elif name == 'err':  # error rate for multi-class
        return lambda logits, target: logits.argmax(dim=1).ne(target).float().mean()
    elif name == 'bacc':  # accuracy for binary classification
        return lambda logits, target: logits.view(-1).ge(0).eq(target.view(-1)).float().mean()
    elif name == 'berr':  # error rate for binary classification
        return lambda logits, target: logits.view(-1).ge(0).ne(target.view(-1)).float().mean()
    else:
        raise NotImplementedError('Unknown metric name: %s' % name)
