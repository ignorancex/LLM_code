from losses.batch_based_classification_loss import BatchBasedClassificationLoss

def loss_factory(config):
    loss_dict = {
        BatchBasedClassificationLoss.code(): BatchBasedClassificationLoss(),
    }
    try:
        return {'metric_loss': loss_dict[config['metric_loss']]}
    except KeyError:
        raise ValueError("Expected metric loss function, but got {}".format(config['metric_loss']))
