from torchmetrics import Accuracy, ConfusionMatrix, F1Score, MeanMetric


class Monitor:

    def __init__(self, logger, num_labels, mode = 'train'):
        self.logger = logger
        self.mode = mode
        self.accuracy = Accuracy()
        self.conf_matrix = ConfusionMatrix(num_labels)
        self.f1_score = F1Score()
        self.loss = MeanMetric()
        self.cls_metrics = {
            'acc': self.accuracy,
            'conf_matrix': self.conf_matrix,
            'f1': self.f1_score,
        }

    def update(self, preds, labels, loss):
        labels = labels.cpu()
        preds = preds.detach().cpu()
        loss = loss.detach().cpu()
        for metric in self.cls_metrics.values():
            metric.update(preds, labels)
        self.loss.update(loss)

    def reset(self):
        for metric in self.cls_metrics.values():
            metric.reset()
        self.loss.reset()

    def emit(self, epoch, reset=True, **kwargs):
        values = {
            'acc': round(self.accuracy.compute().item(), 4),
            'loss': round(self.loss.compute().item(), 5),
            'f1': round(self.f1_score.compute().item(), 4),
            'conf_matrix': self.conf_matrix.compute(),
            'epoch': epoch,
            **kwargs
        }
        self.log(values)
        if reset:
            self.reset()
        return values

    def log(self, values):
        if self.mode == 'train':
            exclude_list = ['epoch', 'conf_matrix']
            log_string = f'[TRAIN] Epoch {values["epoch"]+1:04d} '
            log_string += ' '.join(
                [f'{k}: {v}' for k, v in values.items() if k not in exclude_list]
            )
            self.logger.info(log_string)
        else:
            exclude_list = ['epoch', 'conf_matrix']
            log_string = f'[{self.mode.upper()}] Epoch {values["epoch"]+1:04d} '
            log_string += ' '.join(
                [f'{k}: {v}' for k, v in values.items() if k not in exclude_list]
            )
            self.logger.info(log_string)
            self.logger.info('Confusion matrix:\n' + str(values['conf_matrix'].numpy()))
