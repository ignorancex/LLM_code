from .MultiLoss import MultiLoss


class MultiTaskLoss(MultiLoss):
    def __call__(self, y_hats, ys):
        losses = [
            l(y_hat, y) 
            for l,y_hat,y in zip(self.losses, y_hats, ys)
        ]
        return sum(losses)
