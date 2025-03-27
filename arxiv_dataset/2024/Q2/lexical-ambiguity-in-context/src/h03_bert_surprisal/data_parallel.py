import torch.nn as nn


class TransparentDataParallel(nn.DataParallel):

    def save(self, *args, **kwargs):
        return self.module.save(*args, **kwargs)

    def get_target_tensor(self, *args, **kwargs):
        return self.module.get_target_tensor(*args, **kwargs)
