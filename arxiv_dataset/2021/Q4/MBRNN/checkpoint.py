import logging
import os
import shutil
import time

import torch

model_state = 'model_state.pt'
trainer_state = 'trainer_state.pt'


class Checkpoint():

    def __init__(self, step, epoch, model, optim, path=None, opt=None):
        self.step = step
        self.epoch = epoch
        self.model = model
        self.optim = optim

        self._path = path
        self.opt = opt

        self.logger = logging.getLogger(__name__)

    @property
    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved.")
        return self._path

    @classmethod
    def load(cls, model, optim=None, opt=None):
        logger = logging.getLogger(__name__)

        all_times = sorted(os.listdir(opt.ckpt_fd), reverse=True)
        fchckpt = os.path.join(opt.ckpt_fd, all_times[0])
        logger.info("load checkpoint from %s" % fchckpt)

        resume_model = torch.load(os.path.join(fchckpt, model_state),
                                  map_location=opt.device)
        resume_checkpoint = torch.load(os.path.join(fchckpt, trainer_state),
                                       map_location=opt.device)

        model.load_state_dict(resume_model)
        if optim is not None:
            optim.load_state_dict(resume_checkpoint['optimizer'])

        return Checkpoint(step=resume_checkpoint['step'],
                          epoch=resume_checkpoint['epoch'],
                          model=model,
                          optim=optim,
                          path=opt.ckpt_fd)

    def save(self):
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        path = os.path.join(self.opt.ckpt_fd, date_time)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        torch.save(
            {'epoch': self.epoch,
             'step': self.step,
             'optimizer': self.optim.state_dict()},
            os.path.join(path, trainer_state))
        torch.save(
            self.model.state_dict(), os.path.join(path, model_state))

        log_msg = "Validation loss being smaller than previous "
        log_msg += "minimum, checkpoint is saved at %s" % path
        self.logger.info(log_msg)
        return path
