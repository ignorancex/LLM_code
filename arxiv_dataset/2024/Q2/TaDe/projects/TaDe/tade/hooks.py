from mmengine.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class EpochHook(Hook):
    def before_val_epoch(self, runner) -> None:
        runner.model.epoch = runner.epoch
