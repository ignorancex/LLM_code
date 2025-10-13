from typing import Optional

from mmengine.visualization.vis_backend import force_init_env, WandbVisBackend

from offsetocc.registry import VISBACKENDS


# overload WandbVisBackend class
@VISBACKENDS.register_module()
class WandbVisBackend(WandbVisBackend):

    # def _init_env(self):
    #     super()._init_env()
    #     # define our custom x axis metric
    #     self._wandb.define_metric("iter")
    #     # define which metrics will be plotted against it
    #     self._wandb.define_metric("ValCELoss", step_metric="iter")

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalar's data to wandb.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Useless parameter. Wandb does not
                need this parameter. Defaults to 0.
            file_path (str, optional): Useless parameter. Just for
                interface unification. Defaults to None.
        """
        # dirty hack to have the validation loss aligned with the training loss in wandb
        if 'iter' not in scalar_dict:
            scalar_dict['iter'] = step
        self._wandb.log(scalar_dict, commit=self._commit)
