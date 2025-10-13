import torch
import numpy as np

from get_from_config import get_dataset_from_config, get_module_from_config
from torch.utils.data import Subset
from method.deq_cal_alter_pmri import ImageUpdate, ParameterUpdate, GenericAccelerator, NesterovAccelerator, AndersonAccelerator
from method.warmup import load_warmup


def run(config):
    tra_dataset, val_dataset, tst_dataset = get_dataset_from_config(config)
    tst_dataset = Subset(tst_dataset, [30])

    x_operator = ImageUpdate(get_module_from_config(config, type_='x'), config)
    theta_operator = ParameterUpdate(get_module_from_config(config, type_='theta'), config)

    x_pattern = config['method']['deq_cal']['warmup']['x_ckpt']
    if x_pattern is not None:
        load_warmup(
            target_module=x_operator.cnn,
            dataset=config['setting']['dataset'],
            gt_type='x',
            pattern=x_pattern,
            sigma=config['method']['deq_cal']['warmup']['x_sigma'],
            prefix='net.',
        )

    theta_pattern = config['method']['deq_cal']['warmup']['theta_ckpt']
    if theta_pattern is not None:
        load_warmup(
            target_module=theta_operator.cnn,
            dataset=config['setting']['dataset'],
            gt_type='theta',
            pattern=theta_pattern,
            sigma=config['method']['deq_cal']['warmup']['theta_sigma'],
            prefix='net.'
        )

    accelerator_dict = {
        'generic': lambda x_init: GenericAccelerator(x_init),
        'nesterov': lambda x_init: NesterovAccelerator(x_init),
        'anderson': lambda x_init: AndersonAccelerator(x_init),
    }

    is_joint_cal = False

    for batch_idx in range(len(tst_dataset)):
        x_input, theta_input, y, mask, x_gt, theta_gt = [torch.unsqueeze(i, 0) for i in tst_dataset[batch_idx]]

        x_accelerator = accelerator_dict[config['method']['deq_cal']['accelerator']](x_input)
        theta_accelerator = accelerator_dict[config['method']['deq_cal']['accelerator']](theta_input)

        x_pre, theta_pre = x_input, theta_input
        theta_label = theta_input

        for _ in range(100):

            random_idx = np.random.randint(0, 2)

            if is_joint_cal and random_idx == 0:
                theta_hat, _ = theta_accelerator(
                    theta_operator.forward, theta_pre,
                    x=x_pre, mask=mask, y=y, theta_label=theta_label
                )
            else:
                pass

    exit(0)

    x_accelerator = accelerator_dict[config['method']['deq_cal']['accelerator']](x_pre)
    theta_accelerator = accelerator_dict[config['method']['deq_cal']['accelerator']](theta_pre)
