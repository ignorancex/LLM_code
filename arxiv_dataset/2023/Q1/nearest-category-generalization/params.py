
from utils import ExpExperiments

random_seed = list(range(1))

__all__ = ['OODRobustness', 'OutOfSampleRepr', 'OutOfSampleReprClf',
        'ThreeGauss', 'GetPreds']


class GetPreds(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = ""
        cls.experiment_fn = 'get_preds'
        grid_params = []

        base_params = {
            'dataset': [f'mnistwo{i}' for i in range(10)],
            'optimizer': ['sgd'],
            'momentum': [0.9],
            'weight_decay': [0.],
            'epochs': [70],
            'random_seed': random_seed,
        }
        arch = "CNN002"
        grid_params.append(dict(
            model=[
                f'ce-vtor2-{arch}',
                f'mixupce-vtor2-{arch}',
            ],
            norm=['2'], attack=['cwl2'], eps=[1.0], learning_rate=[0.01], batch_size=[128],
            **base_params,
        ))
        grid_params.append(dict(
            model=[
                f'trades6ce-vtor2-{arch}',
            ],
            norm=['2',], attack=['cwl2'], eps=[2.0, 4.0, 8.0],
            learning_rate=[0.01], batch_size=[128],
            **base_params,
        ))
        grid_params.append(dict(
            model=[
                f'advce-vtor2-{arch}',
            ],
            norm=['2',], attack=['cwl2'], eps=[2.0],
            learning_rate=[0.01], batch_size=[128],
            **base_params,
        ))

        base_params = {
            'dataset': ['cifar10wo0', 'cifar10wo4', 'cifar10wo9', 'cifar100coarsewo0', 'cifar100coarsewo4', 'cifar100coarsewo9', ],
            'optimizer': ['adam'],
            'momentum': [0.],
            'weight_decay': [0.],
            'epochs': [70],
            'learning_rate': [0.01],
            'random_seed': random_seed,
        }
        arch = "WRN_40_10"
        grid_params.append(dict(
            model=[
                f'ce-vtor2-{arch}', # clean training
                f'mixupce-vtor2-{arch}',
            ],
            batch_size=[64], norm=['2'], eps=[1.0], attack=['cwl2'],
            **base_params,
        ))
        grid_params.append(dict(
            model=[f'trades6ce-vtor2-{arch}',],
            batch_size=[64], norm=['2',], eps=[2., 4., 8.], attack=['cwl2'],
            **base_params,
        ))
        grid_params.append(dict(
            model=[f'advce-vtor2-{arch}',],
            batch_size=[64], norm=['2',], eps=[2.], attack=['cwl2'],
            **base_params,
        ))

        base_params = {
            'dataset': [
                'aug10-imgnet100wo0', 'aug10-imgnet100wo1', 'aug10-imgnet100wo2',
            ],
            'optimizer': ['adam'],
            'momentum': [0.],
            'weight_decay': [0.],
            'epochs': [70],
            'learning_rate': [0.01],
            'random_seed': random_seed,
        }
        archs = ["ResNet50Norm01"]
        grid_params.append(dict(
            model=sum([[
                f'ce-vtor2-{arch}', # clean training
                f'mixupce-vtor2-{arch}',
            ] for arch in archs], []),
            batch_size=[128], norm=['2'], eps=[1.0], attack=['cwl2'],
            **base_params,
        ))
        grid_params.append(dict(
            model=sum([[
                f'trades6ce-vtor2-{arch}',
            ] for arch in archs], []),
            batch_size=[128], norm=['2'], eps=[2.0, 4.0, 8.0], attack=['cwl2'],
            **base_params,
        ))
        grid_params.append(dict(
            model=sum([[
                f'advce-vtor2-{arch}',
            ] for arch in archs], []),
            batch_size=[128], norm=['2'], eps=[2.0], attack=['cwl2'],
            **base_params,
        ))

        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)


class OODRobustness(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = ""
        cls.experiment_fn = 'ood_robustness_correct'
        grid_params = []

        base_params = {
            'dataset': [f'mnistwo{i}' for i in range(10)],
            'optimizer': ['sgd'],
            'momentum': [0.9],
            'weight_decay': [0.],
            'epochs': [70],
            'random_seed': random_seed,
        }
        arch = "CNN002"
        grid_params.append(dict(
            model=[
                f'ce-vtor2-{arch}',
            ],
            norm=['2'], attack=['cwl2'], eps=[1.0], learning_rate=[0.01], batch_size=[128],
            **base_params,
        ))
        grid_params.append(dict(
            model=[
                f'trades6ce-vtor2-{arch}',
            ],
            norm=['2',], attack=['cwl2'], eps=[2.0, 4.0, 8.0],
            learning_rate=[0.01], batch_size=[128],
            **base_params,
        ))
        grid_params.append(dict(
            model=[
                f'advce-vtor2-{arch}',
            ],
            norm=['2',], attack=['cwl2'], eps=[2.0],
            learning_rate=[0.01], batch_size=[128],
            **base_params,
        ))

        #base_params = {
        #    'dataset': ['cifar10wo0', 'cifar10wo4', 'cifar10wo9', 'cifar100coarsewo0', 'cifar100coarsewo4', 'cifar100coarsewo9', ],
        #    'optimizer': ['adam'],
        #    'momentum': [0.],
        #    'weight_decay': [0.],
        #    'epochs': [70],
        #    'learning_rate': [0.01],
        #    'random_seed': random_seed,
        #}
        #arch = "WRN_40_10"
        #grid_params.append(dict(
        #    model=[
        #        f'ce-vtor2-{arch}', # clean training
        #    ],
        #    batch_size=[64], norm=['2'], eps=[1.0], attack=['cwl2'],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=[f'trades6ce-vtor2-{arch}',],
        #    batch_size=[64], norm=['2',], eps=[2., 4., 8.], attack=['cwl2'],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=[f'advce-vtor2-{arch}',],
        #    batch_size=[64], norm=['2',], eps=[2.], attack=['cwl2'],
        #    **base_params,
        #))

        base_params = {
            'dataset': [
                'aug10-imgnet100wo0', 'aug10-imgnet100wo1', 'aug10-imgnet100wo2',
            ],
            'optimizer': ['adam'],
            'momentum': [0.],
            'weight_decay': [0.],
            'epochs': [70],
            'learning_rate': [0.01],
            'random_seed': random_seed,
        }
        archs = ["ResNet50Norm01"]
        grid_params.append(dict(
            model=sum([[
                f'ce-vtor2-{arch}', # clean training
            ] for arch in archs], []),
            batch_size=[128], norm=['2'], eps=[1.0], attack=['cwl2'],
            **base_params,
        ))
        grid_params.append(dict(
            model=sum([[
                f'trades6ce-vtor2-{arch}',
            ] for arch in archs], []),
            batch_size=[128], norm=['2'], eps=[2.0, 4.0, 8.0], attack=['cwl2'],
            **base_params,
        ))
        grid_params.append(dict(
            model=sum([[
                f'advce-vtor2-{arch}',
            ] for arch in archs], []),
            batch_size=[128], norm=['2'], eps=[2.0], attack=['cwl2'],
            **base_params,
        ))

        base_params = {
            'dataset': [
                f'calcedreprold-mnistwo{i}-cwl2-128-mnistwo{i}-70-1.0-0.01-ce-vtor2-CNN002-0.9-2-sgd-0-0.0.pkl' for i in range(10)
            ],
            'optimizer': ['sgd'],
            'momentum': [0.9],
            'batch_size': [256],
            'learning_rate': [0.01],
            'weight_decay': [0.],
            'epochs': [70],
            'attack': ['cwl2', 'fbpgdl2_10000_64.0', 'fbpgdl2_10000_32.0', ],
            'random_seed': random_seed,
        }
        arch = "LargeMLP"
        grid_params.append(dict(
            model=[f'trades6ce-vtor2-{arch}',],
            norm=['2'], eps=[2., 4., 8.], **base_params,
        ))
        grid_params.append(dict(
            model=[f'advce-vtor2-{arch}',],
            norm=['2'], eps=[2.], **base_params,
        ))
        grid_params.append(dict(
            model=[f'ce-vtor2-{arch}',],
            norm=['2'], eps=[1.0], **base_params,
        ))

        base_params = {
            'dataset': [
                f'calcedrepr-mnistwo{i}-cwl2-128-mnistwo{i}-70-1.0-0.01-ce-vtor2-CNN002-0.9-2-sgd-0-0.0.pkl' for i in [0, 1, 4, 9]
            ],
            'optimizer': ['sgd'],
            'momentum': [0.9],
            'batch_size': [256],
            'learning_rate': [0.01],
            'weight_decay': [0.],
            'epochs': [70],
            'attack': ['signopt', 'cwl2', 'fbpgdlinf_10000_1.0', 'fbboundary_50000', 'fbboundary', 'fbl2bba', 'fbpgdl2_10000_1.0', 'fbpgdl2_10000_2.0', 'fbpgdl2', 'mtv2', 'mtv2_50', 'pgd_1000'],
            'random_seed': random_seed,
        }
        arch = "LargeMLP"
        grid_params.append(dict(
            model=[f'trades6ce-vtor2-{arch}',],
            norm=['2'], eps=[2., 4., 8.], **base_params,
        ))
        grid_params.append(dict(
            model=[f'advce-vtor2-{arch}',],
            norm=['2'], eps=[2.], **base_params,
        ))
        grid_params.append(dict(
            model=[f'ce-vtor2-{arch}',],
            norm=['2'], eps=[1.0], **base_params,
        ))

        base_params = {
            'dataset': [
                'calcedrepr-cifar10wo0-cwl2-64-cifar10wo0-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar10wo4-cwl2-64-cifar10wo4-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar10wo9-cwl2-64-cifar10wo9-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar100coarsewo0-cwl2-64-cifar100coarsewo0-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar100coarsewo4-cwl2-64-cifar100coarsewo4-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar100coarsewo9-cwl2-64-cifar100coarsewo9-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
            ],
            'optimizer': ['sgd'],
            'momentum': [0.9],
            'batch_size': [128],
            'learning_rate': [0.01],
            'weight_decay': [0.],
            'epochs': [70],
            'norm': ['2'],
            'attack': ['signopt', 'cwl2', 'fbpgdlinf_10000_1.0', 'fbboundary_50000_8.0', 'fbboundary_50000_3.0', 'fbboundary_50000_2.0',
                'fbboundary_25000_1.5', 'fbboundary_25000_2.0', 'fbboundary_50000', 'fbboundary', 'fbl2bba',
                'fbpgdl2_10000_1.0', 'fbpgdl2_10000_2.0', 'fbpgdl2', 'mtv2', 'mtv2_50', 'pgd_1000'],
            'random_seed': random_seed,
        }
        #archs = ["VggMLP", "LargeMLP", "LargeMLPv3"]
        archs = ["LargeMLP"]
        grid_params.append(dict(
            model=sum([[
                f'trades6ce-vtor2-{arch}',
            ] for arch in archs], []),
            eps=[2.0, 4.0, 8.0], **base_params,
        ))
        grid_params.append(dict(
            model=sum([[
                f'advce-vtor2-{arch}',
            ] for arch in archs], []),
            eps=[1.0], **base_params,
        ))
        grid_params.append(dict(
            model=sum([[
                f'ce-vtor2-{arch}',
            ] for arch in archs], []),
            eps=[1.0], **base_params,
        ))

        base_params = {
            'optimizer': ['sgd'],
            'momentum': [0.9],
            'batch_size': [128],
            'weight_decay': [0.],
            'epochs': [70],
            'random_seed': random_seed,
            'dataset': [
                'calcedrepr-aug10-imgnet100wo0-cwl2-128-aug10-imgnet100wo0-70-1.0-0.01-ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-aug10-imgnet100wo1-cwl2-128-aug10-imgnet100wo1-70-1.0-0.01-ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-aug10-imgnet100wo2-cwl2-128-aug10-imgnet100wo2-70-1.0-0.01-ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0.pkl',
            ],
            'attack': ['signopt', 'fbboundary_25000_1.0', 'fbboundary_50000', 'cwl2', 'fbpgdl2', 'mtv2_50', 'pgd_1000'],
        }
        archs = ["LargeMLPv4",]

        #grid_params.append(dict(
        #    model=sum([[
        #        f'ce-vtor2-{arch}',
        #    ] for arch in archs], []),
        #    norm=['2'], eps=[1.0], learning_rate=[0.01], **base_params,
        #))
        #grid_params.append(dict(
        #    model=sum([[
        #        f'trades6ce-vtor2-{arch}',
        #    ] for arch in archs], []),
        #    norm=['2'], eps=[2.0, 4.0, 8.0], learning_rate=[0.01], **base_params,
        #))
        grid_params.append(dict(
            model=sum([[
                f'advce-vtor2-{arch}',
            ] for arch in archs], []),
            norm=['2'], eps=[0.5], learning_rate=[0.1], **base_params,
        ))

        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)

class OutOfSampleReprClf(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "MNIST diff"
        cls.experiment_fn = 'out_of_sample'
        grid_params = []

        base_params = {
            'optimizer': ['sgd'],
            'momentum': [0.9],
            'batch_size': [128],
            'learning_rate': [0.01],
            'weight_decay': [0.],
            'epochs': [70],
            'random_seed': random_seed,
            'dataset': [
                f'calcedrepr-cifar10-madry-cifar10-cifar_nat.pt.pkl'
            ]
        }
        archs = ["LargeMLP", "LargeMLPv3"]
        grid_params.append(dict(
            model=sum([[
                f'ce-vtor2-{arch}',
                f'mixupce-vtor2-{arch}',
            ] for arch in archs], []),
            norm=['2'], attack=['cwl2'], eps=[1.0], **base_params,
        ))
        grid_params.append(dict(
            model=sum([[
                f'trades6ce-vtor2-{arch}',
            ] for arch in archs], []),
            norm=['2'], attack=['cwl2'], eps=[1.0, 2.0, 4.0, 8.0], **base_params,
        ))
        grid_params.append(dict(
            model=sum([[
                f'advce-vtor2-{arch}',
            ] for arch in archs], []),
            norm=['2'], attack=['cwl2'], eps=[1.0, 2.0, 4.0, 8.0], **base_params,
        ))

        base_params = {
            'dataset': [
                f'calcedreprold-mnistwo{i}-cwl2-128-mnistwo{i}-70-1.0-0.01-ce-vtor2-CNN002-0.9-2-sgd-0-0.0.pkl' for i in range(10)
            ],
            'optimizer': ['sgd'],
            'momentum': [0.9],
            'batch_size': [256],
            'learning_rate': [0.01],
            'weight_decay': [0.],
            'epochs': [70],
            'random_seed': random_seed,
        }
        arch = "LargeMLP"
        grid_params.append(dict(
            model=[
                f'ce-vtor2-{arch}',
                f'mixupce-vtor2-{arch}',
            ],
            norm=['2'], attack=['cwl2'], eps=[1.0], **base_params,
        ))
        grid_params.append(dict(
            model=[
                f'trades6ce-vtor2-{arch}',
            ],
            norm=['2'], attack=['cwl2'], eps=[2., 4., 8.], **base_params,
        ))
        grid_params.append(dict(
            model=[
                f'advce-vtor2-{arch}',
            ],
            norm=['2'], attack=['cwl2'], eps=[2.], **base_params,
        ))

        #grid_params.append(dict(
        #    dataset=[
        #        'calcedrepr-mnistwo0-cwl2-128-mnistwo0-70-1.0-0.01-ce-vtor2-CNN002-0.9-2-sgd-0-0.0.pkl',
        #        'calcedrepr-mnistwo4-cwl2-128-mnistwo4-70-1.0-0.01-ce-vtor2-CNN002-0.9-2-sgd-0-0.0.pkl',
        #        'calcedrepr-mnistwo9-cwl2-128-mnistwo9-70-1.0-0.01-ce-vtor2-CNN002-0.9-2-sgd-0-0.0.pkl',
        #    ],
        #    model=[
        #        #f'mixupce-vtor2-{arch}',
        #        #f'ce-vtor2-{arch}',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose.5',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose.75',
        #        #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose2',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose3',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose5',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose10',
        #        #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose20',
        #    ],
        #    norm=['2'], attack=['cwl2'], eps=[1.0], **base_params,
        #))

        #########################
        #####    CIFAR10    #####
        #########################
        #base_params = {
        #    'optimizer': ['sgd'],
        #    'momentum': [0.9],
        #    'batch_size': [128],
        #    'learning_rate': [0.001],
        #    'weight_decay': [0.],
        #    'epochs': [70],
        #    'random_seed': random_seed,
        #}
        #arch = "WRN_40_10_Block2"
        #grid_params.append(dict(
        #    dataset=['crepr-WRN_40_10-cifar100wosp1-pgd-64-cifar100wosp1-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt'] \
        #      + [f'crepr-WRN_40_10-cifar100coarsewo{i}-pgd-64-cifar100coarsewo{i}-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt' for i in range(1, 5)] \
        #      + [f'crepr-WRN_40_10-cifar10wo{i}-pgd-64-cifar10wo{i}-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt' for i in range(1, 10)],
        #    #dataset=[f'crepr-WRN_40_10-cifar100coarsewo{i}-pgd-64-cifar100coarsewo{i}-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt' for i in range(1, 5)],
        #    model=[
        #        f'mixupce-vtor2-{arch}',
        #        f'ce-vtor2-{arch}',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose10',
        #        #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose20',
        #    ],
        #    norm=['2',], attack=['cwl2'], eps=[1.0], **base_params,
        #))
        ##grid_params.append(dict(
        ##    #dataset=[f'crepr-WRN_40_10-cifar100coarsewo{i}-pgd-64-cifar100coarsewo{i}-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt' for i in range(1, 5)],
        ##    dataset=['crepr-WRN_40_10-cifar100wosp1-pgd-64-cifar100wosp1-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt'] \
        ##      + [f'crepr-WRN_40_10-cifar100coarsewo{i}-pgd-64-cifar100coarsewo{i}-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt' for i in range(1, 5)] \
        ##      + [f'crepr-WRN_40_10-cifar10wo{i}-pgd-64-cifar10wo{i}-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt' for i in range(1, 10)],
        ##    model=[
        ##        f'advce-vtor2-{arch}',
        ##        f'trades6ce-vtor2-{arch}',
        ##    ],
        ##    norm=['2',], attack=['cwl2'], eps=[2.0], **base_params,
        ##))

        #grid_params.append(dict(
        #    dataset=[
        #        'crepr-WRN_40_10-cifar10wo0-pgd-64-cifar10wo0-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt',
        #        'crepr-WRN_40_10-cifar100coarsewo0-pgd-64-cifar100coarsewo0-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt',
        #    ],
        #    model=[
        #        f'mixupce-vtor2-{arch}',
        #        f'ce-vtor2-{arch}',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose2',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose3',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose5',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose10',
        #        #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose20',
        #    ],
        #    norm=['2',], attack=['cwl2'], eps=[1.0], **base_params,
        #))
        #grid_params.append(dict(
        #    dataset=[
        #        'crepr-WRN_40_10-cifar10wo0-pgd-64-cifar10wo0-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt',
        #        'crepr-WRN_40_10-cifar100coarsewo0-pgd-64-cifar100coarsewo0-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt',
        #    ],
        #    model=[
        #        f'advce-vtor2-{arch}',
        #        f'trades6ce-vtor2-{arch}',
        #    ],
        #    norm=['2',], attack=['cwl2'], eps=[2.0], **base_params,
        #))
        #grid_params.append(dict(
        #    dataset=[
        #        'crepr-WRN_40_10-cifar10wo0-pgd-64-cifar10wo0-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt',
        #        'crepr-WRN_40_10-cifar100coarsewo0-pgd-64-cifar100coarsewo0-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt'
        #    ],
        #    model=[
        #        f'ce-vtor2-{arch}',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose2',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose3',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose5',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose10',
        #        #f'advce-vtor2-{arch}-halfclose',
        #        #f'trades6ce-vtor2-{arch}-halfclose',
        #    ],
        #    norm=['inf',], attack=['pgd_100'], eps=[1.0], **base_params,
        #))

        base_params = {
            'optimizer': ['sgd'],
            'momentum': [0.9],
            'batch_size': [128],
            'learning_rate': [0.1, 0.01],
            'weight_decay': [0.],
            'epochs': [70],
            'random_seed': random_seed,
            'dataset': [
                f'calcedrepr-cifar10wo{i}-cwl2-64-cifar10wo{i}-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl' for i in range(10)
            ] + [
                f'calcedrepr-cifar100coarsewo{i}-cwl2-64-cifar100coarsewo{i}-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl' for i in range(10)
            ],
        }
        #archs = ["VggMLP", "LargeMLP"]
        archs = ["LargeMLP", "LargeMLPv3"]
        grid_params.append(dict(
            model=sum([[
                f'ce-vtor2-{arch}',
            ] for arch in archs], []),
            norm=['2'], attack=['cwl2'], eps=[1.0], **base_params,
        ))

        base_params = {
            'optimizer': ['sgd'],
            'momentum': [0.9],
            'batch_size': [128],
            'learning_rate': [0.01],
            'weight_decay': [0.],
            'epochs': [70],
            'random_seed': random_seed,
            'dataset': [
                #'calcedrepr-svhnwo9-cwl2-64-svhnwo9-70-1.0-0.01-ce-vtor2-ResNet50-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar10-cwl2-64-cifar10-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar100coarse-cwl2-64-cifar100coarse-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar10wo0-cwl2-64-cifar10wo0-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar10wo4-cwl2-64-cifar10wo4-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar10wo9-cwl2-64-cifar10wo9-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar100coarsewo0-cwl2-64-cifar100coarsewo0-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar100coarsewo4-cwl2-64-cifar100coarsewo4-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar100coarsewo9-cwl2-64-cifar100coarsewo9-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
            ],
        }
        grid_params.append(dict(
            model=sum([[
                #f'trades6ce-vtor2-{arch}-pcaellipbatchada',
                #f'trades6ce-vtor2-{arch}-batchsubvor100randada',
                f'mixupce-vtor2-{arch}',
                f'ce-vtor2-{arch}',
                #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose',
                #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose2',
                #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose3',
                #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose5',
                #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose10',
            ] for arch in archs], []),
            norm=['2'], attack=['cwl2'], eps=[1.0], **base_params,
        ))
        grid_params.append(dict(
            model=sum([[
                f'trades6ce-vtor2-{arch}',
            ] for arch in archs], []),
            norm=['2'], attack=['cwl2'], eps=[0.5, 1.0, 2.0, 4.0, 8.0], **base_params,
        ))

        #base_params = {
        #    'optimizer': ['sgd'],
        #    'momentum': [0.9],
        #    'batch_size': [128],
        #    'learning_rate': [0.01],
        #    'weight_decay': [0.],
        #    'epochs': [70],
        #    'random_seed': random_seed,
        #    'dataset': [
        #        'calcedrepr-cifar10-cwl2-64-cifar10-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
        #        'calcedrepr-cifar100coarse-cwl2-64-cifar100coarse-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
        #        'calcedrepr-cifar10wo0-cwl2-64-cifar10wo0-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
        #        'calcedrepr-cifar10wo4-cwl2-64-cifar10wo4-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
        #        'calcedrepr-cifar10wo9-cwl2-64-cifar10wo9-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
        #        'calcedrepr-cifar100coarsewo0-cwl2-64-cifar100coarsewo0-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
        #        'calcedrepr-cifar100coarsewo4-cwl2-64-cifar100coarsewo4-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
        #        'calcedrepr-cifar100coarsewo9-cwl2-64-cifar100coarsewo9-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
        #    ],
        #}
        #grid_params.append(dict(
        #    model=sum([[
        #        f'advce-vtor2-{arch}',
        #    ] for arch in archs], []),
        #    norm=['2'], attack=['cwl2'], eps=[0.5, 1.0, 2.0], **base_params,
        #))

        base_params = {
            'optimizer': ['adam'],
            'momentum': [0.],
            'batch_size': [128],
            'learning_rate': [0.01],
            'weight_decay': [0.],
            'epochs': [70],
            'random_seed': random_seed,
            'dataset': [
                #'calcedrepr-svhnwo9-cwl2-64-svhnwo9-70-1.0-0.01-ce-vtor2-ResNet50-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar10wo0-cwl2-64-cifar10wo0-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar10wo4-cwl2-64-cifar10wo4-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar10wo9-cwl2-64-cifar10wo9-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar100coarsewo0-cwl2-64-cifar100coarsewo0-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar100coarsewo4-cwl2-64-cifar100coarsewo4-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-cifar100coarsewo9-cwl2-64-cifar100coarsewo9-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-aug10-imgnet100wo0-cwl2-128-aug10-imgnet100wo0-70-1.0-0.01-ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-aug10-imgnet100wo1-cwl2-128-aug10-imgnet100wo1-70-1.0-0.01-ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-aug10-imgnet100wo2-cwl2-128-aug10-imgnet100wo2-70-1.0-0.01-ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0.pkl',
            ],
        }
        archs = ["VggMLP"]

        grid_params.append(dict(
            model=sum([[
                #f'trades6ce-vtor2-{arch}-pcaellipbatchada',
                #f'trades6ce-vtor2-{arch}-batchsubvor100randada',
                f'mixupce-vtor2-{arch}',
                f'ce-vtor2-{arch}',
                #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose',
                #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose2',
                #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose3',
                #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose5',
                #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose10',
            ] for arch in archs], []),
            norm=['2'], attack=['cwl2'], eps=[1.0], **base_params,
        ))
        grid_params.append(dict(
            model=sum([[
                f'trades6ce-vtor2-{arch}',
            ] for arch in archs], []),
            norm=['2'], attack=['cwl2'], eps=[2.0, 4.0, 8.0, 32.0], **base_params,
        ))
        grid_params.append(dict(
            model=sum([[
                f'advce-vtor2-{arch}',
            ] for arch in archs], []),
            norm=['2'], attack=['cwl2'], eps=[2.0], **base_params,
        ))

        base_params = {
            'optimizer': ['sgd'],
            'momentum': [0.9],
            'batch_size': [128],
            'learning_rate': [0.1, 0.01],
            'weight_decay': [0.],
            'epochs': [70],
            'random_seed': random_seed,
            'dataset': [
                f'calcedrepr-aug10-imgnet100wo{i}-cwl2-128-aug10-imgnet100wo{i}-70-1.0-0.01-ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0.pkl' for i in range(10)
            ],
        }
        #archs = ["LargeMLP", "LargeMLPv3", "VggMLP"]
        archs = ["LargeMLPv4",]

        grid_params.append(dict(
            model=sum([[
                f'ce-vtor2-{arch}',
            ] for arch in archs], []),
            norm=['2'], attack=['cwl2'], eps=[1.0], **base_params,
        ))

        base_params = {
            'optimizer': ['sgd'],
            'momentum': [0.9],
            'batch_size': [128],
            'learning_rate': [0.1, 0.01],
            'weight_decay': [0.],
            'epochs': [70],
            'random_seed': random_seed,
            'dataset': [
                'calcedrepr-aug10-imgnet100-cwl2-128-aug10-imgnet100-70-1.0-0.01-ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-aug10-imgnet100wo0-cwl2-128-aug10-imgnet100wo0-70-1.0-0.01-ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-aug10-imgnet100wo1-cwl2-128-aug10-imgnet100wo1-70-1.0-0.01-ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0.pkl',
                'calcedrepr-aug10-imgnet100wo2-cwl2-128-aug10-imgnet100wo2-70-1.0-0.01-ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0.pkl',
            ],
        }
        archs = ["LargeMLPv4",]

        grid_params.append(dict(
            model=sum([[
                #f'trades6ce-vtor2-{arch}-pcaellipbatchada',
                #f'trades6ce-vtor2-{arch}-batchsubvor100randada',
                f'mixupce-vtor2-{arch}',
                f'ce-vtor2-{arch}',
                #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose',
                #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose2',
                #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose3',
                #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose5',
                #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose10',
            ] for arch in archs], []),
            norm=['2'], attack=['cwl2'], eps=[1.0], **base_params,
        ))
        grid_params.append(dict(
            model=sum([[
                f'trades6ce-vtor2-{arch}',
            ] for arch in archs], []),
            norm=['2'], attack=['cwl2'], eps=[2.0, 4.0, 8.0], **base_params,
        ))
        grid_params.append(dict(
            model=sum([[
                f'advce-vtor2-{arch}',
            ] for arch in archs], []),
            norm=['2'], attack=['cwl2'], eps=[0.5, 1.0, 2.0], **base_params,
        ))

        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)

class OutOfSampleRepr(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "OOS"
        cls.experiment_fn = 'oos_repr'
        grid_params = []

        ######################################################################
        ###############               PATHMNIST             ##################
        ######################################################################
        #base_params = {
        #    'dataset': [
        #        'pathmnistwo0',
        #        'dermamnistwo0', 'pathmnistwo1',
        #    ],
        #    'optimizer': ['adam'],
        #    'momentum': [0.],
        #    'weight_decay': [0.],
        #    'epochs': [70],
        #    'random_seed': random_seed,
        #}
        #arch = "preResNet18Norm01"
        #grid_params.append(dict(
        #    model=[
        #        f'ce-vtor2-{arch}',
        #    ],
        #    norm=['2'], attack=['cwl2'], eps=[1.0], learning_rate=[0.01], batch_size=[128],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=[
        #        f'advce-vtor2-{arch}',
        #        f'trades6ce-vtor2-{arch}',
        #    ],
        #    norm=['2',], attack=['cwl2'], eps=[0.5, 1.0],
        #    learning_rate=[0.01], batch_size=[128],
        #    **base_params,
        #))

        #base_params = {
        #    'dataset': ['cifar10wo0',],
        #    'optimizer': ['adam'],
        #    'momentum': [0.],
        #    'weight_decay': [0.],
        #    'epochs': [70],
        #    'learning_rate': [0.01],
        #    'random_seed': random_seed,
        #}
        #arch = "WRN_40_10"
        #grid_params.append(dict(
        #    model=[
        #        f'trades6ce-vtor2-{arch}', #
        #    ],
        #    batch_size=[64], norm=['2',], eps=[0.5], attack=['cwl2'],
        #    **base_params,
        #))
        #arch = "ResNet50Norm02"
        #grid_params.append(dict(
        #    model=[
        #        f'aug09-trades6ce-vtor2-{arch}', #
        #    ],
        #    batch_size=[128], norm=['2',], eps=[1.0], attack=['cwl2'],
        #    **base_params,
        #))


        #base_params = {
        #    'dataset': ['cifar10', 'cifar100coarse'],
        #    'optimizer': ['adam'],
        #    'momentum': [0.],
        #    'weight_decay': [0.],
        #    'epochs': [70],
        #    'learning_rate': [0.01],
        #    'random_seed': random_seed,
        #}
        #arch = "WRN_40_10"
        #grid_params.append(dict(
        #    model=[
        #        f'ce-vtor2-{arch}', # clean training
        #    ],
        #    batch_size=[64], norm=['2'], eps=[1.0], attack=['cwl2'],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=[f'trades6ce-vtor2-{arch}',],
        #    batch_size=[64], norm=['2',], eps=[2.], attack=['cwl2'],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=[f'advce-vtor2-{arch}',],
        #    batch_size=[64], norm=['2',], eps=[2.], attack=['cwl2'],
        #    **base_params,
        #))

        #base_params = {
        #    'dataset': [
        #        'aug10-imgnet100',
        #    ],
        #    'optimizer': ['adam'],
        #    'momentum': [0.],
        #    'weight_decay': [0.],
        #    'epochs': [70],
        #    'learning_rate': [0.01],
        #    'random_seed': random_seed,
        #}
        #archs = ["ResNet50Norm01"]
        #grid_params.append(dict(
        #    model=sum([[
        #        f'ce-vtor2-{arch}', # clean training
        #    ] for arch in archs], []),
        #    batch_size=[128], norm=['2'], eps=[1.0], attack=['cwl2'],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=sum([[
        #        f'trades6ce-vtor2-{arch}',
        #    ] for arch in archs], []),
        #    batch_size=[128], norm=['2'], eps=[2.0], attack=['cwl2'],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=sum([[
        #        f'advce-vtor2-{arch}',
        #    ] for arch in archs], []),
        #    batch_size=[128], norm=['2'], eps=[2.0], attack=['cwl2'],
        #    **base_params,
        #))

        base_params = {
            'dataset': [f'aug10-imgnet100wo{i}' for i in range(10)],
            'optimizer': ['adam'],
            'momentum': [0.],
            'weight_decay': [0.],
            'epochs': [70],
            'learning_rate': [0.01],
            'random_seed': random_seed,
        }
        archs = ["ResNet50Norm01"]
        grid_params.append(dict(
            model=sum([[
                f'ce-vtor2-{arch}', # clean training
                f'mixupce-vtor2-{arch}',
            ] for arch in archs], []),
            batch_size=[128], norm=['2'], eps=[1.0], attack=['cwl2'],
            **base_params,
        ))

        #base_params = {
        #    'dataset': [
        #        'aug10-imgnet100wo0', 'aug10-imgnet100wo1', 'aug10-imgnet100wo2',
        #        #'aug07-imgnet100wo0', 'aug07-imgnet100wo1',
        #        #'aug06-imgnet100wo0', 'aug06-imgnet100wo1', 'aug06-imgnet100wo2'
        #    ],
        #    'optimizer': ['adam'],
        #    'momentum': [0.],
        #    'weight_decay': [0.],
        #    'epochs': [70],
        #    'learning_rate': [0.01],
        #    'random_seed': random_seed,
        #}
        #archs = ["ResNet50Norm01"]
        #grid_params.append(dict(
        #    model=sum([[
        #        f'ce-vtor2-{arch}', # clean training
        #        #f'trades6ce-vtor2-{arch}',
        #    ] for arch in archs], []),
        #    batch_size=[128], norm=['2'], eps=[1.0], attack=['cwl2'],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=sum([[
        #        f'trades6ce-vtor2-{arch}',
        #    ] for arch in archs], []),
        #    batch_size=[128], norm=['2'], eps=[1.0, 2.0, 4.0, 8.0], attack=['cwl2'],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=sum([[
        #        f'advce-vtor2-{arch}',
        #    ] for arch in archs], []),
        #    batch_size=[128], norm=['2'], eps=[2.0], attack=['cwl2'],
        #    **base_params,
        #))

        base_params = {
            'dataset': [f'mnistwo{i}' for i in range(10)],
            'optimizer': ['sgd'],
            'momentum': [0.9],
            'batch_size': [128],
            'learning_rate': [0.01],
            'weight_decay': [0.],
            'epochs': [70],
            'random_seed': random_seed,
        }
        arch = "CNN002"
        grid_params.append(dict(
            model=[
                f'ce-vtor2-{arch}',
                f'mixupce-vtor2-{arch}',
            ],
            norm=['2'], attack=['cwl2'], eps=[1.0],
            **base_params,
        ))
        #grid_params.append(dict(
        #    model=[
        #        f'trades6ce-vtor2-{arch}',
        #    ],
        #    norm=['2',], attack=['cwl2'], eps=[2.0, 4.0, 8.0],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=[
        #        f'advce-vtor2-{arch}',
        #    ],
        #    norm=['2',], attack=['cwl2'], eps=[2.0],
        #    **base_params,
        #))

        #base_params = {
        #    #'dataset': ['fashionwo0', 'mnistwo9', 'mnistwo0', 'mnistwo1', 'mnistwo4', 'mnistwo7'],
        #    'dataset': ['fashionwo0', 'fashionwo4', 'fashionwo9'] + [f'mnistwo{i}' for i in range(10)],
        #    'optimizer': ['sgd'],
        #    'momentum': [0.9],
        #    'weight_decay': [0.],
        #    'epochs': [70],
        #    'random_seed': random_seed,
        #}
        #arch = "CNN002"
        #grid_params.append(dict(
        #    model=[
        #        #f'trades20ce-vtor2-{arch}-batchsubvor100',
        #        f'trades6ce-vtor2-{arch}-batchsubvor100rand',
        #        f'trades6ce-vtor2-{arch}-pcaellipbatch',
        #        #f'trades6ce-vtor2-{arch}-batchsubvor100',
        #        #f'trades20ce-vtor2-{arch}-pcaellipbatch',
        #    ],
        #    norm=['2'], attack=['cwl2'], eps=[2.0, 4.0, 8.0], learning_rate=[0.01], batch_size=[128],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=[
        #        f'ce-vtor2-{arch}',
        #    ],
        #    norm=['2'], attack=['cwl2'], eps=[1.0], learning_rate=[0.01], batch_size=[128],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=[
        #        f'advce-vtor2-{arch}',
        #        f'trades6ce-vtor2-{arch}',
        #    ],
        #    norm=['2',], attack=['cwl2'], eps=[2.0, 4.0, 8.0],
        #    learning_rate=[0.01], batch_size=[128],
        #    **base_params,
        #))

        #########################
        ####### CIFAR10 #########
        #########################

        #base_params = {
        #    'dataset': [f'cifar10wo{i}' for i in range(1, 10)] \
        #            + [f'cifar100coarsewo{i}' for i in range(1, 5)],
        #    #        + ['cifar100wosp1'] \
        #    #'dataset': [f'cifar100coarsewo{i}' for i in range(1, 5)],
        #    'optimizer': ['adam'],
        #    'momentum': [0.],
        #    'weight_decay': [0.],
        #    'epochs': [70],
        #    'learning_rate': [0.01],
        #    'random_seed': random_seed,
        #}
        #arch = "WRN_40_10"
        #grid_params.append(dict(
        #    model=[
        #        #f'mixupce-vtor2-{arch}',
        #        f'ce-vtor2-{arch}', # clean training
        #        #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose2',
        #    ],
        #    batch_size=[64], norm=['2'], eps=[1.0], attack=['cwl2'],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=[
        #        #f'advce-vtor2-{arch}', #
        #        f'trades6ce-vtor2-{arch}', #
        #    ],
        #    batch_size=[64], norm=['2',], eps=[2., 4., 8.], attack=['cwl2'],
        #    **base_params,
        #))

        base_params = {
            'dataset': [f'cifar100coarsewo{i}' for i in range(10)] + [f'cifar10wo{i}' for i in range(10)],
            'optimizer': ['adam'],
            'momentum': [0.],
            'weight_decay': [0.],
            'epochs': [70],
            'learning_rate': [0.01],
            'random_seed': random_seed,
        }
        arch = "WRN_40_10"
        grid_params.append(dict(
            model=[
                f'ce-vtor2-{arch}', # clean training
                f'mixupce-vtor2-{arch}',
            ],
            batch_size=[64], norm=['2'], eps=[1.0], attack=['cwl2'],
            **base_params,
        ))

        #base_params = {
        #    'dataset': ['cifar100coarsewo4', 'cifar100coarsewo9', 'cifar10wo0', 'cifar10wo4', 'cifar10wo9', 'cifar100coarsewo0', ],
        #    'optimizer': ['adam'],
        #    'momentum': [0.],
        #    'weight_decay': [0.],
        #    'epochs': [70],
        #    'learning_rate': [0.01],
        #    'random_seed': random_seed,
        #}
        #archs = ["WRN_40_10",]
        #grid_params.append(dict(
        #    model=sum([[
        #        f'trades6ce-vtor2-{arch}-pcaellipbatchada',
        #        f'trades6ce-vtor2-{arch}-batchsubvor50S50randada',
        #        f'trades6ce-vtor2-{arch}-batchsubvor50S75randada',
        #        f'trades6ce-vtor2-{arch}-batchsubvor50randada',
        #        #f'mixupce-vtor2-{arch}',
        #        f'ce-vtor2-{arch}', # clean training
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose2',
        #        #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose3',
        #        #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose5',
        #        #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose10',
        #    ] for arch in archs], []),
        #    batch_size=[64], norm=['2'], eps=[1.0], attack=['cwl2'],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=sum([[
        #        f'trades6ce-vtor2-{arch}', #
        #    ] for arch in archs], []),
        #    batch_size=[64], norm=['2',], eps=[.25, .5, 1., 2., 4., 8.], attack=['cwl2'],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=sum([[
        #        f'advce-vtor2-{arch}', #
        #    ] for arch in archs], []),
        #    batch_size=[64], norm=['2',], eps=[.25, .5, 1., 2.], attack=['cwl2'],
        #    **base_params,
        #))

        #base_params = {
        #    'dataset': ['cifar10wo0', 'cifar100coarsewo0', 'cifar10wo4', 'cifar100coarsewo4', 'cifar10wo9', 'cifar100coarsewo9',],
        #    'optimizer': ['adam'],
        #    'momentum': [0.],
        #    'weight_decay': [0.],
        #    'epochs': [70],
        #    'learning_rate': [0.01],
        #    'random_seed': random_seed,
        #}
        #archs = ["DenseNet161",]
        #grid_params.append(dict(
        #    model=sum([[
        #        f'ce-vtor2-{arch}', #
        #    ] for arch in archs], []),
        #    batch_size=[64], norm=['2',], eps=[1.], attack=['cwl2'],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=sum([[
        #        f'trades6ce-vtor2-{arch}', #
        #    ] for arch in archs], []),
        #    batch_size=[64], norm=['2',], eps=[2.,], attack=['cwl2'],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=sum([[
        #        f'advce-vtor2-{arch}', #
        #    ] for arch in archs], []),
        #    batch_size=[64], norm=['2',], eps=[2.], attack=['cwl2'],
        #    **base_params,
        #))

        #base_params = {
        #    'dataset': [
        #        'aug10-imgnet100wo0',
        #    ],
        #    'optimizer': ['adam'],
        #    'momentum': [0.],
        #    'weight_decay': [0.],
        #    'epochs': [70],
        #    'learning_rate': [0.01],
        #    'random_seed': random_seed,
        #}
        #archs = ["DenseNet161Norm01"]
        #grid_params.append(dict(
        #    model=sum([[
        #        f'ce-vtor2-{arch}', # clean training
        #    ] for arch in archs], []),
        #    batch_size=[64], norm=['2'], eps=[1.0], attack=['cwl2'],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=sum([[
        #        f'trades6ce-vtor2-{arch}',
        #    ] for arch in archs], []),
        #    batch_size=[48], norm=['2'], eps=[2.0], attack=['cwl2'],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=sum([[
        #        f'advce-vtor2-{arch}',
        #    ] for arch in archs], []),
        #    batch_size=[64], norm=['2'], eps=[2.0], attack=['cwl2'],
        #    **base_params,
        #))

        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)

class ThreeGauss(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "Three Gaussion"
        cls.experiment_fn = 'blind'
        grid_params = []

        base_params = {
            'dataset': [#'mulgaussv1-500-0.01', 'mulgaussv2-1000-0.01',
                        #'mulgaussv3-1000-0.01',
                        'mulgaussv4-500-0.01',
                        'mulgaussv5-500-0.01', 'mulgaussv6-500-0.01',
                        ],
            'attack': ['pgd'],
            'optimizer': ['sgd'],
            'learning_rate': [0.01],
            'epochs': [40],
            'momentum': [0.9],
            'weight_decay': [0.],
            'batch_size': [64],
            'random_seed': random_seed,
        }
        arch = "LargeMLP"
        grid_params.append(dict(
            model=[
                f'ce-vtor2-{arch}',
                f'trades6ce-vtor2-{arch}',
            ],
            norm=['inf'], eps=[1.0],
            **base_params,
        ))
        grid_params.append(dict(
            model=[
                f'trades6ce-vtor2-{arch}-halfclose',
                f'trades6ce-vtor2-{arch}',
                f'trades6ce-vtor2-{arch}-batchsubvor1000rand',
                f'trades6ce-vtor2-{arch}-batchsubvor50rand',
                f'trades6ce-vtor2-{arch}-batchsubvor50',
                f'trades6ce-vtor2-{arch}-pcaellipT0N1000batch',
                f'trades6ce-vtor2-{arch}-pcaellipT0N1000',
                f'trades6ce-vtor2-{arch}-1nnregion',
            ],
            norm=['2'], eps=[1.0, 0.1],
            **base_params,
        ))
        grid_params.append(dict(
            model=[
                f'trades6ce-vtor2-{arch}',
            ],
            norm=['inf'], eps=[0.1, 0.05],
            **base_params,
        ))

        #base_params = {
        #    'dataset': ['threegaussv2-500-0.1',],
        #    'attack': ['pgd'],
        #    'optimizer': ['sgd'],
        #    'learning_rate': [0.01],
        #    'epochs': [20],
        #    'momentum': [0.9],
        #    'weight_decay': [0.],
        #    'batch_size': [64],
        #    'random_seed': random_seed,
        #}

        #arch = "LargeMLP"
        #grid_params.append(dict(
        #    model=[
        #        f'trades6ce-vtor2-{arch}-batchsubvor1000rand',
        #        f'trades6ce-vtor2-{arch}-batchsubvor50rand',
        #        f'trades6ce-vtor2-{arch}-batchsubvor50',
        #        f'trades6ce-vtor2-{arch}-pcaellipT0N1000batch',
        #        f'trades6ce-vtor2-{arch}-pcaellipT0N1000',
        #        f'trades6ce-vtor2-{arch}-1nnregion',
        #    ],
        #    norm=['inf'], eps=[3.0, 5.0],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=[
        #        f'ce-vtor2-{arch}',
        #        f'advce-vtor2-{arch}',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose',
        #    ],
        #    norm=['inf'], eps=[1.0],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=[
        #        f'advce-vtor2-{arch}',
        #        f'trades6ce-vtor2-{arch}',
        #    ],
        #    norm=['inf'], eps=[0.3, 0.1],
        #    **base_params,
        #))

        #base_params = {
        #    'dataset': ['threegauss-1000-0.1',],
        #    #'dataset': ['fourq-10-0.1'],
        #    'norm': ['inf'],
        #    'attack': ['pgd'],
        #    'optimizer': ['sgd'],
        #    'learning_rate': [0.01],
        #    'epochs': [20],
        #    'momentum': [0.9],
        #    'weight_decay': [0.],
        #    'batch_size': [64],
        #    'random_seed': random_seed,
        #}

        #arch = "LargeMLP"
        #grid_params.append(dict(
        #    model=[
        #        f'trades6ce-vtor2-{arch}-pcaellipT1N1000batch',
        #        f'trades6ce-vtor2-{arch}-pcaellipT0N1000batch',
        #        #f'trades6ce-vtor2-{arch}-pcaellipT1N1000',
        #        f'trades6ce-vtor2-{arch}-1nnregion',
        #    ],
        #    eps=[0.5],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=[
        #        f'ce-vtor2-{arch}',
        #        #f'advce-vtor2-{arch}-halfclose',
        #        f'advce-vtor2-{arch}',
        #        f'trades6ce-vtor2-{arch}',
        #        f'trades6ce-vtor2-{arch}-pcaellipN1000',
        #        f'trades6ce-vtor2-{arch}-pcaellipT1N1000',
        #        #f'trades6ce-vtor2-{arch}-halfclose',
        #        f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose',
        #        #f'cusgradtrades6v2autwotimesce-vtor2-{arch}-halfclose2',
        #    ],
        #    eps=[1.0],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=[
        #        f'advce-vtor2-{arch}',
        #        f'trades6ce-vtor2-{arch}',
        #    ],
        #    eps=[0.3, 0.1],
        #    **base_params,
        #))
        #grid_params.append(dict(
        #    model=[
        #        f'trades6ce-vtor2-{arch}-1nnregion',
        #        f'trades6ce-vtor2-{arch}-maxellip',
        #    ],
        #    eps=[5.0],
        #    **base_params,
        #))
        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)
