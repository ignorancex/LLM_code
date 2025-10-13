from .mmsafetybench_loader import mmsafetybenchLoader
from .advbench_loader import advbenchLoader
from .advbench_subset_loader import advbench_Subset_Loader
from .harmbench_loader import harmbenchLoader

class DataLoaderFactory:
    @staticmethod
    def get_loader(config):
        dataset_name = config['dataset_name']
        if dataset_name == 'mmsafetybench':
            return mmsafetybenchLoader(config)
        if dataset_name == 'advbench':
            return advbenchLoader(config)
        if dataset_name == 'advbench_subset':
            return advbench_Subset_Loader(config)
        if dataset_name == 'harmbench':
            return harmbenchLoader(config)
        # elif dataset_name == 'DatasetB':
        #     return DatasetBLoader()
        # elif dataset_name == 'DatasetC':
        #     return DatasetCLoader()
        else:
            raise ValueError(f"未知的数据集类型: {dataset_name}")