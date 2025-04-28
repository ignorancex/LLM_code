from open_biomed.datasets.base_dataset import BaseDataset

class KnowledgeGraph(BaseDataset):
    def __init__(self, path) -> None:
        super().__init__(path)