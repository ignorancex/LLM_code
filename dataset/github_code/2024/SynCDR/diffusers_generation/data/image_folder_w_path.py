from typing import Any, Callable, Optional, Tuple
import torchvision.datasets as tvd
from torchvision.datasets.folder import default_loader
from pathlib import Path

class ImageFolderWPath(tvd.ImageFolder):
    def __init__(
        self, root: str, transform = None, 
        target_transform = None, 
        loader = default_loader, 
        is_valid_file = None):
        
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.root = root
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, str(Path(path).relative_to(self.root))