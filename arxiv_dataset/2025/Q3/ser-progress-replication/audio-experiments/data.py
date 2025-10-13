import opensmile
import torch

from autrainer.datasets.utils import (
    AbstractTargetTransform
)

class OpenSMILE(AbstractTransform):
    def __init__(
        self,
        feature_set: str,
        sample_rate: int,
        functionals: bool = False,
        order: int = -80,
    ) -> None:
        """Overwrite openSMILE to extract LLDs."""
        super().__init__(order=order)
        self.feature_set = feature_set
        self.sample_rate = sample_rate
        self.functionals = functionals
        if self.functionals:
            self.smile = opensmile.Smile(self.feature_set)
            self.smile_de = None
        else:
            self.smile = opensmile.Smile(
                self.feature_set,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            )
            self.smile_de = opensmile.Smile(
                self.feature_set,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas,
            )

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        feats = self.smile.process_signal(data.numpy(), self.sample_rate)
        feats = torch.from_numpy(self.smile.to_numpy(feats))
        if self.smile_de is not None:
            data_de = self.smile.process_signal(data.numpy(), self.sample_rate)
            data_de = torch.from_numpy(self.smile.to_numpy(data_de))
            feats = torch.cat((feats, data_de), axis=-2)
        return feats.squeeze()
