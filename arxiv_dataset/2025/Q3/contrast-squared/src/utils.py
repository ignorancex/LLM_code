import torch
import monai as mn
from monai.transforms.utils_pytorch_numpy_unification import clip, percentile
from monai.data.meta_obj import get_track_meta
from monai.utils.type_conversion import convert_to_dst_type, convert_to_tensor


class ClipPercentilesD(mn.transforms.MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensityRangePercentiles`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        lower: lower percentile.
        upper: upper percentile.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
        relative: whether to scale to the corresponding percentiles of [b_min, b_max]
        channel_wise: if True, compute intensity percentile and normalize every channel separately.
            default to False.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = mn.transforms.ScaleIntensityRangePercentiles.backend

    def __init__(
        self,
        keys,
        lower: float,
        upper: float,
        channel_wise: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.lower = lower
        self.upper = upper
        self.channel_wise = channel_wise

    def _normalize(self, img):
        a_min = percentile(img, self.lower)
        a_max = percentile(img, self.upper)
        img = clip(img, a_min, a_max)
        img = convert_to_tensor(img, track_meta=False)
        return img

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            img = convert_to_tensor(img, track_meta=get_track_meta())
            img_t = convert_to_tensor(img, track_meta=False)
            if self.channel_wise:
                img_t = torch.stack([self._normalize(img=d) for d in img_t])  # type: ignore
            else:
                img_t = self._normalize(img=img_t)
            d[key] = convert_to_dst_type(img_t, dst=img)[0]
        return d
