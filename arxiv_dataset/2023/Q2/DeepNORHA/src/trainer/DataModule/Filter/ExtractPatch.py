import random

from .Transform import Transform

class ExtractPatch(Transform):
    """ExtractPatch

    Extracts a patch of the given shape from the image. Set the dimension in null for the full size.
    If no top_left is given, a random one is chosen.
    
    Example:
     - Type: ExtractPatch:
       patch_shape: [32, 32]
    """
    def __init__(self, patch_shape, top_left = None, **args):
        super().__init__(**args)
        self._patch_shape = patch_shape
        self._top_left = top_left


    def prepare(self, first, others):
        if self._top_left:
            return self._top_left
        return [
            random.randint(0, img_len - (patch_len or img_len))
            for img_len, patch_len in zip(first.shape, self._patch_shape)
        ]

    def apply_one(self,  image, i, prepared):
        return image[
            tuple(slice(x, x + patch_len if patch_len else None )
            for x, patch_len, _ in zip(prepared, self._patch_shape, image.shape))
        ]
        