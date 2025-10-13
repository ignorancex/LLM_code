import torch
import torch.nn.functional as F
import random

# ---------------------- Data Augmentation Functions ---------------------- #
class DataAugmentTorchBase:
    """
    Base class for data augmentation for audio spectrograms using torch.Tensor.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, x: torch.Tensor):
        if self.always_apply:
            return self.apply(x)
        else:
            if torch.rand(1).item() < self.p:
                return self.apply(x)
            else:
                return x

    def apply(self, x: torch.Tensor):
        raise NotImplementedError


class CompositeCutoutTorch(DataAugmentTorchBase):
    """
    This augmentation randomly applies one of the following: RandomCutoutTorch, SpecAugmentTorch, or RandomCutoutHoleTorch.

    Set `use_itfm` to `True` if wish to use the Inter-channel Level-Aware Time-Frequency Masking (ITFM)
    """
    def __init__(self, always_apply: bool = True, p: float = 0.5, aug_channels: int = 2, use_itfm: bool = False):
        super().__init__(always_apply, p)
        self.random_cutout      = RandomCutoutTorch(always_apply=True, aug_channels=aug_channels, use_itfm=use_itfm)
        self.spec_augment       = SpecAugmentTorch(always_apply=True, aug_channels=aug_channels, use_itfm=use_itfm)
        self.random_cutout_hole = RandomCutoutHoleTorch(always_apply=True, aug_channels=aug_channels, use_itfm=use_itfm)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        choice = random.randint(0, 2)
        if choice == 0:
            return self.random_cutout_hole(x)
        elif choice == 1:
            return self.spec_augment(x)
        elif choice == 2:
            return self.random_cutout(x)
        else:
            return x  # fallback


class CompositeFrequencyTorch(DataAugmentTorchBase):
    """
    This augmentation randomly applies one of the following: RandomShiftUpDownTorch or FilterAugmentNormalized.
    """
    def __init__(self, always_apply: bool = True, p: float = 0.5, aug_channels: int = 2):
        super().__init__(always_apply, p)
        self.freq_shift     = RandomShiftUpDownTorch(always_apply=True, aug_channels=aug_channels)
        self.filter_augment = FilterAugmentNormalized(always_apply=True, aug_channels=aug_channels)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        choice = random.randint(0, 1)
        if choice == 0:
            return self.freq_shift(x)
        elif choice == 1:
            return self.filter_augment(x)
        else:
            return x  # fallback


class CompositeEverythingTorch(DataAugmentTorchBase):
    """
    This augmentation randomly applies one of the following:
        - Masking Operations (TFM)
            - RandomCutoutTorch
            - SpecAugmentTorch
            - RandomCutoutHoleTorch

        - Frequency Manipulations (FQM)
            - RandomShiftUpDownTorch
            - FilterAugmentNormalized
    """
    def __init__(self, always_apply: bool = True, p: float = 0.5, aug_channels: int = 2, use_itfm: bool = False):
        super().__init__(always_apply, p)

        # Hybrid/Combination Augmentations
        self.comp_mask = CompositeCutoutTorch(always_apply=True, aug_channels=aug_channels, use_itfm=use_itfm)
        self.comp_freq = CompositeFrequencyTorch(always_apply=True, aug_channels=aug_channels)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        choice = random.randint(0, 1)
        if choice == 0:
            return self.comp_mask(x)
        elif choice == 1:
            return self.comp_freq(x)
        else:
            return x  # fallback


class RandomCutoutTorch(DataAugmentTorchBase):
    """
    This augmentation randomly cutouts a rectangular area from the input spectrogram.
    Supports multi-channel (3D and 4D) tensors.

    Parameters:
        always_apply (bool): If True, the augmentation is always applied.
        p (float): Probability with which the augmentation is applied.
        random_value (float or None): The value used to fill the cutout area. If None, a random value between the min and max of the input is used.
        aug_channels (int): The number of channels to which the augmentation will be applied, starting from the first channel.
        cutout_size_ratio (tuple): A tuple (min_ratio, max_ratio) defining the fraction of the respective dimension to be used for the cutout patch size.
        use_itfm (bool): If True, use the ILD-aware Time-Frequency Masking variation. 
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, random_value: float = None, 
                 aug_channels: int = 2, cutout_size_ratio: tuple = (0.05, 0.2), use_itfm: bool = False):

        super().__init__(always_apply, p)
        self.random_value = random_value
        self.aug_channels = aug_channels
        self.cutout_size_ratio = cutout_size_ratio
        self.use_itfm = use_itfm

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A tensor with shape ((batch_size), channels, time_steps, freq_bins)
        Returns:
            torch.Tensor: The augmented tensor with a random cutout applied.
        """
        # Validate input dimensions.
        if x.dim() == 3:
            n_channels, time_steps, freq_bins = x.shape
        elif x.dim() == 4:
            batch_size, n_channels, time_steps, freq_bins = x.shape
        else:
            raise ValueError("Input must be a 3D tensor with shape (channels, time_steps, freq_bins) or 4D tensor with added batch dimension.")

        # Clone the input tensor to ensure original data isn't modified.
        augmented = x.clone()

        # Compute the overall min and max values for fill value generation if needed.
        min_val = x.min().item()
        max_val = x.max().item()

        # Calculate the minimum and maximum patch sizes in both dimensions.
        min_ratio, max_ratio = self.cutout_size_ratio
        patch_time_min = max(1, int(min_ratio * time_steps))
        patch_time_max = min(time_steps, max(1, int(max_ratio * time_steps)))
        patch_freq_min = max(1, int(min_ratio * freq_bins))
        patch_freq_max = min(freq_bins, max(1, int(max_ratio * freq_bins)))

        # Ensure the patch dimensions are valid relative to the input dimensions.
        if patch_time_min > time_steps or patch_freq_min > freq_bins:
            raise ValueError("Cutout size ratio results in patch dimensions larger than the input tensor.")

        # Randomly select patch dimensions.
        patch_time = random.randint(patch_time_min, patch_time_max)
        patch_freq = random.randint(patch_freq_min, patch_freq_max)

        # Randomly select top-left position such that the patch fits entirely.
        time_start = random.randrange(0, time_steps - patch_time + 1)
        freq_start = random.randrange(0, freq_bins - patch_freq + 1)

        # Determine fill value.
        c = self.random_value if self.random_value is not None else random.uniform(min_val, max_val)

        # Determine the Inter-channel Level Differences
        if self.use_itfm:
            if x.dim() == 3:
                ild = x[1, :, :] - x[0, :, :]
            else:
                ild = x[:, 1, :, :] - x[:, 0, :, :]

        if x.dim() == 3:
            augmented[:self.aug_channels, time_start:time_start + patch_time, freq_start:freq_start + patch_freq] = c
            if self.use_itfm:
                augmented[1, time_start:time_start + patch_time, freq_start:freq_start + patch_freq] += ild[time_start:time_start + patch_time, freq_start:freq_start + patch_freq]
        else:
            augmented[:, :self.aug_channels, time_start:time_start + patch_time, freq_start:freq_start + patch_freq] = c
            if self.use_itfm:
                augmented[:, 1, time_start:time_start + patch_time, freq_start:freq_start + patch_freq] += ild[:, time_start:time_start + patch_time, freq_start:freq_start + patch_freq]

        return augmented


class SpecAugmentTorch(DataAugmentTorchBase):
    """
    This augmentation randomly removes vertical or horizontal stripes (time or frequency) from a multi-channel spectrogram.

    Parameters:
        always_apply (bool): If True, always apply the transformation; otherwise, apply with probability `p`.
        p (float): Probability with which the transformation is applied when always_apply is False.
        time_max_width (int or None): Maximum width (in time steps) of each time stripe to remove. If None, defaults to 5% of the total number of time steps.
        freq_max_width (int or None): Maximum width (in frequency bins) of each frequency stripe to remove. If None, defaults to 5% of the total number of frequency bins.
        n_time_stripes (int): Number of vertical (time) stripes to remove. If None, defaults to 2.
        n_freq_stripes (int): Number of horizontal (frequency) stripes to remove. If None, defaults to 2.
        aug_channels (int): Number of channels to augment (starting from the first channel).
        use_itfm (bool): If True, use the ILD-aware Time-Frequency Masking variation. 
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, time_max_width: int = None, freq_max_width: int = None, 
                 n_time_stripes: int = None, n_freq_stripes: int = None, aug_channels: int = 2, use_itfm: bool = False):
        super().__init__(always_apply, p)
        self.time_max_width = time_max_width
        self.freq_max_width = freq_max_width
        self.n_time_stripes = n_time_stripes
        self.n_freq_stripes = n_freq_stripes
        self.aug_channels = aug_channels
        self.use_itfm = use_itfm

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A tensor with shape ((batch_size), channels, time_steps, freq_bins)
        Returns:
            torch.Tensor: The augmented tensor with a random cutout applied.
        """
        if x.dim() == 3:
            n_channels, time_steps, freq_bins = x.shape
        elif x.dim() == 4:
            batch_size, n_channels, time_steps, freq_bins = x.shape
        else:
            raise ValueError("Input must be a 3D tensor with shape (channels, time_steps, freq_bins) or 4D tensor with added batch dimension.")

        min_value = x.min().item()
        max_value = x.max().item()

        n_freq_stripes = self.n_freq_stripes if self.n_freq_stripes is not None else 2
        n_freq_stripes = max(1, n_freq_stripes)
        n_time_stripes = self.n_time_stripes if self.n_time_stripes is not None else 2
        n_time_stripes = max(1, n_time_stripes)

        time_max_width = self.time_max_width if self.time_max_width is not None else int(0.05 * time_steps)
        time_max_width = max(1, time_max_width)
        freq_max_width = self.freq_max_width if self.freq_max_width is not None else int(0.05 * freq_bins)
        freq_max_width = max(1, freq_max_width)

        new_spec = x.clone()

        if self.use_itfm:
            if x.dim() == 3:
                ild = x[1, :, :] - x[0, :, :]
            else:
                ild = x[:, 1, :, :] - x[:, 0, :, :]

        # Remove frequency stripes.
        for _ in range(n_freq_stripes):
            dur = random.randrange(1, freq_max_width) if freq_max_width > 1 else 1
            start_idx = random.randrange(0, freq_bins - dur) if freq_bins - dur > 0 else 0
            random_val = random.uniform(min_value, max_value)
            if x.dim() == 3:
                new_spec[:self.aug_channels, :, start_idx:start_idx + dur] = random_val
                if self.use_itfm:
                    new_spec[1, :, start_idx:start_idx + dur] += ild[:, start_idx:start_idx + dur]
            else:
                new_spec[:, :self.aug_channels, :, start_idx:start_idx + dur] = random_val
                if self.use_itfm:
                    new_spec[:, 1, :, start_idx:start_idx + dur] += ild[:, :, start_idx:start_idx + dur]

        # Remove time stripes.
        for _ in range(n_time_stripes):
            dur = random.randrange(1, time_max_width) if time_max_width > 1 else 1
            start_idx = random.randrange(0, time_steps - dur) if time_steps - dur > 0 else 0
            random_val = random.uniform(min_value, max_value)
            if x.dim() == 3:
                new_spec[:self.aug_channels, start_idx:start_idx + dur, :] = random_val
                if self.use_itfm:
                    new_spec[1, start_idx:start_idx + dur, :] += ild[start_idx:start_idx + dur, :]
            else:
                new_spec[:, :self.aug_channels, start_idx:start_idx + dur, :] = random_val
                if self.use_itfm:
                    new_spec[:, 1, start_idx:start_idx + dur, :] += ild[:, start_idx:start_idx + dur, :]

        return new_spec


class RandomCutoutHoleTorch(DataAugmentTorchBase):
    """
    Applies random small rectangular holes (cutouts) to a spectrogram tensor.

    Parameters:
        always_apply (bool): If True, always apply the transform.
        p (float): If always_apply is False, p is the probability to apply the transform.
        n_max_holes (int): Maximum number of holes to cut out per channel.
        max_h_size (int): Maximum hole size in the frequency (height) axis. If None, defaults to 8% of frequency bins.
        max_w_size (int): Maximum hole size in the time (width) axis. If None, defaults to 8% of time bins.
        filled_value (float or None): Value to fill the holes. If None, each hole is filled with a random value between the minimum and maximum of the input.
        aug_channels (int): Number of channels to augment (starting from the first channel).
        use_itfm (bool): If True, use the ILD-aware Time-Frequency Masking variation. 
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, n_max_holes: int = 8, max_h_size: int = None,
                 max_w_size: int = None, filled_value: float = None, aug_channels: int = 2, use_itfm: bool = False):
        super().__init__(always_apply, p)
        self.n_max_holes = n_max_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.filled_value = filled_value
        self.aug_channels = aug_channels
        self.use_itfm = use_itfm

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A tensor with shape ((batch_size), channels, time_steps, freq_bins)
        Returns:
            torch.Tensor: The augmented tensor with a random cutout applied.
        """
        if x.dim() == 3:
            n_channels, time_steps, freq_bins = x.shape
        elif x.dim() == 4:
            batch_size, n_channels, time_steps, freq_bins = x.shape
        else:
            raise ValueError("Input must be a 3D tensor with shape (channels, time_steps, freq_bins) or 4D tensor with added batch dimension.")

        # Compute local max_h_size and max_w_size values rather than modifying self attributes.
        max_h_size = max(self.max_h_size, 1) if self.max_h_size is not None else int(0.08 * freq_bins)
        max_w_size = max(self.max_w_size, 1) if self.max_w_size is not None else int(0.08 * time_steps)

        # Clone the input tensor to avoid modifying the original.
        new_spec = x.clone()

        # Determine min and max of the input for generating fill values.
        min_value = x.min().item()
        max_value = x.max().item()

        # Determine Inter-channel Level Differences
        if self.use_itfm:
            if x.dim() == 3:
                ild = x[1, :, :] - x[0, :, :]
            else:
                ild = x[:, 1, :, :] - x[:, 0, :, :]

        for _ in range(self.n_max_holes):
            # Choose random dimensions for the hole.
            hole_width = random.randint(1, max_w_size)   # along time axis
            hole_height = random.randint(1, max_h_size)    # along frequency axis

            # Ensure the hole fits entirely inside the spectrogram.
            time_start = random.randrange(0, time_steps - hole_width + 1) if time_steps - hole_width > 0 else 0
            freq_start = random.randrange(0, freq_bins - hole_height + 1) if freq_bins - hole_height > 0 else 0

            filled_val = self.filled_value if self.filled_value is not None else random.uniform(min_value, max_value)
            if x.dim() == 3:
                new_spec[:self.aug_channels, time_start:time_start + hole_width, freq_start:freq_start + hole_height] = filled_val
                if self.use_itfm:
                    new_spec[1, time_start:time_start + hole_width, freq_start:freq_start + hole_height] += ild[time_start:time_start + hole_width, freq_start:freq_start + hole_height]
            else:
                new_spec[:, :self.aug_channels, time_start:time_start + hole_width, freq_start:freq_start + hole_height] = filled_val
                if self.use_itfm:
                    new_spec[:, 1, time_start:time_start + hole_width, freq_start:freq_start + hole_height] += ild[:, time_start:time_start + hole_width, freq_start:freq_start + hole_height]

        return new_spec


class RandomShiftUpDownTorch(DataAugmentTorchBase):
    """
    This augmentation randomly shifts (upward/downward) the spectrogram along the frequency dimension. 
    
    Parameters:
        always_apply (bool): If True, always apply the transform.
        p (float): If always_apply is False, p is the probability to apply the transform.
        freq_shift_range (int): Maximum number of frequency bands to shift. If None, defaults to 10% of frequency bins.
        direction (str): Direction of shifting (up/down). If None, set at random.
        mode (str): How to shift. Default to 'reflect'
        aug_channels (int): Number of channels to augment (starting from the first channel).
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, freq_shift_range: int = None, 
                 direction: str = None, mode: str = 'reflect', aug_channels: int = 2):
        super().__init__(always_apply, p)
        self.freq_shift_range = freq_shift_range
        self.direction = direction
        self.mode = mode
        self.aug_channels = aug_channels

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A tensor with shape ((batch_size), channels, time_steps, freq_bins)
        Returns:
            torch.Tensor: The augmented tensor with a random cutout applied.
        """
        if x.dim() == 3:
            n_channels, time_steps, freq_bins = x.shape
        elif x.dim() == 4:
            batch_size, n_channels, time_steps, freq_bins = x.shape
        else:
            raise ValueError("Input must be a 3D tensor with shape (channels, time_steps, freq_bins) or 4D tensor with added batch dimension.")

        freq_range = self.freq_shift_range if self.freq_shift_range is not None else max(1, int(freq_bins * 0.05))
        shift_len = random.randrange(1, freq_range) if freq_range > 1 else 1
        direction = self.direction if self.direction is not None else random.choice(['up', 'down'])
        new_spec = x.clone()

        # Helper: shift along the feature (last) dimension.
        def shift_tensor(tensor, shift, direction):
            if direction == 'up':
                padded = F.pad(tensor, (shift, 0), mode=self.mode)
                return padded[:, :-shift]
            else:  # down
                padded = F.pad(tensor, (0, shift), mode=self.mode)
                return padded[:, shift:]

        # Apply shifting only on the first `aug_channels` for each sample.
        if x.dim() == 3:
            for i in range(min(self.aug_channels, n_channels)):
                new_spec[i] = shift_tensor(new_spec[i], shift_len, direction)
        elif x.dim() == 4:
            for b in range(batch_size):
                for i in range(min(self.aug_channels, n_channels)):
                    new_spec[b, i] = shift_tensor(new_spec[b, i], shift_len, direction)
        return new_spec


class FilterAugmentNormalized(DataAugmentTorchBase):
    """
    Applies FilterAugmentNormalized augmentation to normalized spectrograms by applying a 
    frequency-dependent multiplicative gain. This simulates changes in the spectral envelope 
    without assuming the input is in dB scale.

    The frequency axis is divided randomly into several bands. For each band, a gain factor is 
    randomly sampled from gain_range. In the "step" variant, the gain is constant over the band; 
    in the "linear" variant, the gain is linearly interpolated between adjacent band boundaries.

    Parameters:
        always_apply (bool): If True, the augmentation is always applied.
        p (float): Probability with which the augmentation is applied.
        aug_channels (int): Number of channels to augment (applied to the first `aug_channels` channels).
        gain_range (list): Range from which to sample gain factors. Defaults to [0.95, 1.05].
        n_band (list): Range for the number of frequency bands (default: [3, 6]) to sample uniformly from.
        min_bw (int): Minimum bandwidth (in frequency bins) per band.
        filter_type (str or float): Either "linear" or "step". If a float is provided, it is interpreted 
                                    as the probability threshold for using the "step" filter.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, aug_channels: int = 1, gain_range: list = [0.95, 1.05], 
                 n_band: list = [3, 6], min_bw: int = 6, filter_type: str = 'linear'):
        super().__init__(always_apply, p)
        self.aug_channels = aug_channels
        self.gain_range = gain_range
        self.n_band = n_band
        self.min_bw = min_bw
        self.filter_type = filter_type

        # If filter_type is provided as a probability (non-string), choose step vs. linear.
        if not isinstance(self.filter_type, str):
            if torch.rand(1).item() < self.filter_type:
                self.filter_type = "step"
                self.n_band = [2, 5]
                self.min_bw = 4
            else:
                self.filter_type = "linear"
                self.n_band = [3, 6]
                self.min_bw = 6

    def _compute_frequency_gain(self, n_freqs: int, device, dtype) -> torch.Tensor:
        """
        Computes a 1D frequency gain filter.
        """
        # Randomly choose the number of frequency bands.
        n_freq_band = torch.randint(low=self.n_band[0], high=self.n_band[1], size=(1,)).item()
        n_freq_band = max(2, n_freq_band)
        
        # Adjust the minimum bandwidth.
        current_min_bw = self.min_bw
        while n_freqs - n_freq_band * current_min_bw + 1 < 0 and current_min_bw > 1:
            current_min_bw -= 1

        valid_max = n_freqs - n_freq_band * current_min_bw + 1
        if valid_max < 1:
            return torch.ones(n_freqs, device=device, dtype=dtype)
        
        # Generate band boundaries.
        rand_boundaries = torch.randint(0, valid_max, (n_freq_band - 1,), device=device)
        rand_boundaries, _ = torch.sort(rand_boundaries)
        additional = torch.arange(1, n_freq_band, device=device)
        band_boundaries = rand_boundaries + additional * current_min_bw
        band_boundaries = torch.cat((torch.tensor([0], device=device), band_boundaries, torch.tensor([n_freqs], device=device)))

        if self.filter_type == "step":
            # For each band, sample a constant gain.
            band_gains = torch.rand((n_freq_band,), device=device, dtype=dtype) * (self.gain_range[1] - self.gain_range[0]) + self.gain_range[0]
            freq_gain = torch.zeros((n_freqs,), device=device, dtype=dtype)
            for i in range(n_freq_band):
                start = int(band_boundaries[i].item())
                end = int(band_boundaries[i+1].item())
                freq_gain[start:end] = band_gains[i]
        elif self.filter_type == "linear":
            # For each band boundary, sample gain and interpolate.
            band_gains = torch.rand((n_freq_band + 1,), device=device, dtype=dtype) * (self.gain_range[1] - self.gain_range[0]) + self.gain_range[0]
            freq_gain = torch.zeros((n_freqs,), device=device, dtype=dtype)
            for i in range(n_freq_band):
                start = int(band_boundaries[i].item())
                end = int(band_boundaries[i+1].item())
                interp_vals = torch.linspace(band_gains[i], band_gains[i+1], steps=(end - start), device=device, dtype=dtype)
                freq_gain[start:end] = interp_vals
        else:
            freq_gain = torch.ones(n_freqs, device=device, dtype=dtype)
        return freq_gain

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A tensor with shape ((batch_size), channels, time_steps, freq_bins)
        Returns:
            torch.Tensor: The augmented tensor with a random cutout applied.
        """
        if x.ndim == 3:
            n_channels, n_timesteps, n_freqs = x.shape
            new_spec = x.clone()
            # Process only the first aug_channels.
            aug = new_spec[:self.aug_channels].clone()  # shape: (aug_channels, n_time_steps, n_freqs)
            freq_gain = self._compute_frequency_gain(n_freqs, x.device, x.dtype)
            # Reshape to (1, 1, n_freqs) so that it broadcasts over time.
            freq_gain = freq_gain.view(1, 1, n_freqs)
            aug = aug * freq_gain  # multiplicative adjustment.
            new_spec[:self.aug_channels] = aug
            return new_spec

        elif x.ndim == 4:
            batch_size, n_channels, n_timesteps, n_freqs = x.shape
            new_spec = x.clone()
            # Compute one frequency gain filter common to all samples.
            freq_gain = self._compute_frequency_gain(n_freqs, x.device, x.dtype)
            freq_gain = freq_gain.view(1, 1, 1, n_freqs)  # shape: (1, 1, 1, n_freqs)
            new_spec[:, :self.aug_channels, :, :] = new_spec[:, :self.aug_channels, :, :] * freq_gain
            return new_spec
        else:
            raise ValueError("Input tensor must be 3D or 4D.")