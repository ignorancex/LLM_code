import torch
import numpy as np
from scipy import stats
from SIEDD.configs import TransformConfig, DenoisingType


def add_noise(
    frame: torch.Tensor,
    noise_type: DenoisingType,
    gauss_std=0.1,
    noise_amp=1,
):
    N, C, H, W = frame.size()
    frame = frame.clone()
    noise_amount = int(H * W * (1e4 / (1920 * 1080)))

    for i in range(N):
        # generate indices to place noise
        random_noise_ind = np.random.choice(W * H, size=noise_amount, replace=False)
        random_noise_x = random_noise_ind % W
        random_noise_y = random_noise_ind // W

        if noise_type in ["all_white", "all_black", "salt_pepper"]:
            if noise_type == "salt_pepper":
                noise_val = np.random.choice(2, size=noise_amount)
                noise_val = np.stack((noise_val, noise_val, noise_val), axis=1)

                frame[i, :, random_noise_y, random_noise_x] = (
                    torch.from_numpy(noise_val).float().T.to(frame.device)
                )

            elif noise_type == "all_white":
                noise_val = 1
                frame[i, :, random_noise_y, random_noise_x] = noise_val
            elif noise_type == "all_black":
                noise_val = 0
                frame[i, :, random_noise_y, random_noise_x] = noise_val

        elif noise_type == "gaussian":
            noise_val = np.random.normal(0, gauss_std, (noise_amount, 3)) * noise_amp
            frame[i, :, random_noise_y, random_noise_x] += (
                torch.from_numpy(noise_val).float().T.to(frame.device)
            )

        elif noise_type == "random":
            noise_val = (np.random.rand(noise_amount, 3) * 2 - 1) * noise_amp
            # frame[:,random_noise_y, random_noise_x] += torch.from_numpy(noise_val).float().T
            frame[i, :, random_noise_y, random_noise_x] += (
                torch.from_numpy(noise_val).float().T.to(frame.device)
            )

        frame[i].clamp_(0, 1)

    return frame


class Transform:
    def __init__(self, args: TransformConfig):
        self.inverse_map = {}
        self.args = args
        self.method = self.args.transform

    def state(self):
        return self.inverse_map

    def transform(self, data: torch.Tensor, frame: int):
        # data is CHW float on [0, 1] range
        if self.args.denoising:
            data = add_noise(data[None], self.args.denoising_type)[0]
        if self.method == "min_max":
            _min = torch.min(data)
            _max = torch.max(data)
            self.inverse_map[self.method, frame] = (_min.item(), _max.item())
            normed = (data - _min) / (_max - _min)
            normed = self._encode_shift_scale(normed)

        elif self.method == "z_score":
            _mean = data.mean()
            _std = data.std()
            normed = (data - _mean) / (_std + 1e-5)
            self.inverse_map[self.method, frame] = (_mean.item(), _std.item())

        elif self.method == "sym_power":
            _meta, normed = self._sym_power_trans(data)
            self.inverse_map[self.method, frame] = _meta

        elif self.method == "box_cox":
            _min, _max, _lambda, normed = self._box_cox(data)
            self.inverse_map[self.method, frame] = (
                _min.item(),
                _max.item(),
                _lambda.tolist(),
            )

        else:
            raise NotImplementedError

        return normed

    def inverse(self, normed: torch.Tensor, frame: int):
        if self.method == "min_max":
            # scale → shift
            normed = self._decode_shift_scale(normed)
            _min, _max = self.inverse_map[self.method, frame]
            r_data = normed * (_max - _min) + _min

        elif self.method == "z_score":
            _mean, _std = self.inverse_map[self.method, frame]
            r_data = normed * (_std + 1e-5) + _mean

        elif self.method == "sym_power":
            _meta = self.inverse_map[self.method, frame]
            r_data = self._inverse_sym_power_trans(_meta, normed)
        elif self.method == "box_cox":
            _min, _max, _lambda = self.inverse_map[self.method, frame]
            r_data = self._inverse_box_cox(_min, _max, _lambda, normed)
        else:
            raise NotImplementedError

        return r_data

    def _encode_shift_scale(self, data: torch.Tensor):
        nr0, nr1 = self.args.normalization_range
        data = data * (nr1 - nr0) + nr0
        return data

    def _decode_shift_scale(self, data: torch.Tensor):
        nr0, nr1 = self.args.normalization_range
        data = (data - nr0) / (nr1 - nr0)
        return data

    def _sym_power_trans(self, data: torch.Tensor):
        _min = torch.min(data).item()
        _max = torch.max(data).item()

        hist = torch.histc(
            data.flatten(), bins=256, min=data.min().item(), max=data.max().item()
        )
        pdf = hist / hist.sum()

        if self.args.pn_beta <= 0:
            gamma = self._get_gamma_by_half(data)
        else:
            gamma = self._get_gamma_by_edge_claibration(data)

        boundary = self.args.gamma_boundary
        if gamma > 1:
            gamma = min(boundary, gamma)
        else:
            gamma = max(1 / boundary, gamma)

        if self.args.pn_buffer < 0:  # adaptive
            _alpha_len = int(self.args.pn_alpha * 256)
            left_alpha_sum = pdf[:_alpha_len].sum().item()
            right_alpha_sum = pdf[-_alpha_len:].sum().item()
            _left_shift_len = self.args.pn_k * left_alpha_sum
            _right_shift_len = self.args.pn_k * right_alpha_sum

        else:
            _left_shift_len = (_max - _min) * self.args.pn_buffer
            _right_shift_len = (_max - _min) * self.args.pn_buffer

        _shift_len = _left_shift_len + _right_shift_len

        ## gamma transformation
        if self.args.gamma_trans >= 0:
            gamma = self.args.gamma_trans
            _left_shift_len = _right_shift_len = _shift_len = 0

        normed = (data - (_min - _left_shift_len)) / (_max - _min + _shift_len)  # [0,1]
        normed = torch.pow(normed, gamma)
        normed = self._encode_shift_scale(normed)

        return [_min, _max, gamma, _left_shift_len, _shift_len], normed

    def _inverse_sym_power_trans(self, _meta, normalized_data: torch.Tensor):
        ## zero-mean → gamma →  min-max
        normalized_data = self._decode_shift_scale(normalized_data)

        _min, _max, gamma, _left_shift_len, _shift_len = _meta

        # clip to (0,1)
        normalized_data = torch.clamp(normalized_data, min=0.0, max=1.0)

        normalized_data = torch.pow(normalized_data, 1.0 / gamma)
        r_data = normalized_data * (_max - _min + _shift_len) + (_min - _left_shift_len)
        return r_data

    def _get_gamma_by_half(self, data: torch.Tensor):
        hist = torch.histc(
            data.flatten(), bins=256, min=data.min().item(), max=data.max().item()
        )
        pdf = hist / hist.sum()
        cdf = torch.cumsum(pdf, dim=0)
        half_index = torch.searchsorted(cdf, self.args.pn_cum)  # 0.5
        half_perc = half_index / 256
        gamma = np.log(0.5) / np.log(half_perc)
        return gamma

    def _get_gamma_by_edge_claibration(self, data: torch.Tensor):
        """deviation-aware calibration"""
        half_gamma = self._get_gamma_by_half(data)
        hist = torch.histc(
            data.flatten(), bins=256, min=data.min().item(), max=data.max().item()
        )
        pdf = hist / hist.sum()

        min_max_normed = (data - data.min()) / (data.max() - data.min())
        half_gamma_normed = torch.pow(min_max_normed, half_gamma)
        half_gamma_hist = torch.histc(
            half_gamma_normed.flatten(),
            bins=256,
            min=half_gamma_normed.min().item(),
            max=half_gamma_normed.max().item(),
        )
        half_gamma_pdf = half_gamma_hist / half_gamma_hist.sum()

        _beta_len = int(self.args.pn_beta * 256)
        _minmax_bin = pdf[:_beta_len] if half_gamma > 1 else pdf[-_beta_len:]
        _half_gamma_bin = (
            half_gamma_pdf[:_beta_len]
            if half_gamma > 1
            else half_gamma_pdf[-_beta_len:]
        )

        delta_sum = _half_gamma_bin.sum() - _minmax_bin.sum()  # assert > 0
        delta_sum /= self.args.pn_beta
        if half_gamma < 1:
            delta_gamma = 1 / half_gamma - 1
        else:
            delta_gamma = half_gamma - 1

        new_delta_gamma = delta_gamma * min(delta_sum.item(), 1)

        if half_gamma < 1:
            new_gamma = 1 / (1 / half_gamma - new_delta_gamma)
        else:
            new_gamma = half_gamma - new_delta_gamma

        return new_gamma

    def _box_cox(self, data: torch.Tensor):
        ### + shift → boxcox → min-max → zero_mean
        data = data + self.args.box_shift * 256
        _shape = data.shape
        data, _lambda = stats.boxcox(data.flatten().numpy())  # type: ignore
        data = torch.tensor(data)
        data = data.reshape(_shape)

        _min = torch.min(data)
        _max = torch.max(data)
        data = (data - _min) / (_max - _min)  # [0,1]
        normed_data = self._encode_shift_scale(data)
        return _min, _max, _lambda, normed_data

    def _inverse_box_cox(self, _min, _max, _lambda, normaed: torch.Tensor):
        ### zero_mean → min-max → boxcox → - shift
        _lambda = torch.tensor(_lambda)
        normaed = self._decode_shift_scale(normaed)
        normaed = torch.clamp(normaed, min=0.0, max=1.0)
        normaed = normaed * (_max - _min) + _min

        if _lambda == 0:
            normaed = torch.exp(normaed)
        else:
            normaed = (_lambda * normaed + 1) ** (1 / _lambda)

        r_data = normaed - self.args.box_shift * 256
        return r_data
