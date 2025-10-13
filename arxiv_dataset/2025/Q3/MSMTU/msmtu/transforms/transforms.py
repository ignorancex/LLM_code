import torch.nn as nn
import torch


class StandardizeTS(nn.Module):
    """
    Class to standardize our time series
    """

    def __init__(
            self,
            mean,
            std,
            dict_ancillary: dict = None,
    ):
        super().__init__()
        self.mean = mean
        self.std = std

        self.dict_ancillary = dict_ancillary

    def forward(self, rec):
        for direction in ['forward', 'backward']:
            ts_data_direction = rec['ts_data'][direction]

            ts_tensor = torch.as_tensor(ts_data_direction['values'])
            normalization = ((ts_tensor - self.mean) / self.std).tolist()  # * masks_tensor
            ts_data_direction['values'] = normalization

            rec['ts_data'][direction] = ts_data_direction

        if self.dict_ancillary is not None:
            for key, mean_std in self.dict_ancillary.items():
                rec['ancillary_data'][key] = (rec['ancillary_data'][key] - mean_std['mean']) / mean_std['std']

        return rec

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, " \
               f"dict_ancillary={self.dict_ancillary})"
