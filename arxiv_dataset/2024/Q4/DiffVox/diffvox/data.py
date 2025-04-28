import torch

class Dataset_DiffVox(torch.utils.data.Dataset):
    def __len__(self):
        return self.gt_projs.shape[-1]
    
    def __getitem__(self, idx):
        return self.gt_projs[..., idx], self.sources[:,idx, :], self.targets[:, idx, :]

    def get_data(self):
        return self.gt_projs, self.sources, self.targets, self.subject



class FastTensorDataLoader:

    def __init__(
        self, source, target, pixels, subject, init_vol=None, batch_size=None, shuffle=True, pin_memory=True, mask=0):
        assert source.shape[1] == target.shape[1] == pixels.shape[2]
        self.subject = subject
        self.source = source
        self.target = target
        self.pixels = pixels
        self.init_vol = init_vol
        if pin_memory:
            self.pin_memory() # for faster data transfer to GPU, set to false for better memory management

        self.batch_size = batch_size if batch_size is not None else self.__len__()
        self.shuffle = shuffle if batch_size is not None else False

        self.n_batches, remainder = divmod(self.__len__(), self.batch_size)
        self.n_batches += 1 if remainder > 0 else 0
        if mask > 0: # masking the images (work in-progress)
            valid_idx = torch.nonzero(pixels.flatten() >= mask, as_tuple=True)
            self.source = self.source[:, valid_idx[0]]
            self.target = self.target[:, valid_idx[0]]
            self.pixels = self.pixels[..., valid_idx[0]]
        
    def __iter__(self):
        if self.shuffle:
            with torch.no_grad():
                indices = torch.randperm(self.__len__(), dtype=torch.int32, device='cuda').cpu()
                self.source = self.source[:, indices]
                self.target = self.target[:, indices]
                self.pixels = self.pixels[..., indices]
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.__len__():
            raise StopIteration
        source = self.source[:, self.idx : self.idx + self.batch_size]
        target = self.target[:, self.idx : self.idx + self.batch_size]
        pixels = self.pixels[..., self.idx : self.idx + self.batch_size]
        self.idx += self.batch_size
        return source, target, pixels


    def __len__(self):
        return self.source.shape[1]

    def pin_memory(self):
        self.source = self.source.pin_memory()
        self.target = self.target.pin_memory()
        self.pixels = self.pixels.pin_memory()

    def apply_function(self, func, device='cuda'):
        idx = 0
        while idx < self.__len__():
            self.source[:, idx : idx + self.batch_size] = func(self.source[:, idx : idx + self.batch_size].to(device)).cpu()
            self.target[:, idx : idx + self.batch_size] = func(self.target[:, idx : idx + self.batch_size].to(device)).cpu()
            idx += self.batch_size

