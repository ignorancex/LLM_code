"""Same-different tasks"""

from collections import defaultdict
import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from cifar100 import load_data
    from pentomino import pieces
except ImportError:
    from .cifar100 import load_data
    from .pentomino import pieces

pieces = np.array(pieces, dtype='object')

def batch_choice(a, n_elem, batch_size, rng=None):
    if type(a) == int:
        a = np.arange(a)

    assert n_elem <= len(a), f'require n_elem <= len(a), got n_elem={n_elem} and len(a)={len(a)}'

    if rng is None:
        rng = np.random.default_rng(None)

    idxs = np.tile(a, (batch_size, 1))
    idxs = rng.permuted(idxs, axis=1)
    idxs = idxs[:,:n_elem]
    return idxs


def gen_patches(patch_size, n_examples=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(None)

    examples = rng.binomial(1, p=0.5, size=(n_examples, patch_size, patch_size))
    examples = 2 * examples - 1
    return examples
    

class SameDifferentPsvrt:
    def __init__(self, patch_size, n_patches, inc_set=None, exc_set=None, seed=None, batch_size=128) -> None:
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.inc_set = inc_set
        self.exc_set = exc_set
        self.seed = seed
        self.batch_size = batch_size

        self.rng = np.random.default_rng(self.seed)
        self.slack = 10   # extra chances for exc generation
        self.max_retry = 5

    def __next__(self):
        xs = np.zeros((self.batch_size, self.n_patches * self.patch_size, self.n_patches * self.patch_size))
        ys = self.rng.binomial(n=1, p=0.5, size=(self.batch_size + self.slack,))

        if self.inc_set is not None:
            xs_idxs = batch_choice(len(self.inc_set), 2, self.batch_size, self.rng)
            ys = ys[:self.batch_size]

            if np.sum(ys) > 0:
                idxs = ys.astype(bool)
                xs_idxs[idxs,1] = xs_idxs[idxs,0]
            
            xs_patches = self.inc_set[xs_idxs]
        
        elif self.exc_set is not None:
            for _ in range(self.max_retry):
                xs_patches = gen_patches(self.patch_size, n_examples=2*(self.batch_size + self.slack))
                xs_patches = xs_patches.reshape(-1, 2, self.patch_size, self.patch_size)

                if np.sum(ys) > 0:
                    idxs = ys.astype(bool)
                    xs_patches[idxs,1] = xs_patches[idxs,0]  

                excs = np.expand_dims(self.exc_set, axis=(1, 2))
                comps = (excs == xs_patches).sum(axis=(-2, -1))
                bad_idxs = (comps == self.patch_size**2).sum(axis=(0, 2)).astype(bool)
                xs_patches = xs_patches[~bad_idxs]
                ys = ys[~bad_idxs]

                if len(xs_patches) >= self.batch_size:  
                    break
            
            assert len(xs_patches) >= self.batch_size
            xs_patches = xs_patches[:self.batch_size]
            ys = ys[:self.batch_size]

        else:
            xs_patches = gen_patches(self.patch_size, n_examples=2*self.batch_size)
            xs_patches = xs_patches.reshape(-1, 2, self.patch_size, self.patch_size)
            ys = ys[:self.batch_size]

            if np.sum(ys) > 0:
                idxs = ys.astype(bool)
                xs_patches[idxs,1] = xs_patches[idxs,0]  

        xs_locs = batch_choice(self.n_patches**2, 2, self.batch_size)
        for x, x_patch, x_loc in zip(xs, xs_patches, xs_locs):
            a, b = x_patch

            a_x = self.patch_size * (x_loc[0] // self.n_patches)
            a_y = self.patch_size * (x_loc[0] % self.n_patches)

            b_x = self.patch_size * (x_loc[1] // self.n_patches)
            b_y = self.patch_size * (x_loc[1] % self.n_patches)

            x[a_x:a_x+self.patch_size, a_y:a_y+self.patch_size] = a
            x[b_x:b_x+self.patch_size, b_y:b_y+self.patch_size] = b

        return xs, ys

    def __iter__(self):
        return self


class SameDifferentPentomino:
    def __init__(self, ps=None, width=2, blur=0, random_blur=False, batch_size=128):
        self.pieces = ps
        if self.pieces is None:
            self.pieces = np.arange(len(pieces))

        self.width = width
        self.blur = blur
        self.random_blur = random_blur
        self.batch_size = batch_size
        self.rng = np.random.default_rng(None)
    
    def __next__(self):
        xs = np.zeros((self.batch_size, self.width * 7, self.width * 7))
        xs_idxs = batch_choice(self.pieces, 2, self.batch_size)
        xs_patches = batch_choice(self.width**2, 2, self.batch_size)

        ys = self.rng.binomial(n=1, p=0.5, size=(self.batch_size,))
        if np.sum(ys) > 0:
            idxs = ys.astype(bool)
            xs_idxs[idxs,1] = xs_idxs[idxs,0]
        
        xs_pieces = pieces[xs_idxs]
        for x, x_patch, x_piece in zip(xs, xs_patches, xs_pieces):
            a, b = x_piece
            a = self.rng.choice(a)
            b = self.rng.choice(b)

            a_x = 7 * (x_patch[0] // self.width) + 1
            a_y = 7 * (x_patch[0] % self.width) + 1

            b_x = 7 * (x_patch[1] // self.width) + 1
            b_y = 7 * (x_patch[1] % self.width) + 1

            x[a_x:a_x+5, a_y:a_y+5] = a
            x[b_x:b_x+5, b_y:b_y+5] = b

        if self.blur > 0:
            if self.random_blur:
                self.rng.uniform(0, self.blur)

            xs = gaussian_filter(xs, sigma=self.blur, axes=[-2, -1])
        return xs, ys


    def __iter__(self):
        return self


class SameDifferentCifar100:
    def __init__(self, ps, batch_size=128, **loader_kwargs):
        self.pieces = np.array(ps)

        self.batch_size = batch_size
        self.rng = np.random.default_rng(None)

        # lazy loading
        self.loader_kwargs = loader_kwargs
        self.cifar100 = None
        self.label_to_idxs = defaultdict(list)
    
    def __next__(self):
        if self.cifar100 is None:
            self.cifar100 = load_data(**self.loader_kwargs)
            for i, lab in enumerate(self.cifar100['labels']):
                self.label_to_idxs[lab].append(i)

        xs_idxs = batch_choice(self.pieces, 2, self.batch_size)

        ys = self.rng.binomial(n=1, p=0.5, size=(self.batch_size,))
        if np.sum(ys) > 0:
            idxs = ys.astype(bool)
            xs_idxs[idxs,1] = xs_idxs[idxs,0]
        
        xs = []
        for labs in xs_idxs:
            a, b = labs
            a = self.rng.choice(self.label_to_idxs[a])
            b = self.rng.choice(self.label_to_idxs[b])

            x = np.stack((self.cifar100['data'][a], self.cifar100['data'][b]))
            xs.append(x)
        
        return np.stack(xs), ys

    def __iter__(self):
        return self


class SameDifferent:
    def __init__(self, n_symbols=None, task='hard',
                 n_dims=2, noise=0, thresh=0, radius=1, n_patches=2,   # soft/hard params
                 n_seen=None, sample_seen=True,                        # token params
                 seed=None, reset_rng_for_data=True, batch_size=128) -> None:

        if task == 'token':
            assert n_symbols is not None and n_symbols >= 4, 'if task=token, n_symbols should be >= 4'
            
            if n_seen is None:
                n_seen = n_symbols // 2

        self.n_symbols = n_symbols
        self.task = task
        self.n_dims = n_dims
        self.noise = noise
        self.thresh = thresh
        self.radius = radius
        self.n_patches = n_patches
        self.n_seen = n_seen
        self.sample_seen = sample_seen
        self.seed = seed
        self.batch_size = batch_size

        self.rng = np.random.default_rng(seed)
        if self.n_symbols is not None:
            self.symbols = self.rng.standard_normal((self.n_symbols, self.n_dims)) / np.sqrt(self.n_dims)

        if reset_rng_for_data:
            self.rng = np.random.default_rng(None)
    
    def __repr__(self) -> str:
        return f'SD(n_symbols={self.n_symbols}, n_dims={self.n_dims}, noise={self.noise}, n_patches={self.n_patches})'
    
    def __next__(self):
        if self.task == 'soft':
            return self._sample_soft()
        elif self.task == 'hard':
            return self._sample_hard()
        elif self.task == 'token':
            return self._sample_token()
        else:
            raise ValueError(f'unrecognized task type: {self.task}')

    def _sample_soft(self):
        xs = self.rng.standard_normal((self.batch_size, 2, self.n_dims)) / np.sqrt(self.n_dims)
        x0, x1 = xs[:,0], xs[:,1]
        ys = (np.einsum('bi,bi->b', x0, x1) > self.thresh).astype('float')
        return xs, ys.flatten()
    
    def _sample_hard(self):
        if self.n_symbols is None:
            xs = self.rng.standard_normal((self.batch_size, 2, self.n_dims)) / np.sqrt(self.n_dims)
        else:
            sym_idxs = batch_choice(np.arange(self.n_symbols), 2, batch_size=self.batch_size, rng=self.rng)
            xs = self.symbols[sym_idxs]

        ys = self.rng.binomial(n=1, p=0.5, size=(self.batch_size,))
        if np.sum(ys) > 0:
            idxs = ys.astype(bool)
            xs[idxs,1] = xs[idxs,0]
        
        xs = xs + self.rng.standard_normal(xs.shape) * np.sqrt(self.noise / self.n_dims)
        
        if self.n_patches > 2:
            xs_full = np.zeros((self.batch_size, self.n_patches, self.n_dims))
            patch_idxs = batch_choice(np.arange(self.n_patches), 2, batch_size=self.batch_size, rng=self.rng)
            xs_full[np.arange(self.batch_size),patch_idxs[:,0]] = xs[:,0]
            xs_full[np.arange(self.batch_size),patch_idxs[:,1]] = xs[:,1]
            xs = xs_full

        return xs, ys
    
    def _sample_token(self):
        if self.sample_seen:
            xs = batch_choice(np.arange(0, self.n_seen), 2, batch_size=self.batch_size, rng=self.rng)
        else:
            xs = batch_choice(np.arange(self.n_seen, self.n_symbols), 2, batch_size=self.batch_size, rng=self.rng)

        ys = self.rng.binomial(n=1, p=0.5, size=(self.batch_size,))
        if np.sum(ys) > 0:
            idxs = ys.astype(bool)
            xs[idxs,1] = xs[idxs,0]

        return xs, ys

    def __iter__(self):
        return self
    
