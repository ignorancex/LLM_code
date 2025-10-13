from dataclasses import dataclass, field
from typing import Iterator, Optional, Type

import numpy as np
import scipy
import torch
from torch.utils.data import Sampler

from scrstudio.configs.base_config import InstantiateConfig


@dataclass
class CSRGraph:

    indices: torch.Tensor
    indptr: torch.Tensor
    rowsizes: torch.Tensor

    def sample_neighbors(self,start: torch.Tensor,generator: torch.Generator):
        offset = torch.randint(0,torch.iinfo(torch.int32).max,
                               (start.shape[0],),device=start.device,
                                 generator=generator) % self.rowsizes[start]
        return self.indices[self.indptr[start]+offset]
    
    @staticmethod
    def from_csr_array(csr_array: scipy.sparse.csr_array,device):
        return CSRGraph(torch.tensor(np.ascontiguousarray(csr_array.indices),dtype=torch.int32,device=device),
                        torch.tensor(np.ascontiguousarray(csr_array.indptr),dtype=torch.int32,device=device),
                        torch.tensor(np.diff(csr_array.indptr),dtype=torch.int32,device=device))


class PQKNN:
    """KNN search using product quantization."""
    def __init__(self, pq, codes,n_neighbors=10,device='cuda'):
        self.n_neighbors=n_neighbors
        self.device=device
        self.M=pq.M
        self.Ds=pq.Ds
        self.codes = torch.from_numpy(codes).to(torch.int64).to(device)
        self.codewords = torch.from_numpy(pq.codewords).to(device)
        self.row_idx=torch.arange(self.M,dtype=torch.int64,device=self.codes.device).unsqueeze(0).expand_as(self.codes)
    def kneighbors(self,query):
        query_gpu=torch.from_numpy(query).to(self.device)
        diff = query_gpu.view(self.M,1,self.Ds) - self.codewords
        dtable = torch.einsum('ijk,ijk->ij', diff, diff)
        dists=dtable[self.row_idx, self.codes]
        dists=torch.sum(dists.reshape(-1,self.M),dim=1)
        _, indices = torch.topk(dists, self.n_neighbors, largest=False)
        return indices

class RepeatSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.
        Good for caching.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, dataset_size,num_repeats) -> None:

        self.dataset_size = dataset_size
        self.num_repeats = num_repeats


    def __iter__(self) -> Iterator[int]:
        for i in range(self.dataset_size):
            for _ in range(self.num_repeats):
                yield i

    def __len__(self) -> int:
        return self.dataset_size*self.num_repeats
   


@dataclass
class GlobalFeatSamplerConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: GlobalFeatSampler)
    train_covis_graph: Optional[str] = None
    train_covis_thres: float = 0.2
    neighbor_ratio: float = 0.5
    noise_std: float = 0.0
    renorm: bool = False

class GlobalFeatSampler:
    def __init__(self,config: GlobalFeatSamplerConfig,global_feat, generator, 
                 data=None,**kwargs):
        self.config=config
        self.generator=generator
        self.global_feat=global_feat
        if self.config.train_covis_graph is not None:
            assert data is not None
            covis_score=scipy.sparse.load_npz(data/'train'/self.config.train_covis_graph).tocoo()
            covis_score.data[covis_score.data<self.config.train_covis_thres]=0
            covis_score.setdiag(1)
            covis_score.eliminate_zeros()
            covis_score=covis_score.tocsr()
            self.covis_graph=CSRGraph.from_csr_array(covis_score,device=global_feat.device)
        else:
            self.covis_graph=None

    def sample(self,img_idx):
        img_idx=img_idx.clone()
        if self.covis_graph is not None:
            num_neighbors=int(self.config.neighbor_ratio*img_idx.shape[0])
            img_idx[:num_neighbors]=self.covis_graph.sample_neighbors(img_idx[:num_neighbors],self.generator)
        features = self.global_feat[img_idx]
        if self.config.noise_std > 0.0:
            # first self.global_feat_dim is global feature, add gaussian noise and normalize to unit length
            features += torch.empty_like(features).normal_(mean=0,std=self.config.noise_std,generator=self.generator)
            if self.config.renorm:
                features =torch.nn.functional.normalize(features,dim=1)
        return features

@dataclass
class BatchRandomSamplerConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: BatchRandomSampler)
    
    batch_size: int = 5120

class BatchRandomSampler(Sampler[torch.Tensor]):
    dataset_size: int
    batch_size : int
    generator: torch.Generator

    def __init__(self, config: BatchRandomSamplerConfig,
                  dataset_size: int, generator: Optional[torch.Generator] = None,
                  **kwargs
                  ) -> None:
        self.dataset_size = dataset_size
        self.batch_size = config.batch_size
        self.generator = generator if generator is not None else torch.Generator()

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            yield from torch.randperm(self.dataset_size, generator=self.generator,
                                device=self.generator.device).split(self.batch_size)[:-1]

def get_subset_size(dataset_size, world_size):
    return (dataset_size + world_size - 1) // world_size

def get_start(dataset_size, world_size, local_rank):
    subset_size = get_subset_size(dataset_size, world_size)
    offsets=np.linspace(0,dataset_size,world_size+1,dtype=np.int32)
    start = offsets[local_rank]
    start = min(start, dataset_size - subset_size)
    return start

class DistributedSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, dataset_size, world_size,local_rank,generator, shuffle: bool = True) -> None:

        self.dataset_size = dataset_size
        self.world_size = world_size
        self.local_rank = local_rank
        self.generator = generator
        self.shuffle = shuffle
        
        self.subset_size = get_subset_size(dataset_size, world_size)
        self.start = get_start(dataset_size, world_size, local_rank)


    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            while True:
                yield from (torch.randperm(self.subset_size, generator=self.generator) + self.start).tolist()
        else:
            yield from range(self.start, self.start + self.subset_size)


    def __len__(self) -> int:
        return self.subset_size
    