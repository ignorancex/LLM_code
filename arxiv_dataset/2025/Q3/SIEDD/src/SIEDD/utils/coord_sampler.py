import torch
from torch.utils.data import DataLoader, Dataset
import einops
import numpy as np
from .lmc import compute_sobel_edge
from scipy.spatial.distance import pdist, squareform
from ..configs import RunConfig, NoPosEncode
from .coord_utils import generate_coordinates
from fractions import Fraction


def sort_and_select_top_k_percent(losses: torch.Tensor, k: float):
    # Flatten the losses except for the batch dimension
    batch_size = losses.shape[0]
    flattened_losses = losses.view(batch_size, -1)

    # Sort the flattened losses in descending order
    sorted_losses, sorted_indices = torch.sort(flattened_losses, dim=1, descending=True)

    # Calculate the number of elements to keep
    num_elements = flattened_losses.shape[1]
    num_to_keep = int(num_elements * k / 100)

    # Select the top k% of losses and their indices
    top_k_losses = sorted_losses[:, :num_to_keep]
    top_k_indices = sorted_indices[:, :num_to_keep]

    return top_k_losses, top_k_indices


def get_coord_sampler(
    coordinates: list[torch.Tensor] | torch.Tensor,
    cfg: RunConfig,
    images: torch.Tensor,
):
    if isinstance(coordinates, list):
        dataset = CoordXSampler(coordinates, cfg, images)
    else:
        dataset = CoordSampler(coordinates, cfg, images)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
    return loader


class CoordXSampler(Dataset):
    def __init__(
        self,
        coordinates: list[torch.Tensor],
        cfg: RunConfig,
        images: torch.Tensor,
    ) -> None:
        self.coordinates = coordinates
        self.images = images
        self.sample_ratio = cfg.trainer_cfg.coord_sample_frac ** (1 / len(coordinates))
        self.num_total_points = [c.size(0) for c in coordinates]
        self.unif = [torch.ones(c.size(0)) for c in coordinates]

        self.num_samples = [
            int(num_points * self.sample_ratio) for num_points in self.num_total_points
        ]
        self.patch_size = cfg.encoder_cfg.patch_size

        self.images = images  # processsed features
        N, H, W, C = images.shape
        if self.patch_size is not None:
            self.features = einops.rearrange(
                images,
                "n (h ph) (w pw) c -> n h w c ph pw",
                ph=self.patch_size,
                pw=self.patch_size,
            )
        else:
            self.features = einops.rearrange(images, "n h w c -> n h w c")
        # We keep h and w as separate dimensions to make it easier in sampling
        # (we need them separate for coordx)

        self.cfg = cfg
        self.sampling_type = self.cfg.trainer_cfg.sampling
        self.sampling_warmup_steps = self.cfg.trainer_cfg.sampling_warmup
        if self.sampling_type == "edge":
            N, H, W, C = images.shape
            if self.patch_size is None:
                edges = compute_sobel_edge(self.images).mean(0)
                edges = edges.reshape(H, W)
            else:
                edges = compute_sobel_edge(self.images)
                # first dimension is for strainer + patching
                # if this is the case, reduce over meta frame edges as well
                edges = einops.rearrange(
                    edges,
                    "n (h ph) (w pw) -> n h w ph pw",
                    ph=self.patch_size,
                    pw=self.patch_size,
                )
                edges = edges.mean((0, -1, -2))
            # reduces over the opposite axis
            self.edges = [edges.mean(1), edges.mean(0)]

    def __len__(self):
        return 10000000

    def __getitem__(self, iteration: int):
        if (
            self.sample_ratio == 1
            or (iteration < self.sampling_warmup_steps)
            or self.sampling_type is None
        ):
            coords, feats = self.coordinates, self.features.flatten(1, 2)
        elif self.sampling_type == "uniform":
            sampled_indices = [
                unif.multinomial(num_samp)
                for unif, num_samp in zip(self.unif, self.num_samples)
            ]
            coords = [
                c[samp_ind] for c, samp_ind in zip(self.coordinates, sampled_indices)
            ]

            ind_h, ind_w = sampled_indices
            feats = self.features[:, ind_h.unsqueeze(-1), ind_w.unsqueeze(0)]
            feats = feats.flatten(1, 2)
        elif self.sampling_type == "edge":
            sampled_indices = [
                edges.multinomial(num_samp)
                for edges, num_samp in zip(self.edges, self.num_samples)
            ]
            coords = [
                c[samp_ind] for c, samp_ind in zip(self.coordinates, sampled_indices)
            ]
            ind_h, ind_w = sampled_indices
            feats = self.features[:, ind_h.unsqueeze(-1), ind_w.unsqueeze(0)]
            feats = feats.flatten(1, 2)
        else:
            raise ValueError(f"{self.sampling_type} Not Implemented")
        return coords, feats


def get_scaled_image_patches(
    imgs: torch.Tensor,
    scales: list[Fraction],
    patch_size: int,
    normalize_range: tuple[float, float],
):
    imgs = einops.rearrange(imgs, "n h w c -> n c h w")
    scaled_imgs = []
    scaled_coords = []
    pos_encoding = NoPosEncode()
    for scale in scales:
        scaled_img = torch.nn.functional.interpolate(
            imgs, scale_factor=float(scale), mode="bilinear"
        )
        coords = generate_coordinates(
            scaled_img[0].shape, patch_size, normalize_range, pos_encoding, scale
        )
        scaled_img = einops.rearrange(
            scaled_img,
            "n c (h ph) (w pw) -> n (h w) c ph pw",
            ph=patch_size,
            pw=patch_size,
        )
        scaled_imgs.append(scaled_img)
        scaled_coords.append(coords)
    images = torch.cat(scaled_imgs, dim=1).cuda()
    coordinates = torch.cat(scaled_coords, dim=0).cuda()
    return images, coordinates


class CoordSampler(Dataset):
    def __init__(
        self,
        coordinates: torch.Tensor,
        cfg: RunConfig,
        images: torch.Tensor,
    ) -> None:
        self.coordinates = coordinates
        self.sample_ratio = cfg.trainer_cfg.coord_sample_frac
        self.indices: torch.Tensor
        self.num_total_points = int(coordinates.shape[0])
        self.indices = torch.arange(self.num_total_points)

        self.num_samples = int(self.num_total_points * self.sample_ratio)
        self.current_idx = self.num_total_points
        self.patch_size = cfg.encoder_cfg.patch_size
        self.patch_scales = cfg.encoder_cfg.patch_scales
        self.strided_patches = cfg.encoder_cfg.strided_patches
        self.offset_patch_training = cfg.encoder_cfg.offset_patch_training
        self.normalize_range = cfg.encoder_cfg.normalize_range

        self.images = images  # processsed features
        N, H, W, C = images.shape
        if self.patch_size is not None and self.patch_scales is not None:
            self.features, self.coordinates = get_scaled_image_patches(
                images, self.patch_scales, self.patch_size, self.normalize_range
            )
        elif self.patch_size is not None:
            if self.strided_patches:
                order_h, order_w = "(ph h)", "(pw w)"
            else:
                order_h, order_w = "(h ph)", "(w pw)"
            self.features = einops.rearrange(
                images,
                f"n {order_h} {order_w} c -> n (h w) c ph pw",
                ph=self.patch_size,
                pw=self.patch_size,
            )
            if self.offset_patch_training:
                left = self.patch_size // 2
                right = left - self.patch_size
                centroids_h = torch.arange(left, H - self.patch_size, self.patch_size)
                centroids_w = torch.arange(left, W - self.patch_size, self.patch_size)
                # Create grid of centroids
                grid_h, grid_w = torch.meshgrid(centroids_h, centroids_w, indexing="ij")
                coords2 = torch.stack([grid_h, grid_w], dim=-1).reshape((-1, 2))
                nr0, nr1 = self.normalize_range
                coords2 = coords2 / torch.tensor([H - 1, W - 1])
                coords2 = coords2 * (nr1 - nr0) + nr0
                coords2 = coords2.to(self.coordinates.device)
                feats2 = images[:, left:right, left:right]
                feats2 = einops.rearrange(
                    feats2,
                    f"n {order_h} {order_w} c -> n (h w) c ph pw",
                    ph=self.patch_size,
                    pw=self.patch_size,
                ).to(self.features.device)
                self.features = torch.concat((self.features, feats2), dim=1)
                self.coordinates = torch.concat((self.coordinates, coords2), dim=0)
        else:
            self.features = einops.rearrange(images, "n h w c -> n (h w) c")
        self.cfg = cfg

        self.sampling_type = self.cfg.trainer_cfg.sampling
        self.sampling_warmup_steps = self.cfg.trainer_cfg.sampling_warmup

        if self.sampling_type == "edge":
            if self.patch_size is None:
                self.edges = compute_sobel_edge(self.images).mean(0).view(-1)
            else:
                edges = compute_sobel_edge(self.images)
                edges = einops.rearrange(
                    edges,
                    "n (h ph) (w pw) -> n h w ph pw",
                    ph=self.patch_size,
                    pw=self.patch_size,
                )
                edges = edges.mean((0, -1, -2))
                edges = edges.view(-1)
                mu = edges.mean()
                d, delta = cfg.trainer_cfg.edge_d, cfg.trainer_cfg.edge_delta
                alpha = (1 - delta) * mu
                edges = 2 * delta * torch.nn.functional.softmax(edges * d) + alpha
                self.edges = edges

        if self.sampling_type == "unique":
            self.calculate_uniqueness_weights()

    def __len__(self):
        return 10000000

    def __getitem__(self, iteration: int):
        coords: torch.Tensor
        if (
            self.sample_ratio == 1
            or (iteration < self.sampling_warmup_steps)
            or self.sampling_type is None
        ):
            coords, feats = self.coordinates, self.features
        elif self.sampling_type == "uniform":
            """
                This will save the numbner of randperm calls.
                But this is not the best way to sample. 
                As samples are not repeated - we go through all the samples in the dataset.
                Not exactly random and efficient. We need a balance.
            """

            if self.current_idx + self.num_samples > self.num_total_points:
                self.indices = self.indices[torch.randperm(self.num_total_points)]
                self.current_idx = 0

            # random.shuffle(self.indices) extremely slow.
            sampled_indices = self.indices[
                self.current_idx : self.current_idx + self.num_samples
            ]
            self.current_idx += self.num_samples
            coords = self.coordinates[sampled_indices]
            feats = self.features[:, sampled_indices]
        elif self.sampling_type == "edge":
            sampled_indices = torch.multinomial(
                self.edges, self.num_samples, replacement=False
            )
            coords = self.coordinates[sampled_indices]
            feats = self.features[:, sampled_indices]
        elif self.sampling_type == "unique":
            # Sample points with probability proportional to their uniqueness
            sampled_indices = torch.multinomial(
                self.unique_weights, self.num_samples, replacement=False
            )
            sampled_indices = sampled_indices
            coords = self.coordinates[sampled_indices]
            feats = self.features[:, sampled_indices]
        else:
            raise ValueError(f"{self.sampling_type} Not Implemented")
        return coords, feats

    def calculate_uniqueness_weights(self, features: torch.Tensor | None = None):
        if self.patch_size is not None and features is not None:
            # For patch-based approach
            # patches = features.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
            # patches = features.contiguous().view(-1, self.patch_size * self.patch_size * 3)  # Flatten patches
            N, K = features.shape[0], features.shape[1]
            patches = features.view(N, K, -1)
        elif self.images is not None:
            # Doesn't work?
            # For point-based approach
            patches = self.images.view(-1, 3)  # Treat each pixel as a patch
        else:
            raise ValueError()

        # retain batch.
        # features.view(-1,self.patch_size * self.patch_size * 3)
        # Calculate pairwise distances between patches
        unique_weights = []
        for k in range(len(patches)):
            dist = pdist(patches[k].cpu().numpy(), metric="euclidean")
            # distances.append(dist)
            dist_matrix = squareform(dist)
            uniqueness = np.mean(dist_matrix, axis=1)
            uniqueness = (uniqueness - np.min(uniqueness)) / (
                np.max(uniqueness) - np.min(uniqueness)
            )
            unique_weights.append(uniqueness)

        device = (
            self.coordinates[0].device
            if isinstance(self.coordinates, list)
            else self.coordinates.device
        )
        self.unique_weights = (
            torch.from_numpy(np.array(unique_weights)).float().to(device)
        )
