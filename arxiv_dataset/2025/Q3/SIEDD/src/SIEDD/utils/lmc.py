import torch
import torch.nn.functional as F
from ..configs import LMCConfig


class LMC(torch.nn.Module):
    def __init__(
        self,
        images: torch.Tensor,
        coords: torch.Tensor,
        normalize_range: tuple[float, float],
        cfg: LMCConfig,
    ):
        super().__init__()
        """
            Conundrum is do we do it across all images? 
            As in do we select different coords for each image in a group?
            Or do we do it for each image separately?
            No we do not.
            Think about it - for a given batch, the coords need to be the same. 
        """

        self.a = cfg.a
        self.b = cfg.b

        # nchw to nhwc
        images = images.permute(0, 2, 3, 1)

        self.WIDTH = images.shape[2]
        self.HEIGHT = images.shape[1]
        self.NUM_IMGS = images.shape[0]
        self.device = coords.device

        """
            Should we do random selection and sample?
            Or we just pass all coordinates here and leave the sampling to the forward pass?
            I think it is the latter.        
        """

        self.num_points = coords.shape[0]

        self.image_edges = compute_sobel_edge(images.float()).reshape(
            images.shape[0], -1
        )
        self.image_edges = self.image_edges.to(self.device)

        probs = self.image_edges / self.image_edges.sum(dim=-1, keepdim=True)
        cdf = torch.cumsum(probs, dim=-1)
        cdf = torch.nn.functional.pad(cdf, pad=(1, 0), mode="constant", value=0)
        self.cdf = cdf.view(cdf.shape[0], -1)

        self.rand_ten = torch.empty(
            (self.num_points, 2), dtype=torch.float32, device=self.device
        )

        self.noise = torch.empty(
            (self.num_points, 2), dtype=torch.float32, device=self.device
        )

        self.reinit = int(
            cfg.lossminpc * self.num_points
        )  # needs to be across all images.
        self.u_num = int(cfg.minpct * self.num_points)

        self.noise = torch.empty_like(coords)
        self.rand_ten = torch.empty_like(coords)

        # Sampling w/ replacement
        x = torch.randint(0, self.WIDTH, size=(self.num_points,), device=self.device)
        y = torch.randint(0, self.HEIGHT, size=(self.num_points,), device=self.device)
        self.normalize_range = normalize_range
        nr0, nr1 = self.normalize_range
        x = x.float() / (self.WIDTH - 1) * (nr1 - nr0) + nr0
        y = y.float() / (self.HEIGHT - 1) * (nr1 - nr0) + nr0
        self.prev_samples = torch.cat([y[..., None], x[..., None]], dim=1)
        self.prev_samples.clamp_(min=0.0, max=1.0)
        self.HW = torch.tensor([self.HEIGHT - 1, self.WIDTH - 1], device=self.device)

    def forward(self, net_grad: torch.Tensor | None, loss_per_pix: torch.Tensor | None):
        # net_grad: N x 2
        # loss_per_pix: N,
        # prev_samples: N x 2
        if net_grad is not None and loss_per_pix is not None:
            with torch.no_grad():
                self.noise.normal_(mean=0.0, std=1.0)
                self.rand_ten.uniform_()

                net_grad.mul_(self.a).add_(self.noise, alpha=self.b)
                self.prev_samples.add_(net_grad)
                # eq 10 from paper above

                # Everything below is for reinitialization/edge based reinitialization
                threshold, _ = torch.topk(
                    loss_per_pix.view(-1), self.reinit + 1, largest=False
                )
                mask = loss_per_pix <= threshold[-1]
                # mask: all pixels with loss lower than self.reinit+1 lowest loss
                nr0, nr1 = self.normalize_range
                mask = torch.cat(
                    [
                        self.prev_samples < nr0,
                        self.prev_samples > nr1,
                        loss_per_pix.unsqueeze(-1) <= threshold[-1],
                    ],
                    1,
                )
                mask = mask.sum(1).bool()
                # mask: samples that "move out of R or have too low error value"
                # bound_idxs = torch.where(mask)[0] # This makes no sense

                # No idea what this does
                # self.prev_samples[-self.u_num :].copy_(self.rand_ten[-self.u_num :])

                # What is this part
                # if bound_idxs.shape[0] > 0:
                #     # sample from edges
                #     # count: (N,)
                #     count = torch.bincount(bound_idxs, minlength=self.num_points)
                #     # batch1d: (1, N)
                #     batch1d = sample_from_pdf_with_indices(
                #         self.cdf, int(self.num_points / self.NUM_IMGS)
                #     )
                #     # indices: (1, N)
                #     indices = (
                #         torch.arange(batch1d.size(1), device=batch1d.device)
                #         .unsqueeze_(0)
                #         .repeat(batch1d.size(0), 1)
                #     )
                #     mask = indices < count # ???
                #     batch1d = batch1d.masked_select(mask)
                #     self.prev_samples[bound_idxs, 0] = (batch1d // self.WIDTH) / (
                #         self.HEIGHT - 1
                #     )
                #     self.prev_samples[bound_idxs, 1] = (batch1d % self.WIDTH) / (
                #         self.WIDTH - 1
                #     )

                batch1d = sample_from_pdf_with_indices(self.cdf, int(mask.sum()))
                self.prev_samples[mask, 0] = (batch1d // self.WIDTH) / (self.HEIGHT - 1)
                self.prev_samples[mask, 1] = (batch1d % self.WIDTH) / (self.WIDTH - 1)
                self.prev_samples.clamp_(min=0, max=1.0)

        # convert to indices.
        point2d = self.prev_samples * self.HW

        return point2d.round_()


@torch.amp.autocast_mode.autocast(
    dtype=torch.float64, device_type="cuda", enabled=torch.cuda.is_available()
)
def sample_from_pdf_with_indices(cdf: torch.Tensor, num_points: int):
    # Normalize the PDFs
    u = torch.rand(
        (
            cdf.shape[0],
            num_points,
        ),
        device=cdf.device,
        dtype=torch.float64,
    )  # * cdf.max()
    batch1d = torch.searchsorted(cdf, u, right=True) - 1
    return batch1d


def compute_sobel_edge(images):
    # Ensure the input is a torch tensor
    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images)

    if images.max() > 1.0:
        images = images / 255.0

    # Convert the images to grayscale using weighted sum of channels (shape: N x H x W x 1)
    gray_images = (
        0.2989 * images[..., 0] + 0.5870 * images[..., 1] + 0.1140 * images[..., 2]
    )
    gray_images = gray_images.unsqueeze(-1)

    # Transpose the images to the shape (N, C, H, W)
    gray_images = gray_images.permute(0, 3, 1, 2)

    # Define Sobel kernels
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=images.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=images.device
    ).view(1, 1, 3, 3)

    # Compute Sobel edges
    edge_x = F.conv2d(gray_images, sobel_x, padding=1)
    edge_y = F.conv2d(gray_images, sobel_y, padding=1)
    edges = torch.sqrt(edge_x**2 + edge_y**2)

    maxval, _ = edges.max(dim=1)[0].max(dim=1)
    edges = edges / (maxval.unsqueeze(1).unsqueeze(1) + 1e-7)
    edges = torch.clip(edges, min=1e-5, max=1.0)
    return edges.squeeze(1)


# source: https://github.com/NVlabs/tiny-cuda-nn/blob/master/samples/mlp_learning_an_image_pytorch.py
class Image(torch.nn.Module):
    def __init__(self, images, device):
        super(Image, self).__init__()
        self.data = images.to(device, non_blocking=True)
        self.shape = self.data[0].shape

    @torch.amp.autocast_mode.autocast(
        dtype=torch.float32, device_type="cuda", enabled=torch.cuda.is_available()
    )
    def forward(self, iind, ys, xs):
        shape = self.shape

        xy = torch.cat([ys.unsqueeze(1), xs.unsqueeze(1)], dim=1)
        indices = xy.long()
        lerp_weights = xy - indices.float()

        y0 = indices[:, 0].clamp(min=0, max=shape[0] - 1)
        x0 = indices[:, 1].clamp(min=0, max=shape[1] - 1)
        y1 = (y0 + 1).clamp(max=shape[0] - 1)
        x1 = (x0 + 1).clamp(max=shape[1] - 1)

        return (
            self.data[iind, y0, x0]
            * (1.0 - lerp_weights[:, 0:1])
            * (1.0 - lerp_weights[:, 1:2])
            + self.data[iind, y0, x1]
            * lerp_weights[:, 0:1]
            * (1.0 - lerp_weights[:, 1:2])
            + self.data[iind, y1, x0]
            * (1.0 - lerp_weights[:, 0:1])
            * lerp_weights[:, 1:2]
            + self.data[iind, y1, x1] * lerp_weights[:, 0:1] * lerp_weights[:, 1:2]
        )
