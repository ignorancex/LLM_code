import math
import random
import torch
import einops
from guided_diffusion.nn import timestep_embedding


def normalize(image):
    dimension = len(image.shape) - 2
    if dimension == 2:
        dim_reduce = [2, 3]
    elif dimension == 3:
        dim_reduce = [2, 3, 4]
    image_centered = image - torch.mean(image, dim_reduce, keepdim=True)
    stddev = torch.sqrt(torch.mean(image_centered**2, dim_reduce, keepdim=True))
    return image_centered / stddev



class SimilarityBase:
    def __init__(self, isInterpolated=False):
        self.isInterpolated = isInterpolated

class NCC(SimilarityBase):
    def __init__(self):
        super().__init__(isInterpolated=False)

    def __call__(self, image_A, image_B):
        assert image_A.shape == image_B.shape, "The shape of image_A and image_B sould be the same."
        A = normalize(image_A)
        B = normalize(image_B)
        res = torch.mean(A * B)
        return 1 - res

# torch removed this function from torchvision.functional_tensor, so we are vendoring it.
def _get_gaussian_kernel1d(kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()
    return kernel1d

def gaussian_blur(tensor, kernel_size, sigma, padding="same"):
    kernel1d = _get_gaussian_kernel1d(kernel_size=kernel_size, sigma=sigma).to(
        tensor.device, dtype=tensor.dtype
    )
    out = tensor
    group = tensor.shape[1]

    if len(tensor.shape) - 2 == 1:
        out = torch.conv1d(out, kernel1d[None, None, :].expand(group,-1,-1), padding="same", groups=group)
    elif len(tensor.shape) - 2 == 2:
        out = torch.conv2d(out, kernel1d[None, None, :, None].expand(group,-1,-1,-1), padding="same", groups=group)
        out = torch.conv2d(out, kernel1d[None, None, None, :].expand(group,-1,-1,-1), padding="same", groups=group)
    elif len(tensor.shape) - 2 == 3:
        out = torch.conv3d(out, kernel1d[None, None, :, None, None].expand(group,-1,-1,-1,-1), padding="same", groups=group)
        out = torch.conv3d(out, kernel1d[None, None, None, :, None].expand(group,-1,-1,-1,-1), padding="same", groups=group)
        out = torch.conv3d(out, kernel1d[None, None, None, None, :].expand(group,-1,-1,-1,-1), padding="same", groups=group)

    return out


class LNCC(SimilarityBase):
    def __init__(self, sigma):
        super().__init__(isInterpolated=False)
        self.sigma = sigma

    def blur(self, tensor):
        return gaussian_blur(tensor, self.sigma * 4 + 1, self.sigma)

    def __call__(self, image_A, image_B):
        I = image_A
        J = image_B
        assert I.shape == J.shape, "The shape of image I and J sould be the same."

        return torch.mean(
            1
            - (self.blur(I * J) - (self.blur(I) * self.blur(J)))
            / torch.sqrt(
                (torch.relu(self.blur(I * I) - self.blur(I) ** 2) + 0.00001)
                * (torch.relu(self.blur(J * J) - self.blur(J) ** 2) + 0.00001)
            )
        )
    

class NewLNCC(SimilarityBase):
    def __init__(self, diffusion, model, sigma, eps=1e-6):
        super().__init__(isInterpolated=False)
        self.diffusion = diffusion
        self.model = model
        self.sigma = sigma
        self.eps = eps

    def blur(self, tensor):
        return gaussian_blur(tensor, self.sigma * 4 + 1, self.sigma)
    
    def lncc(self, A, B):
        return torch.mean(
            1
            - (self.blur(A * B) - (self.blur(A) * self.blur(B)))
            / torch.sqrt(
                (torch.relu(self.blur(A * A) - self.blur(A) ** 2) + self.eps)
                * (torch.relu(self.blur(B * B) - self.blur(B) ** 2) + self.eps)
            )
        )
    
    def cosine_similarity(self, A, B):
        prod_AB = torch.sum(A * B, dim=1)
        norm_A = torch.sum(A ** 2, dim=1).clamp(self.eps) ** 0.5
        norm_B = torch.sum(B ** 2, dim=1).clamp(self.eps) ** 0.5
        return torch.mean(prod_AB / (norm_A * norm_B))


    def __call__(self, image_A, image_B):

        t = 60
        up_ft_index = 10
        decoder_fts = True

        img_tensor = 2 * torch.vstack([image_A, image_B]) - 1
        img_tensor = img_tensor.repeat(1, 3, 1, 1)
        t = torch.ones((img_tensor.shape[0],), device=img_tensor.device, dtype=torch.int64) * t
        x_t = self.diffusion.q_sample(img_tensor, t, noise=None)
        hs = []
        emb = self.model.time_embed(timestep_embedding(t, self.model.model_channels))
        h = x_t.type(self.model.dtype)
        for i, module in enumerate(self.model.input_blocks):
            h = module(h, emb)
            if not decoder_fts:
                if up_ft_index == i:
                    ft_A = h[:image_A.shape[0]]
                    ft_B = h[image_A.shape[0]:]
                    break
            else:
                hs.append(h)
        if decoder_fts:
            h = self.model.middle_block(h, emb)
            for i, module in enumerate(self.model.output_blocks):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb)
                
                if i == up_ft_index:
                    ft_A = h[:image_A.shape[0]]
                    ft_B = h[image_A.shape[0]:]
                    break
        return self.lncc(ft_A, ft_B)
    

class NewLNCC3D(SimilarityBase):
    def __init__(self, diffusion, model, sigma, eps=1e-6):
        super().__init__(isInterpolated=False)
        self.diffusion = diffusion
        self.model = model
        self.sigma = sigma
        self.eps = eps

    def blur(self, tensor):
        return gaussian_blur(tensor, self.sigma * 4 + 1, self.sigma)
    
    def lncc(self, A, B):
        return torch.mean(
            1
            - (self.blur(A * B) - (self.blur(A) * self.blur(B)))
            / torch.sqrt(
                (torch.relu(self.blur(A * A) - self.blur(A) ** 2) + self.eps)
                * (torch.relu(self.blur(B * B) - self.blur(B) ** 2) + self.eps)
            )
        )
    
    def cosine_similarity(self, A, B):
        prod_AB = torch.sum(A * B, dim=1)
        norm_A = torch.sum(A ** 2, dim=1).clamp(self.eps) ** 0.5
        norm_B = torch.sum(B ** 2, dim=1).clamp(self.eps) ** 0.5
        return torch.mean(prod_AB / (norm_A * norm_B))


    def __call__(self, image_A, image_B):

        t = 50
        up_ft_index = 11
        decoder_fts = False
        
        axis = random.randint(0, 2)
        slices = random.sample(range(0, image_A.shape[axis+2]), 4)
        if axis == 0:
            img_tensor = 2 * torch.vstack([einops.rearrange(image_A[:,:,slices], 'd0 d1 d2 d3 d4 -> (d0 d2) d1 d3 d4'), einops.rearrange(image_B[:,:,slices], 'd0 d1 d2 d3 d4 -> (d0 d2) d1 d3 d4')]) - 1
        elif axis == 1:
            img_tensor = 2 * torch.vstack([einops.rearrange(image_A[:,:,:,slices], 'd0 d1 d2 d3 d4 -> (d0 d3) d1 d2 d4'), einops.rearrange(image_B[:,:,:,slices], 'd0 d1 d2 d3 d4 -> (d0 d3) d1 d2 d4')]) - 1
        elif axis == 2:
            img_tensor = 2 * torch.vstack([einops.rearrange(image_A[:,:,:,:,slices], 'd0 d1 d2 d3 d4 -> (d0 d4) d1 d2 d3'), einops.rearrange(image_B[:,:,:,:,slices], 'd0 d1 d2 d3 d4 -> (d0 d4) d1 d2 d3')]) - 1

        img_tensor = img_tensor.repeat(1, 3, 1, 1)
        t = torch.ones((img_tensor.shape[0],), device=img_tensor.device, dtype=torch.int64) * t
        x_t = self.diffusion.q_sample(img_tensor, t, noise=None)
        hs = []
        emb = self.model.time_embed(timestep_embedding(t, self.model.model_channels))
        h = x_t.type(self.model.dtype)
        for i, module in enumerate(self.model.input_blocks):
            h = module(h, emb)
            if not decoder_fts:
                if i == up_ft_index:
                    ft_A = h[:h.shape[0]//2]
                    ft_B = h[h.shape[0]//2:]
                    break
            else:
                hs.append(h)
        if decoder_fts:
            h = self.model.middle_block(h, emb)
            for i, module in enumerate(self.model.output_blocks):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb)
                
                if i == up_ft_index:
                    ft_A = h[:h.shape[0]//2]
                    ft_B = h[h.shape[0]//2:]
                    break
        return self.lncc(ft_A, ft_B)


class SquaredLNCC(LNCC):
    def __call__(self, image_A, image_B):
        I = image_A
        J = image_B
        assert I.shape == J.shape, "The shape of image I and J sould be the same."

        return torch.mean(
            1
            - ((self.blur(I * J) - (self.blur(I) * self.blur(J)))
            / torch.sqrt(
                (torch.relu(self.blur(I * I) - self.blur(I) ** 2) + 0.00001)
                * (torch.relu(self.blur(J * J) - self.blur(J) ** 2) + 0.00001)
            ))**2
        )

class LNCCOnlyInterpolated(SimilarityBase):
    def __init__(self, sigma):
        super().__init__(isInterpolated=True)
        self.sigma = sigma

    def blur(self, tensor):
        return gaussian_blur(tensor, self.sigma * 4 + 1, self.sigma)

    def __call__(self, image_A, image_B):

        I = image_A[:, :-1]
        J = image_B

        assert I.shape == J.shape, "The shape of image I and J sould be the same."
        lncc_everywhere = 1 - (
            self.blur(I * J) - (self.blur(I) * self.blur(J))
        ) / torch.sqrt(
            (self.blur(I * I) - self.blur(I) ** 2 + 0.00001)
            * (self.blur(J * J) - self.blur(J) ** 2 + 0.00001)
        )

        with torch.no_grad():
            A_inbounds = image_A[:, -1:]

            inbounds_mask = self.blur(A_inbounds) > 0.999

        if len(image_A.shape) - 2 == 3:
            dimensions_to_sum_over = [2, 3, 4]
        elif len(image_A.shape) - 2 == 2:
            dimensions_to_sum_over = [2, 3]
        elif len(image_A.shape) - 2 == 1:
            dimensions_to_sum_over = [2]

        lncc_loss = torch.sum(
            inbounds_mask * lncc_everywhere, dimensions_to_sum_over
        ) / torch.sum(inbounds_mask, dimensions_to_sum_over)

        return torch.mean(lncc_loss)


class BlurredSSD(SimilarityBase):
    def __init__(self, sigma):
        super().__init__(isInterpolated=False)
        self.sigma = sigma

    def blur(self, tensor):
        return gaussian_blur(tensor, self.sigma * 4 + 1, self.sigma)

    def __call__(self, image_A, image_B):
        assert image_A.shape == image_B.shape, "The shape of image_A and image_B sould be the same."
        return torch.mean((self.blur(image_A) - self.blur(image_B)) ** 2)


class AdaptiveNCC(SimilarityBase):
    def __init__(self, level=4, threshold=0.1, gamma=1.5, sigma=2):
        super().__init__(isInterpolated=False)
        self.level = level
        self.threshold = threshold
        self.gamma = gamma
        self.sigma = sigma

    def blur(self, tensor):
        return gaussian_blur(tensor, self.sigma * 2 + 1, self.sigma)

    def __call__(self, image_A, image_B):
        assert image_A.shape == image_B.shape, "The shape of image_A and image_B sould be the same."
        def _nccBeforeMean(image_A, image_B):
            A = normalize(image_A)
            B = normalize(image_B)
            res = torch.mean(A * B, dim=(1, 2, 3, 4))
            return 1 - res

        sims = [_nccBeforeMean(image_A, image_B)]
        for i in range(self.level):
            if i == 0:
                sims.append(_nccBeforeMean(self.blur(image_A), self.blur(image_B)))
            else:
                sims.append(
                    _nccBeforeMean(
                        self.blur(F.avg_pool3d(image_A, 2**i)),
                        self.blur(F.avg_pool3d(image_B, 2**i)),
                    )
                )

        sim_loss = sims[0] + 0
        lamb_ = 1.0
        for i in range(1, len(sims)):
            lamb = torch.clamp(
                sims[i].detach() / (self.threshold / (self.gamma ** (len(sims) - i))),
                0,
                1,
            )
            sim_loss = lamb * sims[i] + (1 - lamb) * sim_loss
            lamb_ *= 1 - lamb

        return torch.mean(sim_loss)

class SSD(SimilarityBase):
    def __init__(self):
        super().__init__(isInterpolated=False)

    def __call__(self, image_A, image_B):
        assert image_A.shape == image_B.shape, "The shape of image_A and image_B sould be the same."
        return torch.mean((image_A - image_B) ** 2)

class SSDOnlyInterpolated(SimilarityBase):
    def __init__(self):
        super().__init__(isInterpolated=True)

    def __call__(self, image_A, image_B):
        if len(image_A.shape) - 2 == 3:
            dimensions_to_sum_over = [2, 3, 4]
        elif len(image_A.shape) - 2 == 2:
            dimensions_to_sum_over = [2, 3]
        elif len(image_A.shape) - 2 == 1:
            dimensions_to_sum_over = [2]

        inbounds_mask = image_A[:, -1:]
        image_A = image_A[:, :-1]
        assert image_A.shape == image_B.shape, "The shape of image_A and image_B sould be the same."

        inbounds_squared_distance = inbounds_mask * (image_A - image_B) ** 2
        sum_squared_distance = torch.sum(inbounds_squared_distance, dimensions_to_sum_over)
        divisor = torch.sum(inbounds_mask, dimensions_to_sum_over)
        ssds = sum_squared_distance / divisor
        return torch.mean(ssds)

class MINDSSC(SimilarityBase):
    def __init__(self, radius:int=2, dilation:int=2):
        """
        Implementation of the MIND-SSC loss function
        See http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for the MIND-SSC descriptor. This ignores the center voxel, but compares adjacent voxels of the 6-neighborhood with each other.
        See http://mpheinrich.de/pub/MEDIA_mycopy.pdf for the original MIND loss function.

        Implementation retrieved from 
        https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/TransMorph/losses.py
        Annotation/Comments and support for 2d by Steffen Czolbe.

        Parameters:
            radius (int): determines size of patches around members of the 6-neighborhood
            dilation (int): determines spacing of members of the 6-neighborhood from the center voxel
        """
        super(MINDSSC, self).__init__()
        self.radius = radius
        self.dilation = dilation

    def pdist_squared(self, x):
        # for a list of length N of interger-valued pixel-coordinates, return an NxN matrix containing the squred euclidian distance between them:
        # 0: coordinates are the same
        # 1: coordinates are neighbours
        # 2: coordinates are diagonal
        # 4: coordinates are opposide, 2 apart

        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC3d(self, img):
        B, C, H, W, D = img.shape
        assert H>1 and W>1 and D>1, "Use 2d implementation for 2d data"

        # Radius: determines size of patches around members of the 6-neighborhood, square kernel
        kernel_size = self.radius * 2 + 1

        # define neighborhood for the self-similarity pattern. These coordinates are centered on [1, 1, 1]
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()

        # squared distances between neighborhood coordinates
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        # we compare adjacent neighborhood pixels (squared distance ==2), and exploid siymmetry to only calculate each pair once (x>y).
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernels to efficiently implement the pairwhise-patch based differences
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :] # 12x3 matrix. Pairing-pixel-coordinates of the first image
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :] # 12x3 matrix. Pairing-pixel-coordinates of the second image
        mshift1 = torch.zeros(12, 1, 3, 3, 3).to(img.device) # shifting-kernels, one channel to 12 channels. Each 3x3x3 kernel has ony a single 1
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).to(img.device)
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        rpad1 = torch.nn.ReplicationPad3d(self.dilation) # Padding to account for borders.
        rpad2 = torch.nn.ReplicationPad3d(self.radius)

        # compute patch-ssd

        # shift-align all 12 patch pairings, implemented via convolution
        h1 = torch.nn.functional.conv3d(rpad1(img), mshift1, dilation=self.dilation)
        h2 = torch.nn.functional.conv3d(rpad1(img), mshift2, dilation=self.dilation)
        # calculate difference 
        diff = rpad2((h1 - h2) ** 2)
        # convolve difference patches via averaging. This makes the loss magnitude invriant of patch size.
        ssd = torch.nn.functional.avg_pool3d(diff, kernel_size, stride=1)


        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0] # normalize by substracting lowest value
        mind_var = torch.mean(mind, 1, keepdim=True) # Mean across neighborhood pixel pairings (channels)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item()) # remove outliers
        mind /= mind_var
        mind = torch.exp(-mind)

        return mind


    def MINDSSC2d(self, img):
        # Radius: determines size of patches around members of the 4-neighborhood, square kernel
        kernel_size = self.radius * 2 + 1

        # define neighborhood for the self-similarity pattern. These coordinates are centered on [1, 1]
        four_neighbourhood = torch.Tensor([[0, 1],
                                            [1, 0],
                                            [2, 1],
                                            [1, 2]]).long()

        # squared distances between neighborhood coordinates
        dist = self.pdist_squared(four_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        # we compare adjacent neighborhood pixels (squared distance ==2), and exploid siymmetry to only calculate each pair once (x>y).
        x, y = torch.meshgrid(torch.arange(4), torch.arange(4))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernels to efficiently implement the pairwhise-patch based differences
        idx_shift1 = four_neighbourhood.unsqueeze(1).repeat(1, 4, 1).view(-1, 2)[mask, :] # 4x2 matrix. Pairing-pixel-coordinates of the first image
        idx_shift2 = four_neighbourhood.unsqueeze(0).repeat(4, 1, 1).view(-1, 2)[mask, :] # 4x2 matrix. Pairing-pixel-coordinates of the second image
        mshift1 = torch.zeros(4, 1, 3, 3).to(img.device) # shifting-kernels, one channel to 4 channels. Each 3x3 kernel has ony a single 1
        mshift1.view(-1)[torch.arange(4) * 9 + idx_shift1[:, 0] * 3 + idx_shift1[:, 1]] = 1
        mshift2 = torch.zeros(4, 1, 3, 3).to(img.device)
        mshift2.view(-1)[torch.arange(4) * 9 + idx_shift2[:, 0] * 3 + idx_shift2[:, 1]] = 1
        rpad1 = torch.nn.ReplicationPad2d(self.dilation) # Padding to account for borders.
        rpad2 = torch.nn.ReplicationPad2d(self.radius)

        # compute patch-ssd

        # shift-align all 4 patch pairings, implemented via convolution
        h1 = torch.nn.functional.conv2d(rpad1(img), mshift1, dilation=self.dilation)
        h2 = torch.nn.functional.conv2d(rpad1(img), mshift2, dilation=self.dilation)
        # calculate difference 
        diff = rpad2((h1 - h2) ** 2)
        # convolve difference patches via averaging. This makes the loss magnitude invriant of patch size.
        ssd = torch.nn.functional.avg_pool2d(diff, kernel_size, stride=1)


        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0] # normalize by substracting lowest value
        mind_var = torch.mean(mind, 1, keepdim=True) # Mean across neighborhood pixel pairings (channels)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item()) # remove outliers
        mind /= mind_var
        mind = torch.exp(-mind)

        return mind

    def forward(self, y_pred, y_true):
        # Get the MIND-SSC descriptor for each image
        if y_pred.dim() == 4:
            true = self.MINDSSC2d(y_true)
            pred = self.MINDSSC2d(y_pred)
        elif y_pred.dim() == 5:
            true = self.MINDSSC3d(y_true)
            pred = self.MINDSSC3d(y_pred)

        # calulate difference
        return torch.mean((true - pred) ** 2)

    def __call__(self, image_A, image_B):
        assert image_A.shape == image_B.shape, "The shape of image_A and image_B sould be the same."
        assert 2 <= len(image_A.shape) - 2 <= 3, "The input image should be 2D or 3D."
        if image_A.dim() == 4:
            return torch.mean((self.MINDSSC2d(image_A) - self.MINDSSC2d(image_B)) ** 2)
        elif image_A.dim() == 5:
            return torch.mean((self.MINDSSC3d(image_A) - self.MINDSSC3d(image_B)) ** 2)
        else:
            raise ValueError("The input image should be 2D or 3D.")
        
class NMI(SimilarityBase):
    """
    Normalized mutual information, using gaussian parzen window estimates.
    Adapted from https://github.com/qiuhuaqi/midir/blob/master/model/loss.py
    """

    def __init__(self,
                 vmin=0.0,
                 vmax=1.0,
                 num_bins=64,
                 sample_ratio=0.1,
                 normalised=True
                 ):
        super(NMI, self).__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.sample_ratio = sample_ratio
        self.normalised = normalised

        # set the std of Gaussian kernel so that FWHM is one bin width
        bin_width = (vmax - vmin) / num_bins
        self.sigma = bin_width * (1/(2 * math.sqrt(2 * math.log(2))))

        # set bin edges
        self.num_bins = num_bins
        self.bins = torch.linspace(
            self.vmin, self.vmax, self.num_bins, requires_grad=False).unsqueeze(1)

    def _compute_joint_prob(self, x, y):
        """
        Compute joint distribution and entropy
        Input shapes (N, 1, prod(sizes))
        """
        # cast bins
        self.bins = self.bins.type_as(x)

        # calculate Parzen window function response (N, #bins, H*W*D)
        win_x = torch.exp(-(x - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_x = win_x / (math.sqrt(2 * math.pi) * self.sigma)
        win_y = torch.exp(-(y - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_y = win_y / (math.sqrt(2 * math.pi) * self.sigma)

        # calculate joint histogram batch
        hist_joint = win_x.bmm(win_y.transpose(1, 2))  # (N, #bins, #bins)

        # normalise joint histogram to get joint distribution
        hist_norm = hist_joint.flatten(
            start_dim=1, end_dim=-1).sum(dim=1) + 1e-5
        # (N, #bins, #bins) / (N, 1, 1)
        p_joint = hist_joint / hist_norm.view(-1, 1, 1)

        return p_joint

    def forward(self, x, y):
        """
        Calculate (Normalised) Mutual Information Loss.
        Args:
            x: (torch.Tensor, size (N, 1, *sizes))
            y: (torch.Tensor, size (N, 1, *sizes))
        Returns:
            (Normalise)MI: (scalar)
        """
        if self.sample_ratio < 1.:
            # random spatial sampling with the same number of pixels/voxels
            # chosen for every sample in the batch
            numel_ = np.prod(x.size()[2:])
            idx_th = int(self.sample_ratio * numel_)
            idx_choice = torch.randperm(int(numel_))[:idx_th]

            x = x.view(x.size()[0], 1, -1)[:, :, idx_choice]
            y = y.view(y.size()[0], 1, -1)[:, :, idx_choice]

        # make sure the sizes are (N, 1, prod(sizes))
        x = x.flatten(start_dim=2, end_dim=-1)
        y = y.flatten(start_dim=2, end_dim=-1)

        # compute joint distribution
        p_joint = self._compute_joint_prob(x, y)

        # marginalise the joint distribution to get marginal distributions
        # batch size in dim0, x bins in dim1, y bins in dim2
        p_x = torch.sum(p_joint, dim=2)
        p_y = torch.sum(p_joint, dim=1)

        # calculate entropy
        ent_x = - torch.sum(p_x * torch.log(p_x + 1e-5), dim=1)  # (N,1)
        ent_y = - torch.sum(p_y * torch.log(p_y + 1e-5), dim=1)  # (N,1)
        ent_joint = - \
            torch.sum(p_joint * torch.log(p_joint + 1e-5), dim=(1, 2))  # (N,1)

        if self.normalised:
            return -torch.mean((ent_x + ent_y) / ent_joint)
        else:
            return -torch.mean(ent_x + ent_y - ent_joint)