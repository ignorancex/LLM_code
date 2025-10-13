import torch
import torch.nn as nn

from src.losses.shared import rand_projections


class ProjNet(nn.Module):
    def __init__(self, size):
        super(ProjNet, self).__init__()
        self.size = size
        self.net = nn.Linear(self.size, self.size)

    def forward(self, input):
        out = self.net(input)
        return out / (torch.sqrt(torch.sum((out) ** 2, dim=1, keepdim=True)) + 1e-20)


class DSW(nn.Module):
    def __init__(self, encoder, embedding_norm, num_projections, projnet, op_projnet,
                 p=2, max_iter=100, lam=1):
        super(DSW, self).__init__()
        self.encoder = encoder
        self.embedding_norm = embedding_norm
        self.embedding_norm = embedding_norm
        self.num_projections = num_projections
        self.projnet = projnet
        self.op_projnet = op_projnet
        self.p = p
        self.max_iter = max_iter
        self.lam = lam

    def __cosine_distance_torch(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return torch.mean(torch.abs(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)))

    @torch.enable_grad()
    @torch.inference_mode(mode=False)
    def distributional_sliced_wasserstein_distance(self, first_samples, second_samples):
        embedding_dim = first_samples.size(1)
        pro = rand_projections(embedding_dim, self.num_projections).to(first_samples)
        # sometimes first_samples and second_samples come from a context where torch.inference_mode is enabled so we need
        # to clone them first.
        first_samples_detach = first_samples.detach().clone()
        second_samples_detach = second_samples.detach().clone()
        # this tmp variable is for debugging purposes.
        tmp = []
        for _ in range(self.max_iter):
            projections = self.projnet(pro)
            cos = self.__cosine_distance_torch(projections, projections)
            reg = self.lam * cos
            encoded_projections = first_samples_detach.matmul(
                projections.transpose(0, 1))
            distribution_projections = second_samples_detach.matmul(
                projections.transpose(0, 1))
            wasserstein_distance = torch.abs(
                (
                        torch.sort(encoded_projections.transpose(0, 1), dim=1)[0]
                        - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]
                )
            )
            wasserstein_distance = torch.pow(
                torch.sum(torch.pow(wasserstein_distance + 1e-20, self.p), dim=1), 1.0 / self.p)
            wasserstein_distance = torch.pow(
                torch.pow(wasserstein_distance + 1e-20, self.p).mean(), 1.0 / self.p)
            loss = reg - wasserstein_distance
            self.op_projnet.zero_grad()
            loss.backward()
            self.op_projnet.step()
            tmp.append(wasserstein_distance.item())
        with torch.inference_mode():
            projections = self.projnet(pro)
        projections = projections.clone()
        encoded_projections = first_samples.matmul(projections.transpose(0, 1))
        distribution_projections = second_samples.matmul(projections.transpose(0, 1))
        wasserstein_distance = torch.abs(
            (
                    torch.sort(encoded_projections.transpose(0, 1), dim=1)[0]
                    - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]
            )
        )
        wasserstein_distance = torch.pow(
            torch.sum(torch.pow(wasserstein_distance, self.p), dim=1), 1.0 / self.p)
        wasserstein_distance = torch.pow(
            torch.pow(wasserstein_distance, self.p).mean(), 1.0 / self.p)
        return wasserstein_distance

    def forward(self, first_sample, second_sample):
        if self.encoder is None:
            data = second_sample
            data_fake = first_sample
        else:
            data = self.encoder(second_sample) / self.embedding_norm
            data_fake = self.encoder(first_sample) / self.embedding_norm
        print(data.shape)
        print(data_fake.shape)
        _dswd = self.distributional_sliced_wasserstein_distance(
            data.view(data.shape[0], -1),
            data_fake.view(data.shape[0], -1)
        )
        return _dswd
