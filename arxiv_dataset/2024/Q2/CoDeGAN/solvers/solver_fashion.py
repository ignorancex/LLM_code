import torch
import torch.nn as nn
from torch import autograd
from utils.util import train_transform_simple, sample_noise, calc_gradient_penalty
from utils.sup_contrastive_loss import SupConLoss
from models.FashionMNIST.model import Generator, Discriminator, Encoder


class Solver:
    def __init__(self, config):
        self.zc_dim = config["zc_dim"]
        self.zn_dim = config["zn_dim"]
        self.fc_dim = config["fc_dim"]

        self.beta_1 = config["beta_1"]
        self.beta_2 = config["beta_2"]

        self.lr = config["lr"]
        self.temp_C = config["temp_C"]
        self.wgan_gp = config["wgan_gp"]
        self.device = "cuda:{}".format(config["gpu_id"])

        self.g_batch_size = config["g_batch_size"]
        self.d_batch_size = config["d_batch_size"]
        self.e_batch_size = config["e_batch_size"]
        self.r_batch_size = config["r_batch_size"]

        self.G = Generator(self.zn_dim, self.zc_dim).to(self.device)
        self.D = Discriminator(self.wgan_gp).to(self.device)
        self.E = Encoder(self.zn_dim, self.fc_dim).to(self.device)

        self.optimizer_G = torch.optim.Adam(
            self.G.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=2.5 * 1e-5
        )
        self.optimizer_D = torch.optim.Adam(
            self.D.parameters(), lr=self.lr, betas=(0.5, 0.9)
        )
        self.optimizer_E = torch.optim.Adam(
            self.E.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=2.5 * 1e-5
        )

        self.bceloss = nn.BCELoss().to(self.device)
        self.mseloss = nn.MSELoss().to(self.device)
        self.C_loss = SupConLoss(temperature=self.temp_C, device=self.device).to(
            self.device
        )

    def forward_G(self, z, zc, real_data):
        fake_q = self.G(z)
        d_fake = self.D(fake_q)

        # add contrastive
        fake_k = train_transform_simple(fake_q).to(self.device)

        fz, fc_q = self.E(fake_q)
        _, fc_k = self.E(fake_k)

        # add few labels
        if real_data is not None:
            (real_q, real_label) = real_data
            real_q = real_q.to(self.device)
            real_k = train_transform_simple(real_q)

            zc = torch.cat([zc, real_label.to(self.device)], dim=0)
            fc_q = torch.cat([fc_q, self.E(real_q)[1]], dim=0)
            fc_k = torch.cat([fc_k, self.E(real_k)[1]], dim=0)

        fc = torch.cat([fc_q.unsqueeze(1), fc_k.unsqueeze(1)], dim=1)

        if not self.wgan_gp:
            self.lvalid = autograd.Variable(
                torch.FloatTensor(fake_q.size(0), 1).fill_(1.0), requires_grad=False
            ).to(self.device)

        return d_fake, fz, fc, zc

    def optimize_parametersG(self, batch_size=None, real_data=None):
        batch_size = self.g_batch_size if batch_size == None else batch_size
        self.set_requires_grad(self.D, False)
        self.set_requires_grad(self.G, True)
        self.set_requires_grad(self.E, False)
        self.optimizer_G.zero_grad()

        z, zn, zc = sample_noise(batch_size, self.zn_dim, self.zc_dim, self.device)
        d_fake, fz, fc, zc = self.forward_G(z, zc, real_data)

        ecloss = self.C_loss(fc, zc)
        ezloss = self.mseloss(fz, zn)
        eloss = self.beta_1 * ecloss + self.beta_2 * ezloss
        if self.wgan_gp:
            gloss = torch.mean(d_fake) + eloss
        else:
            gloss = self.bceloss(d_fake, self.lvalid) + eloss

        gloss.backward(retain_graph=True)
        self.optimizer_G.step()
        return gloss.item()

    def forward_E(self, z, zc, real_data):
        # add contrastive
        fake_q = self.G(z)
        fake_k = train_transform_simple(fake_q)

        fz, fc_q = self.E(fake_q)
        _, fc_k = self.E(fake_k)

        # add few labels
        if real_data is not None:
            (real_q, real_label) = real_data
            real_q = real_q.to(self.device)
            real_k = train_transform_simple(real_q)
            zc = torch.cat([zc, real_label.to(self.device)], dim=0)
            fc_q = torch.cat([fc_q, self.E(real_q)[1]], dim=0)
            fc_k = torch.cat([fc_k, self.E(real_k)[1]], dim=0)

        fc = torch.cat([fc_q.unsqueeze(1), fc_k.unsqueeze(1)], dim=1)

        return fz, fc, zc

    def optimize_parametersE(self, batch_size=None, real_data=None, ec=False):
        if ec == False:
            return 0, 0
        batch_size = self.e_batch_size if batch_size == None else batch_size
        self.set_requires_grad(self.D, False)
        self.set_requires_grad(self.G, False)
        self.set_requires_grad(self.E, True)
        self.optimizer_E.zero_grad()

        z, zn, zc = sample_noise(batch_size, self.zn_dim, self.zc_dim, self.device)
        fz, fc, zc = self.forward_E(z, zc, real_data)

        ecloss = self.C_loss(fc, zc)
        ezloss = self.mseloss(fz, zn)
        eloss = ecloss + (self.beta_2 / self.beta_1) * ezloss

        eloss.backward()
        self.optimizer_E.step()
        return ecloss.item(), ezloss.item()

    def forward_D(self, real, fake):
        d_real = self.D(real)
        d_fake_detach = self.D(fake.detach())

        if not self.wgan_gp:
            self.lvalid = autograd.Variable(
                torch.FloatTensor(fake.size(0), 1).fill_(1.0), requires_grad=False
            ).to(self.device)
            self.lfake = autograd.Variable(
                torch.FloatTensor(fake.size(0), 1).fill_(0.0), requires_grad=False
            ).to(self.device)
        return d_real, d_fake_detach

    def optimize_parametersD(self, data, batch_size=None):
        batch_size = self.d_batch_size if batch_size == None else batch_size
        self.set_requires_grad(self.D, True)
        self.set_requires_grad(self.G, False)
        self.set_requires_grad(self.E, False)
        self.optimizer_D.zero_grad()

        z = sample_noise(batch_size, self.zn_dim, self.zc_dim, self.device)[0]
        real = data.to(self.device)
        fake = self.G(z)
        d_real, d_fake_detach = self.forward_D(real, fake)

        if self.wgan_gp:
            grad_penalty = calc_gradient_penalty(real, fake, self.D)
            dloss = torch.mean(d_real) - torch.mean(d_fake_detach) + grad_penalty
        else:
            real_loss = self.bceloss(d_real, self.lvalid)
            fake_loss = self.bceloss(d_fake_detach, self.lfake)
            dloss = (real_loss + fake_loss) / 2

        dloss.backward()
        self.optimizer_D.step()
        return dloss.item()

    def set_requires_grad(self, nets, requires_grad=False):
        for param in nets.parameters():
            param.requires_grad_(requires_grad)

    def train(
        self,
    ):
        self.G.train()
        self.E.train()
        self.D.train()

    def eval(
        self,
    ):
        self.G.eval()
        self.E.eval()

    def save_model(self, save_path, epoch=0):
        torch.save(self.G.state_dict(), save_path + "/{}_G.pth".format(epoch + 1))
        torch.save(self.D.state_dict(), save_path + "/{}_D.pth".format(epoch + 1))
        torch.save(self.E.state_dict(), save_path + "/{}_E.pth".format(epoch + 1))
