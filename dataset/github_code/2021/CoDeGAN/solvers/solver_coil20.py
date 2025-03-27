import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import *
from utils.sup_contrastive_loss import SupConLoss
from models.COIL20.Generator import Generator
from models.COIL20.Discriminator import Discriminator
from models.COIL20.Encoder import Classifier, EncoderC, EncoderZ


class Solver:
    def __init__(self, config):
        self.zn_dim = config["zn_dim"]
        self.zc_dim = config["zc_dim"]
        self.fc_dim = config["fc_dim"]

        self.beta_2 = config["beta_2"]
        self.beta_1 = config["beta_1"]

        self.lr = config["lr"]
        self.temp_C = config["temp_C"]
        self.loss_metric = config["loss_metric"]
        self.device = "cuda:{}".format(config["gpu_id"])

        self.g_batch_size = config["g_batch_size"]
        self.d_batch_size = config["d_batch_size"]
        self.ec_batch_size = config["ec_batch_size"]
        self.ez_batch_size = config["ez_batch_size"]

        self.D = Discriminator().to(self.device)
        self.G = Generator(dim_z=self.zn_dim + self.zc_dim).to(self.device)
        self.EC = EncoderC(name="resnet18", feat_dim=self.fc_dim).to(self.device)
        self.EZ = EncoderZ(name="resnet18", feat_dim=self.zn_dim).to(self.device)
        self.Classifier = Classifier().to(self.device)

        self.optimizer_G = torch.optim.Adam(
            self.G.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.D.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )
        self.optimizer_EC = torch.optim.Adam(
            self.EC.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )
        self.optimizer_EZ = torch.optim.Adam(
            self.EZ.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )

        self.bceloss = nn.BCELoss().to(self.device)
        self.mseloss = nn.MSELoss().to(self.device)
        self.C_loss = SupConLoss(device=self.device, temperature=self.temp_C).to(
            self.device
        )

    def forward_G(self, z, zc, batch_size, real_data):
        img_q = self.G(z)
        d_fake = self.D(img_q)

        # add contrastive
        img_k = train_transform(img_q).to(self.device)
        fz = self.EZ(img_q)

        images = torch.cat([img_q, img_k], dim=0)
        fc = self.EC(images)
        f1, f2 = torch.split(fc, [batch_size, batch_size], dim=0)
        fc = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        return d_fake, fz, fc, zc

    def optimize_parametersG(self, batch_size=None, real_data=None):
        batch_size = self.g_batch_size if batch_size == None else batch_size
        self.set_requires_grad(self.D, False)
        self.set_requires_grad(self.G, True)
        self.set_requires_grad(self.EC, False)
        self.set_requires_grad(self.EZ, False)
        self.optimizer_G.zero_grad()

        z, zn, zc = sample_noise(batch_size, self.zn_dim, self.zc_dim, self.device)
        d_fake, fz, fc, zc = self.forward_G(z, zc, batch_size, real_data)

        ecloss = self.C_loss(fc, zc)
        ezloss = self.mseloss(fz, zn)
        eloss = self.beta_1 * ecloss + self.beta_2 * ezloss

        if self.loss_metric == "hinge":
            gloss = -torch.mean(d_fake.view(-1))
        elif self.loss_metric == "wgan-gp":
            gloss = torch.mean(d_fake)
        else:
            raise ValueError("Unknown Loss Metric")

        loss = gloss + eloss
        loss.backward()
        self.optimizer_G.step()
        return gloss.item()

    def forward_EC(self, z, zc, batch_size, real_data):
        # add contrastive
        img_q = self.G(z)
        img_k = train_transform(img_q).to(self.device)

        images = torch.cat([img_q, img_k], dim=0)
        fc = self.EC(images)
        f1, f2 = torch.split(fc, [batch_size, batch_size], dim=0)
        fc = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        return fc, zc

    def optimize_parametersEC(self, batch_size, real_data=None):
        self.set_requires_grad(self.D, False)
        self.set_requires_grad(self.G, False)
        self.set_requires_grad(self.EC, True)
        self.set_requires_grad(self.EZ, False)
        self.optimizer_EC.zero_grad()

        z, _, zc = sample_noise(batch_size, self.zn_dim, self.zc_dim, self.device)
        fc, zc = self.forward_EC(z, zc, batch_size, real_data)

        closs = self.C_loss(fc, zc)

        closs.backward()
        self.optimizer_EC.step()
        return closs.item()

    def forward_EZ(self, z):
        fake = self.G(z)
        fz = self.EZ(fake)
        return fz

    def optimize_parametersEZ(self, batch_size):
        self.set_requires_grad(self.D, False)
        self.set_requires_grad(self.G, False)
        self.set_requires_grad(self.EC, False)
        self.set_requires_grad(self.EZ, True)
        self.optimizer_EZ.zero_grad()

        z, zn, _ = sample_noise(batch_size, self.zn_dim, self.zc_dim, self.device)
        fz = self.forward_EZ(z)

        zloss = self.mseloss(fz, zn)

        zloss.backward()
        self.optimizer_EZ.step()
        return zloss.item()

    def optimize_parametersE(self, real_data=None, ec=False):
        ez_loss = self.optimize_parametersEZ(
            batch_size=self.ez_batch_size,
        )
        if ec:
            ec_loss = self.optimize_parametersEC(
                batch_size=self.ec_batch_size, real_data=real_data
            )
        else:
            ec_loss = 0
        return ez_loss, ec_loss

    def forward_D(self, real, fake):
        d_real = self.D(real)
        d_fake_detach = self.D(fake.detach())
        return d_real, d_fake_detach

    def optimize_parametersD(self, data, batch_size=None):
        batch_size = self.d_batch_size if batch_size == None else batch_size
        self.set_requires_grad(self.D, True)
        self.set_requires_grad(self.G, False)
        self.set_requires_grad(self.EC, False)
        self.set_requires_grad(self.EZ, False)
        self.optimizer_D.zero_grad()

        z = sample_noise(batch_size, self.zn_dim, self.zc_dim, self.device)[0]
        real = data.to(self.device)
        fake = self.G(z)
        d_real, d_fake_detach = self.forward_D(real, fake)

        if self.loss_metric == "hinge":
            dloss = torch.mean(F.relu(1 - d_real.view(-1))) + torch.mean(
                F.relu(1 + d_fake_detach.view(-1))
            )
        elif self.loss_metric == "wgan-gp":
            dloss = (
                torch.mean(d_real)
                - torch.mean(d_fake_detach)
                + calc_gradient_penalty(real, fake.detach(), self.D)
            )
        else:
            raise ValueError("Unknown Loss Metric")

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
        self.D.train()
        self.EC.train()
        self.EZ.train()

    def eval(
        self,
    ):
        self.G.eval()
        self.EC.eval()

    def save_model(self, save_path, epoch=0):
        torch.save(self.G.state_dict(), save_path + "/{}_G.pth".format(epoch + 1))
        torch.save(self.D.state_dict(), save_path + "/{}_D.pth".format(epoch + 1))
        torch.save(self.EZ.state_dict(), save_path + "/{}_EZ.pth".format(epoch + 1))
        torch.save(self.EC.state_dict(), save_path + "/{}_EC.pth".format(epoch + 1))
