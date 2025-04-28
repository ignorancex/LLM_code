import itertools
from models.unit_networks import *
import torch
from collections import OrderedDict
import os
import sys
import itertools
import torch
from util.image_pool import ImagePool
from util.util import *
from .base_model import BaseModel, OrderedDict
from . import networks
import os
import pyiqa
from .vgg import VGG
from loss_functions.attention import Self_Attn
from metrics.FID import *
from loss_functions.perceptual_loss import perceptual_similarity_loss
from loss_functions.texture_loss import texture_loss
from metrics.mse_psnr_ssim_vif import *
from models.networks import init_net
from piq import brisque


def compute_kl(mu):
    mu_2 = torch.pow(mu, 2)
    loss = torch.mean(mu_2)
    return loss


def ada_in(content_feature, mean_s, std_s, epsilon=1e-5):
    # Calculate mean and standard deviation of content feature
    mean_c = torch.mean(content_feature, dim=(2, 3), keepdim=True)
    std_c = torch.std(content_feature, dim=(2, 3), keepdim=True) + epsilon

    # Apply AdaIN formula
    normalized_content = std_s.unsqueeze(2).unsqueeze(3) * (content_feature - mean_c) / (std_c + 1e-5) + mean_s.unsqueeze(2).unsqueeze(3)

    return normalized_content


class UNITMODIFIEDModel(BaseModel):

    def __init__(self, opt):
        """Initialize the UNIT class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterionTexture = torch.nn.L1Loss(reduction='none')
        self.criterion_pixel = torch.nn.L1Loss()
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        input_shape = (opt.channels, opt.img_height, opt.img_width)
        self.Tensor = torch.FloatTensor if str(self.device).find("cpu") != -1 else torch.Tensor

        # Dimensionality (channel-wise) of image embedding
        shared_dim = opt.dim * 2 ** opt.n_downsample

        # Initialize generator and discriminator
        shared_E = ResidualBlock(features=shared_dim)
        self.E1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)
        self.E2 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)
        shared_G = ResidualBlock(features=shared_dim)
        self.G1 = Generator_AdaIn(dim=opt.dim, n_upsample=opt.n_downsample, shared_block=shared_G)
        self.G2 = Generator_AdaIn(dim=opt.dim, n_upsample=opt.n_downsample, shared_block=shared_G)
        self.D1 = Discriminator(input_shape)
        self.D2 = Discriminator(input_shape)

        self.Es1 = StyleEncoder().to(self.device)
        self.Es2 = StyleEncoder().to(self.device)

        self.E1 = self.E1.to(self.device)
        self.E2 = self.E2.to(self.device)
        self.G1 = self.G1.to(self.device)
        self.G2 = self.G2.to(self.device)
        self.D1 = self.D1.to(self.device)
        self.D2 = self.D2.to(self.device)
        self.criterion_GAN.to(self.device)
        self.criterion_pixel.to(self.device)

        # AdaIN params
        self.mean_s = 0
        self.std_s = 0
        self.test_mean_s = 0
        self.test_std_s = 0

        # Initialize weights
        self.E1.apply(weights_init_normal)
        self.E2.apply(weights_init_normal)
        self.G1.apply(weights_init_normal)
        self.G2.apply(weights_init_normal)
        self.D1.apply(weights_init_normal)
        self.D2.apply(weights_init_normal)

        # Optimizers
        if opt.texture_criterion == 'attention':
            self.attention = init_net(Self_Attn(1, 'relu'))
            self.weight_X1 = list()
            self.weight_X2 = list()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.E1.parameters(), self.E2.parameters(),
                                                                self.G1.parameters(), self.G2.parameters(),
                                                                self.attention.parameters()),
                                                lr=opt.lr,
                                                betas=(opt.b1, opt.b2),
                                                )
            self.model_names = ['E1', 'E2', 'G1', 'G2', 'D1', 'D2', "attention"]
        else:
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.E1.parameters(), self.E2.parameters(), self.G1.parameters(), self.G2.parameters()),
                lr=opt.lr,
                betas=(opt.b1, opt.b2),
            )
            self.model_names = ['E1', 'E2', 'G1', 'G2', 'D1', 'D2']

        self.optimizer_D1 = torch.optim.Adam(self.D1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D2 = torch.optim.Adam(self.D2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        # Learning rate update schedulers
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
        )
        self.lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
        )
        self.lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
        )

        if opt.lambda_perceptual > 0.0:
            if opt.vgg_pretrained:
                self.vgg = VGG().eval()
            else:
                self.vgg = VGG(num_classes=4, pretrained=False, init_type='finetuned', saved_weights_path=opt.vgg_model_path).eval()

        # dictionary to store training loss
        if opt.lambda_texture > 0:
            self.loss_names = ['GAN_1', 'GAN_2', 'KL_1', 'KL_2', 'ID_1', 'ID_2', 'KL_1_', 'KL_2_', 'cyc_1', 'cyc_2', "cycle_texture_X1", "cycle_texture_X2"]
        elif opt.lambda_perceptual > 0:
            self.loss_names = ['GAN_1', 'GAN_2', 'KL_1', 'KL_2', 'ID_1', 'ID_2', 'KL_1_', 'KL_2_', 'cyc_1', 'cyc_2', "perceptual_X1", "perceptual_X2"]
        else:
            self.loss_names = ['GAN_1', 'GAN_2', 'KL_1', 'KL_2', 'ID_1', 'ID_2', 'KL_1_', 'KL_2_', 'cyc_1', 'cyc_2']

        self.error_store = OrderedDict()
        for key in self.loss_names:
            self.error_store[key] = list()

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir

        # dictionary to store metrics
        self.metric_names = ['psnr', 'mse', 'ssim', 'vif', 'NIQE', 'FID', 'brisque']
        self.metrics_eval = OrderedDict()
        for key in self.metric_names:
            self.metrics_eval[key] = list()

        # metrics initialization
        self.fid_object = GANMetrics('cpu', detector_name='inceptionv3', batch_size=64)
        self.real_test_buffer = []
        self.fake_test_buffer = []
        self.raps = list()
        # NIQE metric
        self.niqe = pyiqa.create_metric('niqe', device=torch.device('cpu'), as_loss=False)

        # standardized buffer, not standardized buffer
        self.standardized_test_buffer = []
        self.not_standardized_test_buffer = []

        # Folders initialization
        self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
        self.img_dir = os.path.join(self.web_dir, f'{opt.image_folder}')
        self.test_dir = os.path.join(self.web_dir, f'{opt.test_folder}')
        self.loss_dir = os.path.join(self.web_dir, f'{opt.loss_folder}')
        self.metric_dir = os.path.join(self.web_dir, f'{opt.metric_folder}')
        print('create web directory %s...' % self.web_dir)
        mkdirs([self.web_dir, self.img_dir]) if (opt.image_folder is not None) else "pass"
        mkdirs([self.web_dir, self.loss_dir]) if (opt.loss_folder is not None) else "pass"
        mkdirs([self.web_dir, self.test_dir]) if (opt.test_folder is not None) else "pass"
        mkdirs([self.web_dir, self.metric_dir]) if (opt.metric_folder is not None) else "pass"

        self.opt = opt

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        if self.opt.dataset_mode == "LIDC_IDRI":
            self.X = input['img'].type(self.Tensor).expand(-1, 3, -1, -1).to(self.device)
            self.domain = input['domain'][0]
        else:
            # Set model input
            self.X1 = input["A"].type(self.Tensor).expand(-1, 3, -1, -1).to(self.device)
            self.X2 = input["B"].type(self.Tensor).expand(-1, 3, -1, -1).to(self.device)
            self.image_paths = input['A_paths']

            # Adversarial ground truths
            self.valid = Variable(self.Tensor(np.ones((self.X1.size(0), *self.D1.output_shape))), requires_grad=False)
            self.fake = Variable(self.Tensor(np.zeros((self.X1.size(0), *self.D1.output_shape))), requires_grad=False)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.optimizer_G.zero_grad()

        # Get shared latent representation
        self.mu1, self.Z1 = self.E1(self.X1)
        self.mu2, self.Z2 = self.E2(self.X2)

        self.mean_1, self.std_1 = self.Es1(self.X1)
        self.mean_2, self.std_2 = self.Es2(self.X2)

        # Reconstruct images
        self.recon_X1 = self.G1(self.Z1, self.mean_1, self.std_1)
        self.recon_X2 = self.G2(self.Z2, self.mean_2, self.std_2)

        # Translate images
        self.fake_X1 = self.G1(self.Z2, self.mean_1, self.std_1)
        self.fake_X2 = self.G2(self.Z1, self.mean_2, self.std_2)

        # Cycle translation
        self.mu1_, self.Z1_ = self.E1(self.fake_X1)
        self.mu2_, self.Z2_ = self.E2(self.fake_X2)

        self.mean_1, self.std_1 = self.Es1(self.fake_X1)
        self.mean_2, self.std_2 = self.Es2(self.fake_X2)

        self.cycle_X1 = self.G1(self.Z2_, self.mean_1, self.std_1)
        self.cycle_X2 = self.G2(self.Z1_, self.mean_2, self.std_2)

    def test(self):
        with torch.no_grad():
            if self.domain == "B30":
                _, self.Z1 = self.E1(self.X)
                self.fake = self.G1(self.Z1, self.test_mean_s, self.test_std_s)  # standardized

            elif self.domain == "D45":
                _, self.Z2 = self.E2(self.X)
                self.fake = self.G2(self.Z2, self.test_mean_s, self.test_std_s)  # standardized

    def backwark_encoderGen(self):
        self.loss_GAN_1 = self.opt.lambda_0 * self.criterion_GAN(self.D1(self.fake_X1), self.valid)
        self.loss_GAN_2 = self.opt.lambda_0 * self.criterion_GAN(self.D2(self.fake_X2), self.valid)
        self.loss_KL_1 = self.opt.lambda_1 * compute_kl(self.mu1)
        self.loss_KL_2 = self.opt.lambda_1 * compute_kl(self.mu2)
        self.loss_ID_1 = self.opt.lambda_2 * self.criterion_pixel(self.recon_X1, self.X1)
        self.loss_ID_2 = self.opt.lambda_2 * self.criterion_pixel(self.recon_X2, self.X2)
        self.loss_KL_1_ = self.opt.lambda_3 * compute_kl(self.mu1_)
        self.loss_KL_2_ = self.opt.lambda_3 * compute_kl(self.mu2_)
        self.loss_cyc_1 = self.opt.lambda_4 * self.criterion_pixel(self.cycle_X1, self.X1)
        self.loss_cyc_2 = self.opt.lambda_4 * self.criterion_pixel(self.cycle_X2, self.X2)

        # Texture Loss
        if self.opt.lambda_texture > 0:
            if self.opt.texture_criterion == 'attention':
                loss_texture_X1, _, weight_X1 = texture_loss(self.recon_X1, self.X1, self.criterionTexture, self.opt, self.attention)
                loss_texture_X2, _, weight_X2 = texture_loss(self.recon_X2, self.X2, self.criterionTexture, self.opt, self.attention)

                self.loss_cycle_texture_X1 = loss_texture_X1
                self.loss_cycle_texture_X2 = loss_texture_X2
                self.weight_X1.append(weight_X1.item())
                self.weight_X2.append(weight_X2.item())

            elif self.opt.texture_criterion == 'max':
                loss_texture_X1, _, _ = texture_loss(self.recon_X1, self.X1, self.criterionTexture, self.opt)
                loss_texture_X2, _, _ = texture_loss(self.recon_X2, self.X2, self.criterionTexture, self.opt)

                self.loss_cycle_texture_X1 = loss_texture_X1 * self.opt.lambda_texture
                self.loss_cycle_texture_X2 = loss_texture_X2 * self.opt.lambda_texture

            elif self.opt.texture_criterion == 'average':
                loss_texture_X1 = texture_loss(self.recon_X1, self.X1, self.criterionTexture, self.opt)
                loss_texture_X2 = texture_loss(self.recon_X2, self.X2, self.criterionTexture, self.opt)

                self.loss_cycle_texture_X1 = loss_texture_X1 * self.opt.lambda_texture
                self.loss_cycle_texture_X2 = loss_texture_X2 * self.opt.lambda_texture

            elif self.opt.texture_criterion == 'Frobenius':
                loss_texture_X1 = texture_loss(self.recon_X1, self.X1, self.criterionTexture, self.opt)
                loss_texture_X2 = texture_loss(self.recon_X2, self.X2, self.criterionTexture, self.opt)

                self.loss_cycle_texture_X1 = loss_texture_X1 * self.opt.lambda_texture
                self.loss_cycle_texture_X2 = loss_texture_X2 * self.opt.lambda_texture
            else:
                raise NotImplementedError
        else:
            self.loss_cycle_texture_X1 = 0
            self.loss_cycle_texture_X2 = 0

        # Perceptual loss
        if self.opt.lambda_perceptual > 0:
            loss_perceptual_X1 = perceptual_similarity_loss(self.recon_X1, self.X1, self.vgg, self.opt.perceptual_layers)
            loss_perceptual_X2 = perceptual_similarity_loss(self.recon_X2, self.X2, self.vgg, self.opt.perceptual_layers)
            self.loss_perceptual_X1 = loss_perceptual_X1 * self.opt.lambda_perceptual
            self.loss_perceptual_X2 = loss_perceptual_X2 * self.opt.lambda_perceptual
        else:
            self.loss_perceptual_X1 = 0
            self.loss_perceptual_X2 = 0

        # Total loss
        self.loss_G = (
                self.loss_KL_1
                + self.loss_KL_2
                + self.loss_ID_1
                + self.loss_ID_2
                + self.loss_GAN_1
                + self.loss_GAN_2
                + self.loss_KL_1_
                + self.loss_KL_2_
                + self.loss_cyc_1
                + self.loss_cyc_2
                + self.loss_cycle_texture_X1
                + self.loss_cycle_texture_X2
                + self.loss_perceptual_X1
                + self.loss_perceptual_X2

        )
        self.loss_G.backward()

    def backward_D1(self):
        self.loss_D1 = self.criterion_GAN(self.D1(self.X1), self.valid) + self.criterion_GAN(self.D1(self.fake_X1.detach()), self.fake)
        self.loss_D1.backward()

    def backward_D2(self):
        self.loss_D2 = self.criterion_GAN(self.D2(self.X2), self.valid) + self.criterion_GAN(self.D2(self.fake_X2.detach()), self.fake)
        self.loss_D2.backward()

    def optimize_parameters(self):
        self.forward()

        self.backwark_encoderGen()
        self.optimizer_G.step()

        self.optimizer_D1.zero_grad()
        self.backward_D1()
        self.optimizer_D1.step()

        self.optimizer_D2.zero_grad()
        self.backward_D2()
        self.optimizer_D2.step()

        self.combine_adain_params([self.mean_1, self.mean_2], [self.std_1, self.std_2])

    def combine_adain_params(self, means, stds):
        self.mean_s = sum(means) / len(means)
        self.std_s = sum(stds) / len(stds)


    def update_learning_rate(self):
        # Update learning rates
        self.lr_scheduler_G.step()
        self.lr_scheduler_D1.step()
        self.lr_scheduler_D2.step()

    #############################################
    def train(self):
        """Make models train mode after test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def get_current_losses(self):
        """Return traning losses / errors"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def track_current_losses(self):
        """Return traning losses / errors so far."""
        for name in self.loss_names:
            if isinstance(name, str):
                self.error_store[name].append(
                    float(getattr(self, 'loss_' + name)))  # float(...) works for both scalar tensor and float number

        return self.error_store

    def print_current_loss(self, epoch, i, dataset_len, time_left):
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
            % (epoch, self.opt.n_epochs, dataset_len, i, (self.loss_D1 + self.loss_D2).item(), self.loss_G.item(), time_left)
        )

    def print_time(self, epoch, delta_time):
        message = 'End of epoch %d / %d \t Time Taken: %d sec' % (epoch, self.opt.n_epochs, delta_time)
        sys.stdout.write(message)

    def plot_current_losses(self, epoch, tracked_losses, name_title):
        loss_path = os.path.join(self.loss_dir, f'epoch.png')
        if self.opt.n_epochs == epoch:
            save_ordered_dict_as_csv(tracked_losses, f'{self.loss_dir}/loss.csv')
        save_plots(tracked_losses, loss_path, name_title)  # save the plots to the disk

    def save_networks(self, epoch):
        torch.save(self.E1.state_dict(), f"{self.save_dir}/E1_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.E2.state_dict(), f"{self.save_dir}/E2_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.G1.state_dict(), f"{self.save_dir}/G1_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.G2.state_dict(), f"{self.save_dir}/G2_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.D1.state_dict(), f"{self.save_dir}/D1_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.D2.state_dict(), f"{self.save_dir}/D2_ep{epoch}_{self.opt.experiment_name}.pth")

    def compute_metrics(self):
        if self.opt.dataset_mode == "LIDC_IDRI":
            # NIQE
            fake_X2_3channels = self.fake_X2.expand(-1, 3, -1, -1)
            self.NIQE = self.niqe(fake_X2_3channels).item()
            self.raps.append(azimuthalAverage(np.squeeze(self.fake_X2[0, 0, :, :].cpu().detach().numpy())).tolist())
            self.brisque = 0
            self.mse = 0
            self.psnr = 0
            self.ssim = 0
            self.vif = 0
        elif self.opt.test == "elcap_complete":
            # NIQE
            fake_X2_3channels = self.fake_X2.expand(-1, 3, -1, -1)
            self.NIQE = self.niqe(fake_X2_3channels).item()
            self.raps.append(azimuthalAverage(np.squeeze(self.fake_X2[0, 0, :, :].cpu().detach().numpy())).tolist())
            self.brisque = 0
            self.mse = 0
            self.psnr = 0
            self.ssim = 0
            self.vif = 0
        else:
            x = tensor2im2(self.X2)
            y = tensor2im2(self.fake_X2)
            # MSE
            self.mse = mean_squared_error(x, y)
            # PSNR
            self.psnr = peak_signal_to_noise_ratio(x, y)
            # SSIM
            self.ssim = structural_similarity_index(x, y)
            # NIQE
            fake_X2_3channels = self.fake_X2.expand(-1, 3, -1, -1)
            self.NIQE = self.niqe(fake_X2_3channels).item()
            self.brisque = 0
            # VIF
            self.vif = vif(x, y)
            self.fake_test_buffer.append(self.fake_X2)
            self.real_test_buffer.append(self.X2)

    def compute_FrechetInceptionDistance(self):
        real_buffer = torch.cat(self.real_test_buffer, dim=0)
        fake_buffer = torch.cat(self.fake_test_buffer, dim=0)

        fid_score = self.fid_object.compute_fid(fake_buffer, real_buffer, self.opt.dataset_len)
        self.metrics_eval['FID'].append(fid_score)

    def track_metrics(self):
        for name in self.metric_names:
            if isinstance(name, str) and name != 'FID':
                self.metrics_eval[name].append(
                    float(getattr(self, name)))  # float(...) works for both scalar tensor and float number

    def get_epoch_performance(self):
        return self.metrics_eval

    def save_metrics(self, epoch_performance, epoch):
        csv_path2 = os.path.join(self.metric_dir, f'metrics_{self.opt.test}_epoch{epoch}.csv')
        save_ordered_dict_as_csv(epoch_performance, csv_path2)
        empty_dictionary(self.metrics_eval)

    def save_raps(self, epoch):
        save_json(self.raps, f"{self.metric_dir}/raps_{self.opt.test}_epoch{epoch}")
        self.raps = []

    def save_test_images(self, epoch):
        standardized_test = torch.cat(self.standardized_test_buffer, dim=0)
        not_standardized_test = torch.cat(self.not_standardized_test_buffer, dim=0)
        torch.save(standardized_test, f'{self.test_dir}/standardized_{self.opt.test}_epoch{epoch}.pth')
        torch.save(not_standardized_test, f'{self.test_dir}/not_standardized_{self.opt.test}_epoch{epoch}.pth')
        self.standardized_test_buffer = []
        self.not_standardized_test_buffer = []

    def save_adaIN(self, epoch):
        torch.save(self.mean_s, f"{self.save_dir}/mean_s_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.std_s, f"{self.save_dir}/std_s_ep{epoch}_{self.opt.experiment_name}.pth")

    def average_adaiN(self):

        self.test_mean_s = torch.mean(self.mean_s, dim=0, keepdim=True)
        self.test_std_s = torch.mean(self.std_s, dim=0, keepdim=True)
