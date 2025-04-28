import itertools
from models.unit_networks import *
import torch
from collections import OrderedDict
import os
import sys
import itertools
import torch
from data.storage import *
from util.image_pool import ImagePool
from util.util import *
from .base_model import BaseModel, OrderedDict
from . import networks
import os
import pyiqa
from .vgg import VGG
from loss_functions.attention import Self_Attn
from metrics.FID import *
from loss_functions.glcm_soft_einsum import _texture_loss, _GridExtractor
from loss_functions.perceptual_loss import perceptual_similarity_loss
# from loss_functions.texture_loss import texture_loss
from metrics.mse_psnr_ssim_vif import *
from models.networks import init_net
from piq import brisque
from scipy.stats import skew
from torchmetrics.functional.image import image_gradients
import cv2 as cv
import time
from metrics.piqe import *
from loss_functions.edge_loss import *
from piq import SSIMLoss
from metrics.piqe import *
from models.autoencoder_perceptual import *


def compute_kl(mu):
    mu_2 = torch.pow(mu, 2)
    loss = torch.mean(mu_2)
    return loss


# Function to add a patient to the dictionary
def add_patient(data, patient_id, other_patient_fields=None):
    if patient_id not in data:
        data[patient_id] = {'images': {}}
        if other_patient_fields:
            data[patient_id].update(other_patient_fields)


# Function to add an image to a patient in the dictionary
def add_image(data, patient_id, image_id, other_image_fields=None):
    image_data = {'computed_values': {}}
    if other_image_fields:
        image_data.update(other_image_fields)
    data[patient_id]['images'][image_id] = image_data


# Function to add computed values to an image for a patient in the dictionary
def add_computed_values(data, patient_id, image_id, values):
    data[patient_id]['images'][image_id]['computed_values'] = values


def template_matching(ld, hd, deno, template, method='cv.TM_CCOEFF_NORMED'):
    method = eval(method)
    res_1 = cv.matchTemplate(ld[0, 0, :, :], template, method)
    min_val_1, max_val_1, min_loc_1, max_loc_1 = cv.minMaxLoc(res_1)
    res_2 = cv.matchTemplate(deno[0, 0, :, :], template, method)
    min_val_2, max_val_2, min_loc_2, max_loc_2 = cv.minMaxLoc(res_2)
    res_3 = cv.matchTemplate(hd[0, 0, :, :], template, method)
    min_val_3, max_val_3, min_loc_3, max_loc_3 = cv.minMaxLoc(res_3)

    return {"ld": [min_val_1, max_val_1, min_loc_1, max_loc_1], "deno": [min_val_2, max_val_2, min_loc_2, max_loc_2], "hd": [min_val_3, max_val_3, min_loc_3, max_loc_3]}


def _template_matching(ld, deno, template, method='cv.TM_CCOEFF_NORMED'):
    method = eval(method)
    res_1 = cv.matchTemplate(ld[0, 0, :, :], template, method)
    min_val_1, max_val_1, min_loc_1, max_loc_1 = cv.minMaxLoc(res_1)
    res_2 = cv.matchTemplate(deno[0, 0, :, :], template, method)
    min_val_2, max_val_2, min_loc_2, max_loc_2 = cv.minMaxLoc(res_2)

    return {"ld": [min_val_1, max_val_1, min_loc_1, max_loc_1], "deno": [min_val_2, max_val_2, min_loc_2, max_loc_2]}


margin = 48
crop_size = 32
x_coord = [(margin, margin + crop_size), (margin + 2 * crop_size, margin + 3 * crop_size), (margin + 4 * crop_size, margin + 5 * crop_size),
           (margin, margin + crop_size), (margin + 2 * crop_size, margin + 3 * crop_size), (margin + 4 * crop_size, margin + 5 * crop_size),
           (margin, margin + crop_size), (margin + 2 * crop_size, margin + 3 * crop_size), (margin + 4 * crop_size, margin + 5 * crop_size)]

y_coord = [(margin, margin + crop_size), (margin, margin + crop_size), (margin, margin + crop_size),
           (margin + 2 * crop_size, margin + 3 * crop_size), (margin + 2 * crop_size, margin + 3 * crop_size), (margin + 2 * crop_size, margin + 3 * crop_size),
           (margin + 4 * crop_size, margin + 5 * crop_size), (margin + 4 * crop_size, margin + 5 * crop_size), (margin + 4 * crop_size, margin + 5 * crop_size)]


class UNITModel(BaseModel):

    def __init__(self, opt):
        """Initialize the UNIT class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
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
        self.E1 = Encoder(in_channels=opt.channels, dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E).to(self.device)
        self.E2 = Encoder(in_channels=opt.channels, dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E).to(self.device)
        shared_G = ResidualBlock(features=shared_dim)
        self.G1 = Generator(out_channels=opt.channels, dim=opt.dim, n_upsample=opt.n_downsample, shared_block=shared_G).to(self.device)
        self.G2 = Generator(out_channels=opt.channels, dim=opt.dim, n_upsample=opt.n_downsample, shared_block=shared_G).to(self.device)
        self.D1 = Discriminator(input_shape).to(self.device)
        self.output_shape = self.D1.output_shape
        self.D2 = Discriminator(input_shape).to(self.device)

        self.E1 = nn.DataParallel(self.E1, device_ids=[0, 1])  # .to(self.device)
        self.E2 = nn.DataParallel(self.E2, device_ids=[0, 1])  # .to(self.device)
        self.G1 = nn.DataParallel(self.G1, device_ids=[0, 1])  # .to(self.device)
        self.G2 = nn.DataParallel(self.G2, device_ids=[0, 1])  # .to(self.device)
        self.D1 = nn.DataParallel(self.D1, device_ids=[0, 1])  # .to(self.device)
        self.D2 = nn.DataParallel(self.D2, device_ids=[0, 1])  # .to(self.device)
        self.criterion_GAN.to(self.device)
        self.criterion_pixel.to(self.device)

        # Initialize weights
        self.E1.apply(weights_init_normal)
        self.E2.apply(weights_init_normal)
        self.G1.apply(weights_init_normal)
        self.G2.apply(weights_init_normal)
        self.D1.apply(weights_init_normal)
        self.D2.apply(weights_init_normal)

        self.time = []
        # Optimizers
        if opt.texture_criterion == 'attention':
            self.attention = init_net(Self_Attn(1, 'relu')).to(self.device)
            self.weight_A = list()
            self.weight_B = list()
            self.attention_A = list()
            self.attention_B = list()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.E1.parameters(), self.E2.parameters(),
                                                                self.G1.parameters(), self.G2.parameters(),
                                                                self.attention.parameters()),
                                                lr=opt.lr,
                                                betas=(opt.b1, opt.b2),
                                                )
            self.model_names = ['E1', 'E2', 'G1', 'G2', 'D1', 'D2', "attention"]
        else:
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.E1.parameters(), self.E2.parameters(), self.G1.parameters(), self.G2.parameters()), lr=opt.lr,
                                                betas=(opt.b1, opt.b2),
                                                )
            self.model_names = ['E1', 'E2', 'G1', 'G2', 'D1', 'D2']

        self.optimizer_D1 = torch.optim.Adam(self.D1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D2 = torch.optim.Adam(self.D2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        # Learning rate update schedulers
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
        self.lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
        self.lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

        # SSIM loss function
        if opt.ssim_loss == True:
            self.ssim_loss = SSIMLoss(data_range=1)

        # Edge loss function
        if opt.edge_loss == True:
            self.edge_loss = EdgeLoss()

        if opt.autoencoder_loss == True:
            self.autoencoder = Autoencoder().eval().to("cuda:1")
            checkpoint = torch.load("/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/cycleGAN/autoencoder/early_autoencoder_7.pth")
            self.autoencoder.load_state_dict(checkpoint)

        if opt.lambda_perceptual > 0.0:
            if opt.vgg_pretrained:
                self.vgg = VGG().eval().to("cuda:1")
            else:
                self.vgg = VGG(num_classes=4, pretrained=False, init_type='finetuned', saved_weights_path=opt.vgg_model_path).eval().to("cuda:1")

        # dictionary to store training loss
        if opt.lambda_texture > 0:
            self.loss_names = ['GAN_1', 'GAN_2', 'KL_1', 'KL_2', 'ID_1', 'ID_2', 'KL_1_', 'KL_2_', 'cyc_1', 'cyc_2', "cycle_texture_X1", "cycle_texture_X2"]
        elif opt.lambda_perceptual > 0:
            self.loss_names = ['GAN_1', 'GAN_2', 'KL_1', 'KL_2', 'ID_1', 'ID_2', 'KL_1_', 'KL_2_', 'cyc_1', 'cyc_2', "perceptual_X1", "perceptual_X2"]
        elif opt.experiment_name.find('ssim') != -1:
            self.loss_names = ['GAN_1', 'GAN_2', 'KL_1', 'KL_2', 'ID_1', 'ID_2', 'KL_1_', 'KL_2_', 'cyc_1', 'cyc_2', 'ssim_X1', 'ssim_X2']
        elif opt.experiment_name.find('edge') != -1:
            self.loss_names = ['GAN_1', 'GAN_2', 'KL_1', 'KL_2', 'ID_1', 'ID_2', 'KL_1_', 'KL_2_', 'cyc_1', 'cyc_2', 'edge_X1', 'edge_X2']
        elif opt.experiment_name.find('autoencoder') != -1:
            self.loss_names = ['GAN_1', 'GAN_2', 'KL_1', 'KL_2', 'ID_1', 'ID_2', 'KL_1_', 'KL_2_', 'cyc_1', 'cyc_2', 'autoencoder_X1', 'autoencoder_X2']
        else:
            self.loss_names = ['GAN_1', 'GAN_2', 'KL_1', 'KL_2', 'ID_1', 'ID_2', 'KL_1_', 'KL_2_', 'cyc_1', 'cyc_2']

        self.error_store = OrderedDict()
        for key in self.loss_names:
            self.error_store[key] = list()

            self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir

            # dictionary to store metrics
            self.metric_names = ['psnr', 'mse', 'ssim', 'vif', 'NIQE', 'PIQE', 'paq2piq', 'FID_ImNet', 'FID_random', 'brisque']
            self.metrics_eval = OrderedDict()
            for key in self.metric_names:
                self.metrics_eval[key] = list()

            # dictionary to store metrics per patient
            self.metrics_data_1 = init_storing(test_1_ids, self.metric_names)
            self.metrics_data_2 = init_storing(test_2_ids, self.metric_names)
            self.metrics_data_3 = init_storing(test_3_ids, self.metric_names)
            self.raps_data_2 = init_storing(test_2_ids, ['raps'])
            self.raps_data_3 = init_storing(test_3_ids, ['raps'])
            self.metrics_data_4 = init_storing(test_4_ids, self.metric_names)
            self.raps_data_4 = init_storing(test_4_ids, ['raps'])

            self.skweness_2 = init_storing(test_2_ids, ["skew"])
            self.skweness_3 = init_storing(test_3_ids, ["skew"])
            self.skweness_4 = init_storing(test_4_ids, ["skew"])

            self.difference_heatmaps_2 = init_image_buffer(test_2_ids)
            self.difference_heatmaps_3 = init_image_buffer(test_3_ids)
            self.difference_heatmaps_4 = init_image_buffer(test_4_ids)

            self.grad_real_A_2 = init_image_buffer(test_2_ids)
            self.grad_real_A_3 = init_image_buffer(test_3_ids)
            self.grad_real_A_4 = init_image_buffer(test_4_ids)
            self.grad_fake_B_2 = init_image_buffer(test_2_ids)
            self.grad_fake_B_3 = init_image_buffer(test_3_ids)
            self.grad_fake_B_4 = init_image_buffer(test_4_ids)

            self.gradient_difference_2 = init_image_buffer(test_2_ids)
            self.gradient_difference_3 = init_image_buffer(test_3_ids)
            self.gradient_difference_4 = init_image_buffer(test_4_ids)

            self.grad_correlation_2 = init_storing(test_2_ids, ["corr"])
            self.grad_correlation_3 = init_storing(test_3_ids, ["corr"])
            self.grad_correlation_4 = init_storing(test_4_ids, ["corr"])

            self.template_matching_2 = {}
            self.template_matching_3 = {}
            self.template_matching_elcap = {}

            # metrics initialization
            self.fid_object_1 = GANMetrics('cpu', detector_name='inceptionv3', batch_size=64, pretrained=True)
            self.fid_object_2 = GANMetrics('cpu', detector_name='inceptionv3', batch_size=64, pretrained=False)

            # self.real_test_buffer = []
            # self.fake_test_buffer = []
            self.real_buffer_2 = []  # init_image_buffer(test_2_ids)
            self.fake_buffer_2 = []  # init_image_buffer(test_2_ids)

            self.raps = 0
            self.skweness = []

            self.grid_extractor = _GridExtractor()
            # Wrap the texture_extractor in DataParallel if you have multiple GPUs
            if torch.cuda.device_count() > 1:
                self.grid_extractor = nn.DataParallel(self.grid_extractor, device_ids=[0, 1])

            # NIQE metric
            self.niqe = pyiqa.create_metric('niqe', device=torch.device('cpu'), as_loss=False)
            self.BRISQUE = pyiqa.create_metric('brisque', device=torch.device('cpu'), as_loss=False)
            self.Paq2Piq = pyiqa.create_metric('paq2piq', device=torch.device('cpu'), as_loss=False)

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

            # standardized buffer, not standardized buffer
            self.standardized_test_buffer = []
            self.not_standardized_test_buffer = []

            self.opt = opt

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        if self.opt.dataset_mode == "LIDC_IDRI":
            self.X1 = input['img'].type(self.Tensor).expand(-1, self.opt.channels, -1, -1).to(self.device)
            self.id = input['patient'][0]
            self.domain = input['domain'][0]
        else:
            # Set model input
            self.X1 = input["A"].type(self.Tensor).expand(-1, self.opt.channels, -1, -1).to(self.device)
            self.X2 = input["B"].type(self.Tensor).expand(-1, self.opt.channels, -1, -1).to(self.device)
            self.id = input['patient'][0]

            # Adversarial ground truths
            self.valid = Variable(self.Tensor(np.ones((self.opt.batch_size, *self.output_shape))), requires_grad=False).to(self.device)
            self.fake = Variable(self.Tensor(np.zeros((self.opt.batch_size, *self.output_shape))), requires_grad=False).to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.optimizer_G.zero_grad()

        # Get shared latent representation
        self.mu1, self.Z1 = self.E1(self.X1)
        self.mu2, self.Z2 = self.E2(self.X2)

        # Reconstruct images
        self.recon_X1 = self.G1(self.Z1)
        self.recon_X2 = self.G2(self.Z2)

        # Translate images
        self.fake_X1 = self.G1(self.Z2)
        self.fake_X2 = self.G2(self.Z1)

        # Cycle translation
        self.mu1_, self.Z1_ = self.E1(self.fake_X1)
        self.mu2_, self.Z2_ = self.E2(self.fake_X2)
        self.cycle_X1 = self.G1(self.Z2_)
        self.cycle_X2 = self.G2(self.Z1_)

    def test(self):
        with torch.no_grad():
            _, self.Z1 = self.E1(self.X1)
            self.fake_X2 = self.G2(self.Z1)
            # diff = abs(self.fake_X2[0, 0, :, :] - self.X1[0, 0, :, :])
            # self.skweness.append(skew(diff.detach().cpu().flatten()))

            self.compute_metrics()

            self.track_metrics_per_patient(self.id)

    def test_2(self):
        with torch.no_grad():
            if self.domain == "B30":
                _, self.Z1 = self.E1(self.X1)
                self.fake = self.G1(self.Z1)  # standardized
            elif self.domain == "D45":
                _, self.Z2 = self.E2(self.X1)
                self.fake = self.G2(self.Z2)  # standardized

            self.standardized_test_buffer.append(self.fake[:, 0, :, :])
            self.not_standardized_test_buffer.append(self.X1[:, 0, :, :])

            # self.compute_metrics()
            # self.track_metrics()

    def test_tm(self, iter):

        with torch.no_grad():

            for i in zip(x_coord, y_coord):
                _, self.Z1 = self.E1(self.X1)
                self.fake = self.G1(self.Z1)
                noise_template = self.X1[0, 0, :, :].detach().cpu().numpy().copy()
                noise_template = noise_template[i[0][0]:i[0][1], i[1][0]:i[1][1]]
                if self.opt.test == "test_2":
                    values = template_matching(self.X1.detach().cpu().numpy(), self.fake.detach().cpu().numpy(), self.X2.detach().cpu().numpy(), noise_template)
                    add_patient(self.template_matching_2, self.id)
                    add_image(self.template_matching_2, self.id, iter)
                    add_computed_values(self.template_matching_2, self.id, iter, values)
                elif self.opt.test == "test_3":
                    values = _template_matching(self.X1.detach().cpu().numpy(), self.fake.detach().cpu().numpy(), noise_template)
                    add_patient(self.template_matching_3, self.id)
                    add_image(self.template_matching_3, self.id, iter)
                    add_computed_values(self.template_matching_3, self.id, iter, values)
                elif self.opt.test == "elcap_complete":
                    values = _template_matching(self.X1.detach().cpu().numpy(), self.fake.detach().cpu().numpy(), noise_template)
                    add_patient(self.template_matching_elcap, self.id)
                    add_image(self.template_matching_elcap, self.id, iter)
                    add_computed_values(self.template_matching_elcap, self.id, iter, values)

    def test_3(self):
        with torch.no_grad():
            _, self.Z1 = self.E1(self.X1)
            self.fake_X2 = self.G2(self.Z1)
            # diff = abs(self.fake_X2[0, 0, :, :] - self.X1[0, 0, :, :])
            # self.skweness.append(skew(diff.detach().cpu().flatten()))

            self.fake_buffer_2.append(self.fake_X2)
            self.real_buffer_2.append(self.X2)

    def test_4(self):
        with torch.no_grad():
            _, self.Z1 = self.E1(self.X1)
            self.fake_X2 = self.G2(self.Z1)
            self.gradient_and_correlation()
            diff = abs(self.fake_X2 - self.X1)
            if self.opt.test == "test_2":
                self.skweness_2[self.id]["skew"].append(skew(diff.cpu().flatten()))
                self.difference_heatmaps_2[self.id].append(diff)
            elif self.opt.test == "test_3":
                self.skweness_3[self.id]["skew"].append(skew(diff.cpu().flatten()))
                self.difference_heatmaps_3[self.id].append(diff)
            elif self.opt.test == "elcap_complete":
                self.skweness_4[self.id]["skew"].append(skew(diff.cpu().flatten()))
                self.difference_heatmaps_4[self.id].append(diff)

    def gradient_and_correlation(self):
        grad_real_A_x, grad_real_A_y = image_gradients(self.X1.cpu())
        grad_fake_B_x, grad_fake_B_y = image_gradients(self.fake_X2.cpu())
        grad_real_A = torch.sqrt(grad_real_A_x ** 2 + grad_real_A_y ** 2)
        grad_fake_B = torch.sqrt(grad_fake_B_x ** 2 + grad_fake_B_y ** 2)
        gradient_difference = grad_fake_B[0, 0, :, :] - grad_real_A[0, 0, :, :]
        grad_correlation = np.corrcoef(grad_real_A[0, 0, :, :].cpu().flatten(), grad_fake_B[0, 0, :, :].cpu().flatten(), rowvar=False)[0, 1]

        if self.opt.test == "test_2":
            self.grad_real_A_2[self.id].append(grad_real_A)
            self.grad_fake_B_2[self.id].append(grad_fake_B)
            self.gradient_difference_2[self.id].append(gradient_difference)
            self.grad_correlation_2[self.id]["corr"].append(grad_correlation)
        elif self.opt.test == "test_3":
            self.grad_real_A_3[self.id].append(grad_real_A)
            self.grad_fake_B_3[self.id].append(grad_fake_B)
            self.gradient_difference_3[self.id].append(gradient_difference)
            self.grad_correlation_3[self.id]["corr"].append(grad_correlation)
        elif self.opt.test == "elcap_complete":
            self.grad_real_A_4[self.id].append(grad_real_A)
            self.grad_fake_B_4[self.id].append(grad_fake_B)
            self.gradient_difference_4[self.id].append(gradient_difference)
            self.grad_correlation_4[self.id]["corr"].append(grad_correlation)

    def compute_metrics(self):
        if self.opt.test == "test_3":
            # NIQE
            fake_X2_3channels = self.fake_X2.expand(-1, 3, -1, -1)
            self.NIQE = self.niqe(fake_X2_3channels).item()
            self.brisque = 0  # self.BRISQUE(fake_X2_3channels).item()
            self.paq2piq = 0  # self.Paq2Piq(fake_X2_3channels).item()
            # self.raps = azimuthalAverage(np.squeeze(self.fake_X2[0, 0, :, :].cpu().detach().numpy())).tolist()
            self.PIQE, _, _, _ = piqe((((fake_X2_3channels + 1) * 127.5).clamp(0, 255).to(torch.uint8))[0, 0, :, :].detach().cpu().numpy())
            self.brisque = 0
            self.mse = 0
            self.psnr = 0
            self.ssim = 0
            self.vif = 0

        elif self.opt.test == "elcap_complete":
            # NIQE
            fake_X2_3channels = self.fake_X2.expand(-1, 3, -1, -1)
            self.NIQE = self.niqe(fake_X2_3channels).item()
            self.brisque = 0  # self.BRISQUE(fake_X2_3channels).item()
            self.paq2piq = 0  # self.Paq2Piq(fake_X2_3channels).item()
            # self.raps = azimuthalAverage(np.squeeze(self.fake_X2[0, 0, :, :].cpu().detach().numpy())).tolist()
            self.PIQE, _, _, _ = piqe((((fake_X2_3channels + 1) * 127.5).clamp(0, 255).to(torch.uint8))[0, 0, :, :].detach().cpu().numpy())
            self.brisque = 0
            self.mse = 0
            self.psnr = 0
            self.ssim = 0
            self.vif = 0

        elif self.opt.test == "test_2":
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
            self.NIQE = 0  # self.niqe(fake_X2_3channels).item()
            # self.raps = azimuthalAverage(np.squeeze(self.fake_X2[0, 0, :, :].cpu().detach().numpy())).tolist()
            self.PIQE = 0  # piqe((((fake_X2_3channels+1)*127.5).clamp(0, 255).to(torch.uint8))[0,0,:,:].detach().cpu().numpy()) self.PIQE,_,_,_
            self.brisque = 0  # self.BRISQUE(fake_X2_3channels).item()
            self.paq2piq = 0  # self.Paq2Piq(fake_X2_3channels).item()
            # VIF
            self.vif = 0  # vif(x, y)

            # self.fake_buffer_2[self.id].append(self.fake_X2)
            # self.real_buffer_2[self.id].append(self.X2)

        elif self.opt.test == "test_1":
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

        """if self.opt.lambda_texture > 0:
            if self.opt.texture_criterion == 'attention':
                loss_texture_X1, attention_X1, weight_X1 = texture_loss(self.recon_X1, self.X1, self.criterionTexture, self.opt, self.attention)
                loss_texture_X2, attention_X2, weight_X2 = texture_loss(self.recon_X2, self.X2, self.criterionTexture, self.opt, self.attention)

                self.loss_cycle_texture_X1 = loss_texture_X1
                self.loss_cycle_texture_X2 = loss_texture_X2
                self.weight_X1.append(weight_X1.item())
                self.weight_X2.append(weight_X2.item())
                self.attention_X1.append(attention_X1.detach().clone().numpy())
                self.attention_X2.append(attention_X2.detach().clone().numpy())

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
            self.loss_cycle_texture_X2 = 0"""

        # SSIM loss
        if self.opt.ssim_loss == True:
            # self.loss_ssim_X1 = self.ssim_loss((self.recon_X1+1)*0.5, (self.X1+1)*0.5)
            # self.loss_ssim_X2 = self.ssim_loss((self.recon_X2+1)*0.5, (self.X2+1)*0.5)
            self.loss_ssim_X1 = self.ssim_loss((self.cycle_X1 + 1) * 0.5, (self.X1 + 1) * 0.5)
            self.loss_ssim_X2 = self.ssim_loss((self.cycle_X2 + 1) * 0.5, (self.X2 + 1) * 0.5)
        else:
            self.loss_ssim_X1 = 0
            self.loss_ssim_X2 = 0

        # Edge loss
        if self.opt.edge_loss == True:
            # self.loss_edge_X1 = 10*self.edge_loss(self.recon_X1, self.X1)
            # self.loss_edge_X2 = 10*self.edge_loss(self.recon_X2, self.X2)
            self.loss_edge_X1 = 10 * self.edge_loss(self.cycle_X1, self.X1)
            self.loss_edge_X2 = 10 * self.edge_loss(self.cycle_X2, self.X2)
        else:
            self.loss_edge_X1 = 0
            self.loss_edge_X2 = 0

            # Perceptual auto-encoder loss
        if self.opt.autoencoder_loss == True:
            # self.loss_autoencoder_X1 = 1000*perceptual_autoencoder_loss(self.recon_X1, self.X1, self.autoencoder)
            # self.loss_autoencoder_X2 = 1000*perceptual_autoencoder_loss(self.recon_X2, self.X2, self.autoencoder)
            self.loss_autoencoder_X1 = 1000 * perceptual_autoencoder_loss(self.cycle_X1, self.X1, self.autoencoder)
            self.loss_autoencoder_X2 = 1000 * perceptual_autoencoder_loss(self.cycle_X2, self.X2, self.autoencoder)
        else:
            self.loss_autoencoder_X1 = 0
            self.loss_autoencoder_X2 = 0

            # Texture loss
        if self.opt.lambda_texture > 0:
            if self.opt.texture_criterion == 'attention':
                # loss_texture_X1, map_A, weight_A = _texture_loss(self.recon_X1, self.X1, self.opt, self.grid_extractor, self.attention)
                # loss_texture_X2, map_B, weight_B = _texture_loss(self.recon_X2, self.X2, self.opt, self.grid_extractor, self.attention)
                loss_texture_X1, map_A, weight_A = _texture_loss(self.cycle_X1, self.X1, self.opt, self.grid_extractor, self.attention)
                loss_texture_X2, map_B, weight_B = _texture_loss(self.cycle_X2, self.X2, self.opt, self.grid_extractor, self.attention)

                self.loss_cycle_texture_X1 = loss_texture_X1
                self.loss_cycle_texture_X2 = loss_texture_X2
                self.weight_A.append(weight_A.item())
                self.weight_B.append(weight_B.item())
                self.attention_A.append(map_A.detach().cpu().clone().numpy())
                self.attention_B.append(map_B.detach().cpu().clone().numpy())

            elif self.opt.texture_criterion == 'max':
                # loss_texture_X1 = _texture_loss(self.recon_X1, self.X1, self.opt, self.grid_extractor)
                # loss_texture_X2 = _texture_loss(self.recon_X2, self.X2, self.opt, self.grid_extractor)
                loss_texture_X1 = _texture_loss(self.cycle_X1, self.X1, self.opt, self.grid_extractor)
                loss_texture_X2 = _texture_loss(self.cycle_X2, self.X2, self.opt, self.grid_extractor)
                # save the index of the maximum texture descriptor for each image in the batch
                '''for i in range(self.opt.batch_size):
                    self.index_texture_A.append(torch.nonzero(delta_grids_A == criterion_texture_A[i]).squeeze())
                    self.index_texture_B.append(torch.nonzero(delta_grids_B == criterion_texture_B[i]).squeeze())'''

                # compute the loss function by averaging over the batch
                self.loss_cycle_texture_X1 = loss_texture_X1 * self.opt.lambda_texture
                self.loss_cycle_texture_X2 = loss_texture_X2 * self.opt.lambda_texture

            elif self.opt.texture_criterion == 'average':
                # loss_texture_X1 = _texture_loss(self.recon_X1, self.X1, self.opt, self.grid_extractor)
                # loss_texture_X2 = _texture_loss(self.recon_X2, self.X2, self.opt, self.grid_extractor)
                loss_texture_X1 = _texture_loss(self.cycle_X1, self.X1, self.opt, self.grid_extractor)
                loss_texture_X2 = _texture_loss(self.cycle_X2, self.X2, self.opt, self.grid_extractor)

                self.loss_cycle_texture_X1 = loss_texture_X1 * self.opt.lambda_texture
                self.loss_cycle_texture_X2 = loss_texture_X2 * self.opt.lambda_texture

            elif self.opt.texture_criterion == 'Frobenius':
                # loss_texture_X1 = _texture_loss(self.recon_X1, self.X1, self.opt, self.grid_extractor)
                # loss_texture_X2 = _texture_loss(self.recon_X2, self.X2, self.opt, self.grid_extractor)
                loss_texture_X1 = _texture_loss(self.cycle_X1, self.X1, self.opt, self.grid_extractor)
                loss_texture_X2 = _texture_loss(self.cycle_X2, self.X2, self.opt, self.grid_extractor)

                self.loss_cycle_texture_X1 = loss_texture_X1 * self.opt.lambda_texture
                self.loss_cycle_texture_X2 = loss_texture_X2 * self.opt.lambda_texture
            else:
                raise NotImplementedError
        else:
            self.loss_cycle_texture_X1 = 0
            self.loss_cycle_texture_X2 = 0

        # Perceptual loss
        if self.opt.lambda_perceptual > 0:
            # loss_perceptual_X1 = perceptual_similarity_loss(self.recon_X1, self.X1, self.vgg, self.opt.perceptual_layers)
            # loss_perceptual_X2 = perceptual_similarity_loss(self.recon_X2, self.X2, self.vgg, self.opt.perceptual_layers)

            loss_perceptual_X1 = perceptual_similarity_loss(self.cycle_X1, self.X1, self.vgg, self.opt.perceptual_layers)
            loss_perceptual_X2 = perceptual_similarity_loss(self.cycle_X2, self.X2, self.vgg, self.opt.perceptual_layers)

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
                + self.loss_ssim_X1
                + self.loss_ssim_X2
                + self.loss_edge_X1
                + self.loss_edge_X2
                + self.loss_autoencoder_X1
                + self.loss_autoencoder_X2
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
        ts = time.time()
        self.backwark_encoderGen()
        self.optimizer_G.step()

        self.optimizer_D1.zero_grad()
        self.backward_D1()
        self.optimizer_D1.step()

        self.optimizer_D2.zero_grad()
        self.backward_D2()
        self.optimizer_D2.step()
        tf = time.time()
        self.time.append(tf - ts)

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
        # torch.save(self.attention.state_dict(), f"{self.save_dir}/attention_ep{epoch}_{self.opt.experiment_name}.pth")

    def setup1(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.load_networks_1(opt.epoch, opt.experiment_name)
        # self.print_networks(opt.verbose)

    def load_networks_1(self, epoch, exp):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = 'net_%s_ep%s_unit_%s.pth' % (name, epoch, exp)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                new_state_dict = {}
                for key, value in state_dict.items():
                    # Remove the specific word or prefix you want to eliminate
                    new_key = key.replace('module.', '')
                    new_state_dict[new_key] = value

                # if hasattr(state_dict, '_metadata'):
                #    del new_state_dict._metadata

                """# patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))"""
                net.load_state_dict(new_state_dict)

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

    def track_metrics_per_patient(self, id):
        if self.opt.test == "test_1":
            for name in self.metric_names:
                if isinstance(name, str) and name != 'FID_ImNet' and name != 'FID_random':
                    self.metrics_data_1[id][name].append(
                        float(getattr(self, name)))  # float(...) works for both scalar tensor and float number
            return self.metrics_data_1
        elif self.opt.test == "test_2":
            for name in self.metric_names:
                if isinstance(name, str) and name != 'FID_ImNet' and name != 'FID_random':
                    self.metrics_data_2[id][name].append(
                        float(getattr(self, name)))  # float(...) works for both scalar tensor and float number
            # self.raps_data_2[id]['raps'].append(list(getattr(self, 'raps')))
            return self.metrics_data_2
        elif self.opt.test == "test_3":
            for name in self.metric_names:
                if isinstance(name, str) and name != 'FID_ImNet' and name != 'FID_random':
                    self.metrics_data_3[id][name].append(
                        float(getattr(self, name)))  # float(...) works for both scalar tensor and float number
            # self.raps_data_3[id]['raps'].append(list(getattr(self, 'raps')))
            return self.metrics_data_3, self.raps_data_3

        elif self.opt.test == "elcap_complete":
            for name in self.metric_names:
                if isinstance(name, str) and name != 'FID_ImNet' and name != 'FID_random':
                    self.metrics_data_4[id][name].append(
                        float(getattr(self, name)))  # float(...) works for both scalar tensor and float number

            #  self.raps_data_4[id]['raps'].append(list(getattr(self, 'raps')))

            return self.metrics_data_4  # , self.raps_data_4

    def get_epoch_performance(self):
        return self.metrics_eval

    def save_times(self):
        np.save(f"{self.metric_dir}/times.npy", np.array(self.time))

    def save_metrics(self, epoch_performance, epoch):
        csv_path2 = os.path.join(self.metric_dir, f'metrics_{self.opt.test}_epoch{epoch}.csv')
        save_ordered_dict_as_csv(epoch_performance, csv_path2)
        empty_dictionary(self.metrics_eval)

    def save_raps(self, epoch):
        save_json(self.raps, f"{self.metric_dir}/raps_{self.opt.test}_epoch{epoch}")
        self.raps = []

    def save_attention_maps(self):
        np.save(f"{self.loss_dir}/attention_X1.npy", np.array(self.attention_A))
        np.save(f"{self.loss_dir}/attention_X2.npy", np.array(self.attention_B))

    def save_attention_weights(self):
        np.save(f"{self.loss_dir}/weight_X1.npy", np.array(self.weight_A))
        np.save(f"{self.loss_dir}/weight_X2.npy", np.array(self.weight_B))

    def fid_compute(self):

        # for key in tqdm(self.real_buffer_2.keys()):
        # print(key)
        real_buffer = torch.cat(self.real_buffer_2, dim=0)
        fake_buffer = torch.cat(self.fake_buffer_2, dim=0)
        print(real_buffer.shape)
        print(fake_buffer.shape)
        fid_score_1 = self.fid_object_1.compute_fid(fake_buffer, real_buffer, len(real_buffer))
        fid_score_2 = self.fid_object_2.compute_fid(fake_buffer, real_buffer, len(real_buffer))
        for key in self.metrics_data_2.keys():
            self.metrics_data_2[key]['FID_ImNet'].append(fid_score_1)
            self.metrics_data_2[key]['FID_random'].append(fid_score_2)

        self.real_buffer_2 = []
        self.real_buffer_1 = []
        # empty_Dictionary(self.real_buffer_2, nesting=0)
        # empty_Dictionary(self.fake_buffer_2, nesting=0)

    def save_test_4(self, epoch):
        if self.opt.test == "test_2":
            save_to_json(self.skweness_2, f"{self.metric_dir}/skewness_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.skweness_2, nesting=2)
            save_to_json(self.grad_correlation_2, f"{self.metric_dir}/grad_correlation_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.grad_correlation_2, nesting=2)
            torch.save(self.difference_heatmaps_2, f'{self.metric_dir}/difference_heatmaps_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.difference_heatmaps_2, nesting=1)
            torch.save(self.grad_real_A_2, f'{self.metric_dir}/grad_real_A_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.grad_real_A_2, nesting=1)
            torch.save(self.grad_fake_B_2, f'{self.metric_dir}/grad_fake_B_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.grad_fake_B_2, nesting=1)
            torch.save(self.gradient_difference_2, f'{self.metric_dir}/gradient_difference_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.gradient_difference_2, nesting=1)

        elif self.opt.test == "test_3":
            save_to_json(self.skweness_3, f"{self.metric_dir}/skewness_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.skweness_3, nesting=2)
            save_to_json(self.grad_correlation_3, f"{self.metric_dir}/grad_correlation_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.grad_correlation_3, nesting=2)
            torch.save(self.difference_heatmaps_3, f'{self.metric_dir}/difference_heatmaps_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.difference_heatmaps_3, nesting=1)
            torch.save(self.grad_real_A_3, f'{self.metric_dir}/grad_real_A_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.grad_real_A_3, nesting=1)
            torch.save(self.grad_fake_B_3, f'{self.metric_dir}/grad_fake_B_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.grad_fake_B_3, nesting=1)
            torch.save(self.gradient_difference_3, f'{self.metric_dir}/gradient_difference_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.gradient_difference_3, nesting=1)
        elif self.opt.test == "elcap_complete":
            save_to_json(self.skweness_4, f"{self.metric_dir}/skewness_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.skweness_4, nesting=2)
            save_to_json(self.grad_correlation_4, f"{self.metric_dir}/grad_correlation_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.grad_correlation_4, nesting=2)
            torch.save(self.difference_heatmaps_4, f'{self.metric_dir}/difference_heatmaps_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.difference_heatmaps_4, nesting=1)
            torch.save(self.grad_real_A_4, f'{self.metric_dir}/grad_real_A_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.grad_real_A_4, nesting=1)
            torch.save(self.grad_fake_B_4, f'{self.metric_dir}/grad_fake_B_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.grad_fake_B_4, nesting=1)
            torch.save(self.gradient_difference_4, f'{self.metric_dir}/gradient_difference_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.gradient_difference_4, nesting=1)

    def save_raps(self, epoch):
        save_json(self.raps, f"{self.metric_dir}/raps_{self.opt.test}_epoch{epoch}")
        self.raps = []

    def save_raps_per_patient(self, epoch):
        if self.opt.test == "test_3":
            save_to_json(self.raps_data_3, f"{self.metric_dir}/raps_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.raps_data_3, nesting=2)
        elif self.opt.test == "test_2":
            save_to_json(self.raps_data_2, f"{self.metric_dir}/raps_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.raps_data_2, nesting=2)
        elif self.opt.test == "elcap_complete":
            save_to_json(self.raps_data_4, f"{self.metric_dir}/raps_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.raps_data_4, nesting=2)

    def save_metrics_per_patient(self, epoch):
        if self.opt.test == "test_1":
            save_to_json(self.metrics_data_1, f"{self.metric_dir}/metrics_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.metrics_data_1, nesting=2)
        elif self.opt.test == "test_2":
            save_to_json(self.metrics_data_2, f"{self.metric_dir}/metrics_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.metrics_data_2, nesting=2)
        elif self.opt.test == "test_3":
            save_to_json(self.metrics_data_3, f"{self.metric_dir}/metrics_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.metrics_data_3, nesting=2)
        elif self.opt.test == "elcap_complete":
            save_to_json(self.metrics_data_4, f"{self.metric_dir}/metrics_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.metrics_data_4, nesting=2)
        else:
            raise ("NotImplemented")

    def save_template(self, epoch):
        if self.opt.test == "test_2":
            save_to_json(self.template_matching_2, f"{self.metric_dir}/tm_{self.opt.test}_epoch{epoch}")
        elif self.opt.test == "test_3":
            save_to_json(self.template_matching_3, f"{self.metric_dir}/tm_{self.opt.test}_epoch{epoch}")
        elif self.opt.test == "elcap_complete":
            save_to_json(self.template_matching_elcap, f"{self.metric_dir}/tm_{self.opt.test}_epoch{epoch}")

    def save_test_images(self, epoch):
        standardized_test = torch.cat(self.standardized_test_buffer, dim=0)
        not_standardized_test = torch.cat(self.not_standardized_test_buffer, dim=0)
        torch.save(standardized_test, f'{self.test_dir}/standardized_{self.opt.test}_epoch{epoch}.pth')
        torch.save(not_standardized_test, f'{self.test_dir}/not_standardized_{self.opt.test}_epoch{epoch}.pth')
        self.standardized_test_buffer = []
        self.not_standardized_test_buffer = []

    def save_time(self):
        np.save(f"{self.metric_dir}/time.npy", np.array(self.time))

    def save_noise_metrics(self, epoch):
        # torch.save(self.kurtosis, f'{self.metric_dir}/kurtosis_{self.opt.test}_epoch{epoch}.pth')
        torch.save(self.skweness, f'{self.metric_dir}/skweness_{self.opt.test}_epoch{epoch}.pth')
        # torch.save(self.shannon_entropy, f'{self.metric_dir}/shannon_entropy_{self.opt.test}_epoch{epoch}.pth')
