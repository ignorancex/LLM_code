from .base_model import BaseModel, OrderedDict
from . import networks
import os
import itertools
from .vgg import VGG
from data.storage import *
from util.util import *
import pyiqa
from loss_functions.attention import Self_Attn
from metrics.FID import *
from metrics.mse_psnr_ssim_vif import *
from metrics.piqe import *
# from loss_functions.texture_loss import texture_loss
from loss_functions.glcm_soft_einsum import _texture_loss, _GridExtractor
from loss_functions.perceptual_loss import perceptual_similarity_loss
from models.networks import init_net
from piq import brisque
from scipy.stats import skew, kurtosis
from torchmetrics.functional.image import image_gradients
from loss_functions.glcm import *
import cv2 as cv
import time
from loss_functions.edge_loss import *
from piq import SSIMLoss
from models.autoencoder_perceptual import *
import matplotlib.patches as patches

def save_fig_in_pdf(img, idx, tag, bbox):
    image_array = img.cpu().numpy()  # Assuming image_tensor is a PyTorch tensor

    # Create a figure without axis
    fig, ax = plt.subplots()
    ax.axis('off')  # Turn off axis labels and ticks

    # Plot the image
    ax.imshow(image_array[0, 0, :, :], cmap='gray')

    if bbox:
        # Create a rectangle patch
        rect = patches.Rectangle((100, 158), 50, 50, linewidth=1, edgecolor='r', facecolor='none') # test 2: 158, test 3,4: 148

        # Add the rectangle to the plot
        ax.add_patch(rect)
    # plt.show()
    # Save the figure as a PDF without axis
    plt.savefig(f'{tag}_{idx}.pdf', format='pdf', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()  # Close the figure to free up memory

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


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        parser.add_argument('--experiment_name', type=str, default="default", help='experiment name')
        parser.add_argument('--image_folder', type=str, default=None, help='folder to save images during training')
        parser.add_argument('--metric_folder', type=str, default=None, help='folder to save metrics')
        parser.add_argument('--loss_folder', type=str, default=None, help='folder to save losses')
        parser.add_argument('--test_folder', type=str, default=None, help='folder to save test images')

        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_texture', type=float, default=0.001, help='use texture loss.')
            parser.add_argument('--texture_criterion', type=str, default="max", help='texture loss criterion.')
            parser.add_argument('--texture_offsets', type=str, default="all", help='texture offsets.')
            parser.add_argument('--vgg_pretrained', type=str, default=True, help='pretraining flag.')
            parser.add_argument('--vgg_model_path', type=str, default=None, help='finetuned vgg model path.')
            parser.add_argument('--lambda_perceptual', type=float, default=0.1, help='use perceptual loss.')
            parser.add_argument('--perceptual_layers', type=str, default='all', help='choose the perceptual layers.')
            parser.add_argument('--edge_loss', type=bool, default=False, help='decide to use edge loss.')
            parser.add_argument('--ssim_loss', type=bool, default=False, help='decide to use ssim loss.')
            parser.add_argument('--autoencoder_loss', type=bool, default=False, help='choose to use edge loss.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if opt.experiment_name.find('texture') != -1:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'texture']
        elif opt.experiment_name.find('perceptual') != -1:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'perceptual']
        elif opt.experiment_name.find('baseline') != -1:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        elif opt.experiment_name.find('ssim') != -1:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'ssim']
        elif opt.experiment_name.find('edge') != -1:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'edge']

            # dictionary to store training loss
        self.error_store = OrderedDict()
        for key in self.loss_names:
            self.error_store[key] = list()

        self.test_visual_names = ['real_A', 'fake_B', 'real_B']
        self.metric_names = ['psnr', 'mse', 'ssim', 'vif', 'NIQE', 'PIQE', 'FID_ImNet', 'FID_random', 'brisque', 'paq2piq']
        self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
        self.loss_dir = os.path.join(self.web_dir, f'{opt.loss_folder}')
        self.test_dir = os.path.join(self.web_dir, f'{opt.test_folder}')
        self.metric_dir = os.path.join(self.web_dir, f'{opt.metric_folder}')

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

        # self.avg_metrics_test_1 = OrderedDict()
        # self.avg_metrics_test_2 = OrderedDict()
        # self.avg_metrics_test_3 = OrderedDict()

        # for key in self.metric_names:
        #     self.avg_metrics_test_1[key] = OrderedDict()
        #     self.avg_metrics_test_2[key] = OrderedDict()
        #     self.avg_metrics_test_3[key] = OrderedDict()

        #     self.avg_metrics_test_1[key]['mean'] = list()
        #     self.avg_metrics_test_1[key]['std'] = list()
        #     self.avg_metrics_test_2[key]['mean'] = list()
        #     self.avg_metrics_test_2[key]['std'] = list()
        #     self.avg_metrics_test_3[key]['mean'] = list()
        #     self.avg_metrics_test_3[key]['std'] = list()

        self.template_matching_2 = {}
        self.template_matching_3 = {}
        self.template_matching_elcap = {}

        self.dictionary_2 = {}
        self.dictionary_3 = {}
        self.dictionary_elcap = {}

        # metrics initialization
        self.fid_object_1 = GANMetrics('cpu', detector_name='inceptionv3', batch_size=64, pretrained=True)
        self.fid_object_2 = GANMetrics('cpu', detector_name='inceptionv3', batch_size=64, pretrained=False)
        # self.real_test_buffer = []
        # self.fake_test_buffer = []
        # self.real_buffer_2 = init_image_buffer(test_2_ids)
        # self.fake_buffer_2 = init_image_buffer(test_2_ids)
        self.real_buffer_2 = []
        self.fake_buffer_2 = []

        self.raps = 0
        self.time = []

        # NIQE metric
        self.niqe = pyiqa.create_metric('niqe', device=torch.device('cpu'), as_loss=False)
        self.BRISQUE = pyiqa.create_metric('brisque', device=torch.device('cpu'), as_loss=False)
        self.Paq2Piq = pyiqa.create_metric('paq2piq', device=torch.device('cpu'), as_loss=False)

        if self.isTrain:
            if opt.texture_criterion == 'attention':
                self.model_names = ['G', 'D']
            else:
                self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']  # "attention"
            self.texture_grids_2 = init_image_buffer(test_2_ids)
            self.attention_maps_2 = init_image_buffer(test_2_ids)

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain)
        if opt.texture_criterion == 'attention':
            self.netattention = init_net(Self_Attn(1, 'relu')).to(self.device)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

        self.skweness = []

        self.grid_extractor = _GridExtractor()
        # Wrap the texture_extractor in DataParallel if you have multiple GPUs
        if torch.cuda.device_count() > 1:
            self.grid_extractor = nn.DataParallel(self.grid_extractor)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionTexture = torch.nn.L1Loss(reduction='none').to(self.device)

            # SSIM loss function
            if opt.ssim_loss == True:
                self.ssim_loss = SSIMLoss(data_range=1)

            # Edge loss function
            if opt.edge_loss == True:
                self.edge_loss = EdgeLoss()

                # Perceptual autoencoder loss
            if opt.autoencoder_loss == True:
                self.autoencoder = Autoencoder().eval().to("cuda:1")
                saved_model_path = '/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/cycleGAN/autoencoder/early_autoeencoder_7.pth'
                checkpoint = torch.load(saved_model_path)
                self.autoencoder.load_state_dict(checkpoint['state_dict'])

            if opt.texture_criterion == 'attention':
                self.netattention = init_net(Self_Attn(1, 'relu')).to(self.device)
                self.weight = list()
                self.attention_maps = list()
                # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netattention.parameters()), lr=opt.lr,
                                                    betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if opt.lambda_perceptual > 0.0:
                if opt.vgg_pretrained == True:
                    self.vgg = VGG().eval().to(int(opt.gpu_ids[1]))
                else:
                    self.vgg = VGG(num_classes=4, pretrained=False, init_type='finetuned', saved_weights_path=opt.vgg_model_path).to(int(opt.gpu_ids[1]))

            self.index_texture = list()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        if self.opt.dataset_mode == "LIDC_IDRI":
            self.real_A = input['img'].to(self.device)
            self.id = input['patient'][0]
        else:
            AtoB = self.opt.direction == 'AtoB'
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.id = input['patient'][0]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def test(self):

        with torch.no_grad():
            self.fake_B = self.netG(self.real_A)
            # diff = abs(self.fake_B[0, 0, :, :] - self.real_A[0, 0, :, :])
            # self.skweness.append(skew(diff.detach().cpu().flatten()))

            self.compute_metrics()

            self.track_metrics_per_patient(self.id)

    def test_visuals(self, iter, list_index):

        with torch.no_grad():
            if iter in list_index:
                self.fake_B = self.netG(self.real_A)
                grad_fake_B_x, grad_fake_B_y = image_gradients(self.fake_B.cpu())
                grad_fake_B = torch.sqrt(grad_fake_B_x ** 2 + grad_fake_B_y ** 2)
                save_fig_in_pdf(self.fake_B, iter, self.opt.experiment_name, False)
                save_fig_in_pdf(grad_fake_B, iter, f"grad_{self.opt.experiment_name}", True)
                save_fig_in_pdf(grad_fake_B[:, :, 158:208, 100:150], iter, f"zoom_{self.opt.experiment_name}", False)  # test 2: 158:208, test 3,4: 148:198

    def test2(self, iter):

        with torch.no_grad():

            for i in zip(x_coord, y_coord):
                self.fake_B = self.netG(self.real_A)
                noise_template = self.real_A[0, 0, :, :].detach().cpu().numpy().copy()
                noise_template = noise_template[i[0][0]:i[0][1], i[1][0]:i[1][1]]
                if self.opt.test == "test_2":
                    values = template_matching(self.real_A.detach().cpu().numpy(), self.fake_B.detach().cpu().numpy(), self.real_B.detach().cpu().numpy(), noise_template)
                    add_patient(self.template_matching_2, self.id)
                    add_image(self.template_matching_2, self.id, iter)
                    add_computed_values(self.template_matching_2, self.id, iter, values)
                elif self.opt.test == "test_3":
                    values = _template_matching(self.real_A.detach().cpu().numpy(), self.fake_B.detach().cpu().numpy(), noise_template)
                    add_patient(self.template_matching_3, self.id)
                    add_image(self.template_matching_3, self.id, iter)
                    add_computed_values(self.template_matching_3, self.id, iter, values)
                elif self.opt.test == "elcap_complete":
                    values = _template_matching(self.real_A.detach().cpu().numpy(), self.fake_B.detach().cpu().numpy(), noise_template)
                    add_patient(self.template_matching_elcap, self.id)
                    add_image(self.template_matching_elcap, self.id, iter)
                    add_computed_values(self.template_matching_elcap, self.id, iter, values)

    def test4(self, iter):

        with torch.no_grad():
            self.fake_B = self.netG(self.real_A)

            if self.opt.test == "test_2":
                grad_ld_x, grad_ld_y = image_gradients(self.real_A)
                grad_ld = torch.sqrt(grad_ld_x ** 2 + grad_ld_y ** 2)
                grad_deno_x, grad_deno_y = image_gradients(self.fake_B)
                grad_deno = torch.sqrt(grad_deno_x ** 2 + grad_deno_y ** 2)
                grad_hd_x, grad_hd_y = image_gradients(self.real_B)
                grad_hd = torch.sqrt(grad_hd_x ** 2 + grad_hd_y ** 2)
                add_patient(self.dictionary_2, self.id)
                add_image(self.dictionary_2, self.id, iter)
                add_computed_values(self.dictionary_2, self.id, iter, {'ld': skew(grad_ld.detach().cpu().numpy().flatten()), "hd": skew(grad_hd.detach().cpu().numpy().flatten()),
                                                                       "deno": skew(grad_deno.detach().cpu().numpy().flatten())})
            elif self.opt.test == "test_3":
                grad_ld_x, grad_ld_y = image_gradients(self.real_A)
                grad_ld = torch.sqrt(grad_ld_x ** 2 + grad_ld_y ** 2)
                grad_deno_x, grad_deno_y = image_gradients(self.fake_B)
                grad_deno = torch.sqrt(grad_deno_x ** 2 + grad_deno_y ** 2)
                add_patient(self.dictionary_3, self.id)
                add_image(self.dictionary_3, self.id, iter)
                add_computed_values(self.dictionary_3, self.id, iter,
                                    {'ld': skew(grad_ld.detach().cpu().numpy().flatten()), "deno": skew(grad_deno.detach().cpu().numpy().flatten())})
            elif self.opt.test == "elcap_complete":
                grad_ld_x, grad_ld_y = image_gradients(self.real_A)
                grad_ld = torch.sqrt(grad_ld_x ** 2 + grad_ld_y ** 2)
                grad_deno_x, grad_deno_y = image_gradients(self.fake_B)
                grad_deno = torch.sqrt(grad_deno_x ** 2 + grad_deno_y ** 2)
                add_patient(self.dictionary_elcap, self.id)
                add_image(self.dictionary_elcap, self.id, iter)
                add_computed_values(self.dictionary_elcap, self.id, iter,
                                    {'ld': skew(grad_ld.detach().cpu().numpy().flatten()), "deno": skew(grad_deno.detach().cpu().numpy().flatten())})

    def test_3(self):
        with torch.no_grad():
            self.fake_B = self.netG(self.real_A)
            self.gradient_and_correlation()
            diff = abs(self.fake_B - self.real_A)
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
        grad_real_A_x, grad_real_A_y = image_gradients(self.real_A.cpu())
        grad_fake_B_x, grad_fake_B_y = image_gradients(self.fake_B.cpu())
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
            fake_B_3channels = self.real_A.expand(-1, 3, -1, -1)
            self.NIQE = self.niqe((fake_B_3channels + 1) * 0.5).item()
            self.brisque = 0  # self.BRISQUE(fake_B_3channels).item()
            self.paq2piq = 0  #  self.Paq2Piq(fake_B_3channels).item()
            self.PIQE, _, _, _ = piqe((((fake_B_3channels + 1) * 127.5).clamp(0, 255).to(torch.uint8))[0, 0, :, :].detach().cpu().numpy())
            # self.raps = azimuthalAverage(np.squeeze(self.fake_B[0, 0, :, :].cpu().detach().numpy())).tolist()

            self.mse = 0
            self.psnr = 0
            self.ssim = 0
            self.vif = 0

        elif self.opt.test == "elcap_complete":
            # NIQE
            fake_B_3channels = self.real_A.expand(-1, 3, -1, -1)  # +1)*0.5
            self.NIQE = self.niqe((fake_B_3channels + 1) * 0.5).item()
            self.brisque = 0  # self.BRISQUE(fake_B_3channels).item()
            self.paq2piq = 0  # self.Paq2Piq(fake_B_3channels).item()
            self.PIQE, _, _, _ = piqe((((fake_B_3channels + 1) * 127.5).clamp(0, 255).to(torch.uint8))[0, 0, :, :].detach().cpu().numpy())
            # self.raps = azimuthalAverage(np.squeeze(self.fake_B[0, 0, :, :].cpu().detach().numpy())).tolist()

            self.mse = 0
            self.psnr = 0
            self.ssim = 0
            self.vif = 0

        elif self.opt.test == "test_2":
            x = tensor2im2(self.real_B)
            y = tensor2im2(self.fake_B)
            # MSE
            self.mse = mean_squared_error(x, y)
            # PSNR
            self.psnr = peak_signal_to_noise_ratio(x, y)
            # SSIM
            self.ssim = structural_similarity_index(x, y)
            # NIQE
            fake_B_3channels = self.real_A.expand(-1, 3, -1, -1)
            self.NIQE = self.niqe((fake_B_3channels + 1) * 0.5).item()
            self.brisque = 0  # self.BRISQUE(fake_B_3channels).item()
            self.paq2piq = 0  # self.Paq2Piq(fake_B_3channels).item()
            self.PIQE, _, _, _ = piqe((((fake_B_3channels + 1) * 127.5).clamp(0, 255).to(torch.uint8))[0, 0, :, :].detach().cpu().numpy())
            # self.raps = azimuthalAverage(np.squeeze(self.fake_B[0, 0, :, :].cpu().detach().numpy())).tolist()

            # VIF
            self.vif = 0  # vif(x, y)

            # self.fake_buffer_2[self.id].append(self.fake_B)
            # self.real_buffer_2[self.id].append(self.real_B)

        elif self.opt.test == "test_1":
            x = tensor2im2(self.real_B)
            y = tensor2im2(self.fake_B)
            # MSE
            self.mse = mean_squared_error(x, y)
            # PSNR
            self.psnr = peak_signal_to_noise_ratio(x, y)
            # SSIM
            self.ssim = structural_similarity_index(x, y)
            # NIQE
            fake_B_3channels = self.fake_B.expand(-1, 3, -1, -1)
            self.NIQE = self.niqe(fake_B_3channels).item()

            self.brisque = 0
            # VIF
            self.vif = vif(x, y)

    def save_dictionary(self, epoch):
        if self.opt.test == "test_2":
            save_to_json(self.dictionary_2, f"{self.metric_dir}/grad_skew_{self.opt.test}_epoch{epoch}")
        elif self.opt.test == "test_3":
            save_to_json(self.dictionary_3, f"{self.metric_dir}/grad_skew_{self.opt.test}_epoch{epoch}")
        elif self.opt.test == "elcap_complete":
            save_to_json(self.dictionary_elcap, f"{self.metric_dir}/grad_skew_{self.opt.test}_epoch{epoch}")

    """def compute_fid(self, idx):
        if idx == self.opt.dataset_len - 1:
            self.real_test_buffer = torch.cat(self.real_test_buffer)
            self.fake_test_buffer = torch.cat(self.fake_test_buffer)
            fid_index = calculate_frechet(self.real_test_buffer, self.fake_test_buffer, self.inception_model)
            self.metrics_eval['FID'].append(fid_index)
            self.real_test_buffer = []
            self.fake_test_buffer = []
        else:
            pass"""

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        # SSIM loss
        if self.opt.ssim_loss:
            self.loss_ssim = self.ssim_loss((self.fake_B + 1) * 0.5, (self.real_B + 1) * 0.5)
            print(self.opt.ssim_loss)
            print(self.loss_ssim)

        else:
            self.loss_ssim = 0
            print(self.opt.ssim_loss)
            print(self.loss_ssim)

        # Edge loss
        if self.opt.edge_loss == True:
            self.loss_edge = 10 * self.edge_loss(self.fake_B, self.real_B)
            print(self.loss_edge)
        else:
            self.loss_edge = 0

        # Perceptual auto-encoder loss
        if self.opt.autoencoder_loss == True:
            self.loss_autoencoder = 10 * perceptual_autoencoder_loss(self.fake_B, self.real_B, self.autoencoder)
        else:
            self.loss_autoencoder = 0

        # Texture loss
        lambda_texture = self.opt.lambda_texture
        if lambda_texture > 0:

            if self.opt.texture_criterion == 'attention':
                loss_texture, att_map, weight = _texture_loss(self.fake_B, self.real_B, self.opt, self.grid_extractor, self.netattention)
                self.loss_texture = loss_texture
                self.weight.append(weight.item())
                self.attention_maps.append(att_map.detach().clone().cpu().numpy())


            elif self.opt.texture_criterion == 'max':
                loss_texture = _texture_loss(self.fake_B, self.real_B, self.opt, self.grid_extractor)  # , self.texture_extractor

                # save the index of the maximum texture descriptor for each image in the batch
                # for i in range(self.opt.batch_size):
                #     self.index_texture.append(torch.nonzero(delta_grids_B == criterion_texture_B[i]).squeeze())

                # compute the loss function by averaging over the batch
                self.loss_texture = loss_texture * lambda_texture

            elif self.opt.texture_criterion == 'average':
                loss_texture = _texture_loss(self.fake_B, self.real_B, self.opt, self.grid_extractor)
                self.loss_texture = loss_texture * lambda_texture

            elif self.opt.texture_criterion == 'Frobenius':
                loss_texture = _texture_loss(self.fake_B, self.real_B, self.opt, self.grid_extractor)
                self.loss_texture = loss_texture * lambda_texture
        else:
            self.loss_texture = 0

        # Perceptual loss
        lambda_perceptual = self.opt.lambda_perceptual
        if lambda_perceptual > 0:
            loss_perceptual = perceptual_similarity_loss(self.fake_B, self.real_B, self.vgg, self.opt.perceptual_layers)
            self.loss_perceptual = loss_perceptual * lambda_perceptual
        else:
            self.loss_perceptual = 0

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_texture + self.loss_perceptual + self.loss_ssim + self.loss_edge + self.loss_autoencoder

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        ts = time.time()
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # update G's weights
        tf = time.time()
        self.time.append(tf - ts)

    def save_test_2(self, epoch):
        save_to_json(self.texture_grids_2, f"{self.metric_dir}/delta_grids_test_2_ep{epoch}")
        empty_Dictionary(self.texture_grids_2, nesting=1)
        save_to_json(self.attention_maps_2, f"{self.metric_dir}/attention_maps_ep{epoch}")
        empty_Dictionary(self.attention_maps_2, nesting=1)

    def save_texture_indexes(self):
        save_list_to_csv(self.index_texture, f'{self.loss_dir}/idx_texture.csv')

    def save_attention_maps(self):
        np.save(f"{self.loss_dir}/attention_B.npy", np.array(self.attention_maps))

    def save_attention_net(self, epoch):
        torch.save(self.netattention.state_dict(), f"{self.save_dir}/netattention_ep{epoch}_{self.opt.experiment_name}.pth")

    def save_attention_weights(self):
        np.save(f"{self.loss_dir}/weight.npy", np.array(self.weight))

    def save_list_images(self, epoch):
        real_buffer = torch.cat(self.real_test_buffer, dim=0)
        fake_buffer = torch.cat(self.fake_test_buffer, dim=0)

        fid_score = self.fid_object.compute_fid(fake_buffer, real_buffer, self.opt.dataset_len)
        self.metrics_eval['FID'].append(fid_score)

        # torch.save(real_buffer, f'{self.test_dir}/real_buffer_{self.opt.test}_epoch{epoch}.pth')
        # torch.save(fake_buffer, f'{self.test_dir}/fake_buffer_{self.opt.test}_epoch{epoch}.pth')

        self.real_test_buffer = []
        self.fake_test_buffer = []

    def save_raps(self, epoch):
        save_json(self.raps, f"{self.metric_dir}/raps_{self.opt.test}_epoch{epoch}")
        self.raps = []

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

    def save_test_3(self, epoch):
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

    def save_template(self, epoch):
        if self.opt.test == "test_2":
            save_to_json(self.template_matching_2, f"{self.metric_dir}/tm_{self.opt.test}_epoch{epoch}")
        elif self.opt.test == "test_3":
            save_to_json(self.template_matching_3, f"{self.metric_dir}/tm_{self.opt.test}_epoch{epoch}")
        elif self.opt.test == "elcap_complete":
            save_to_json(self.template_matching_elcap, f"{self.metric_dir}/tm_{self.opt.test}_epoch{epoch}")

    def save_time(self):
        np.save(f"{self.metric_dir}/time.npy", np.array(self.time))

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

    def save_noise_metrics(self, epoch):
        # torch.save(self.kurtosis, f'{self.metric_dir}/kurtosis_{self.opt.test}_epoch{epoch}.pth')
        torch.save(self.skweness, f'{self.metric_dir}/skweness_{self.opt.test}_epoch{epoch}.pth')
        # torch.save(self.shannon_entropy, f'{self.metric_dir}/shannon_entropy_{self.opt.test}_epoch{epoch}.pth')
