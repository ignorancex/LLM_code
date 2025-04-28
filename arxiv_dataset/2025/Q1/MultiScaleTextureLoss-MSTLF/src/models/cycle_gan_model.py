import itertools
import matplotlib.pyplot as plt
from util.image_pool import ImagePool
from util.util import *
from .base_model import BaseModel, OrderedDict
from data.storage import *
from . import networks
import os
import pyiqa
from .vgg import VGG
from loss_functions.attention import Self_Attn
from metrics.FID import *
from loss_functions.perceptual_loss import perceptual_similarity_loss
# from loss_functions.texture_loss import texture_loss
from loss_functions.glcm_soft_einsum import _texture_loss_d5, _GridExtractor_d5, _texture_loss, _GridExtractor
from metrics.mse_psnr_ssim_vif import *
from models.networks import init_net
from piq import brisque
from scipy.stats import skew
from torchmetrics.functional.image import image_gradients
from loss_functions.attention import Self_Attn
from metrics.FID import *
from metrics.mse_psnr_ssim_vif import *
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
from metrics.piqe import *
from models.autoencoder_perceptual import *
from loss_functions.kl_divergence import *
import matplotlib.patches as patches


def save_fig_to_pdf(img, idx, tag, bbox):
    image_array = img.cpu().numpy()  # Assuming image_tensor is a PyTorch tensor

    # Create a figure without axis
    fig, ax = plt.subplots()
    ax.axis('off')  # Turn off axis labels and ticks

    # Plot the image
    ax.imshow(image_array[0, 0, :, :], cmap='gray')

    if bbox:
        # Create a rectangle patch
        rect = patches.Rectangle((100, 148), 50, 50, linewidth=1, edgecolor='r', facecolor='none') # test 2: 158, test 3,4: 148

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


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters: parser          -- original option parser is_train (bool) -- whether training phase or test
        phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the
        following losses. A (source domain), B (target domain). Generators: G_A: A -> B; G_B: B -> A. Discriminators:
        D_A: G_A(A) vs. B; D_B: G_B(B) vs. A. Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the
        paper) Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper) Identity loss (optional):
        lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from
        paintings" in the paper) Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--experiment_name', type=str, default="default", help='experiment name')
        parser.add_argument('--image_folder', type=str, default=None, help='folder to save images during training')
        parser.add_argument('--metric_folder', type=str, default=None, help='folder to save metrics')
        parser.add_argument('--loss_folder', type=str, default=None, help='folder to save losses')
        parser.add_argument('--test_folder', type=str, default=None, help='folder to save test images')
        parser.add_argument('--test', type=str, default="test_1", help='folder to save test images')
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of '
                                     'scaling the weight of the identity mapping loss. For example, if the weight of '
                                     'the identity loss should be 10 times smaller than the weight of the '
                                     'reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_texture', type=float, default=0.001, help='use texture loss.')
            parser.add_argument('--texture_criterion', type=str, default="max", help='texture loss criterion.')
            parser.add_argument('--texture_offsets', type=str, default="all", help='texture offsets.')
            parser.add_argument('--vgg_pretrained', type=str, default=True, help='pretraining flag.')
            parser.add_argument('--vgg_model_path', type=str, default=None, help='finetuned vgg model path.')
            parser.add_argument('--lambda_perceptual', type=float, default=0.1, help='use perceptual loss.')
            parser.add_argument('--perceptual_layers', type=str, default='all', help='choose the perceptual layers.')
            parser.add_argument('--attention_loss', type=bool, default=False,
                                help='choose to use attention mechanism for loss computation.')
            parser.add_argument('--ssim_loss', type=bool, default=False, help='choose to use ssim loss.')
            parser.add_argument('--edge_loss', type=bool, default=False, help='choose to use edge loss.')
            parser.add_argument('--divergence_loss', type=bool, default=False, help='choose to use divergence loss.')
            parser.add_argument('--autoencoder_loss', type=bool, default=False, help='choose to use edge loss.')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call
        # <BaseModel.get_current_losses>
        if opt.experiment_name.find('texture') != -1:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'cycle_texture_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'cycle_texture_B']
        elif opt.experiment_name.find('perceptual') != -1:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'perceptual_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'perceptual_B']
        elif opt.experiment_name.find('baseline') != -1:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        elif opt.experiment_name.find('ssim') != -1:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'ssim_A', 'ssim_B']
        elif opt.experiment_name.find('edge') != -1:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'edge_A', 'edge_B']
        elif opt.experiment_name.find('divergence') != -1:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'divergence_A', 'divergence_B']
        elif opt.experiment_name.find('autoencoder') != -1:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'autoencoder_A', 'autoencoder_B']

        # dictionary to store training loss
        self.error_store = OrderedDict()
        for key in self.loss_names:
            self.error_store[key] = list()

        # specify the images you want to save/display. The training/test scripts will call
        # <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(
            # B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        #####
        self.test_visual_names = ['real_A', 'fake_B', 'real_B']

        self.metric_names = ['psnr', 'mse', 'ssim', 'vif', 'NIQE', 'PIQE', 'FID_ImNet', 'FID_random', 'brisque']
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
        self.raps_data_3 = init_storing(test_3_ids, ['raps'])
        self.raps_data_2 = init_storing(test_2_ids, ['raps'])
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

        self.time = []

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

        # self.fid_object_1 = GANMetrics('cuda', detector_name='inceptionv3', batch_size=64, pretrained=True)
        # self.fid_object_2 = GANMetrics('cuda', detector_name='inceptionv3', batch_size=64, pretrained=False)

        # self.real_test_buffer = []
        # self.fake_test_buffer = []
        self.real_buffer_2 = []
        self.fake_buffer_2 = []

        self.raps = 0

        # NIQE
        self.niqe = pyiqa.create_metric('niqe', device=torch.device('cpu'), as_loss=False)
        # self.brisque = pyiqa.create_metric('brisque', device=torch.device('cpu'), as_loss=False)

        # #### specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']  # , 'G_B', "attention"
            self.texture_grids_2 = init_image_buffer(test_2_ids)
            self.attention_maps_2 = init_image_buffer(test_2_ids)

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain)
        if opt.texture_criterion == 'attention':
            self.netattention = init_net(Self_Attn(1, 'relu')).to(self.device)

        self.skweness = []

        self.grid_extractor = _GridExtractor()
        # Wrap the texture_extractor in DataParallel if you have multiple GPUs
        if torch.cuda.device_count() > 1:
            self.grid_extractor = nn.DataParallel(self.grid_extractor)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            # define loss_functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionTexture = torch.nn.L1Loss(reduction='none')  # operator

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if opt.texture_criterion == 'attention':
                self.netattention = init_net(Self_Attn(1, 'relu')).to(self.device)

                # self.optimizer_att = torch.optim.Adam(self.attention.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.weight_A = list()
                self.weight_B = list()
                self.attention_A = list()
                self.attention_B = list()

                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), self.netattention.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # SSIM loss function
            if opt.ssim_loss == True:
                self.ssim_loss = SSIMLoss(data_range=1)

            # Divergence loss function
            if opt.divergence_loss == True:
                self.img_min = opt.window_center - opt.window_width // 2
                self.img_max = opt.window_center + opt.window_width // 2
                self.histogramming = Histogramming(100, self.img_min, self.img_max, sigma=4)

            # Edge loss function
            if opt.edge_loss == True:
                self.edge_loss = EdgeLoss()

            # Perceptual autoencoder loss
            if opt.autoencoder_loss == True:
                self.autoencoder = Autoencoder().eval().to("cuda:1")
                checkpoint = torch.load("/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/cycleGAN/autoencoder/early_autoencoder_7")
                self.autoencoder.load_state_dict(checkpoint)

            if opt.lambda_perceptual > 0.0:
                if opt.vgg_pretrained == True:
                    self.vgg = VGG().eval().to("cuda:1")
                else:
                    self.vgg = VGG(num_classes=4, pretrained=False, init_type='finetuned',
                                   saved_weights_path=opt.vgg_model_path)
            self.index_texture_A = list()
            self.index_texture_B = list()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
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
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def test(self):

        with torch.no_grad():
            self.fake_B = self.netG_A(self.real_A)
            # diff = abs(self.fake_B[0, 0, :, :] - self.real_A[0, 0, :, :])
            # self.skweness.append(skew(diff.detach().cpu().flatten()))
            self.compute_metrics()

            self.track_metrics_per_patient(self.id)

    def test_visuals(self, iter, list_index):

        with torch.no_grad():
            if iter in list_index:
                self.fake_B = self.netG_A(self.real_A)
                grad_fake_B_x, grad_fake_B_y = image_gradients(self.fake_B.cpu())
                grad_fake_B = torch.sqrt(grad_fake_B_x ** 2 + grad_fake_B_y ** 2)
                save_fig_to_pdf(self.fake_B, iter, self.opt.experiment_name, False)
                # save_fig_in_pdf(grad_fake_B, iter, self.opt.experiment_name, True)
                # save_fig_in_pdf(grad_fake_B[:, :, 148:198, 100:150], iter, f"zoom_{self.opt.experiment_name}", False)  # test 2: 158:208, test 3,4: 148:198

    def test2(self, iter):

        with torch.no_grad():

            for i in zip(x_coord, y_coord):
                self.fake_B = self.netG_A(self.real_A)
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

    def test_3(self):
        with torch.no_grad():
            self.fake_B = self.netG_A(self.real_A)
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
            fake_B_3channels = self.fake_B.expand(-1, 3, -1, -1)
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
            fake_B_3channels = self.fake_B.expand(-1, 3, -1, -1)  # +1)*0.5
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
            fake_B_3channels = self.fake_B.expand(-1, 3, -1, -1)
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

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        lambda_texture = self.opt.lambda_texture

        # Divergence loss
        if self.opt.divergence_loss == True:
            print(self.real_A.min(), self.real_A.max())
            print(self.rec_A.min(), self.rec_A.max())
            self.loss_divergence_A = compute_kl_divergence(self.real_A, self.rec_A, self.img_min, self.img_max, self.histogramming)
            self.loss_divergence_B = compute_kl_divergence(self.real_B, self.rec_B, self.img_min, self.img_max, self.histogramming)
            print(self.loss_divergence_A, self.loss_divergence_B)
        else:
            self.loss_divergence_A = 0
            self.loss_divergence_B = 0

        # SSIM loss
        if self.opt.ssim_loss == True:
            self.loss_ssim_A = self.ssim_loss((self.rec_A + 1) * 0.5, (self.real_A + 1) * 0.5)
            self.loss_ssim_B = self.ssim_loss((self.rec_B + 1) * 0.5, (self.real_B + 1) * 0.5)
        else:
            self.loss_ssim_A = 0
            self.loss_ssim_B = 0

        # Edge loss
        if self.opt.edge_loss == True:
            self.loss_edge_A = 10 * self.edge_loss(self.rec_A, self.real_A)
            self.loss_edge_B = 10 * self.edge_loss(self.rec_B, self.real_B)
        else:
            self.loss_edge_A = 0
            self.loss_edge_B = 0

        # Perceptual auto-encoder loss
        if self.opt.autoencoder_loss == True:
            self.loss_autoencoder_A = 10 * perceptual_autoencoder_loss(self.rec_A, self.real_A, self.autoencoder)
            self.loss_autoencoder_B = 10 * perceptual_autoencoder_loss(self.rec_B, self.real_B, self.autoencoder)
        else:
            self.loss_autoencoder_A = 0
            self.loss_autoencoder_B = 0

            # Texture loss
        if lambda_texture > 0:
            if self.opt.texture_criterion == 'attention':
                loss_texture_A, map_A, weight_A = _texture_loss(self.rec_A, self.real_A, self.opt, self.grid_extractor,
                                                                self.netattention)
                loss_texture_B, map_B, weight_B = _texture_loss(self.rec_B, self.real_B, self.opt, self.grid_extractor,
                                                                self.netattention)

                self.loss_cycle_texture_A = loss_texture_A
                self.loss_cycle_texture_B = loss_texture_B
                self.weight_A.append(weight_A.item())
                self.weight_B.append(weight_B.item())
                self.attention_A.append(map_A.detach().clone().cpu().numpy())
                self.attention_B.append(map_B.detach().clone().cpu().numpy())

            elif self.opt.texture_criterion == 'max':
                loss_texture_A = _texture_loss_d5(self.rec_A, self.real_A, self.opt, self.grid_extractor)
                loss_texture_B = _texture_loss_d5(self.rec_B, self.real_B, self.opt, self.grid_extractor)

                # save the index of the maximum texture descriptor for each image in the batch
                '''for i in range(self.opt.batch_size):
                    self.index_texture_A.append(torch.nonzero(delta_grids_A == criterion_texture_A[i]).squeeze())
                    self.index_texture_B.append(torch.nonzero(delta_grids_B == criterion_texture_B[i]).squeeze())'''

                # compute the loss function by averaging over the batch
                self.loss_cycle_texture_A = loss_texture_A * lambda_texture
                self.loss_cycle_texture_B = loss_texture_B * lambda_texture

            elif self.opt.texture_criterion == 'average':
                loss_texture_A = _texture_loss_d5(self.rec_A, self.real_A, self.opt, self.grid_extractor)
                loss_texture_B = _texture_loss_d5(self.rec_B, self.real_B, self.opt, self.grid_extractor)

                self.loss_cycle_texture_A = loss_texture_A * lambda_texture
                self.loss_cycle_texture_B = loss_texture_B * lambda_texture

            elif self.opt.texture_criterion == 'Frobenius':
                loss_texture_A = _texture_loss_d5(self.rec_A, self.real_A, self.opt, self.grid_extractor)
                loss_texture_B = _texture_loss_d5(self.rec_B, self.real_B, self.opt, self.grid_extractor)

                self.loss_cycle_texture_A = loss_texture_A * lambda_texture
                self.loss_cycle_texture_B = loss_texture_B * lambda_texture
            else:
                raise NotImplementedError
        else:
            self.loss_cycle_texture_A = 0
            self.loss_cycle_texture_B = 0

        # Perceptual loss
        lambda_perceptual = self.opt.lambda_perceptual
        if lambda_perceptual > 0:
            loss_perceptual_A = perceptual_similarity_loss(self.rec_A, self.real_A, self.vgg,
                                                           self.opt.perceptual_layers)
            loss_perceptual_B = perceptual_similarity_loss(self.rec_B, self.real_B, self.vgg,
                                                           self.opt.perceptual_layers)
            self.loss_perceptual_A = loss_perceptual_A * lambda_perceptual
            self.loss_perceptual_B = loss_perceptual_B * lambda_perceptual
        else:
            self.loss_perceptual_A = 0
            self.loss_perceptual_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        self.loss_G = (self.loss_G_A
                       + self.loss_G_B
                       + self.loss_cycle_A
                       + self.loss_cycle_B
                       + self.loss_idt_A
                       + self.loss_idt_B
                       + self.loss_cycle_texture_A
                       + self.loss_cycle_texture_B
                       + self.loss_perceptual_A
                       + self.loss_perceptual_B
                       + self.loss_ssim_A
                       + self.loss_ssim_B
                       + self.loss_edge_A
                       + self.loss_edge_B
                       + self.loss_autoencoder_A
                       + self.loss_autoencoder_B
                       + self.loss_divergence_A
                       + self.loss_divergence_B
                       )

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B

        ts = time.time()
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero

        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        tf = time.time()

        self.time.append(tf - ts)

    def save_test_2(self, epoch):
        save_to_json(self.texture_grids_2, f"{self.metric_dir}/delta_grids_B_test_2_ep{epoch}")
        empty_Dictionary(self.texture_grids_2, nesting=1)
        save_to_json(self.attention_maps_2, f"{self.metric_dir}/attention_maps_B_ep{epoch}")
        empty_Dictionary(self.attention_maps_2, nesting=1)

    def save_texture_indexes(self):
        save_list_to_csv(self.index_texture_A, f'{self.loss_dir}/idx_texture_A.csv')
        save_list_to_csv(self.index_texture_B, f'{self.loss_dir}/idx_texture_B.csv')

    def save_attention_maps(self):
        np.save(f"{self.loss_dir}/attention_A.npy", np.array(self.attention_A))
        np.save(f"{self.loss_dir}/attention_B.npy", np.array(self.attention_B))

    def save_attention_net(self, epoch):
        torch.save(self.netattention.state_dict(), f"{self.save_dir}/netattention_ep{epoch}_{self.opt.experiment_name}.pth")

    def save_attention_weights(self):
        np.save(f"{self.loss_dir}/weight_A.npy", np.array(self.weight_A))
        np.save(f"{self.loss_dir}/weight_B.npy", np.array(self.weight_B))

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

    def save_raps(self, epoch):
        save_json(self.raps, f"{self.metric_dir}/raps_{self.opt.test}_epoch{epoch}")
        self.raps = []

    def save_time(self):
        np.save(f"{self.metric_dir}/time.npy", np.array(self.time))

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


if __name__ == "__main__":
    print(2)
