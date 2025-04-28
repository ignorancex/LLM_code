from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument("--num_layers", default=10, type=int)
        parser.add_argument("--num_channels", default=32, type=int)
        # Need D conv_dim 64
        parser.add_argument("--g_lr", default=1e-4, type=float)
        parser.add_argument("--d_lr", default=1e-4, type=float)
        parser.add_argument("--d_iter", default=1, type=int)
        parser.add_argument("--cutmix_prob", default=0.5, type=float)
        parser.add_argument("--img_gen_loss_weight", default=0.1, type=float)
        parser.add_argument("--grad_gen_loss_weight", default=0.1, type=float)
        parser.add_argument("--pix_loss_weight", default=1., type=float)
        parser.add_argument("--grad_loss_weight", default=20., type=float)
        parser.add_argument("--cr_loss_weight", default=1.0, type=float)
        parser.add_argument("--cutmix_warmup_iter", default=1000, type=int)
        parser.add_argument("--use_grad_discriminator", help='use_grad_discriminator', type=bool, default=True)
        parser.add_argument("--moving_average", default=0.999, type=float)
        parser.add_argument("--repeat_num", default=6, type=int)
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
        parser.add_argument("--dataset_name", type=str, default="apple2orange", help="name of the dataset")
        parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
        parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
        parser.add_argument("--img_height", type=int, default=256, help="size of image height")
        parser.add_argument("--img_width", type=int, default=256, help="size of image width")
        parser.add_argument("--channels", type=int, default=3, help="number of image channels")
        parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator samples")
        parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
        parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
        parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
        parser.add_argument("--lambda_0", type=float, default=10, help="GAN loss weight")
        parser.add_argument("--lambda_1", type=int, default=0.1, help="KL (encoded images) weight")
        parser.add_argument("--lambda_2", type=int, default=100, help="ID pixel-wise weight")
        parser.add_argument("--lambda_3", type=int, default=0.1, help="KL (encoded translated images) weight")
        parser.add_argument("--lambda_4", type=int, default=100, help="Cycle pixel-wise weight")
        parser.add_argument("--lambda_texture", type=int, default=0.001, help="Texture loss weight")
        parser.add_argument("--lambda_perceptual", type=int, default=0.1, help="Perceptual loss weight")
        parser.add_argument('--perceptual_layers', type=str, default='all', help='choose the perceptual layers.')
        parser.add_argument('--vgg_pretrained', type=str, default=True, help='pretraining flag.')
        parser.add_argument("--texture_criterion", type=str, default="max", help="select the aggregation rule")
        parser.add_argument('--image_folder', type=str, default=None, help='folder to save images during training')
        parser.add_argument('--metric_folder', type=str, default=None, help='folder to save metrics')
        parser.add_argument('--loss_folder', type=str, default=None, help='folder to save losses')
        parser.add_argument('--test_folder', type=str, default=None, help='folder to save test images')
        parser.add_argument('--test', type=str, default="test_1", help='folder to save test images')
        parser.add_argument('--experiment_name', type=str, default=None, help='experiment name')
        parser.add_argument('--texture_offsets', type=str, default="all", help='texture offsets.')
        self.isTrain = True
        return parser