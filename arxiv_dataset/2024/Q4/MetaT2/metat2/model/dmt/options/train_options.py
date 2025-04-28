from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for trans training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=60, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')

        # for unet training
        parser.add_argument('--seg_lr', type=float, default=0.0001, help='initial learning rate for unet')

        # for initialize model
        parser.add_argument('--pretrained_netG', type=str, help='pretrained netg path')
        parser.add_argument('--pretrained_unet', type=str, help='pretrained unet path')

        # for meta model
        parser.add_argument('--meta_model_save_path', type=str, help='meta model save path')
        parser.add_argument('--train_image_dir', type=str, help='train image dir')
        parser.add_argument('--vali_image_dir', type=str, help='validation image dir')        
        parser.add_argument('--test_image_dir', type=str, help='test image dir')        
        parser.add_argument('--croot_modality', type=str, default='t2w', help='croot modality')        
        parser.add_argument('--sroot_modality', type=str, default='dwi', help='sroot modality')        
        parser.add_argument('--start_epoch', type=int, default=60, help='start epoch for netg')        

        # for inference
        parser.add_argument('--save_inference_results', action='store_true', default=True, help='if save inference results.')        
        parser.add_argument('--inference_save_path', type=str, help='inference results save path')        

        # for discriminators
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
        parser.add_argument('--lambda_kld', type=float, default=0.05)

        # for DMT
        parser.add_argument('--diffusion_path', type=str, default=None, help='Path to pre-traiend DPM.')
        parser.add_argument('--denoise_step', type=int, default=50, help='Timestep to diffuse.')
        parser.add_argument('--use_fp16', action='store_true', help='Whether to use fp16 for DPM.')
        parser.add_argument('--saved_ckpt', type=str, default=None, help='Path to pre-traiend DPM.')
        parser.add_argument('--isInference', action='store_true', help='Whether to use fp16 for DPM.')

        self.isTrain = True
        return parser
