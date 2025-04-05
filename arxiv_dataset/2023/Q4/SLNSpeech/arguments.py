import argparse
import os


class ArgParser(object):
    def __init__(self):
        name = 'avs'
        parser = argparse.ArgumentParser()
        # Model related arguments
        parser.add_argument('--id', default=name, help="a name for identifying the model")
        parser.add_argument('--store_path', default='./result' + '/' + name, type=str, help="store_path")

        parser.add_argument('--mode', default='train', help="train/eval")
        parser.add_argument('--new_train', default=True, help="")
        parser.add_argument('--list_train', default='')
        parser.add_argument('--list_val', default='')
        parser.add_argument('--dup_trainset', default=1, type=int, help='duplicate so that one epoch has more iters')

        parser.add_argument('--arch_sound', default='resunet_middle', help="architecture of net_sound: resunet_middle")
        parser.add_argument('--arch_frame', default='resnet18fc', help="architecture of net_frame: vggface / resnet18fc / resnet50fc")
        parser.add_argument('--arch_sign', default='3dresnet', help="architecture of net_sign: 3dresnet")
        parser.add_argument('--fusion_type', default='pcc', help="choose fusion type: concat / pcc / transunet")
        parser.add_argument('--arch_optimizer', default='sgd', help="architecture of net_optimizer: sgd / adam")
        parser.add_argument('--pretrained_frame', default=False, help="weather to use pretrained model")
        parser.add_argument('--pretrained_sign', default=False, help="weather to use pretrained model")

        parser.add_argument('--lr_frame', default=1e-1, type=float, help='LR')
        parser.add_argument('--lr_sound', default=1e-1, type=float, help='LR')
        parser.add_argument('--lr_sign', default=1e-1, type=float, help='LR')
        parser.add_argument('--lr_steps', nargs='+', type=int, default=[40, 80, 120], help='steps to drop LR in epochs')
        parser.add_argument('--beta1', default=0.9, type=float, help='momentum for sgd, beta1 for adam')
        parser.add_argument('--weight_decay', default=1e-4, type=float, help='weights regularizer')

        parser.add_argument('--seed', default=1234, type=int, help='manual seed')
        parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
        parser.add_argument('--num_epoch', default=150, type=int, help='epochs to train for')
        parser.add_argument('--eval_epoch', type=int, default=10, help='frequency to evaluate')
        parser.add_argument('--num_gpus', default=1, type=int, help='number of gpus to use')
        parser.add_argument('--batch_size_per_gpu', default=5, type=int, help='input batch size')
        parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
        parser.add_argument('--num_val', default=1200, type=int, help='max number of images to evalutate')
        # parser.add_argument('--num_vis', default=20, type=int, help='frequency to display')
        # parser.add_argument('--disp_iter', type=int, default=400, help='frequency to display')

        parser.add_argument('--num_mix', default=2, type=int, help="number of sounds to mix")
        parser.add_argument('--num_channels', default=512, type=int, help='number of channels') 
        parser.add_argument('--img_pool', default='maxpool', help="avg or max pool image features")
        parser.add_argument('--img_activation', default='sigmoid', help="activation on the image features")
        parser.add_argument('--sound_activation', default='no', help="activation on the sound features")
        parser.add_argument('--output_activation', default='sigmoid', help="activation on the output")
        parser.add_argument('--aud_channels', default=1, type=int, help='number of audio channels, 2 if complex else 1')
        parser.add_argument('--mask_type', default='binary', type=str, help="type of mask: binary / ratio / complex")
        parser.add_argument('--mask_binary_threshold', default=0.5, type=float, help="threshold in the case of binary masks")
        parser.add_argument('--mask_clip_threshold', default=5., type=float, help="threshold in the case of ratio masks")
        parser.add_argument('--compression_type', type=str, default='none', choices=('hyperbolic', 'sigmoidal', 'none'), help="type of compression on masks")
        parser.add_argument('--hyperbolic_compression_K', type=int, default=10, help="hyperbolic compression K")
        parser.add_argument('--hyperbolic_compression_C', type=float, default=0.1, help="hyperbolic compression C")
        parser.add_argument('--sigmoidal_compression_a', type=float, default=0.1, help="sigmoidal compression a")
        parser.add_argument('--sigmoidal_compression_b', type=int, default=0, help="sigmoidal compression b")
        parser.add_argument('--log_freq', default=1, type=int, help="log frequency scale")
        parser.add_argument('--loss', default='bce', help="loss function to use: bce / l1 / l2")
        parser.add_argument('--weighted_loss', default=0, type=int, help="weighted loss")

        # Data related arguments
        parser.add_argument('--audLen', default=51100, type=int, help='sound length')
        parser.add_argument('--audRate', default=16000, type=int, help='sound sampling rate')
        parser.add_argument('--stft_frame', default=1022, type=int, help="stft frame length")
        parser.add_argument('--stft_hop', default=160, type=int, help="stft hop length")
        parser.add_argument('--stft_win', default=400, type=int, help="stft hop length")

        parser.add_argument('--frameRate', default=8, type=float, help='video frame sampling rate') 
        parser.add_argument('--num_frames', default=3, type=int, help='number of frames') 
        parser.add_argument('--stride_frames', default=8, type=int, help='sampling stride of frames')  
        parser.add_argument('--imgSize', default=224, type=int, help='size of input frame')
        parser.add_argument('--signSize', default=140, type=int, help='size of input sign')

        self.parser = parser

    def parse_arguments(self):
        args = self.parser.parse_args()
        return args
