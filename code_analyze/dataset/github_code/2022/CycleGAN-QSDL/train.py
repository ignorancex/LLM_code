import os
import h5py

from models.cyclegan import CycleGAN
from util.parser import training_parser

os.environ['CUDA_VISIBLE_DEVICES']='0'

def main():
	args = training_parser().parse_args()

	name = args.name
	restore = args.restore
	restore_ckpt = True if restore else False

	# f = h5py.File('./bp_gan_norm_chuyi_MAX.h5', 'r')
	# data = f.get('fbp')  # input size 64*64
	# label = f.get('label')  # label size 64*64
	# data = f.get('nac')  # input size 64*64
	# label = f.get('ac')  # label size 64*64

	# print(data)
	#File paths
	train_dir = os.path.join('Network/', name)

	cyclegan = CycleGAN(args, True, restore_ckpt)
	# cyclegan.train(data.value, label.value)
	cyclegan.train()

if __name__ == '__main__':
    main()