import argparse

def ParseArgs():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--difflr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--batch', default=2048, type=int, help='batch size')
	parser.add_argument('--tstBat', default=1024, type=int, help='number of users in a testing batch')
	parser.add_argument('--reg', default=3e-2, type=float, help='weight decay regularizer')
	parser.add_argument('--patience',   type=int,   default=20)
	#retain params
	parser.add_argument('--threshold', default=0.5, type=float, help='threshold to filter users')
	parser.add_argument('--data', default='retail_rocket', type=str, help='name of dataset')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')


	#gcn_setting
	parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
	parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
	parser.add_argument('--decay_step', type=int,   default=1)
	parser.add_argument('--init', default=False, type=bool, help='whether initial embedding')
	parser.add_argument('--latdim', default=64, type=int, help='embedding size')
	parser.add_argument('--gcn_layer', default=2, type=int, help='number of gcn layers')
	parser.add_argument('--uugcn_layer', default=2, type=int, help='number of gcn layers')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--topk', default=20, type=int, help='K of top K')
	parser.add_argument('--dropRate', default=0.5, type=float, help='rate for dropout layer')
	parser.add_argument('--gpu', default='0', type=str, help='indicates which gpu to use')
	
	
	#diffusion setting
	parser.add_argument('--dims', type=str, default='[64]')
	parser.add_argument('--d_emb_size', type=int, default=8)
	parser.add_argument('--norm', type=bool, default=True)
	parser.add_argument('--steps', type=int, default=200)
	parser.add_argument('--noise_scale', type=float, default=1e-4)
	parser.add_argument('--noise_min', type=float, default=0.0001)
	parser.add_argument('--noise_max', type=float, default=0.001)
	parser.add_argument('--sampling_steps', type=int, default=0)



	return parser.parse_args()
args = ParseArgs()


#tmall :lr 1e-3  batch:4096 2048   layer:2 reg:3e-2 steps:200 noise-scale:1e-4

#retail_rocket :lr 1e-3  batch:4096 2048   layer:2 reg:3e-2 steps:150 noise-scale:1e-4