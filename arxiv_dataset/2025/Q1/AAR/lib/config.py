from argparse import ArgumentParser

BATCHNORM_MOMENTUM = 0.01

class Config(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """
        Defaults
        """
        self.save_path = None
        self.model_path = None
        self.seq_model_path = None
        self.data_path = None
        self.conf_path = None
        self.optimizer = None
        self.lr_scheduler = None
        self.semantic = None
        self.mlp_layers = None
        self.pool_type = None
        self.concept_net = None
        self.model_type = None
        self.visualize = None
        self.lr = 1e-5
        self.enc_layer = 1
        self.dec_layer = 3
        self.nepoch = 20
        self.num_frames = 0
        self.num_workers = 0
        self.emb_out = 512
        self.resume = None
        self.infer_last = None
        self.top_k = None
        self.task = None
        self.seq_layer = 1
        self.seq_model = None
        self.seq_model_mlp_layers = None
        self.resume_epoch = None
        self.cross_attention = False
        self.predict_flag = False
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.__dict__.update(self.args)

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')
        parser.add_argument('--save_path', default='data/', type=str)
        parser.add_argument('--model_path', default=None, type=str)
        parser.add_argument('--seq_model_path', default=None, type=str)
        parser.add_argument('--conf_path', default=None, type=str)
        parser.add_argument('--data_path', default='/media/ntu/clement/dataset/ag/', type=str)
        parser.add_argument('--optimizer', help='adamw/adam/sgd', default='adamw', type=str)
        parser.add_argument('--lr', dest='lr', help='learning rate', default=1e-5, type=float)
        parser.add_argument('--nepoch', help='epoch number', default=10, type=int)
        parser.add_argument('--enc_layer', dest='enc_layer', help='spatial encoder layer', default=1, type=int)
        parser.add_argument('--dec_layer', dest='dec_layer', help='temporal decoder layer', default=3, type=int)
        parser.add_argument('--mlp_layers', dest='mlp_layers', help='number of mlp layers', default=2, type=int)
        parser.add_argument('--num_workers', dest='num_workers', help='number of workers in the dataloader', default=0, type=int)
        parser.add_argument('--num_frames', dest='num_frames', type=int, help='number of past frame actions to consider (inclusive of the current frame)', default=2) 
        parser.add_argument('--lr_scheduler', help='whether to use lr scheduler or not', action='store_true', default=True)
        parser.add_argument('--semantic', help='whether to use semantic features or not', action='store_true', default=True)
        parser.add_argument('--pool_type', help='whether to use max/avg pooling on features of the models.', default='max', dest='pool_type', type=str)
        parser.add_argument('--model_type', help='either mlp, transformer, GNNED, RBP, BiGED, or Relational', default='mlp', dest='model_type', type=str)
        parser.add_argument('--concept_net', help='whether to use concept_net embeddings or not', action='store_true', dest='concept_net')
        parser.add_argument('--emb_out', help='out size for BiLinearGraphEncDec', default=512, type=int)
        parser.add_argument('--resume', default=None, type=str)
        parser.add_argument('--visualize', default=None, action='store_true', dest='visualize')
        parser.add_argument('--top_k', help='top_k for recall computation', type=int, default=10, dest='top_k')
        parser.add_argument('--task', help='select task: either set or sequence prediction', type=str, default="set", dest='task')
        parser.add_argument('--infer_last', help='whether to only perform inference on the last frame', action='store_true', dest='infer_last')
        parser.add_argument('--seq_layer', help='number of gru/transformer layers for the sequence task', type=int, default=1, dest='seq_layer')
        parser.add_argument('--seq_model', help='gru or transformer decoder', default='gru', dest='seq_model', type=str)
        parser.add_argument('--hidden_dim', help='dimension of hidden layers in GRU or transformer for the sequence task', type=int, default=1936, dest='hidden_dim')
        parser.add_argument('--seq_model_mlp_layers', dest='seq_model_mlp_layers', help='number of mlp layers in the classifier of the sequential model', default=2, type=int)
        parser.add_argument('--cross_attention', dest='cross_attention', help='whether to use cross-attention in transformer decoder', default=False, action='store_true')
        parser.add_argument('--predict_flag', dest='predict_flag', help='whether to use fasterRCNN labels or not', default=False, action='store_true')
        
        return parser