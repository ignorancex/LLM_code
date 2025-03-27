class BasicArguments:
    def __init__(self, parser):
        self.parser = parser
        # The Ultimate
        self.parser.add_argument('--no_seed', action='store_true',
                            help='Do not freeze random seed. Use this option if you want to explore your model\'s robustness.')
        self.parser.add_argument('--seed', type=int, default=32,
                            help='Set global random seed.')
        self.parser.add_argument('--cuda', action='store_true', 
                            help="Set it to true if you want to use GPU to accelerate model training. The framework will silently fallback to CPU if cuda is unavailable on this machine.")
        self.parser.add_argument("--ngpus", type=int, default=1,
                            help="If you want to train your model on multiple GPUs, please set this parameter with integer bigger than 1. Multi-GPU training is unverified, so this functionality might not work.")

        # The number of Dataloader worker
        self.parser.add_argument('--n_worker', default=0, type=int,
                  help='The number of dataloader workers. For most datasets, multiprocessing can speed up the training procedure. But you should set it to lower value, even 0 \
                      if you meet the \'received 0 items of ancdata\' exception.')