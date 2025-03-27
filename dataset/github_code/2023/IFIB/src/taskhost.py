import datetime, os, sys, torch, importlib, random
import numpy as np

import torch.distributed as dist
import torch.multiprocessing as mp

from src.taskhost_utils import getLogger

'''
The TaskHost executes tasks using pytorch.multiprocessing. Credits to the neural_stpp created by Ricky TQ Chen from Facebook.
'''
logger = getLogger('TaskHost')

class TaskHost:
    def __init__(self, parser, root_path):
        self.opt = parser.parse_args()
        self.root_path = root_path

        procedure = importlib.import_module('src.' + self.opt.procedure)
        self.opt = getattr(procedure, f'{self.opt.task_category}_postprocess')(self.opt, self.root_path)


    def start(self):
        '''
        All source files related to the specific procedure should locate in src, and the folder name
        should match the given procedure name.
        
        Caveats:
        1. The arguments loader should be named as "main_procedure + sub_procedure + Arguments"(no whitespace).
        2. The name of the entry function should be 'work'.
        '''
        logger.info(f'Root path: {self.root_path}')
        logger.info(f'Main procedure name: {self.opt.procedure}, sub-procedure name: {self.opt.task_category}.')

        '''
        Reproducibility.
        '''
        if self.opt.no_seed:
            import time
            logger.warning(f'Reproducibility only presents when you give me a random seed. If you really request reproducible results. Please ABORT this run ASAP and manually assign a random seed using argument \'--seed\'')
            logger.warning(f'No explicit random seed is available, so the framework will spontaneously select a number as the random seed.')
            random.seed(int(time.time()) % 65535)
            self.opt.seed = random.randint(0, 65535)
            logger.info(f'The model prefers {self.opt.seed} this time.')
        else:
            logger.info(f'Following your request, the random seed is {self.opt.seed}.')

        # Check cuda availability. We will force using CPU if cuda is unavailable even the user script wants to use cuda.
        if self.opt.cuda and not torch.cuda.is_available():
            logger.warning('You expect cuda acceleration but cuda is unavailable in this machine. Please check your cuda configuration and make sure that you have installed pytorch with cuda support.')
            logger.warning('We use cpu now.')
            self.opt.cuda = False

        # Prepare for multithreading
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(int(np.random.randint(30000, 65535)))

        '''
        Please check https://pytorch.org/docs/stable/notes/randomness.html?highlight=reproducibility for furhter information about
        reproducibility
        '''
        random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)
        np.random.seed(self.opt.seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        # For debug usage
        # torch.autograd.set_detect_anomaly(True)

        try:
            mp.set_start_method("forkserver")
            mp.spawn(self.main, nprocs=self.opt.ngpus, join=True)
        except Exception:
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)
    

    def main(self, rank):
        '''
        Multiprocessing training controller.
        '''
        dist.init_process_group("nccl" if self.opt.cuda else 'gloo', rank=rank, world_size=self.opt.ngpus, timeout=datetime.timedelta(minutes=30))

        procedure = importlib.import_module('src.' + self.opt.procedure)
        self.worker = getattr(procedure, self.opt.procedure + self.opt.task_category)()

        if rank == 0:
            logger.info(f'PyTorch Version: {torch.__version__}.')
            '''
            Avoid pytorch issue #36313
            '''
            if torch.__version__ == '1.4.0':
                raise logger.exception('Due to the pytorch issue #36313(https://github.com/pytorch/pytorch/issues/36313),\
                several learning rate schedulers including LambdaLR used by this architecture fail to run. Please update PyTorch to 1.5.0 or above.')
    
        '''
        Report device status
        '''
        self.opt.device = torch.device(
            f'cuda:{rank:d}' if self.opt.cuda else 'cpu')
    
        if rank == 0:
            if self.opt.device.type == 'cuda':
                logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    logger.info('{} \t Memory: {:.2f}GiB'.format(props.name, props.total_memory / (1024**3)))
            else:
                logger.info('WARNING: Using device {}'.format(self.opt.device))
    
        try:
            self.worker.work(rank = rank, opt = self.opt)
        except:
            import traceback
            logger.error(traceback.format_exc())
            raise
    
        dist.destroy_process_group()