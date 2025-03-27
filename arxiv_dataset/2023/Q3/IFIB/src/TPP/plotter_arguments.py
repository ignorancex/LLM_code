import os, argparse
from src.arguments import BasicArguments
from src.TPP.utils import suffix


class TPPPlotterArguments(BasicArguments):
    def __init__(self, parser, root_path):
        super().__init__(parser)

        self.root_path = root_path        
        # Input data
        self.parser.add_argument('--dataset_name', type=str, default=None, help='Name of the used dataset. All datasets should be placed in {root}/data/input.')
        self.parser.add_argument('--dataset_type', type=str, default='json', help='File type of the selected dataset.')
        self.parser.add_argument('--dataloader_name', default=None, help='Name of the used dataloader. All dataloaders are stored in {root}/src/TPP/dataloader.')
        self.parser.add_argument('--dataloader_config', type=str, default=None, help='Relative path to the custom dataloader config file. This absolute file path is {root}/config/{model_name}/{dataloader_config}.')
        self.parser.add_argument('--used_dataloader_config', type=str, default = None, help='The name of dataloader config file used during training. We only need the filename, not the relative path.')
        self.parser.add_argument('--custom_collator', action='store_true',\
                help='seems useless now. I might remove it in the future.')

        # Training procedure related hyperparameters
        self.parser.add_argument('--n_training_steps', type=int, default=10000, help='How many steps did we use to train this model?')
        self.parser.add_argument('--agg_update_step', type=int, default=1, help='The number of minibatches between two adjacent optimizer steps.\
                                                                                 The number of practical training steps is agg_update_step * n_training_steps')

        # Model save and log management
        self.parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best', help='Store all model checkpoints or only store the best one.')
        
        # Training procedure related hyperparameters
        self.parser.add_argument('-ub', '--used_batch_size', type=int, default=2048, help='Batch size used for training the model.')
        
        # Model-related hyperparameters
        self.parser.add_argument('--model_name', default=None, help="The model name.")
        self.parser.add_argument('--model_config', type=str, default=None, help="Relative path to the custom model config file used for training. This absolute file path is {root}/config/{model_name}/{model_config}.")
        
        # Optimizer-related hyperparameters
        self.parser.add_argument('--lr', type=float, default=0.1, 
                            help='The learning rate used when training the model.')

        # plotter specific
        parser.add_argument('--figure_count', type = int, help='We will select \{figure_count\} records from training set(if set),\
                                                  test set(if set), and evaluation set(if set), respectively. So there will be\
                                                  \{enabled_dataset\} * figure_count plots when the plotter finish running.')
        parser.add_argument('--train', action='store_true', help='Evaluation on the training dataset.')
        parser.add_argument('--test', action='store_true', help='Evaluation on the test dataset.')
        parser.add_argument('--evaluation', action='store_true', help='Evaluation on the evaluation dataset.')
        parser.add_argument('--plot_type', type=str, choices=['intensity', 'probability', 'integral', 'debug'], default = 'intensity')
        parser.add_argument('--resolution', type=int, default=100, help='How many interpolating points may each time interval have?')
        parser.add_argument('--sample_amount', type=int, default=500, help='The number of samples per dim of a high-dimensional space.')
        parser.add_argument('--task_name', type=str, help='Define which evaluation task you\'d like to start.')

        # identification mark
        self.parser.add_argument('--procedure', type = str, default = 'TPP',
                            help=argparse.SUPPRESS)
        self.parser.add_argument('--task_category', type = str, default = 'Plotter',
                            help=argparse.SUPPRESS)
        

def Plotter_postprocess(opt, root_path):
    '''
    The preprocessing function for TPPPlotterArguments.
    '''
    if opt.agg_update_step > 1:
        opt.n_training_steps *= opt.agg_update_step

    opt.batch_size = 1
    opt.data_path = os.path.join(root_path, 'data', opt.procedure, opt.dataset_name)
    opt.abs_dataloader_config = os.path.join(root_path, 'config', opt.model_name, opt.dataloader_config) if opt.dataloader_config else None
    opt.dataloader_config = os.path.basename(opt.abs_dataloader_config) if opt.dataloader_config else None
    opt.abs_model_config = os.path.join(root_path, 'config', opt.model_name, opt.model_config) if opt.model_config else None
    opt.model_config = os.path.basename(opt.abs_model_config) if opt.model_config else None

    # locate where checkpoints are stored.
    model_hyperparameters = suffix(opt, 'model_name', 'lr', 'used_batch_size', 'n_training_steps', 'used_dataloader_config', 'model_config')
    folder_suffix = 'output_' + model_hyperparameters
    opt.checkpoint_folder = os.path.join(root_path, 'model', opt.dataset_name, folder_suffix)

    # where figures, records are stored.
    opt.store_dir = os.path.join(root_path, 'output', opt.dataset_name, 'output_' + model_hyperparameters)

    return opt