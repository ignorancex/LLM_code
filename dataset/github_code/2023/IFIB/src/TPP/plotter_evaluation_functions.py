import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pickle as pkl

from einops import pack
from tqdm import tqdm
from src.TPP.utils import getLogger


logger = getLogger(name = __file__)


def draw(model, minibatch, desc, batch_idx, opt):
    '''
    This function will be called when task_name = graph

    In the new pipeline, each plot is defined as a instruction list. draw_features() should extract and
    call correct seaborn APIs with expected kwargs. The structure of the dict goes as follows:
    {
        ...
        '[plot name]':
        [
            ...
            {
                'plot_type': '[plot_type]'
                'length': [diagram length],
                'height': [diagram height],
                'kwargs':
                {
                    ...'[arguments sent to seaborn APIs.]'
                }
            }
            ...
        ]
        ...
    }
    '''

    plots = model.plot(minibatch, opt)
    
    # Create the plot storing directory if not exist.
    plot_store_dir_for_this_batch = os.path.join(opt.store_dir, opt.plot_type, desc, str(batch_idx))
    if not os.path.exists(plot_store_dir_for_this_batch):
        os.makedirs(plot_store_dir_for_this_batch)
    
    plt.rcParams.update({'font.size': 22, 'figure.figsize': (9, 7)})
    for plot_name, plot_instructions in plots.items():
        fig = plt.figure()
        ax = None
        for instruction in plot_instructions:
            if instruction.get('plot_type') == 'text':
                ax.text(**instruction['kwargs'])
            else:
                if instruction.get('length') and instruction.get('height'):
                    fig.set_size_inches(instruction.get('length'), instruction.get('height'))
                ax = getattr(sns, instruction['plot_type'])(ax = ax, **instruction['kwargs'])
        
        logger.info(f'Image saved for {plot_name} for No.{batch_idx} minibatch in {desc} dataset.')
        plt.savefig(os.path.join(plot_store_dir_for_this_batch, plot_name + '.png'), dpi = 1000)
        plt.close(fig = fig)


def spearman_and_l1(model, dataset, desc, opt):
    '''
    This function is called when task_name = spearman_and_l1.

    This function calculates the average of spearman and L^1 distance between the learned probability distribution
    and the ground truth on all synthetic event sequences.
    '''
    spearman = 0
    l1 = 0
    size_of_dataset = len(dataset)
    for minibatch in tqdm(dataset, desc = f'Spearman and L1 for {desc}'):
        spearman_for_this_batch, l1_for_this_batch = model.get_spearman_and_l1(minibatch, opt)               
                                                                               # [batch_size, seq_len * resolution]
        spearman += spearman_for_this_batch
        l1 += l1_for_this_batch
    
    spearman = spearman / size_of_dataset
    l1 = l1 / size_of_dataset

    if not os.path.exists(opt.store_dir):
        os.makedirs(opt.store_dir)
    result_file = os.path.join(opt.store_dir, f'{desc}_spearman_and_l1.txt')
    f = open(result_file, 'w')
    f.write(f'For the {desc} of {opt.dataset_name}, we announce that the average spearman coefficient is {spearman} and average L1 distance is {l1}.')
    f.close()


def mae_and_f1(model, dataset, desc, opt):
    '''
    This function is called when task_name = mae_and_f1.

    This function calculates the average of mae and macro-f1 between the model prediction based on history
    and the ground truth on all available event sequences.
    We dump all mae values for calculating Q1, Q2, and Q3 later.
    '''
    mae = None
    f1 = 0
    elapsed_time = 0
    data_size = 0
    size_of_dataset = len(dataset)
    
    with tqdm(dataset, desc = f'MAE and macro-f1 for {desc}') as progress_bar:
        for minibatch in progress_bar:
            mae_per_seq, f1_per_seq = model.get_mae_and_f1(minibatch, opt)
                                                                               # [batch_size, seq_len]
            if mae is None:
                mae = mae_per_seq.flatten()
            else:
                mae, mae_ps = pack((mae, mae_per_seq.flatten()), '*')
            f1 += f1_per_seq
        elapsed_time = progress_bar.format_dict['elapsed']
        data_size = progress_bar.format_dict['total']

    f1 = f1 / size_of_dataset
    mean_mae = mae.mean().item()

    if not os.path.exists(opt.store_dir):
        os.makedirs(opt.store_dir)

    '''
    Report the average of mae and f1.
    '''
    result_file = os.path.join(opt.store_dir, f'{desc}_mae_and_macro-f1.txt')
    f = open(result_file, 'w')
    f.write(f'For the {desc} of {opt.dataset_name}, we announce that the average MAE is {mean_mae} and average macro-F1 is {f1}.\n Evaluation speed: {elapsed_time/data_size}s per sequence.')
    f.close()

    '''
    Dump all maes for further usage.
    '''
    mae_dist_file = os.path.join(opt.store_dir, f'{desc}_mae.pkl')
    f = open(mae_dist_file, 'wb')
    pkl.dump(mae, f)
    f.close()


def mae_e_and_f1(model, dataset, desc, opt):
    '''
    This function is called when task_name = mae_e_and_f1.

    This function calculates the average of mae_e and macro-f1 between the model prediction based on history
    and the ground truth on all available event sequences.
    We dump all mae_e values for calculating Q1, Q2, and Q3 later.
    '''
    mae_e = None
    f1 = []
    probability_sum = None
    elapsed_time = 0
    data_size = 0

    with tqdm(dataset, desc = f'MAE-E and macro-f1 for {desc}') as progress_bar:
        for minibatch in progress_bar:
            mae_e_per_seq, f1_per_seq, probability_sum_per_seq = model.get_mae_e_and_f1(minibatch, opt)
                                                                               # [batch_size, seq_len]
            if mae_e is None:
                mae_e = mae_e_per_seq.flatten()
            else:
                mae_e, mae_e_ps = pack((mae_e, mae_e_per_seq.flatten()), '*')

            if probability_sum is None:
                probability_sum = probability_sum_per_seq.flatten()
            else:
                probability_sum, probability_sum_ps = pack((probability_sum, probability_sum_per_seq.flatten()), '*')

            f1 += f1_per_seq
        elapsed_time = progress_bar.format_dict['elapsed']
        data_size = progress_bar.format_dict['total']

    f1 = np.array(f1).mean()
    mean_mae_e = mae_e.mean().item()
    mean_probability_sum = probability_sum.mean().item()

    if not os.path.exists(opt.store_dir):
        os.makedirs(opt.store_dir)
    
    '''
    Report the average of mae-e and f1.
    '''
    result_file = os.path.join(opt.store_dir, f'{desc}_mae_e_and_macro-f1.txt')
    f = open(result_file, 'w')
    f.write(f'For the {desc} of {opt.dataset_name}, we announce that the average MAE-E is {mean_mae_e} and average macro-F1 is {f1}. The sum of p(t) is {mean_probability_sum}. \n Evaluation speed: {elapsed_time/data_size}s per sequence.')
    f.close()

    '''
    Dump all of mae-es for further usage.
    '''
    mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_mae_e.pkl')
    f = open(mae_e_dist_file, 'wb')
    pkl.dump(mae_e, f)
    f.close()


def mae_and_distance(model, dataset, desc, opt):
    mae = None
    distance = None
    elapsed_time = 0
    data_size = 0

    with tqdm(dataset, desc = f'MAE and macro-f1 for {desc}') as progress_bar:
        for minibatch in progress_bar:
            mae_per_seq, distance_per_seq = model.get_mae_and_distance(minibatch, opt)
                                                                               # [batch_size, seq_len]
            if mae is None:
                mae = mae_per_seq.flatten()
            else:
                mae, mae_ps = pack((mae, mae_per_seq.flatten()), '*')

            if distance is None:
                distance = distance_per_seq.flatten()
            else:
                distance, distance_ps = pack((distance, distance_per_seq.flatten()), '*')

        elapsed_time = progress_bar.format_dict['elapsed']
        data_size = progress_bar.format_dict['total']

    mean_distance = distance.mean().item()
    mean_mae = mae.mean().item()

    if not os.path.exists(opt.store_dir):
        os.makedirs(opt.store_dir)
    
    '''
    Report the average of mae-e and f1.
    '''
    result_file = os.path.join(opt.store_dir, f'{desc}_mae_and_distance.txt')
    f = open(result_file, 'w')
    f.write(f'For the {desc} of {opt.dataset_name}, we announce that the average MAE is {mean_mae} and average distance between prediction and ground truth is {mean_distance}. \n Evaluation speed: {elapsed_time/data_size}s per sequence.')
    f.close()

    '''
    Dump the detailed distribution of mae and distance for further usage.
    '''
    mae_dist_file = os.path.join(opt.store_dir, f'{desc}_mae.pkl')
    f = open(mae_dist_file, 'wb')
    pkl.dump(mae, f)
    f.close()

    distance_dist_file = os.path.join(opt.store_dir, f'{desc}_distance.pkl')
    f = open(distance_dist_file, 'wb')
    pkl.dump(distance, f)
    f.close()


def mae_e_and_distance(model, dataset, desc, opt):
    mae_e = None
    distance = None
    probability_sum = None
    elapsed_time = 0
    data_size = 0

    with tqdm(dataset, desc = f'MAE-E and macro-f1 for {desc}') as progress_bar:
        for minibatch in progress_bar:
            mae_e_per_seq, distance_per_seq, probability_sum_per_seq = model.get_mae_e_and_distance(minibatch, opt)
                                                                               # [batch_size, seq_len]
            if mae_e is None:
                mae_e = mae_e_per_seq.flatten()
            else:
                mae_e, mae_e_ps = pack((mae_e, mae_e_per_seq.flatten()), '*')

            if distance is None:
                distance = distance_per_seq.flatten()
            else:
                distance, distance_ps = pack((distance, distance_per_seq.flatten()), '*')
            
            if probability_sum is None:
                probability_sum = probability_sum_per_seq.flatten()
            else:
                probability_sum, probability_sum_ps = pack((probability_sum, probability_sum_per_seq.flatten()), '*')

        elapsed_time = progress_bar.format_dict['elapsed']
        data_size = progress_bar.format_dict['total']

    mean_distance = distance.mean().item()
    mean_mae_e = mae_e.mean().item()
    mean_probability_sum  = probability_sum.mean().item()

    if not os.path.exists(opt.store_dir):
        os.makedirs(opt.store_dir)
    
    '''
    Report the average of mae-e and f1.
    '''
    result_file = os.path.join(opt.store_dir, f'{desc}_mae_e_and_distance.txt')
    f = open(result_file, 'w')
    f.write(f'For the {desc} of {opt.dataset_name}, we announce that the average MAE-E is {mean_mae_e} and average distance between prediction and ground truth is {mean_distance}. The sum of p(t) is {mean_probability_sum}. \n Evaluation speed: {elapsed_time/data_size}s per sequence.')
    f.close()

    '''
    Dump the detailed distribution of mae-e and distance for further usage.
    '''
    mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_mae_e.pkl')
    f = open(mae_e_dist_file, 'wb')
    pkl.dump(mae_e, f)
    f.close()

    distance_dist_file = os.path.join(opt.store_dir, f'{desc}_distance_with_mae_e.pkl')
    f = open(distance_dist_file, 'wb')
    pkl.dump(distance, f)
    f.close()