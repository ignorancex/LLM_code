import pickle
from run_ppl import *
import seaborn as sns
import matplotlib.pyplot as plt
import functools

def get_ppl_(logits, generated_ids):
    generated_ids = generated_ids.detach().cpu()
    logits = logits[:-1]
    generated_ids = generated_ids[1:]
    generated_probs = torch.softmax(logits, dim=-1)
    generated_per_token_perplexity = torch.exp(-torch.log(generated_probs.gather(dim=-1, index=generated_ids.unsqueeze(1))).mean())
    generated_perplexity = generated_per_token_perplexity.item()
    
    loss_function = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_function(logits, generated_ids.view(-1))
    if any(torch.isnan(loss)):
        nan_indices = torch.nonzero(torch.isnan(loss)).flatten()
        logit_nan = logits[nan_indices]
        ids_nan = generated_ids[nan_indices]
        print('some items is nan')
    
    perplexity = torch.exp(torch.mean(loss))
    # perplexity = torch.mean(loss)
    if torch.isnan(perplexity):
        perplexity = 0
    return -perplexity

def get_eos_probability(logits, generated_ids):
    generated_ids = generated_ids.detach().cpu()
    logits = logits[-1]
    probs = torch.softmax(logits, dim=-1)
    eos_token_id = 50256
    eos_prob = probs[eos_token_id].item()
    return -eos_prob

def get_probability_margin(logits, generated_ids, log=False, cnt_zero=False):
    generated_ids = generated_ids.detach().cpu()
    logits = logits.squeeze()
    logits = logits[:-1]
    generated_ids = generated_ids[1:]
    probs = F.softmax(logits, dim=1)

    input_probs = torch.gather(probs, dim=1, index=generated_ids.unsqueeze(1))
    max_probs, max_indexes = torch.max(probs, dim=1)
    prob_margin = (max_probs - input_probs.squeeze()) / max_probs
    
    if cnt_zero:
        count_zero = float(torch.sum(prob_margin == 0).item() + 1.)
        if log:
            prob_margin = np.log(count_zero)
        return count_zero
    elif log:
        prob_margin = torch.exp(prob_margin)
    # probs.scatter_(1, max_indexes.unsqueeze(1), 0)  # Zero out max probabilities
    # second_max_probs, second_max_indexes = torch.max(probs, dim=1)
    # prob_margin = max_probs - second_max_probs
    
    prob_margin = torch.mean(prob_margin)
    return -prob_margin.item()

def get_rank(logits, labels, log=False):
    if logits.shape[0] != 1:
        logits = logits.unsqueeze(0)
    logits = logits[:, :-1]
    labels = labels[1:]
    # get ank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:,-1], matches[:,-2]

    # make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1 # convert to 1-indexed rank
    if log:
        ranks = torch.log(ranks)

    return -ranks.float().mean().item()
    
def get_ppl_of_all(logits, input_ids, generate_ids=None, num_tokens=500):
    ppls = []
    for i in range(len(logits)):
        input_id = input_ids[i].squeeze()
        len_input_id = input_id.shape[-1]
        
        input_logits = logits[i][:num_tokens + len_input_id]
        generate_id = generate_ids[i][:num_tokens + len_input_id]
        ppls.append(get_ppl(input_logits, generate_id))
        
    return ppls

def get_ppl_of_input(logits, input_ids, generate_ids=None, func=get_probability_margin):
    ppls = []
    for i in range(len(logits)):
        input_id = input_ids[i].squeeze()
        len_input_id = input_id.shape[-1]
        
        input_logits = logits[i][:len_input_id]
        ppls.append(func(input_logits, input_id))
    return ppls

def get_entropy_of_generation(logits, num_tokens=500):
    entropies = []
    for i in range(len(logits)):
        
        input_logits = logits[i].squeeze()
        
        softmax_probs = F.softmax(input_logits, dim=-1)
        entropy = -torch.sum(softmax_probs * torch.log(softmax_probs), dim=-1)
        
        entropies.append(entropy.mean().item())
        
    return np.mean(entropies)

def get_generation_length(generated_ids, input_ids):
    lengthes = []
    
    for i in range(len(generated_ids)):
        ids_generated = generated_ids[i].squeeze()
        id_input = input_ids[i].squeeze()
        
        len_generated = ids_generated.shape[-1]
        len_input = id_input.shape[-1]
        
        generated_len = len_generated - len_input
        lengthes.append(generated_len)
    
    lengthes = np.array(lengthes)
    return np.min(lengthes), np.mean(lengthes), np.median(lengthes)

# def plt_vector_difference(real_samples, fake_samples, name):
#     # Create a DataFrame for the data
#     import pandas as pd
#     data = pd.DataFrame({'Real samples': real_samples, 'Fake samples': fake_samples})

#     # Create a figure with subplots
#     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

#     # Plot a violin plot
#     sns.violinplot(data=data, ax=axes[0])
#     axes[0].set_title('Violin Plot Comparison')

#     # Plot a box plot
#     sns.boxplot(data=data, ax=axes[1])
#     axes[1].set_title('Box Plot Comparison')
    
#     sns.lineplot(data=data, ax=axes[2])
#     axes[1].set_title('Line Plot Comparison')
#     plt.savefig(name)
#     plt.close()        
            
def get_metrics(scores_real, scores_sample, name):
    fpr, tpr, roc_auc = get_roc_metrics(scores_real, scores_sample)
    result90 = get_native_detection_metric(np.array(scores_real), np.array(scores_sample), ratio=0.9)
    result = get_native_detection_metric(np.array(scores_real), np.array(scores_sample))
    result99 = get_native_detection_metric(np.array(scores_real), np.array(scores_sample), ratio=0.99)
    p, r, pr_auc = get_precision_recall_metrics(scores_real, scores_sample)
    mean_real, min_real, max_real = cal_statics(scores_real)
    mean_fake, min_fake, max_fake = cal_statics(scores_sample)
    print(f'{name} of real: min: {min_real} \t mean : {mean_real} \t median {max_real}')
    print(f'{name} of model: min: {min_fake} \t mean : {mean_fake} \t median {max_fake}')
    # plt_vector_difference(scores_real, scores_sample, name=f'{name}.jpg')
    print(f"{name} ROC AUC: {roc_auc}, PR AUC: {pr_auc} fpr90: {result90['FPR']}, fpr95: {result['FPR']}, fpr99: {result99['FPR']}")



def get_entropy_of_dataset(file_path, dataset_name):
    with open(file_path, "rb") as f:
        results = pickle.load(f)
    logits = {
            'real': [x["original_crit"]['logits'] for x in results],
            'samples': [x["sampled_crit"]['logits'] for x in results],
        }
    entropy_first = [get_entropy_of_generation(logits['real']), get_entropy_of_generation(logits['samples'])]
    print(f'Entropy of {dataset_name} real {entropy_first[0]}, Entropy of {dataset_name} sample {entropy_first[1]}')


datas = [ 'squad', 'writing', 'code']#'xsum',
dataset1, dataset2 = 'xsum', 'code'

# for data_name in datas:
    
#     file_path = f"generation_{data_name}_llama2_7B_num500_main.pkl" if data_name != 'squad' else f"generation_{data_name}_llama2_7B_num312_main.pkl"
#     get_entropy_of_dataset(file_path, data_name)

cache_dir = '/n/holyscratch01/barak_lab/Users/hanlinzhang/cache_detectgpt'
for dataset_pair in [('writing', 'code'),  ('writing', 'squad')]:#,('xsum', 'squad'),('squad', 'code'),  ('writing', 'code'),  ('writing', 'squad'),  ('squad', 'code')
    
    print(f'************************************** {dataset_pair} **************************************')
    data1, data2 = dataset_pair
    
    file_path = f"{cache_dir}/generation_{data1}_llama2_7B_num500_main.pkl" if data1 != 'squad' else f"{cache_dir}/generation_{data1}_llama2_7B_num312_main.pkl"
    with open(file_path, "rb") as f:
        results = pickle.load(f)

    file_path_code = f"{cache_dir}/generation_{data2}_llama2_7B_num500_main.pkl" if data2 != 'squad' else f"{cache_dir}/generation_{data2}_llama2_7B_num312_main.pkl"
    with open(file_path_code, "rb") as f:
        results_code = pickle.load(f)
        
    logits = {
        'real': [x["original_crit"]['logits'] for x in results],
        'samples': [x["sampled_crit"]['logits'] for x in results],
    }

    logits_code = {
        'real': [x["original_crit"]['logits'] for x in results_code],
        'samples': [x["sampled_crit"]['logits'] for x in results_code],
    }

    generate_ids = {
        'real': None, #[x["original_crit"]['generated_ids'] for x in results],
        'samples': None #[x["sampled_crit"]['generated_ids'] for x in results],
    }

    input_ids = {
        'real': [x["original_crit"]['input_ids'] for x in results],
        'samples': [x["sampled_crit"]['input_ids'] for x in results],
    }

    input_ids_code = {
        'real': [x["original_crit"]['input_ids'] for x in results_code],
        'samples': [x["sampled_crit"]['input_ids'] for x in results_code],
    }

    logp = {
        'real': [x["original_crit"]['log p'] for x in results],
        'samples': [x["sampled_crit"]['log p'] for x in results],
    }

    logp_code = {
        'real': [x["original_crit"]['log p'] for x in results_code],
        'samples': [x["sampled_crit"]['log p'] for x in results_code],
    }

    def cal_statics(x):
        x = np.array(x)
        return np.mean(x), np.min(x), np.max(x)

    log_rank = functools.partial(get_rank, log=True)

    for func, name in zip([get_rank, None, log_rank, get_probability_margin], ['Rank', 'Log_p', 'Log_rank', 'Margin']):
    # for func, name in zip([get_probability_margin], ['Margin']):
        if func is None:
            metric_real = [logp['real'][i] for i in range(len(logp['real']))]
            metric_samples = [logp['samples'][i] for i in range(len(logp['samples']))]
            metric_real_code = [logp_code['real'][i] for i in range(len(logp_code['real']))]
            metric_samples_code = [logp_code['samples'][i] for i in range(len(logp_code['samples']))]
        else:
            metric_real = get_ppl_of_input(logits['real'], input_ids['real'], generate_ids=generate_ids['real'], func=func)
            metric_samples = get_ppl_of_input(logits['samples'], input_ids['samples'], generate_ids=generate_ids['samples'], func=func)
            metric_real_code = get_ppl_of_input(logits_code['real'], input_ids_code['real'], generate_ids=generate_ids['real'], func=func)
            metric_samples_code = get_ppl_of_input(logits_code['samples'], input_ids_code['samples'], generate_ids=generate_ids['samples'], func=func)
        
        
        get_metrics(metric_real, metric_samples,f'{name}_{data1}')
        get_metrics(metric_real_code, metric_samples_code,f'{name}_{data2}')
        
        get_metrics(metric_real, metric_samples_code,f'{name}_{data1}_real_{data2}_machine')
        get_metrics(metric_real_code, metric_samples,f'{name}_{data2}_real_{data1}_machine')
        
        get_metrics(metric_real + metric_real_code, metric_samples,f'{name}_{data1}_{data2}_real_{data1}_machine')
        get_metrics(metric_real + metric_real_code, metric_samples_code,f'{name}_{data1}_{data2}_real_{data2}_machine')
        get_metrics(metric_real + metric_real_code, metric_samples + metric_samples_code,f'{name}_{data1}_{data2}_real_{data1}_{data2}_machine')