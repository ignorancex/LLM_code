from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from model import InferenceModel
from datasets import load_dataset, get_dataset_config_names
import pandas as pd
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from tqdm import tqdm
import argparse
from kaggle_utils import *
from data_sets import *


parser = argparse.ArgumentParser()
parser.add_argument("--hf_logintoken", type=str, default=None, help="HF Login")
parser.add_argument("-m", "--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")

parser.add_argument("--dataset_name", type=str, default="AmazonScience/massive", help="Dataset to use")
parser.add_argument("--split", type=str, default="test", help="Dataset split")
parser.add_argument("--max_instances", type=int, default=None, help="Maximum number of instances")
parser.add_argument("--max_lang", type=int, default=None, help="Maximum number of languages to use")
parser.add_argument("--selected_langs", nargs='+', default=None, help="List of selected languages")
parser.add_argument("--is_predict", action="store_true", help="Whether to run predictions")
parser.add_argument("--apply_template", action="store_true", help="Whether to apply template")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--take_whole", action="store_true", help="Take whole neuron from prompt if is_predict True")
parser.add_argument("--max_tokens_overzeros", type=int, default=10000, help="Max tokens overzeros")
parser.add_argument("--max_sentence_avgs", type=int, default=500, help="Max sentences to count average of for map")
parser.add_argument("--kaggle_dataname_to_save", type=str, default=None, help="Dataset name for saving to Kaggle NO USERNAME!")
parser.add_argument("--is_update", action='store_true', help="Flag to update Kaggle dataset")
parser.add_argument("--parent_dir_to_save", type=str, default=None, help="Parent directory to save like /workspace for runpod")
parser.add_argument("--batch_size", type=int, default=None, help="Batch size")




# MODIF FOR DEBUG
args = parser.parse_args()
# args, unknown = parser.parse_known_args()
# args.model = "Qwen/Qwen2.5-0.5B-Instruct"
# args.dataset_name = "Muennighoff/flores200"
# args.split = "dev"
# args.max_tokens_overzeros = 100000
# args.selected_langs = [
#     "deu_Latn", "eng_Latn"]
# args.max_instances = 500
# args.max_sentence_avgs = 500
# args.batch_size = 32
# args.debug = True

# END MODIF
parent_dir = args.parent_dir_to_save if args.parent_dir_to_save else ""
login(args.hf_logintoken)
model_name = args.model

infer_model = InferenceModel(model_name)
raw_values_avg_tokens = {}
# raw_values_last_token = {}
full_raw_values_avg_tokens = []
# full_raw_values_last_token = []
# full_raw_values = []
num_layers = infer_model.num_layers
intermediate_size = infer_model.intermediate_size
over_zeros = torch.zeros(num_layers, intermediate_size, dtype=torch.int32)
over_zeros_dict = {"lang":0,"num" : 0, "over_zero" : torch.tensor([])}


def save_raw_vals_to_dict(name, dictname, savees):
    # print(f"dictname.keys: {list(raw_values_avg_tokens.keys())}")
    if not name in dictname:
        dictname[name] = savees
    else:
        dictname[name] = torch.cat([dictname[name], savees], dim=0)
    
def clean_hooks(infer_model):
    for i in range(len(infer_model.model.model.layers)):
        mlp = infer_model.model.model.layers[i].mlp
        mlp.act_fn._forward_hooks.clear()

def get_activation_mlp(name, slice_ranges, sequence_lengths, max_tokens_overzeros = 10000, do_overzero=True, num_row_ozs=None, do_sent_avg=True, num_row_s_avg=None): 
    """
        name (str): buat namain layer
        is_averaged_tokens (boolean): true if avergaed across tokens, else get the last token
    """
    def hook_fn(module, input, output):
        global raw_values_avg_tokens
        # global raw_values_last_token
        global over_zeros
        global over_zeros_dict

        if do_overzero:
            slice_oz = slice_ranges[:num_row_ozs]
            overzero_perlayer = [
                (output[i, start:end, :] > 0).detach().sum(dim=0).to(dtype=torch.int32).cpu()
                for i, (start, end) in enumerate(slice_oz)
            ]
            over_zeros[int(name), :] += sum(overzero_perlayer)
            over_zeros_dict['over_zero'] = over_zeros
        if do_sent_avg:
            slice_savg = slice_ranges[:num_row_s_avg]
            means = [output[i, start:end, :].half().mean(dim=0).cpu() for i, (start, end) in enumerate(slice_savg)]
            avg_output = torch.stack(means, dim=0).cpu() 
            save_raw_vals_to_dict(name, raw_values_avg_tokens, avg_output)
            # print(f"kalo avg_output: {avg_output.shape}")
            # print(f"kalo raw_values_avg_tokens.shape: {raw_values_avg_tokens['0'].shape}")
        
    return hook_fn

def concat_neuron_layers(raw_values_avg_tokens):
    full_raw_values_avg_tokens = list(raw_values_avg_tokens.values())[0]
    
    
    for i in list(raw_values_avg_tokens.values())[1:]:
        # print(i.shape)
        full_raw_values_avg_tokens = torch.cat([full_raw_values_avg_tokens, i], dim=-1).cpu()
   
    return full_raw_values_avg_tokens


# def merge_avg_last(full_raw_values_avg_tokens, full_raw_values_last_token):
#     # print(f"shape full_raw_values_last_token {full_raw_values_last_token.shape}")
#     full_raw_values = torch.stack((full_raw_values_avg_tokens, full_raw_values_last_token), dim=0).cpu()
#     return full_raw_values


def register_hook(infer_model, handlers, slice_ranges, sequence_lengths, do_overzero=True, num_row_ozs=None, do_sent_avg=True, num_row_s_avg=None): 
    # remove_hooks(handlers)  
    clean_hooks(infer_model)
    num_layers = infer_model.num_layers
    remove_hooks(handlers)
    
    for i in range (num_layers):
        mlp = infer_model.model.model.layers[i].mlp
        handlers.append(mlp.act_fn.register_forward_hook(get_activation_mlp(f"{i}", slice_ranges, sequence_lengths, do_overzero=do_overzero, num_row_ozs=num_row_ozs, do_sent_avg=do_sent_avg, num_row_s_avg=num_row_s_avg)))
    # print(infer_model.model.model.layers[1].mlp.act_fn._forward_hooks)

    # for handler in handlers:
    #     handler.remove()
    # return handlers


def remove_hooks(handlers):
    for handler in handlers:
        handler.remove()
    handlers.clear()
    

def cleanup():
    raw_values_avg_tokens.clear()
    # raw_values_last_token.clear()
    full_raw_values_avg_tokens.clear()
    # full_raw_values_last_token.clear()
    # full_raw_values.clear()

def all_languages_dict_to_tensor(all_languages_dict):
    batch_size = 1000
    chunks = [torch.stack(all_languages_dict[i:i+batch_size], dim=0).cpu() 
              for i in range(0, len(all_languages_dict), batch_size)]
    full_languages_raw_values = torch.cat(chunks, dim=0)  # reassemble after stacking
    del chunks
    return full_languages_raw_values
def concat_languages(tensor, savees):
    perlanguage = savees.unsqueeze(0)
    if tensor.numel() == 0:
        return perlanguage
    else:
        return torch.cat([tensor, perlanguage], dim=0)

path_res = f"{parent_dir}res/act/{model_name.split('/')[-1]}"


os.makedirs(path_res, exist_ok=True)
def get_neurons(
    # models properties
    model_name,
    infer_model,
    # dataset properties
    dataset_name, split, max_instances=None, max_lang=None, selected_langs=None, is_predict=True, apply_template=True,
    debug=False,take_whole=False,
    # neuron retrieval properties
    max_tokens_overzeros=10000, max_sentence_avgs=500,
    # save to kaggle properties
    kaggle_dataname_to_save=None, is_update=False, batch_size=1, 

):
    infer_model.model.eval()
    print(f"num layers: {infer_model.num_layers}")
    """
    infer_model: HF model InferenceModel
    dataset_name: dataset name from HF
    split: split dataset "test", "devtest", etc
    max_instances: max_instances to be ran
    max_lang: max language to be ran. from language 0 to language max_lang. default None means all.
    selected_langs: specifiy selected_langs to be ran.
    """
    # infer_model = InferenceModel(model_name)
    global over_zeros_dict, over_zeros, raw_values_avg_tokens
    # raw_values_avg_tokens = {}
    # raw_values_last_token = {}
    full_raw_values_avg_tokens = []
    # full_raw_values_last_token = []
    # full_raw_values = []

    batch_size = batch_size if batch_size else 1
    eval_result = {}
    all_languages = []
    all_languages_over_zero = []
    handlers = []
    configs = get_dataset_config_names(dataset_name, trust_remote_code=True)
    n_instances = 0
    n_lang = 0
    language_dict = {}
    selected_langs = selected_langs if selected_langs != None else configs
    print(selected_langs)
    full_languages_raw_values = torch.tensor([])
    for lang in selected_langs:
        if lang.startswith("all"):
            continue
        if max_lang and n_lang >= max_lang:
            break
        cleanup()
        over_zeros_dict["lang"] = lang
        language_dict[n_lang] = lang 
        # ds = load_dataset(dataset_name, lang, split=split, trust_remote_code=True, cache_dir=f"{parent_dir}/data/hf_datasets")
        ds = load_dataset(dataset_name, lang, split=split, trust_remote_code=True)

        dataset_instance = dataset_instance = Dataset(dataset_name, ds)
        if dataset_name.endswith("massive"):
            dataset_instance = Massive(dataset_name, ds)
        elif dataset_name.endswith("xcopa"): 
            dataset_instance = Xcopa(dataset_name, ds)
        elif dataset_name.endswith("xwinograd"):
            dataset_instance = Xwinograd(dataset_name, ds)
        elif dataset_name.endswith("flores200"):
            dataset_instance = Flores(dataset_name, ds)
        elif dataset_name.endswith("MLAMA-dod-185"):
            dataset_instance = Mlama(dataset_name, ds)
        else:
            raise ValueError("Dataset is not available!")
        dataset_name = dataset_instance.dataset_name
        max_instances = min(max_instances, len(ds)) if max_instances else len(ds)
        do_sent_avg = True
        do_overzero = True
        end_idx = 0
        for start_idx in tqdm(range(0, max_instances, batch_size), desc=f"Processing {lang} Examples in batches", leave=False):
            end_idx = min(start_idx + batch_size, max_instances)
            batch_data = ds.select(range(start_idx, end_idx))

            texts = []
            slice_ranges = []
            seq_lengths = []
            print(f"processing data {start_idx} to {end_idx}")
            for data in batch_data:
                detail_data = dataset_instance.get_detail_inference(data)
                id_prompt_start, id_prompt_end, len_sentence, prompt_whole = dataset_instance.get_index_start_end_prompt(
                    infer_model, detail_data, infer_model.model_name, is_predict, take_whole
                )
                
                id_prompt_start = id_prompt_start if apply_template else 2
                slice_ranges.append((id_prompt_start, id_prompt_end))
                seq_lengths.append(len_sentence)
                text = infer_model.get_templated_prompt(prompt_whole, apply_template)
                texts.append(text)

            
            token_lens = [end - start for (start, end) in slice_ranges]
            accumulated = 0
            cut_index = len(texts)
            limit_sent_idx = max(0, max_sentence_avgs - start_idx)
            cut_sent_avg = min(len(texts), limit_sent_idx)
            do_sent_avg = cut_sent_avg > 0
            print(f"cut_sent_avg: {cut_sent_avg}, end_idx: {end_idx}, max_sentence_avgs: {max_sentence_avgs}")
            for i, tok_len in enumerate(token_lens):
                if over_zeros_dict['num'] + accumulated + tok_len > max_tokens_overzeros:
                    cut_index = i
                    break
                accumulated += tok_len
             

            if cut_index <= 0:
                do_overzero = False
            
            model_inputs, input_lengths, slice_ranges = infer_model.batch_tokenize(texts, slice_ranges)
            clean_hooks(infer_model)
            over_zeros_dict['num'] += accumulated if do_overzero else 0
            print(f"do_sent_avg: {do_sent_avg}")
            print(f"do_overzero: {do_overzero}")
            
            register_hook(infer_model, handlers, slice_ranges, seq_lengths, do_overzero=do_overzero, num_row_ozs=cut_index, do_sent_avg=do_sent_avg, num_row_s_avg=cut_sent_avg)
            generated_texts = infer_model.batch_inference(texts, model_inputs, slice_ranges, max_new_tokens=1, debug=debug)

            clean_hooks(infer_model)
            remove_hooks(handlers)

            n_instances += len(texts)
            if do_sent_avg == False and do_overzero == False:
                print("stopping because of stopping criteria.")
                break
            print(f"raw_values_avg_tokens.shape: {raw_values_avg_tokens['0'].shape}")
        print(f"stop at {end_idx} sentences.")
        full_raw_values_avg_tokens = concat_neuron_layers(raw_values_avg_tokens)
        
        # full_raw_values = merge_avg_last(full_raw_values_avg_tokens, full_raw_values_last_token)
        full_languages_raw_values = concat_languages(full_languages_raw_values, full_raw_values_avg_tokens)
        print(f"full_languages_raw_values.shape: {full_languages_raw_values.shape}")
        all_languages.append(full_raw_values_avg_tokens)
        all_languages_over_zero.append(over_zeros_dict)
        over_zeros = torch.zeros(infer_model.num_layers, infer_model.intermediate_size, dtype=torch.int32)
        over_zeros_dict = {"lang":0,"num" : 0, "over_zero" : torch.tensor([])}
        # print(full_raw_values.shape)
        n_lang += 1
        print(f"Done with lang: {lang}")
        for handler in handlers:
            handler.remove()
        
        n_instances = 0
        
        torch.save(all_languages, f"{path_res}/L_{lang}_{dataset_name.split('/')[-1]}_{max_instances}_{is_predict}.pt")
        # break
        
    cleanup()
    del raw_values_avg_tokens
    # del raw_values_last_token
    del full_raw_values_avg_tokens
    # del full_raw_values_last_token
    # del full_raw_values 

    
    print(f"FINAL full_languages_raw_values.shape: {full_languages_raw_values.shape}")
    torch.save(full_languages_raw_values, f"{path_res}/act_{dataset_name.split('/')[-1]}_{max_instances}_{is_predict}.pt")
    torch.save(all_languages_over_zero, f"{path_res}/oz_{dataset_name.split('/')[-1]}_{max_instances}_{is_predict}")
    torch.save(language_dict, f"{path_res}/ld_{dataset_name.split('/')[-1]}")

    if kaggle_dataname_to_save:
        save_to_kaggle(dataset_name= kaggle_dataname_to_save, data_dir=path_res, is_update=is_update)

    return full_languages_raw_values, all_languages_over_zero, language_dict

a, b, c = get_neurons(
        model_name=args.model,
        infer_model=infer_model,
        dataset_name=args.dataset_name,
        split=args.split,
        max_instances=args.max_instances,
        max_lang=args.max_lang,
        selected_langs=args.selected_langs,
        is_predict=args.is_predict,
        apply_template=args.apply_template,
        debug=args.debug,
        take_whole=args.take_whole,
        max_tokens_overzeros=args.max_tokens_overzeros,
        max_sentence_avgs = args.max_sentence_avgs,
        kaggle_dataname_to_save=args.kaggle_dataname_to_save, 
        is_update=args.is_update,
        batch_size=args.batch_size
    )