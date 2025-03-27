import argparse
import os
import json
from typing import Optional
from dataclasses import dataclass, field

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    OPTForCausalLM,
    BloomTokenizerFast,
    AutoConfig
)

from transformers import BitsAndBytesConfig, LlamaForCausalLM

from optimum.bettertransformer import BetterTransformer

from safetensors.numpy import save_file

try:
    from data_collator import DataCollatorForNI
except:
    from evaluation.data_collator import DataCollatorForNI

try: 
    from args_for_corruptions import args_for_corruptions
except:
    from evaluation.args_for_corruptions import args_for_corruptions

torch.set_grad_enabled(False)


OPT_30B_DEVICE_MAP = {
    'model.decoder.embed_tokens': 0,
    'model.decoder.embed_positions': 0,
    'model.decoder.final_layer_norm': 0,
    'model.decoder.layers.0': 0,
    'model.decoder.layers.1': 0,
    'model.decoder.layers.2': 0,
    'model.decoder.layers.3': 0,
    'model.decoder.layers.4': 0,
    'model.decoder.layers.5': 0,
    'model.decoder.layers.6': 0,
    'model.decoder.layers.7': 0,
    'model.decoder.layers.8': 0,
    'model.decoder.layers.9': 0,
    'model.decoder.layers.10': 0,
    'model.decoder.layers.11': 0,
    'model.decoder.layers.12': 0,
    'model.decoder.layers.13': 0,
    'model.decoder.layers.14': 0,
    'model.decoder.layers.15': 0,
    'model.decoder.layers.16': 0,
    'model.decoder.layers.17': 0,
    'model.decoder.layers.18': 0,
    'model.decoder.layers.19': 0,
    'model.decoder.layers.20': 0,
    'model.decoder.layers.21': 0,
    'model.decoder.layers.22': 0,
    'model.decoder.layers.23': 0,
    'model.decoder.layers.24': 1,
    'model.decoder.layers.25': 1,
    'model.decoder.layers.26': 1,
    'model.decoder.layers.27': 1,
    'model.decoder.layers.28': 1,
    'model.decoder.layers.29': 1,
    'model.decoder.layers.30': 1,
    'model.decoder.layers.31': 1,
    'model.decoder.layers.32': 1,
    'model.decoder.layers.33': 1,
    'model.decoder.layers.34': 1,
    'model.decoder.layers.35': 1,
    'model.decoder.layers.36': 1,
    'model.decoder.layers.37': 1,
    'model.decoder.layers.38': 1,
    'model.decoder.layers.39': 1,
    'model.decoder.layers.40': 1,
    'model.decoder.layers.41': 1,
    'model.decoder.layers.42': 1,
    'model.decoder.layers.43': 1,
    'model.decoder.layers.44': 1,
    'model.decoder.layers.45': 1,
    'model.decoder.layers.46': 1,
    'model.decoder.layers.47': 1,
    'lm_head': 1,
}

OPT_66B_INT8_DEVICE_MAP = device_map = {
 'model.decoder.embed_tokens': 0,
 'model.decoder.embed_positions': 0,
 'model.decoder.final_layer_norm': 0,
 'model.decoder.layers.0': 0,
 'model.decoder.layers.1': 0,
 'model.decoder.layers.2': 0,
 'model.decoder.layers.3': 0,
 'model.decoder.layers.4': 0,
 'model.decoder.layers.5': 0,
 'model.decoder.layers.6': 0,
 'model.decoder.layers.7': 0,
 'model.decoder.layers.8': 0,
 'model.decoder.layers.9': 0,
 'model.decoder.layers.10': 0,
 'model.decoder.layers.11': 0,
 'model.decoder.layers.12': 0,
 'model.decoder.layers.13': 0,
 'model.decoder.layers.14': 0,
 'model.decoder.layers.15': 0,
 'model.decoder.layers.16': 0,
 'model.decoder.layers.17': 0,
 'model.decoder.layers.18': 0,
 'model.decoder.layers.19': 0,
 'model.decoder.layers.20': 0,
 'model.decoder.layers.21': 0,
 'model.decoder.layers.22': 0,
 'model.decoder.layers.23': 0,
 'model.decoder.layers.24': 0,
 'model.decoder.layers.25': 0,
 'model.decoder.layers.26': 0,
 'model.decoder.layers.27': 0,
 'model.decoder.layers.28': 0,
 'model.decoder.layers.29': 0,
 'model.decoder.layers.30': 0,
 'model.decoder.layers.31': 0,
 'model.decoder.layers.32': 1,
 'model.decoder.layers.33': 1,
 'model.decoder.layers.34': 1,
 'model.decoder.layers.35': 1,
 'model.decoder.layers.36': 1,
 'model.decoder.layers.37': 1,
 'model.decoder.layers.38': 1,
 'model.decoder.layers.39': 1,
 'model.decoder.layers.40': 1,
 'model.decoder.layers.41': 1,
 'model.decoder.layers.42': 1,
 'model.decoder.layers.43': 1,
 'model.decoder.layers.44': 1,
 'model.decoder.layers.45': 1,
 'model.decoder.layers.46': 1,
 'model.decoder.layers.47': 1,
 'model.decoder.layers.48': 1,
 'model.decoder.layers.49': 1,
 'model.decoder.layers.50': 1,
 'model.decoder.layers.51': 1,
 'model.decoder.layers.52': 1,
 'model.decoder.layers.53': 1,
 'model.decoder.layers.54': 1,
 'model.decoder.layers.55': 1,
 'model.decoder.layers.56': 1,
 'model.decoder.layers.57': 1,
 'model.decoder.layers.58': 1,
 'model.decoder.layers.59': 1,
 'model.decoder.layers.60': 1,
 'model.decoder.layers.61': 1,
 'model.decoder.layers.62': 1,
 'model.decoder.layers.63': 1,
 'lm_head': 1
}

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    task_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions tasks json files."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=2000,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=10,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_num_instances_per_task: int = field(
        default=None, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=100, metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_task_definition: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task definition before the task input."}
    )
    num_pos_examples: Optional[int] = field(
        default=4,
        metadata={"help": "number of in-context positive examples."}
    )
    num_neg_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context negative examples."}
    )
    add_explanation: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to add explanation for both the postive examples and negtive examples."}
    )
    def __post_init__(self):
        pass

@dataclass
class OPTArguments(DataTrainingArguments):
    taskdirectory: str = field(
        default="data_generator/processed_tasks", metadata={"help": "task files"}
    )
    output_dir: str = field(
        default="../new_predictions/", metadata={"help": "output directory where prediction files will be saved"}
    )
    modelname: str = field(
        default="gpt2", metadata={"help": "model name"}
     )
    corruption_base_file: str = field(
        default="baseline", metadata={"help": "corruption is applied on corruption base file"}
    )
    batch_size: int = field(
        default=16, metadata={"help": "batch size for evaluation"}
    )
    corruptions_name_list: str = field(
        default= "all", metadata={"help": "list of corruption name e.g. [empty, intr_n_demo]"}
    )
    placement_based_corruption: str = field(
        default=None, metadata={"help": "placement based corruptions, it can be input_empty, label_empty or instr_after_demo"}
    )
    task_set_no: str = field(
        default="set100", metadata={"help": "set the task set number, e.g. set1 or set2 or set3, this will save the result in e.g. set100 folder"}
    )

    
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = HfArgumentParser((OPTArguments,))
    args, = parser.parse_args_into_dataclasses()
   
    # load the tokenizer and model, loading once for all corruption in the corruptions_name_list and for all tasks in test_tasks.txt
    modelname = args.modelname
    tokenizer = AutoTokenizer.from_pretrained(
            modelname, 
            return_tensors="pt"
        )
    tokenizer.padding_side = "left"  # for batch inferencecle
  
    if device == "cpu": # to test the code on local system
        model = AutoModelForCausalLM.from_pretrained(
            modelname
        )
    elif modelname == "facebook/opt-30b":   
        model = AutoModelForCausalLM.from_pretrained(
            modelname, 
            device_map="auto", 
            load_in_8bit=True
        )
    elif modelname == "facebook/opt-66b":
        model = AutoModelForCausalLM.from_pretrained(
            modelname,
            device_map="auto",
            load_in_4bit=True,
        )
        # model = BetterTransformer.transform(model, keep_original_model=False)
    elif "llama" in modelname:
        config = AutoConfig.from_pretrained(modelname)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=False,
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
            llm_int8_enable_fp32_cpu_offload=False,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=None,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=False,
        )
        model = LlamaForCausalLM.from_pretrained(
            modelname,
            config=config,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            modelname, 
            device_map="auto", 
            load_in_4bit=True
        )
    
    if 'gpt' in modelname or 'llama' in modelname or 'falcon' in modelname or 'pythia' in modelname:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # model: OPTForCausalLM  # for type hints
    model.eval()

    corruptions_name_list = args.corruptions_name_list  # e.g. [empty, intr_n_demo]
    
    if corruptions_name_list == "all":
        corruptions_name_list = [
        'input_randomwords' ,
        'all_instr_n_demo',
        'instr_n_demo',
        'inline_n_demo',
        'empty',
        'only_instruction',
        'only_inline',
        'both_instructions',
        'only_demostration',
        'instr_randomwords',
        'wrong_labels_labelspace',
        'labels_randomwords',
        'input_empty', 
        'label_empty', 
        'instr_after_demo',
        'input_ood',
        'inline_instr_in_0_demo',
        'inline_instr_in_1_demo',
        'inline_instr_in_2_demo',
        'inline_instr_in_3_demo',
        'randomwords_inline_instr_in_0_demo',
        'randomwords_inline_instr_in_1_demo',
        'randomwords_inline_instr_in_2_demo',
        'randomwords_inline_instr_in_3_demo',
        ]
    else:
        corruptions_name_list = corruptions_name_list[1:-1].split(",") # e.g. ["empty", "intr_n_demo"]

    for each_corruption in corruptions_name_list:
        
        # get args for the specific corruption
        corruption_basefile, add_task_definition, num_pos_examples, _, placement_based_corruption = args_for_corruptions(each_corruption)

        # load the dataset (loads all files one after the other sequencially)
        raw_datasets = load_dataset(
            "evaluation/loaddataset.py", 
            data_dir="evaluation/",  # dir for test_tasks.txt
            task_dir=args.taskdirectory, 
            max_num_instances_per_eval_task=args.max_num_instances_per_eval_task,
            corruption_base_file=corruption_basefile
        )
        
        # data collator
        data_collator = DataCollatorForNI(
            tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            add_task_definition=add_task_definition,
            num_pos_examples=num_pos_examples,
            num_neg_examples=False,
            add_explanation=False, 
            placement_based_corruption=placement_based_corruption    
        )

        eval_dataloader = torch.utils.data.DataLoader(
            raw_datasets['test'],
            batch_size=args.batch_size,
            collate_fn=data_collator,
            shuffle=False,
        )

        # uncomment this and comment model load and evaluation loop to print prompts
        # checking the prompt, comment this section once prompts look okay
        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                print("######################## ", each_corruption,  " #############################")
                print(batch["inputs"][0] + "\n\n")
                break
            

        # create output directory
        n_params = sum(p.numel() for p in model.parameters()) / 1e9
        prefix = f"{n_params:.2f}B_"
        model_name_for_save = prefix + modelname.split('/')[-1]

        output_dir = args.output_dir + '/' + model_name_for_save + '/' + args.task_set_no + '/' 
        os.makedirs(output_dir, exist_ok=True)

        output_dir_config = args.output_dir + '/' + model_name_for_save + '/'  + args.task_set_no + '/' + 'config' + '/'
        os.makedirs(output_dir_config, exist_ok=True)

        output_dir_correct_incorrect_pred = args.output_dir + '/' + model_name_for_save +  '/' + args.task_set_no + '/' + 'cor_incor_pred' + '/'
        os.makedirs(output_dir_correct_incorrect_pred, exist_ok=True)

        from collections import defaultdict
        saved_correct_prediction_heads = defaultdict(lambda: False)
        saved_incorrect_prediction_heads = defaultdict(lambda: False)

        with torch.no_grad():
            with open(os.path.join(output_dir, "predicted_examples_{}.jsonl".format(each_corruption)), "w") as fout:
                # Save the configuration of evaluation
                with open(os.path.join(output_dir_config, "config_{}.json".format(each_corruption)), "w") as fout_config:
                    json.dump(args.__dict__, fout_config, indent=4)
        
                # evaluation loop for all tasks
                for i, batch in enumerate(tqdm(eval_dataloader)):
                    
                    for j in range(len(batch['label'])):
                        # strip the whitespace in input and target
                        batch['inputs'][j] = batch['inputs'][j].strip()
                        batch['label'][j] = batch['label'][j].strip()
                        break
                
                    tok_input = tokenizer(batch['inputs'], return_tensors="pt", padding=True)
                
                    # outputs: generated ids
                    outputs = model.generate(
                        input_ids=tok_input['input_ids'].to(device), # https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517/6
                        attention_mask=tok_input['attention_mask'].to(device),
                        max_new_tokens=args.max_target_length,
                    )

                    # save predictions
                    for k in range(len(batch['label'])):
                        complete_dict = {}
                       
                        n_input_tokens = len(tok_input['input_ids'][k])
                        response_ids = outputs[k][n_input_tokens:] # remove input tokens from generated tokens

                        response = tokenizer.decode(
                            response_ids,
                            skip_special_tokens=True, 
                        ) 
                                                    
                        full_generation = tokenizer.decode(
                            outputs[k],
                            skip_special_tokens=True,
                        )

                        # Note: Following original paper, we cut the generated text at the first period, since the language model sometimes generates more than one sentences.
                        response = response.strip().split(".")[0]
                        
                        complete_dict = {
                            'id': batch['id'][k], 
                            'Task': batch['Task'][k],
                            'Corruption_name' : each_corruption,
                            'Corruption_base_file' : corruption_basefile,
                            'Categories': batch['Categories'][k],
                            'Reasoning': batch['Reasoning'][k],
                            'Instance': {
                                "input" : batch['inputs'][k], 
                                "output" : [batch['label'][k]]
                            },
                            'Target': batch['label'][k],
                            'Prediction': response,
                            'Full generation': full_generation,  # useful if we want to feed this into a visualization
                        }
                        fout.write(json.dumps(complete_dict) + "\n")