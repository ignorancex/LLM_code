# Import necessary modules
import torch
import torch.nn as nn
import sys
# Import get_loaders function from data module within the same directory
from .data import get_loaders
import fnmatch
from pdb import set_trace as st

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(model, tokenizer, device=torch.device("cuda:0"), dataset="wikitext2"):

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl = eval_ppl_dataset(model, testloader, 1, device)
    return ppl 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_dataset(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # st()
        
        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to("cuda:1")
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)


        # print ("nlls",nlls)
        sys.stdout.flush()

    
    print ('begin calcualte ppl')
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    # torch.cuda.empty_cache()

    return ppl.item()




def eval_zero_shot(model_name, task_list=["qqp","rte","mnli","mrpc","cola", "qnli", "stsb"], 
        num_fewshot=0, use_accelerate=True, add_special_tokens=False):
    from lm_eval import tasks, evaluator 
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    model_args = f"pretrained={model_name},cache_dir=./llm_weights"

    if use_accelerate:
        model_args = f"pretrained={model_name},use_accelerate=True,device_map_option=\"auto\""
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        max_batch_size=None,
        device=None,
        no_cache=True,
        # limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=False,
        output_base_path=None
    )
    print("********************************")
    print("zero_shot evaluation results")
    print(evaluator.make_table(results))
    # st()
    return results 