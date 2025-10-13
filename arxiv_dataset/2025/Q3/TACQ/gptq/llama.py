import os
import random
import time

import torch
import torch.nn as nn
import sys
import numpy as np

# Get the directory of the current file
file_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_directory)

from gptq import *
from gptq.gptq import *
from modelutils import *
from quant import *
from transformers import AutoModelForCausalLM

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    from peft import AutoPeftModelForCausalLM
    if model.endswith("lora_model"):
        model = AutoPeftModelForCausalLM.from_pretrained(model, torch_dtype='auto')
        model = model.merge_and_unload()
    elif "Qwen" in model:
        model = AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto')
    else:
        model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    
    if getattr(model.model, 'rotary_emb', None):
        # for llama3 and qwen models when transformers >=4.45.0
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps_list = []            # list of hidden states from the 1st layer
    attn_masks_list = []      # list of attention masks
    position_ids_list = []    # list of position IDs
    # Only caches the tokenized inputs from the first layer?
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps_list.append(inp.detach())
            attn_masks_list.append(kwargs.get('attention_mask', None))
            position_ids_list.append(kwargs.get('position_ids', None))
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError as e:
            pass
    layers[0] = layers[0].module
    
    if getattr(model.model, 'rotary_emb', None):
        # for llama3 and qwen models when transformers >=4.45.0
        model.model.rotary_emb = model.model.rotary_emb.cpu()

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs_list = [None] * len(inps_list)

    # Load importances to calculate important_mask: parts that will be kept in higher precision
    if args.important_mask:
        with open(args.important_mask, "rb") as f:
            important_mask = torch.load(f)
        print(f"Important mask loaded {args.important_mask}")
    else:
        important_mask = {}
        for key, param in model.named_parameters():
            important_mask[key] = torch.zeros_like(param)

    print('Ready.')
    overall_greedy_importances = {}
    overall_final_mask = {}
    overall_important_mask = {} # for debugging only
    quantizers = {}
    for i in range(len(layers)):
        print(f"Considering layer {i}")
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}

            if args.nearest:  # A baseline
                for name in subset:
                    print(i, name)
                    print('RTN Quantizing ...')
                    quantizer = Quantizer()
                    quantizer.configure(
                        args.wbits, perchannel=True, sym=False, mse=False
                    )
                    W = subset[name].weight.data
                    quantizer.find_params(W, weight=True)
                    subset[name].weight.data = quantize(
                        W, quantizer.scale, quantizer.zero, quantizer.maxq
                    ).to(next(iter(layer.parameters())).dtype)
            else:
                for name in subset:
                    # Different bitwidths for different modules are defined based on configs / other means
                    config_key = "model.layers." + str(i) + "." + name + ".weight"
                    if args.fine_wbits_yaml is not None:
                        import yaml
                        layer_weight_bits = yaml.safe_load(open(args.fine_wbits_yaml, "r"))[config_key]
                    elif args.wbits_yaml is not None:
                        import yaml
                        layer_weight_bits = yaml.safe_load(open(args.wbits_yaml, "r"))[name] 
                    else:
                        layer_weight_bits = args.wbits

                    if args.important_mask:
                        current_matrix_mask = important_mask[config_key].to(torch.bool).to("cuda").detach()
                    else:
                        current_matrix_mask = None
                    
                    # Configures the run: Both GPTQ and Quantizer are initialized. Quantizers is configured to set bitwidth when "find_params" is called. find_params is called in gptq.py#fasterquant.
                    gptq[name] = GPTQ(subset[name], current_matrix_mask)  # This initiates gptq[name].layer to a reference to an nn.Module, which is either a linear layer or a conv layer.
                    gptq[name].quantizer = Quantizer()
                    gptq[name].quantizer.configure(
                        layer_weight_bits, perchannel=True, sym=args.sym, mse=True
                    )
                activations_cache = {}
                def add_batch(name):
                    def tmp(_, inp, out):
                        gptq[name].add_batch(inp[0].data, out.data)
                    return tmp
                
                # Collect activations for each GPTQ instance [every matrix]
                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))  # Hidden states are hooked to be collected.
                for j in range(len(dataloader)):
                    # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                    # We use the same inps because we are re-running the entire layer, but just collecting statistics from one matrix in the layer.
                    outs_list[j] = layer(inps_list[j], attention_mask=attn_masks_list[j], position_ids=position_ids_list[j])[0].detach()
                for h in handles:
                    h.remove()
                    

                for name in subset:
                    print(f"\nCurrently Quantizing layer {i} component {name}")
                    print(i, name, f"bits = {gptq[name].quantizer.bits}")
                    print('Quantizing ...')
                    gptq[name].fasterquant(
                        percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups, save_quantization = bool(args.save)
                    )
                    quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer

                    # Save in SPQR format:
                    if args.save:
                        if not gptq[name].save_quant_dict:
                            raise Exception("Error: save_quant_dict not populated, ensure gptq[name].fasterquant is ran with save_quantization = True")
                        gptq[name].save_quant_dict["sublayer_name"] = name
                        full_path = args.save + "/" + str(i) + "/"
                        os.makedirs(full_path, exist_ok=True)
                        torch.save(gptq[name].save_quant_dict, full_path + name)

                    gptq[name].free()

        for j in range(len(dataloader)):
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            outs_list[j] = layer(inps_list[j], attention_mask=attn_masks_list[j], position_ids=position_ids_list[j])[0].detach()

        # Remove this transformer block from GPU and move on to the next one
        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        # inps, outs = outs, inps
        inps_list, outs_list = outs_list, inps_list

    model.config.use_cache = use_cache

    print("Quantization complete.")
    return quantizers

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)
    
    # for transformers >= 4.45.0
    if getattr(model.model, 'rotary_emb', None):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    
    # for transformers >= 4.45.0
    if getattr(model.model, 'rotary_emb', None):
        model.model.rotary_emb = model.model.rotary_emb.cpu()

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        
        # if args.nearest:
        #     subset = find_layers(layer)
        #     for name in subset:
        #         quantizer = Quantizer()
        #         quantizer.configure(
        #             args.wbits, perchannel=True, sym=False, mse=False
        #         )
        #         W = subset[name].weight.data
        #         quantizer.find_params(W, weight=True)
        #         subset[name].weight.data = quantize(
        #             W, quantizer.scale, quantizer.zero, quantizer.maxq
        #         ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

def llama_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str,
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=3, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--save_in_16bits', type=str, default='',
        help='Save quantized checkpoint under this name in 16bits'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--no-eval', action='store_true',
        help='Whether to skip evaluation.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    parser.add_argument(
        '--wbits-yaml', type=str, default=None,
        help='YAML file with the number of bits for each layer.'
    )
    parser.add_argument(
        '--fine-wbits-yaml', type=str, default=None,
        help='YAML file with the number of bits for each 2D weight matrix.'
    )
    parser.add_argument(
        '--important_mask', type=str, default=None,
        help='Path to a important_mask which specifies the parameters to keep in 16 bits. Type dictionary of boolean torch tensors.'
    )
    parser.add_argument(
        '--disable_placeholder', action="store_true", help="This does not effect performance and should be enabled at all times."
    )
    parser.add_argument(
        '--save_in_dtype_float16', action="store_true", help="For debugging purposes"
    )

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.device = "cuda"
    print("args.model for get_llama:", args.model)
    model = get_llama(args.model)

    # Update args.model to the base model name and loadstring
    args.model = args.model.split("/")[-1].split("_")[0]
    from utils.model_utils import model_loadstring_dict
    args.model = model_loadstring_dict[args.model] + "/" + args.model
    print("args.model for post processing:", args.model)

    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.wbits < 16:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV)
        print(time.time() - tick)

    datasets = ['wikitext2', 'ptb', 'c4'] 
    if args.new_eval:
        datasets = ['wikitext2', 'ptb-new', 'c4-new']
    if args.no_eval:
        datasets = []
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        llama_eval(model, testloader, DEV)

    if args.save:
        llama_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save)

    if args.save:
        # SPQR SAVING STRATEGY
        torch.save(vars(args), args.save + "/args.pt")
        # try:
        #     already_saved_weights = set()
        #     for name, layer in nn.ModuleList(model.model.layers).named_modules():
        #         if isinstance(layer, (nn.Conv2d, nn.Linear)):
        #             already_saved_weights.add(layer.weight)
        #     not_quantized_weights = {
        #         name: param for name, param in model.named_parameters() if param not in already_saved_weights
        #     }
        #     torch.save(not_quantized_weights, args.save + "/not_quantized_weights.pt")
        # except:
        #     print("Saving unquantized model failed")

    if args.save_in_16bits:
        def print_model_layer_dtype(model):
            print('\nModel dtypes:')
            for name, param in model.named_parameters():
                print(f"Parameter: {name}, Data type: {param.dtype}")
        # print_model_layer_dtype(model)
        if args.save_in_dtype_float16:
            model.half()
        else:
            model.to(torch.bfloat16)
        torch.save(model.state_dict(), args.save_in_16bits)
        print("Saved in 16bits at:", args.save_in_16bits)
