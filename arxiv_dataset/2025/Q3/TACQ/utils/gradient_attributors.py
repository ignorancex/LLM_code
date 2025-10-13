import time
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from utils.measurement_utils import filter_importances_dict
from utils.model_utils import load_model
try:
    from transformers.utils import logging
    logging.set_verbosity_error()
except:
    print("Logging change failed.")

"""A simplified version of the gradient capturing used for TACQ, which no longer needs to use decomposition but simply moves gradients to CPU once computed."""

#DEBUG
# from torch.utils.viz._cycles import warn_tensor_cycles
# warn_tensor_cycles()
from torch.cuda.memory import _record_memory_history, _dump_snapshot

def sample_abs(key, accumulated_gradient, param):
    return accumulated_gradient[key] + torch.abs(param.grad).to("cpu")

@torch.no_grad
def weight_prod_contrastive_postprocess(attributed_matrices, model, corrupt_model):
    """Contrasting with the difference in model weights and simulated quantized model weights. An aboslute value is taken when used for quantization mask, meaning an absolute value need not be taken here."""
    model = {key: param for key, param in model.named_parameters()}
    corrupt_model = {key: param for key, param in corrupt_model.named_parameters()}
    for module_name in attributed_matrices:
        try:
            print(f"Producing final output for {module_name}")
            print(
                "\nGradients:"
                f"{attributed_matrices[module_name].abs().mean()=}",
                f"{attributed_matrices[module_name].abs().median()=}",
                )
        except:
            print("Print failed")
        attributed_matrices[module_name] = attributed_matrices[module_name] * (model[module_name] - corrupt_model[module_name]) * model[module_name]
        try:
            print(
                "\nFinal:",
                f"{attributed_matrices[module_name].abs().mean()=}", 
                f"{attributed_matrices[module_name].abs().median()}",  # Added median
                "\nContrastive:",
                f"{(model[module_name] - corrupt_model[module_name]).abs().mean()}", 
                f"{(model[module_name] - corrupt_model[module_name]).abs().median()}",  # Added median
                "\nClean Weights:",
                f"{model[module_name].abs().mean()}", 
                f"{model[module_name].abs().median()}",  # Added median
                "\nCorrupt Weights:",
                f"{corrupt_model[module_name].abs().mean()}", 
                f"{corrupt_model[module_name].abs().median()}"  # Added median
            )    
        except:
            print("Warning: Print statement failed")
    return attributed_matrices



def grad_attributor(args, model_name, corrupt_model_name, dataset, masking_function=None, 
                    loss_func=CrossEntropyLoss(), checkpoints_dir=None, attributor_function=sample_abs, 
                    postprocess_function=lambda x, y, z: x, record_memory_history=False, backward_in_full_32_precision=True):
    ## Define Gradient Capturing Aparatus
    accumulated_gradient = {}
    def make_clear_grad_hook(key, accumulated_gradient):
        def hook(param):
            if key in accumulated_gradient:
                accumulated_gradient[key] = attributor_function(key, accumulated_gradient, param)
            param.grad = None
        return hook
    ## Load model
    model = load_model(engine=model_name, checkpoints_dir=checkpoints_dir, full_32_precision=backward_in_full_32_precision, brainfloat=False)["model"]
    ## Setup gradients to accumulate
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            accumulated_gradient.update({".".join((name, key)): torch.zeros_like(val).detach().to("cpu") for key, val in module.named_parameters() if key == "weight"})
    accumulated_gradient = filter_importances_dict(accumulated_gradient)
    ## Setup hooks
    hook_handles = []
    for name, param in model.named_parameters():
        if isinstance(param, torch.Tensor):
            grad_hook = make_clear_grad_hook(name, accumulated_gradient)
            hook_handle = param.register_post_accumulate_grad_hook(grad_hook)
            hook_handles.append(hook_handle)
    ## Cache all gradients for the clean model
    start_time = time.time()
    cumulation_counter = 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for example in dataloader:
        example["input_ids"] = example["input_ids"].to(args.device)
        try:
            (cumulation_counter % 5 == 0) and print(f"Processing sample {cumulation_counter}")
        except:
            pass
        outputs = model(**example)
        shift_logits = outputs.logits[..., :-1, :].contiguous()  # Get rid of the prediction from the last token, since we don't have a label for it
        shift_labels = example["input_ids"][..., 1:].contiguous()  # Get rid of the label from the first token, since no predictions are made for it
        if masking_function != None:  # Optionally mask calculation of loss, for example, to ignore loss on the prompt.
            shift_logits, shift_labels = masking_function(args, shift_logits, shift_labels)
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        False and args.logger.info(f"shift logits and labels, {shift_logits.shape}, {shift_labels.shape}")
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_func(shift_logits, shift_labels) 
        loss.backward()
        if record_memory_history:
            for i, x in model.named_parameters():
                print("Grad should be None if save_memory=True:", f"{x.grad=}, x should not require grad {x.requires_grad=} {x.is_leaf=} {x.device=}")
                break
        del shift_logits, shift_labels, loss
        del outputs, example
        cumulation_counter += 1
    print(f"samples processed: {cumulation_counter}")
    # Remove all handles, dataloader, model, and clear gpu
    for hook in hook_handles:
        hook.remove()
    del dataloader, hook_handles, model
    torch.cuda.empty_cache()

    # Postprocess
    model = load_model(engine=model_name, checkpoints_dir=checkpoints_dir, full_32_precision=False, brainfloat=False, device_map="cpu")["model"].to("cpu")
    corrupt_model = load_model(engine=corrupt_model_name, checkpoints_dir=checkpoints_dir, full_32_precision=False, brainfloat=False, device_map="cpu")["model"].to("cpu")
    total_time = time.time() - start_time
    print(f"Time Used To Capture Importances: {total_time}")
    print(f"Scores {accumulated_gradient=}")
    tcq_scores = postprocess_function(accumulated_gradient, model, corrupt_model)
    return tcq_scores

