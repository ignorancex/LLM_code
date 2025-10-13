import os
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_loadstring_dict = {"Qwen2.5-32B-Instruct": "Qwen", "Qwen2.5-7B": "Qwen", "Qwen2.5-7B-Instruct": "Qwen", "gemma-2b": "google", "gpt2-large": "openai-community", "Llama-2-7b-hf": "meta-llama", "Meta-Llama-3-8B-Instruct": "meta-llama", "Meta-Llama-3-8B": "meta-llama", "Mistral-7B-v0.3": "mistralai", "Meta-Llama-3-70B-Instruct": "meta-llama"}

def load_model(engine, checkpoints_dir, device_map = "auto", full_32_precision=False, brainfloat=False):
    """Can handle many types of models."""

    if engine.endswith("quantized_model"):  # FULLLY SAVED MODEL
        if "+" in engine:
            base_model_name = engine.split("+")[0].split("_")[0]
        else:
            base_model_name = engine.split("_")[0]
        loadstring = model_loadstring_dict[base_model_name] + "/" +  base_model_name
        tokenizer = AutoTokenizer.from_pretrained(loadstring)
        model = AutoModelForCausalLM.from_pretrained(loadstring, device_map="auto")
        print("Base model loaded, now replacing with saved state dict.")
        devices_mapper = {}
        for name, module in model.named_parameters():
          devices_mapper[name] = module.dtype
        loaded_state_dict = torch.load(os.path.join(checkpoints_dir, engine+".pt"))
        model.load_state_dict(loaded_state_dict)
        for key, param in model.named_parameters():
            param.data = param.data.to(devices_mapper[key])
        # model.to(device)
    elif engine.endswith("qlora_model"): 
        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            )
        base_model_name = engine.split("_")[0]
        loadstring = model_loadstring_dict[base_model_name] + "/" +  base_model_name
        tokenizer = AutoTokenizer.from_pretrained(loadstring)
        model = AutoPeftModelForCausalLM.from_pretrained(os.path.join(checkpoints_dir, engine), quantization_config=bnb_config, device_map=device_map)
    elif engine.endswith("lora_model"):  
        base_model_name = engine.split("_")[0]
        loadstring = model_loadstring_dict[base_model_name] + "/" +  base_model_name
        tokenizer = AutoTokenizer.from_pretrained(loadstring)
        model = AutoPeftModelForCausalLM.from_pretrained(os.path.join(checkpoints_dir, engine), device_map=device_map)
        print(f"Model loaded: {type(model)}")
    else:
        print(engine)
        loadstring = model_loadstring_dict[engine] + "/" +  engine
        model = AutoModelForCausalLM.from_pretrained(loadstring, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(loadstring)

    print("Model loaded of type:", type(model))
    if not full_32_precision:
        if brainfloat:
            model = model.to(torch.bfloat16)
            print("Model activations converted to bf16 bit precision")
        else:
            model = model.half()
            print("Model activations converted to fp16 bit precision")
    else:
        model = model.to(torch.float32)
        print("Model activations converted to fp32 bit precision")
    unique_dtypes = set()
    for name, param in model.named_parameters():
        unique_dtypes.add(param.dtype)
    print("Activation Dtypes for model:", unique_dtypes)

    return {"model": model, "tokenizer": tokenizer}
