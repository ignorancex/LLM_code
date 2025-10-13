def load_model_and_preprocess(args):
    import torch
    
    API_KEY_gpt = ""

    ORG = ""
    
    if args.model_family == "llava":

        from transformers import AutoProcessor, LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name, 
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=f"{args.work_dir}/.cache/huggingface/hub",
        )
        processor = AutoProcessor.from_pretrained(args.model_name)

    elif args.model_family == "instructblip":

        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        model = InstructBlipForConditionalGeneration.from_pretrained(
            args.model_name, 
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=f"{args.work_dir}/.cache/huggingface/hub",
        )
        processor = InstructBlipProcessor.from_pretrained(args.model_name)

    elif args.model_family == "llava-onevision":

        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            args.model_name, 
            torch_dtype=torch.float16, 
            device_map="auto",
            cache_dir=f"{args.work_dir}/.cache/huggingface/hub",
        )

        processor = AutoProcessor.from_pretrained(args.model_name)
        processor.tokenizer.padding_side = 'left'

    elif args.model_family == "qwen":

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor 

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            cache_dir=f"{args.work_dir}/.cache/huggingface/hub",
        )
        min_pixels = 32*32
        max_pixels = 512*512
        processor = AutoProcessor.from_pretrained(args.model_name, min_pixels=min_pixels, max_pixels=max_pixels)
        processor.tokenizer.padding_side = 'left'

        if hasattr(args, "alpha") and args.alpha != 1.0:
            n_heads = model.config.num_attention_heads
            d_head  = int(model.config.hidden_size // n_heads)

            args.model_name += "||"

            with torch.no_grad():
                for layer_idx, head_idx in zip(args.layer_idx, args.head_idx):
                    model.model.layers[layer_idx].self_attn.o_proj.weight[ : , head_idx*d_head : head_idx*d_head+d_head ] *= args.alpha
                    args.model_name += f"L{layer_idx}H{head_idx}A{args.alpha:.1f},"
    elif args.model_family == "gpt":
        from openai import OpenAI
        # OpenAI API Key TODO TODO invalidate this api key
        api_key_gpt = API_KEY_gpt

        client = OpenAI(
            organization=ORG,
            api_key=api_key_gpt
        )

        model = client
        processor = None

    if args.caption_type == "text_only" and (args.model_family in ["llava", "custom_llava", "instructblip"]):
        model = model.language_model
        
    return model, processor