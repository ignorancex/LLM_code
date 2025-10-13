import torch
import copy
from PIL import Image

def _prepare_messages(
    prompts,
    processing_class,
    max_prompt_length
):
    prompts_lists = []
    input_images_lists = []

    for msgs in prompts:
        copy_msgs = copy.deepcopy(msgs)
        
        images = []
        for i, msg in enumerate(copy_msgs):
            role, content = msg["role"], msg["content"]
            
            if isinstance(content,str):
                content = [content]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_msgs.append(c)
            msg['content'] = "\n".join(cur_msgs)
        
        prompts_lists.append(
            processing_class.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True)
        )
        input_images_lists.append(images)
    
    ret = processing_class(
        prompts_lists,
        input_images_lists,
        return_tensors="pt",
        max_length=max_prompt_length
    )
    
    
    return {
        **ret
    }

def _create_inputs(
    processing_class,
    prompt_inputs,
    completions,
):
    # now handle completion_ids and completion_mask
    pad_token_id = getattr(processing_class,"pad_token_id", getattr(processing_class.tokenizer,"pad_token_id",None))
    if pad_token_id is None:
        pad_token_id = 0
    completion_ids = torch.full((len(prompt_inputs["input_ids"]),max(map(len,completions))), pad_token_id , dtype=prompt_inputs["input_ids"].dtype,device=prompt_inputs["input_ids"].device)
    for idx,completion in enumerate(completions):
        completion_ids[idx,:len(completion)] = completion

    # Mask everything after the first EOS token
    im_eos = completion_ids == processing_class.tokenizer.convert_tokens_to_ids('<|im_end|>')
    s_eos = completion_ids == processing_class.tokenizer.convert_tokens_to_ids('</s>')
    is_eos = im_eos | s_eos
    
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long,device=completion_ids.device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1)).expand(is_eos.size(0), -1).to(device=eos_idx.device)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    
    
    prompt_inputs["input_ids"] = torch.cat([prompt_inputs["input_ids"],completion_ids],dim=-1).to(dtype=torch.int64)
    prompt_inputs["attention_mask"] = torch.cat([prompt_inputs["attention_mask"], completion_mask], dim=1)  # (B, P+C)
    
    return prompt_inputs,completion_mask

def _process_inputs(
    inputs, 
    processing_class,
    max_prompt_length
):
    prompts = []
    completions = []
    advantages = []
    rewards = []
    ids = []
    step_ids = []
    for inp in inputs:
        ids.append(inp["id"])
        prompts.append(inp["prompt"])
        completions.append(inp["completion_ids"])
        advantages.append(inp["advantage"])
        rewards.append(inp["reward"])
        step_ids.append(inp.get("step_id",0))
        
    ids = torch.tensor(ids)
    advantages = torch.tensor(advantages)
    step_ids = torch.tensor(step_ids)

    prompt_inputs = _prepare_messages(prompts,processing_class,max_prompt_length)
    prompt_len = prompt_inputs["input_ids"].size(1)
    prompt_inputs["rewards"] = torch.tensor(rewards)

    prompt_inputs,completion_mask = _create_inputs(processing_class,prompt_inputs,completions)
    return {
        "prompt_inputs": prompt_inputs,
        "completion_mask": completion_mask,
        "advantages": advantages,
        "prompt_len": prompt_len,
        "step_ids": step_ids
    }