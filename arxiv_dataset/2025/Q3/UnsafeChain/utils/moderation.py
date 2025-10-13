import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llamaguard(model_id="meta-llama/Llama-Guard-3-8B", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return tokenizer, model

def moderate(prompt, response, tokenizer, model, device="cuda"):
    try:
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(input_ids=input_ids, temperature=0.0, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip().lower()
        if not isinstance(generated, str):
            return "unsafe"
        return "safe" if generated.startswith("safe") else "unsafe"
    except Exception as e:
        print(f"[moderate error] {e}")
        return "unsafe"
