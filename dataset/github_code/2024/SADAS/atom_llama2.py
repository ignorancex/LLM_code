import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = 'FlagAlpha/Atom-7B-Chat'
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16, load_in_8bit=True)
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

input1 = '<s>Human: 你是一个对话助手，请根据你的知识回复用户。\n</s><s>Human: 介绍一下中国\n</s><s>Assistant: '
input2 = "<s>Human: 你是一个对话助手，你的任务是改写对话中最后一句话，使其符合社会规范。在谈判游戏中，谈判的对话如下：User1: 你好，请问贵公司有一种特殊的工业商品吗？\nUser2: 你好，我们公司产能充足，能够以一次性的方式向贵司出售所需的工业商品。这种工业商品的单元价格是$50。\nUser1: 你会考虑以每单元$30的价格进行出售吗？\nUser2: 【violation】很抱歉，我们不会接受低于$40的价格。\n在这个对话中，最后一句话\"很抱歉，我们不会接受低于$40的价格\"违反了社会规范，请对这句话进行改写:\n</s><s>Assistant: "

input_ids = tokenizer([input2], return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":0.3,
    "repetition_penalty":1.3,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
}
generate_ids = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)