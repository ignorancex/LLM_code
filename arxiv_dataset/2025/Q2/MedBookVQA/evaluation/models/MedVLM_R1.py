from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader

class MedVLM_R1:
    def __init__(self, args):
        self.args = args
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        self.temp_generation_config = GenerationConfig(
            max_new_tokens=32768,
            do_sample=False,  
            temperature=0, 
            num_return_sequences=1,
            pad_token_id=151643,
        )


    def inference(self, prompt_batch, image_path_batch):
        
        batch_messages = []
        for prompt, img_path in zip(prompt_batch, image_path_batch):
            message = [{
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": img_path
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
            batch_messages.append(message)
        
        
        # Preparation for inference
        text = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, use_cache=True, max_new_tokens=32768, do_sample=False, generation_config=self.temp_generation_config)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return batch_output_text


###########################################################################

###########################################################################

def format_input(item):
    choice_content = [item["correct_choice"] + ". " + item["Answer"]]
    for i in range(3):
        choice_content.append(item["other_choices"][i] + ". " + item["Distractors"][i])

    choice_content = sorted(choice_content)
    choice_content = " | ".join(choice_content)

    input = f"""
    Question: {item["Question"]}
    Choices : {choice_content}
    """
    return input

def format_prompt_reasoning(input):
    return f"""
    {input} 
    Your task: 
    1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags. 
    2. Then provide the correct single-letter choice (A, B, C, D) inside <answer>...</answer> tags.
    3. No extra information or text outside of these tags.
    """

class dataset_MC(Dataset):
    def __init__(self, data, reasoning=True):
        self.data = data
        self.reasoning = reasoning

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        _id = item["_id"]
        image_path = item["image_path"]
        input = format_input(item)
        prompt = format_prompt_reasoning(input)
        return prompt, image_path, _id
    

def evaluate_MC(all_MC, res_path, model, args):
    dataset = dataset_MC(all_MC, reasoning=args.reasoning)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    with open(res_path, 'a') as f:
        for batch in tqdm(dataloader):
            prompt_batch, image_path_batch, _id_batch = batch
            response_batch = model.inference(prompt_batch, image_path_batch)

            for i, response in enumerate(response_batch):
                res_item = {
                    "response": response,
                    "prompt": prompt_batch[i].split("Your task: ")[0],
                    "image_path": image_path_batch[i],
                    "_id": _id_batch[i]
                }
                f.write(json.dumps(res_item) + "\n")
                f.flush()

def parse_reasoning_response(res_path, parsed_path, error_path):
    with open(res_path, 'r') as f:
        res = [json.loads(line) for line in f.readlines()]
    parsed = []
    error = []

    for item in res:
        response = item["response"]
        try:
            Answer = response.split("<answer>")[1].split("</answer>")[0].strip()
            item["response"] = Answer
            item["reasoning"] = response
        except:
            pass
        if "A." in item["response"] or "B." in item["response"] or "C." in item["response"] or "D." in item["response"]:
            item["response"] = item["response"].split(".")[0].strip()
        if f"A. {response}" in item["prompt"]:
            item["response"] = "A"
        elif f"B. {response}" in item["prompt"]:
            item["response"] = "B"
        elif f"C. {response}" in item["prompt"]:
            item["response"] = "C"
        elif f"D. {response}" in item["prompt"]:
            item["response"] = "D"
        if item["response"] not in ["A", "B", "C", "D"]:
            error.append(item)
            print(item["response"])
            item["response"] = "ERROR"
            item["reasoning"] = response
        parsed.append(item)

    print(len(error))

    with open(error_path, 'w') as f:
        json.dump(error, f, indent=2)
    with open(parsed_path, 'w') as f:
        json.dump(parsed, f, indent=2)
    with open(parsed_path+'l', 'w') as f:
        for item in parsed:
            f.write(json.dumps(item) + "\n")
            f.flush()
    print(f"{len(error)} errors.")