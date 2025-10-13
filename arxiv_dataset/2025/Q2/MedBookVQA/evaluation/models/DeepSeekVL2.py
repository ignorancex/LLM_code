
import torch
from transformers import AutoModelForCausalLM

from models.deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from models.deepseek_vl2.utils.io import load_pil_images

import json

class DeepSeekVL2:
    def __init__(self, args):
        self.args = args
        self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
        self.tokenizer = self.processor.tokenizer

        self.model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, device_map="auto", low_cpu_mem_usage=True)
        self.model = self.model.to(torch.bfloat16).cuda().eval()

    def inference(self, prompt_batch, image_path_batch):
        output_batch =[]
        for prompt, image_path in zip(prompt_batch, image_path_batch):
            conversation = [
                {
                    "role": "<|User|>",
                    "content": prompt,
                    "images": [image_path],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            # load images and prepare for inputs
            pil_images = load_pil_images(conversation)
            prepare_inputs = self.processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt=""
            ).to(self.model.device)

            # run image encoder to get the image embeddings
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

            # run the model to get the response
            outputs = self.model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=32768,
                do_sample=False,
                use_cache=True
            )

            answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            output_batch.append(answer)
        return output_batch
    


def parse_reasoning_response(res_path, parsed_path, error_path):
    with open(res_path, 'r') as f:
        res = [json.loads(line) for line in f.readlines()]
    parsed = []
    error = []

    for item in res:
        response = item["response"]
        try: # [ANSWER: A]
            Answer = response.split("[ANSWER: ")[1].split("]")[0]
            if Answer not in ["A", "B", "C", "D"]:
                error.append(item)
                print(Answer)
            item["response"] = Answer
            item["reasoning"] = response
            parsed.append(item)
        except:
            try: # Answer: A
                Answer = response.split("\nAnswer: ")[1].strip()
                if ". " in Answer:
                    Answer = Answer.split(". ")[0].strip()
                if Answer not in ["A", "B", "C", "D"]:
                    error.append(item)
                    print(Answer)
                item["response"] = Answer
                item["reasoning"] = response
                parsed.append(item)
            except:
                try:
                    Answer = response.split("the correct choice is:\n\n")[1].strip()
                    Answer = Answer.split(".")[0].strip()
                    if Answer not in ["A", "B", "C", "D"]:
                        error.append(item)
                        print(Answer)
                    item["response"] = Answer
                    item["reasoning"] = response
                    parsed.append(item)
                except:
                    error.append(item)
                    print("ERROR")
                    item["response"] = "ERROR"
                    item["reasoning"] = response
                    parsed.append(item)

    with open(error_path, 'w') as f:
        json.dump(error, f, indent=2)
    with open(parsed_path, 'w') as f:
        json.dump(parsed, f, indent=2)
    print(f"{len(error)} errors.")