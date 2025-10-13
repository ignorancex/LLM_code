from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoTokenizer
from qwen_vl_utils import process_vision_info
from eval_utils import get_device
import torch


class QwenVL_model:
    def __init__(self, args):
        self.args = args
        # self.model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     args.model_path, torch_dtype="auto", device_map="auto" # torch.float16
        # )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map="auto" 
        )
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        self.processor.tokenizer.padding_side = "left"  # Set padding_side to 'left'
        
    def inference(self, prompt_batch, image_path_batch):
        messages_batch = []
        for prompt, image_path in zip(prompt_batch, image_path_batch):
            if "VL-Reasoner" in self.args.model_path:
                prompt = prompt.split("<multiple choice question end>")[0]
                prompt += "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            messages_batch.append(messages)

        try:
            # Apply chat templates to all messages in the batch
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages_batch
            ]
            # Process vision information for the batch
            image_inputs, video_inputs = process_vision_info(messages_batch)
            # Prepare inputs for the model
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            # Perform inference with memory optimization
            with torch.cuda.amp.autocast(enabled=True):
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=32768)

            # Trim generated IDs and decode outputs
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # Free memory
            del inputs, generated_ids, generated_ids_trimmed
            torch.cuda.empty_cache()

            return output_texts
        except Exception as e:
            print(f"Error during batch inference: {e}")
            return None




########## VL-Reasoner-7B ##########
import json

def parse_reasoning_response(res_path, parsed_path, error_path):
    with open(res_path, 'r') as f:
        res = [json.loads(line) for line in f.readlines()]
    parsed = []
    error = []

    for item in res:
        response = item["response"]
        try: # match with \\boxed{}
            Answer = response.split("\\boxed{")[1].split("}")[0]
            if Answer not in ["A", "B", "C", "D"]:
                if Answer.startswith("A. ") or Answer.startswith("B. ") or Answer.startswith("C. ") or Answer.startswith("D. "):
                    Answer = Answer[0]
                else:
                    error.append(item)
                    print(Answer)
            item["response"] = Answer
            item["reasoning"] = response
            parsed.append(item)
        except:
            error.append(item)
            print(response)
            item["response"] = "ERROR"
            item["reasoning"] = response
            parsed.append(item)

    with open(error_path, 'w') as f:
        json.dump(error, f, indent=2)
    with open(parsed_path, 'w') as f:
        json.dump(parsed, f, indent=2)
    with open(parsed_path+'l', 'w') as f:
        for item in parsed:
            f.write(json.dumps(item) + "\n")
            f.flush()
    print(f"{len(error)} errors.")
