from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import json

class KimiVL:
    def __init__(self, args):
        self.args = args
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            offload_buffers=True,
            attn_implementation="flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"  # Set padding_side to 'left'

    # def inference(self, prompt_batch, image_path_batch):    
    #     response_batch = []
    #     for image_path, promt in zip(image_path_batch, prompt_batch):
    #         image = Image.open(image_path)
    #         messages = [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image", "image": image},
    #                     {"type": "text", "text": promt},
    #                 ],
    #             }
    #         ]
    #         text = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    #         inputs = self.processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
    #         generated_ids = self.model.generate(**inputs, max_new_tokens=32768)
    #         generated_ids_trimmed = [
    #             out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    #         ]
    #         response = self.processor.batch_decode(
    #             generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    #         )[0]
    #         response_batch.append(response)
    #         # free up memory
    #         del image, inputs, generated_ids, generated_ids_trimmed
    #         torch.cuda.empty_cache()
    #     return response_batch

    def inference(self, prompt_batch, image_path_batch):
        messages_batch = []
        for prompt, image_path in zip(prompt_batch, image_path_batch):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": Image.open(image_path),
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

            # Prepare inputs for the model
            inputs = self.processor(
                text=texts,
                images=[Image.open(image_path) for image_path in image_path_batch],
                padding=True,
                return_tensors="pt",
                truncation=True,
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
        
def parse_reasoning_response(res_path, parsed_path, error_path):
    with open(res_path, 'r') as f:
        res = [json.loads(line) for line in f.readlines()]
    parsed = []
    error = []

    for item in res:
        try: # [ANSWER: A]
            response = item["response"]
            Answer = response.split("[ANSWER: ")[1].split("]")[0]
            if Answer not in ["A", "B", "C", "D"]:
                if Answer.startswith("A. ") or Answer.startswith("B. ") or Answer.startswith("C. ") or Answer.startswith("D. "):
                    Answer = Answer[0]
            if Answer in ["A", "B", "C", "D"]:
                item["response"] = Answer
                item["reasoning"] = item["response"]
                parsed.append(item)
                continue
        except:
            pass
        try: # \n\n**ANSWER: A**
            response = item["response"]
            Answer = response.split("\n\n**ANSWER: ")[1].split("**")[0]
            if Answer not in ["A", "B", "C", "D"]:
                if Answer.startswith("A. ") or Answer.startswith("B. ") or Answer.startswith("C. ") or Answer.startswith("D. "):
                    Answer = Answer[0]
            if Answer in ["A", "B", "C", "D"]:
                item["response"] = Answer
                item["reasoning"] = item["response"]
                parsed.append(item)
                continue
        except:
            pass
        try: # \n\n**ANSWER: A**
            response = item["response"]
            Answer = response.split("**ANSWER: ")[1].split("**")[0]
            if Answer not in ["A", "B", "C", "D"]:
                if Answer.startswith("A. ") or Answer.startswith("B. ") or Answer.startswith("C. ") or Answer.startswith("D. "):
                    Answer = Answer[0]
            if Answer in ["A", "B", "C", "D"]:
                item["response"] = Answer
                item["reasoning"] = item["response"]
                parsed.append(item)
                continue
        except:
            pass
        try: # \n\nANSWER: A
            response = item["response"]
            Answer = response.split("\n\nANSWER: ")[1].strip()
            if Answer not in ["A", "B", "C", "D"]:
                if Answer.startswith("A.") or Answer.startswith("B.") or Answer.startswith("C.") or Answer.startswith("D."):
                    Answer = Answer[0]
            if Answer in ["A", "B", "C", "D"]:
                item["response"] = Answer
                item["reasoning"] = item["response"]
                parsed.append(item)
                continue
        except:
            pass
        try: # \n\\[ \\boxed{B} \\]
            response = item["response"]
            Answer = response.split("\\[ \\boxed{")[1].split("} \\]")[0]
            if Answer not in ["A", "B", "C", "D"]:
                if Answer.startswith("A.") or Answer.startswith("B.") or Answer.startswith("C.") or Answer.startswith("D."):
                    Answer = Answer[0]
            if Answer in ["A", "B", "C", "D"]:
                item["response"] = Answer
                item["reasoning"] = item["response"]
                parsed.append(item)
                continue
        except:
            pass
        try: # \n\\boxed{A}
            response = item["response"]
            Answer = response.split("\\boxed{")[1].split("}")[0]
            if Answer not in ["A", "B", "C", "D"]:
                if Answer.startswith("A.") or Answer.startswith("B.") or Answer.startswith("C.") or Answer.startswith("D."):
                    Answer = Answer[0]
            if Answer in ["A", "B", "C", "D"]:
                item["response"] = Answer
                item["reasoning"] = item["response"]
                parsed.append(item)
                continue
        except:
            pass
        error.append(item)
        print(item["response"])
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

