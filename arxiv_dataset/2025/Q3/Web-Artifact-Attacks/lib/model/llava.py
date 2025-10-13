import torch
import torch.nn as nn 
from transformers import LlavaForConditionalGeneration, AutoProcessor

class LLaVAModel(nn.Module): 

    def __init__(self, args):
        super(LLaVAModel, self).__init__()

        self.args = args
        model_id = "llava-hf/llava-1.5-13b-hf"

        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.image_processor.do_center_cropq = False
        self.processor.image_processor.size = {"height": 336, "width": 336}

        # self.caption_prompt = "Provide a detailed visual description in order to answer the question about [TOPIC]. Describe any logos or text artificially pasted on the image and discuss if they are misleading with respect to the visual content of the image."
        # self.caption_prompt = "Is there any text/logos artifically pasted on the image? If yes, what is it and does it contradict the visual content of the image?"
        self.caption_prompt = "Given the following question: [TOPIC]. Is there any text/logos artifically pasted on the image that interfers with the quesiton? If yes, what is it and does it contradict the visual content of the image?"

    def get_llava_prompt(self, lvlm_prompt): 
        return [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": lvlm_prompt},
                        ],
                    },
                ]

    def get_llava_caption_prompt(self, caption_prompt, question_prompt, response):
        return [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": caption_prompt},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": response},]

                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": question_prompt},]
                    },
                ]

    def forward(self, batch):
        images = [img[0] for img in batch["images"]]
        

        if self.args.add_caption: 
            # prompts_caption = [self.processor.apply_chat_template(self.get_llava_prompt(self.caption_prompt.replace("[TOPIC]", self.args.topic)), add_generation_prompt=True) for _ in range(len(images))]
            prompts_caption = [self.processor.apply_chat_template(self.get_llava_prompt(self.caption_prompt.replace("[TOPIC]", self.args.lvlm_prompt.replace("Answer with the corresponding number only", ""))), add_generation_prompt=True) for _ in range(len(images))]

            input = self.processor(images=images, text=prompts_caption, return_tensors="pt", padding="max_length", max_length=200).to("cuda")
            outputs = self.model.generate(**input, max_new_tokens=300, do_sample=True, temperature=0.8)
            outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)
            outputs = [output.split("ASSISTANT:")[1].lower().strip() for output in outputs]  
            prompts = [self.processor.apply_chat_template(self.get_llava_caption_prompt(self.caption_prompt.replace("[TOPIC]", self.args.lvlm_prompt.replace("Answer with the corresponding number only", "")), self.args.lvlm_prompt, response), add_generation_prompt=True) for response in outputs]

            # prompts = [self.processor.apply_chat_template(self.get_llava_caption_prompt(self.caption_prompt.replace("[TOPIC]", self.args.topic), self.args.lvlm_prompt, response), add_generation_prompt=True) for response in outputs]


        else: 
            prompts = [self.processor.apply_chat_template(self.get_llava_prompt(self.args.lvlm_prompt), add_generation_prompt=True) for _ in range(len(images))]


        input = self.processor(images=images, text=prompts, return_tensors="pt", padding='longest', max_length=200).to("cuda")
        outputs = self.model.generate(**input, max_new_tokens=300, do_sample=True, temperature=0.1)
        outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)        

        outputs = [output.split("ASSISTANT:")[-1].lower().strip() for output in outputs]

        # for output in outputs:
        #     print(output)
        #     print("="*20)
        
        # quit()

        return outputs
