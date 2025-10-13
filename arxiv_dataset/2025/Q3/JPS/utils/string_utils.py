import torch
from qwen_vl_utils import process_vision_info
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "../model/LLaVA_NeXT")
sys.path.append(os.path.abspath(model_path))

# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX


# from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token

import re

class autodan_SuffixManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):

        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string

    def get_prompt(self, adv_string=None, image=None, model_name = None):
        
        if self.conv_template is not None:
            # if adv_string is not None:
            #     self.adv_string = adv_string.replace('[REPLACE]', self.instruction)
            # else:
            #     self.adv_string = self.instruction
                

            # self.conv_template.append_message(self.conv_template.roles[0], f"{self.instruction}")
            # self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
            # prompt = self.conv_template.get_prompt()

            # encoding = self.tokenizer(prompt)
            # toks = encoding.input_ids
            if hasattr(self.conv_template, 'name') and self.conv_template.name=='internlm2-chat':
            # if hasattr(self.conv_template, 'name') and self.conv_template.name=='internvl2_5':
                self.conv_template.messages = []
                sys_stop_pos = len(self.tokenizer(self.conv_template.get_prompt()).input_ids)

                self.conv_template.append_message(self.conv_template.roles[0], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._user_role_slice = slice(sys_stop_pos, len(toks))

                # update last message
                self.conv_template.messages[-1][1] = f"{self.instruction}"
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids

                self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks) - 1))
                self._control_slice = self._goal_slice

                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

                # update last message
                self.conv_template.messages[-1][1] = f"{self.target}"
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids

                self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 1)
                self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 2)

                prompt = self.conv_template.get_prompt()

            # elif hasattr(self.conv_template, 'version') and self.conv_template.version=='llama_v3':
            #     eot_ids = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

            #     self.conv_template.messages = []

            #     self.conv_template.append_message(self.conv_template.roles[0], None)
            #     toks = (
            #         tokenizer_image_token(self.conv_template.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            #         # .unsqueeze(0)
            #         # .cuda()
            #     )
 
            #     self._user_role_slice = slice(None, len(toks))

            #     # update last message
            #     self.conv_template.messages[-1][1] = f"{self.instruction}"
            #     toks = (
            #         tokenizer_image_token(self.conv_template.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            #         # .unsqueeze(0)
            #         # .cuda()
            #     )
            #     goal_end_indices = torch.nonzero(toks[self._user_role_slice.stop:]== eot_ids)[0].item()
            #     self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, self._user_role_slice.stop + goal_end_indices))
            #     self._control_slice = self._goal_slice

            #     self.conv_template.append_message(self.conv_template.roles[1], None)
            #     toks = (
            #         tokenizer_image_token(self.conv_template.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            #         # .unsqueeze(0)
            #         # .cuda()
            #     )
            #     self._assistant_role_slice = slice(self._control_slice.stop + 1, len(toks))

            #     # update last message
            #     self.conv_template.messages[-1][1] = f"{self.target}"
            #     toks = (
            #         tokenizer_image_token(self.conv_template.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            #         # .unsqueeze(0)
            #         # .cuda()
            #     )

            #     toks = torch.cat((toks[:self._assistant_role_slice.stop], torch.tensor([198],device=toks.device), toks[self._assistant_role_slice.stop:]))

            #     target_end_indices = torch.nonzero(toks[self._assistant_role_slice.stop:]== eot_ids)[0].item()
            #     self._target_slice = slice(self._assistant_role_slice.stop, self._assistant_role_slice.stop + target_end_indices)
            #     self._loss_slice = slice(self._assistant_role_slice.stop - 1, self._assistant_role_slice.stop + target_end_indices - 1)

            #     prompt = self.conv_template.get_prompt()
            
            elif hasattr(self.conv_template, 'name') and self.conv_template.name=='minigpt4':
                self.conv_template.messages = []
                sys_stop_pos = len(self.tokenizer(self.conv_template.get_prompt()).input_ids)

                self.conv_template.append_message(self.conv_template.roles[0], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._user_role_slice = slice(sys_stop_pos, len(toks))

                # update last message
                self.conv_template.messages[-1][1] = f"{self.instruction}"
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids

                self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))
                self._control_slice = self._goal_slice

                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

                # update last message
                self.conv_template.messages[-1][1] = f"{self.target}"
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids

                self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
                self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)

                prompt = self.conv_template.get_prompt()
                import pdb;pdb.set_trace()

            else:
                raise KeyError(f"Unknown conv_template version: {self.conv_template.version}")
            self.conv_template.messages = []
        else:
            
            if self.tokenizer.__class__.__name__=='Qwen2VLProcessor':
                eos_ids = self.tokenizer.tokenizer.eos_token_id
                messages = []
                messages.append({
                    "role": "user",
                    "content":""
                })
                toks = self.qwen_tokenizer(messages=messages, with_content=False)
                self._user_role_slice = slice(None, len(toks))

                messages[0]['content']= [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": self.instruction},
                ]
                toks = self.qwen_tokenizer(messages=messages, with_content=True)
                self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks) - 1))
                self._control_slice = self._goal_slice

                messages.append({
                    "role": "assistant",
                    "content":""
                })
                toks = self.qwen_tokenizer(messages=messages, with_content=False)
                self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

                messages[1]['content']= [
                    {"type": "text", "text": self.target},
                ]
        
                toks = self.qwen_tokenizer(messages=messages, with_content=True)
                target_end_indices = torch.nonzero(toks[self._assistant_role_slice.stop:]== eos_ids)[0].item()
                self._target_slice = slice(self._assistant_role_slice.stop, self._assistant_role_slice.stop + target_end_indices)
                self._loss_slice = slice(self._assistant_role_slice.stop - 1, self._assistant_role_slice.stop + target_end_indices - 1)
                
                prompt = messages
            else:
                raise KeyError(f"Unknown conv_template version: {self.conv_template.version}")
        return prompt

    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])

        return input_ids
    
    def get_image_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])

        return input_ids
    
    def get_qwen_image_input_ids(self, adv_string=None, image=None):
        prompt = self.get_prompt(adv_string=adv_string, image=image)
        toks = self.qwen_tokenizer(messages=prompt)
        input_ids = toks[:self._target_slice.stop]
        return input_ids, prompt

    # def get_llava_image_input_ids(self, adv_string=None):
    #     prompt = self.get_prompt(adv_string=adv_string)
    #     toks = (
    #         tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    #         # .unsqueeze(0)
    #         .cuda()
    #     )
    #     toks = torch.cat((toks[:self._assistant_role_slice.stop], torch.tensor([198],device=toks.device), toks[self._assistant_role_slice.stop:]))
    #     input_ids = toks[:self._target_slice.stop]

    #     return input_ids

    def qwen_tokenizer(self,messages, with_content=True):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        if with_content is False:
            text = text[:-len("<|im_end|>\n")]

        text = [text]
        image_inputs, video_inputs = process_vision_info(messages)
        toks = self.tokenizer(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).input_ids[0]

        return toks