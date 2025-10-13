from random import choice as rand_choice

class PromptGenerator:
    def __init__(self, args):
        self.args = args
        self.prompt_template = self._get_prompt_template()

    def _get_image_part(self):
        if self.args.caption_type == "text_only":
            image_part = ""
        elif self.args.model_family in ["llava", "custom_llava", "llava-onevision"]:
            image_part = "<image>\n"
            image_part = f"{'Image: ' if self.args.use_pointers == 1 else ''}{image_part}"
        elif self.args.model_family in ["qwen"]:
            image_part = "<|vision_start|><|image_pad|><|vision_end|>"
            image_part = f"{'Image: ' if self.args.use_pointers == 1 else ''}{image_part}"
        elif self.args.model_family in ["instructblip"]:
            image_part = ""
        else:
            raise NotImplementedError(f"{self.args.model_family} is not supported yet.")
        return image_part

    def _get_caption_part(self):
        if self.args.model_family == "human_study":
            return f"{'Caption: ' if self.args.use_pointers == 1 else ''}{rand_choice(self.args.caption_template)}\n"
        else:
            if self.args.caption_type in ["no_caption"]:
                return ""
            else:
                return f"{'Caption: ' if self.args.use_pointers == 1 else ''}{rand_choice(self.args.caption_template)}"
        
    def _get_question_part(self):
        question_part = rand_choice(self.args.question_template)

        if self.args.n_candidates > 0:
            question_part += " " + self.args.candidates_template
            question_part += ", ".join([
                "{}" for _ in range(self.args.n_candidates)
            ])
            question_part += "."
        
        question_part += " " + self.args.further_instruction + " "

        if self.args.is_explicit_helper:
            question_part = self.args.explicit_helper_template + question_part
        
        question_part = f"Question: {question_part}"

        return question_part
    
    def _get_answer_part(self):
        return self.args.answer_template

    def _get_parts(self):
        return {
            "image": self._get_image_part(),
            "caption": self._get_caption_part(),
            "question": self._get_question_part(),
            "answer": self._get_answer_part(),
        }
        
    def _get_image_caption_question_prompt(self, parts):
        res = []
        for part in self.args.order:
            if part == "i":
                res.append(parts["image"])
            elif part == "c":
                res.append(parts["caption"])
            elif part == "q":
                res.append(parts["question"])
            else:
                raise ValueError(f"{part} is not supported.")
        res = "".join(res)
        res = res.lstrip()
        return res

    def _get_prompt_template(self):
        parts = self._get_parts()
        image_caption_question_prompt = self._get_image_caption_question_prompt(parts)

        if self.args.model_family == "human_study":
            prompt_template = image_caption_question_prompt
        else: 
            if self.args.is_assistant_prompt:

                if self.args.model_family == "qwen":
                    prompt_template = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n {image_caption_question_prompt}<|im_end|>\n<|im_start|>assistant\n{parts['answer']}"
                elif self.args.model_family == "llava-onevision":
                    prompt_template = f"<|im_start|>user {image_caption_question_prompt}\n<|im_end|><|im_start|>assistant\n{parts['answer']}"
                elif self.args.model_family in ["llava", "instructblip"]:
                    prompt_template = f"USER: {image_caption_question_prompt}\nASSISTANT: {parts['answer']}"
                else:
                    print(f"model family not supported: {self.args.model_family}.")
                    raise RuntimeError
                
            else:
                prompt_template = image_caption_question_prompt + parts["answer"]

        return prompt_template

    def __call__(self, kargs=None):
        prompt_template = self._get_prompt_template()
        return prompt_template.format(*kargs) if kargs else prompt_template
