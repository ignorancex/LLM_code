from transformers import AutoModelWithLMHead, AutoTokenizer


class T5_summarizer:
    def __init__(self, model_path="philschmid/bart-large-cnn-samsum"):
        self.model = AutoModelWithLMHead.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def summarize(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
        max_length = len(text) if len(text) > 150 else 150

        generated_ids = self.model.generate(input_ids=input_ids, num_beams=2, max_length=max_length,
                                            repetition_penalty=2.5,
                                            length_penalty=1.0, early_stopping=True)

        preds = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                 generated_ids]

        return preds[0]


# text = """Where was the claim first published. It was first published on Sccopertino. What kind of website is
# Scoopertino. Scoopertino is an imaginary news organization devoted to ferreting out the most relevant stories in the
# world of Apple, whether or not they actually occurred - says their about page"""
#
# t5_sum = T5_summarizer()
# output = t5_sum.summarize(text)
# print(output)
