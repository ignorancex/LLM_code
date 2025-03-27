from convlab2.nlg.nlg import NLG


class GenerativeModel(NLG):

    def __init__(self):
        """Load model and tokenizer"""
        super(GenerativeModel, self).__init__()

    def generate_text(self, prompt_text, temperature, tolerance):
        """Generate the text given the prompt.
            Return the only the new generated text"""
        pass
