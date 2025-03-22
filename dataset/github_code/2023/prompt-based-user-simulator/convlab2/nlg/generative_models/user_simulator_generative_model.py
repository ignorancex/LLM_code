from copy import deepcopy
from convlab2.nlg.generative_models.generative_model import (
    GenerativeModel)


class UserSimulatorGenerativeModel(GenerativeModel):

    def __init__(self, debug_prompt_processing=False):
        super(UserSimulatorGenerativeModel, self).__init__()
        self.last_generate = ''

    def _generate_text(
            self, prompt_text, temperature=0.8,
            debug_prompt_processing=False):

        padded_sequence = self.tokenizer(
            prompt_text, padding=True, return_tensors="pt")

        input_ids = padded_sequence.input_ids.to(self.device)
        attention_mask = padded_sequence.attention_mask.to(self.device)
        generated_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
            # pad_token_id=self.tokenizer.pad_token_id,
            max_length=input_ids.shape[1]+30,
            do_sample=True)
        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        return generated_text

    @staticmethod
    def _process_generated_text(generated_text, cut_off_n_chars=150):

        # look for first occurrance of 'CUSTOMER: '
        # concatenate all texts starting with 'CUSTOMER: '
        # remove all 'CUSTOMER: '
        if 'CUSTOMER: ' in generated_text:
            generated_text = ''.join(generated_text.split(
                'CUSTOMER: ')[1:]).strip()
            generated_text = generated_text.replace('CUSTOMER', '')

        if 'ASSISTANT' in generated_text:
            # extract the generated_text until the AGENT's generated_text
            generated_text = generated_text.split(
                'ASSISTANT')[0].strip()

        if '\n' in generated_text:
            generated_text = generated_text.split(
                '\n')[0].strip()

        if len(generated_text) > cut_off_n_chars:  # cut off sentences that are too long  # noqa
            new_generated_text = ''
            for sentence in generated_text.split('.'):
                if len(new_generated_text + sentence) < cut_off_n_chars:
                    new_generated_text += sentence + "."
                else:
                    return new_generated_text
            generated_text = new_generated_text

        return generated_text

    def generate(
            self, prompt_text, temperature, tolerance=20,
            debug_prompt_processing=False):

        processed_text = ''
        iteration = 0
        prompt_text += '\n'

        while len(processed_text) < 2:  # empty utterance

            # generate
            if debug_prompt_processing:
                print('\n==========================================================')  # noqa
                print('PROMPTED TEXT: {}'.format(prompt_text))
                print('==========================================================\n')  # noqa
            generated_text = self._generate_text(
                prompt_text, temperature=temperature)

            if debug_prompt_processing:
                print('\n==========================================================')  # noqa
                print('GENERATED TEXT: {}'.format(generated_text))
                print('==========================================================\n')  # noqa

            # remove prompt text if necessary
            processed_text = deepcopy(generated_text)
            if processed_text.startswith(prompt_text):
                processed_text = processed_text.replace(prompt_text, '')

            # process
            processed_text = self._process_generated_text(
                generated_text=processed_text)
            if debug_prompt_processing:
                print('\n==========================================================')  # noqa
                print('PROCESSED TEXT: {}'.format(processed_text))
                print('==========================================================\n\n\n\n\n\n')  # noqa

            if iteration > tolerance:
                return "I DONT KNOW"

            iteration += 1

        self.last_generate = {
            'prompted_text': prompt_text,
            'generated_text': generated_text,
            'processed_text': processed_text
        }

        return processed_text
