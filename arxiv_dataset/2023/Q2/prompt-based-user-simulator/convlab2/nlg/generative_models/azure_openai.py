import openai
import traceback
import logging
import os
from convlab2.nlg.generative_models.user_simulator_generative_model import (
    UserSimulatorGenerativeModel)


class AzureOpenAIModel(UserSimulatorGenerativeModel):
    def __init__(self, model_id='gpt-35-turbo'):
        super(UserSimulatorGenerativeModel, self).__init__()
        self.engine = model_id.split('/')[1]

    @staticmethod
    def _prepare_prompt(prompt_text):
        '''
        [
                            {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},  # noqa
                            {"role": "user", "content": "What's the difference between garbanzo beans and chickpeas?"}  # noqa
        ]
        '''

        messages = []

        conversations = prompt_text.split('Conversation:')

        # use everything up to last Conversation: as system message
        system_message_content = 'Conversation:'.join(conversations[:-1]) + 'Conversation:\n'  # noqa
        messages.append({"role": "system", "content": system_message_content})  # noqa

        current_conv = conversations[-1]
        for line in current_conv.split('\n'):
            if line.startswith('CUSTOMER:'):
                utt = line.replace('CUSTOMER:', '').strip()
                messages.append({"role": "user", "content": utt})
            elif line.startswith('ASSISTANT:'):
                utt = line.replace('ASSISTANT:', '').strip()
                messages.append({"role": "assistant", "content": utt})

        return messages

    # for run_model.py
    def _generate_text(self, prompt_text, **kwargs):

        openai.api_type = "azure"
        openai.api_version = "2023-03-15-preview"
        openai.api_base = os.getenv("OPENAI_API_BASE")  # Your Azure OpenAI resource's endpoint value.  # noqa
        assert openai.api_base, "Need OPENAI_API_BASE env var"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        assert openai.api_key, "Need OPENAI_API_KEY env var"

        # this did not help, it confused the model
        # messages = self._prepare_prompt(prompt_text)
        # so assign everything to role system:
        messages = [{"role": "system", "content": prompt_text}]

        try:
            response = openai.ChatCompletion.create(
                engine=self.engine, # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.  # noqa
                messages=messages
            )
            generated_text = response['choices'][0]['message']['content']
        except Exception as exc:
            logging.error('Cannot generate: {} Traceback: {}'.format(
                exc, traceback.format_exc()))
            generated_text = ''

        return generated_text
