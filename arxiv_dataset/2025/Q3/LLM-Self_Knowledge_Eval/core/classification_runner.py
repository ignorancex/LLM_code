import ast
from core.llm_interface import create_and_send_prompt, Prompt
from utils.task_class_prompts import vanilla_class_prompt, challenge_qap_class_prompt
from utils.helper_functions import find_outer_key
from utils.infeasible_reasons import infeasible_reasons

class TaskClassificationPromptRunner:
    def __init__(self, prompt_type:str, task: str, model: str):
        self.task = task
        self.prompt_type = prompt_type
        self.model:str = model
        if self.prompt_type.lower() == 'vanilla':
            self.system_prompt:str = vanilla_class_prompt
        elif self.prompt_type.lower().startswith('challenge'):
            self.system_prompt:str = challenge_qap_class_prompt
        self.user_prompt = '\nUSER TASK: ' + self.task
        
    @create_and_send_prompt
    def _send_prompt(self):
        return Prompt(system_prompt=self.system_prompt, user_prompt=self.user_prompt, model=self.model)

    def get_response(self):
        self.raw_response = self._send_prompt()
        if self.raw_response.startswith("```python"):
            self.raw_response = self.raw_response.replace('```python', '').replace('```', '').strip()
        self.response = ast.literal_eval(self.raw_response)
        if 'reason' in self.response and (self.response['reason'] != '' or self.response['reason'] is not None):
            self.response['type_of_self_knowledge'] = find_outer_key(infeasible_reasons, self.response['reason'])