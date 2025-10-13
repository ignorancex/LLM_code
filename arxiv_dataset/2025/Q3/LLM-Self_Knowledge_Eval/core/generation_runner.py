import ast, random
from core.llm_interface import create_and_send_prompt, Prompt
from utils.infeasible_reasons import feasible_explanations
from utils.task_gen_prompts import vanilla_feasible_task_generation, vanilla_infeasible_task_generation, challenge_qap_feasible_task_generation, challenge_qap_infeasible_task_generation   

class TaskGenerationPromptRunner:
    def __init__(self, task_type: str, prompt_type:str, infeasibility_reason:str, type_of_self_knowledge: str, model: str):
        self.task_type = task_type
        self.prompt_type = prompt_type
        self.infeasibility_reason = infeasibility_reason
        if self.task_type.lower() == 'feasible':
            self.type_of_self_knowledge = (type_of_self_knowledge,feasible_explanations[type_of_self_knowledge])
            print(self.type_of_self_knowledge)
            if self.prompt_type.startswith('vanilla'):
                self.user_prompt: str = vanilla_feasible_task_generation.format(reason_name=self.type_of_self_knowledge[0], reason_description=self.type_of_self_knowledge[1])
            elif self.prompt_type.startswith('challenge'):
                self.user_prompt: str = challenge_qap_feasible_task_generation.format(reason_name=self.type_of_self_knowledge[0], reason_description=self.type_of_self_knowledge[1])
        elif self.task_type.lower() == 'infeasible':
            self.type_of_self_knowledge = type_of_self_knowledge
            if self.prompt_type.startswith('vanilla'):
                self.user_prompt: str = vanilla_infeasible_task_generation.format(reason_name=self.infeasibility_reason[0], reason_description=self.infeasibility_reason[1])
            elif self.prompt_type.startswith('challenge'):
                self.user_prompt: str = challenge_qap_infeasible_task_generation.format(reason_name=self.infeasibility_reason[0], reason_description=self.infeasibility_reason[1])
        self.model:str = model
        self.system_prompt:str = 'You are a helpful assistant. Follow the instruction and return only the required sictionary in your response.'
        

    @create_and_send_prompt
    def _send_prompt(self):
        return Prompt(system_prompt=self.system_prompt, user_prompt=self.user_prompt, model=self.model)

    def get_response(self):
        self.raw_response = self._send_prompt()
        if self.raw_response.startswith("```python"):
            self.raw_response = self.raw_response.replace('```python', '').replace('```', '').strip()
        self.response = ast.literal_eval(self.raw_response)