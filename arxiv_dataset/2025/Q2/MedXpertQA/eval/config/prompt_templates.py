class PromptTemplates:
    def __init__(self):
        pass

    def load_templates(self, dataset, model):
        # System Role
        self.zero_shot_system_role = "You are a helpful assistant."
        self.zero_shot_ao_system_role = "Please answer only."
        self.few_shot_system_role = "Follow the given examples and answer the question."

        # Few-shot Trigger
        self.few_shot_trigger = "The answer is"

        # Few-shot Prompt
        self.few_shot_prompt = "Q: {question}\nA:"

        if dataset in ["medqa", "medmcqa", "mmlu_medical"]:
            self.zero_shot_system_role = "You are a helpful medical assistant."

            self.zero_shot_ao_trigger = "Among A through D, the answer is"
            self.zero_shot_cot_trigger = "Therefore, among A through D, the answer is"
        elif dataset in ["medxpertqa", "medxpertqa_sampled"]:
            self.zero_shot_system_role = "You are a helpful medical assistant."

            self.zero_shot_ao_trigger = "Among {start} through {end}, the answer is"
            self.zero_shot_cot_trigger = "Therefore, among {start} through {end}, the answer is"
        else:
            raise ValueError("Dataset prompt template is not defined...")

        if model == "deepseek-reasoner":
            self.zero_shot_cot_trigger = "Put your final answer within \\boxed{{}}. " + self.zero_shot_cot_trigger
        elif "qvq" in model.lower():
            self.zero_shot_ao_system_role = "You are a helpful medical assistant."
            self.zero_shot_ao_trigger = "Let's think step by step."

        self.zero_shot_ao_prompt = "Q: {question}\nA: " + self.zero_shot_ao_trigger
        self.zero_shot_cot_prompt = "Q: {question}\nA: Let's think step by step."

        return self
