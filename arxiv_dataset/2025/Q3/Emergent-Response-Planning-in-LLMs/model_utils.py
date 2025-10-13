import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm


class ModelWrapper:
    """
    Wrapper for language models with different generation settings.
    For keeping the same configs among different operations in the same experiment.
    """
    
    def __init__(self, model_path: str, device: str = "auto", load_model=True, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # support only load tokenizer
        if load_model:
            # settings
            use_sampling = kwargs.get("use_sampling", False)
            temperature = kwargs.get("temperature", 0.7)
            top_p = kwargs.get("top_p", 0.9)
            self.max_new_tokens = kwargs.get("max_new_tokens", 2048)
            num_samples = kwargs.get("num_samples", 1)
            self.use_vllm = kwargs.get("use_vllm", False)
            self.lora_path = kwargs.get("lora_path", None)


            # load model
            if not self.use_vllm:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.bfloat16, 
                    device_map=device
                )
                self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
                # generation configs
                if use_sampling:
                    self._set_sampling_config(temperature, top_p, num_samples)
                else:
                    self._set_greedy_config()
            else:
                from vllm import LLM
                from vllm.lora.request import LoRARequest
                print('Available GPUs:', torch.cuda.device_count())
                self.vllm_model = LLM(
                    model=model_path,
                    dtype="bfloat16",
                    enforce_eager=True,
                    gpu_memory_utilization=0.95,  # Can be adjusted based on needs
                    tensor_parallel_size=torch.cuda.device_count(),  # Adjust based on number of GPUs
                    enable_lora=True,
                    max_lora_rank=64,
                )
                # set lora
                if self.lora_path:
                    self.lora_request = LoRARequest(
                        "lora_adapter",
                        1,
                        self.lora_path,
                    )

            # stopping_criteria
            self.is_chat_model = any(x in model_path.split("/")[-1].lower() for x in ["chat", "instruct"])
            # truncate for base models
            self.max_new_tokens = min(self.max_new_tokens, 500) if not self.is_chat_model else self.max_new_tokens

        else:
            self.model = None
            self.vllm_model = None
    
    def _set_greedy_config(self):
        """Set greedy decoding configuration"""
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.model.generation_config.num_beams = 1
        self.model.generation_config.do_sample = False
        self.num_samples = 1
    
    def _set_sampling_config(self, temperature: float = 0.7, top_p: float = 0.9, num_samples: int = 1):
        """Set sampling configuration"""
        self.model.generation_config.temperature = temperature
        self.model.generation_config.top_p = top_p
        self.model.generation_config.top_k = None
        self.model.generation_config.num_beams = 1
        self.model.generation_config.do_sample = True
        self.num_samples = num_samples
        
    
    def generate(self, batch) -> list[str]:
        """Generate responses for a list of prompts"""
        if not self.use_vllm:
            with torch.no_grad():
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
                input_length = inputs['input_ids'].shape[1]
                batch_responses_list = []

                # stopping criteria: for non-chat models
                for _ in range(self.num_samples):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        return_dict_in_generate=True,
                        num_beams=self.model.generation_config.num_beams,
                        do_sample=self.model.generation_config.do_sample
                    )
                    # Only include newly generated tokens
                    new_tokens = outputs.sequences[:, inputs['input_ids'].shape[1]:]
                    # batch > 1: batch decoding
                    if new_tokens.ndim == 2:
                        batch_responses = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                    else:
                        batch_responses = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
                        # add dim
                        batch_responses = [batch_responses]
                    batch_responses_list.append(batch_responses)
                # if num_samples>1, then rearrange the responses to be a list of lists, where each sub list contains the responses for a single prompt
                if self.num_samples > 1:
                    batch_responses_by_prompt = []
                    for i in range(len(batch)):
                        prompt_responses = [batch_responses_list[j][i] for j in range(self.num_samples)]
                        batch_responses_by_prompt.append(prompt_responses)
                else:
                    batch_responses_by_prompt = batch_responses_list[0]
        else:
            # VLLM
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest

            # Configure sampling parameters for greedy decoding
            sampling_params = SamplingParams(
                temperature=0.0,  # Set to 0 for greedy decoding
                top_p=1.0,       # Set to 1 for greedy decoding
                max_tokens=self.max_new_tokens,
                n=1,            # Always 1 for greedy decoding
            )
            
            # Generate with vLLM
            outputs = self.vllm_model.generate(
                batch, 
                sampling_params,
                lora_request=self.lora_request if self.lora_path else None,
            )
            # Process outputs to match the original format
            batch_responses = []
            batch_responses_by_prompt = [output.outputs[0].text for output in outputs]
                            
        return batch_responses_by_prompt
    
    def forward(self, batch, chosen_layers) -> torch.Tensor:
        """Get hidden states for the last token of a list of prompts"""
        with torch.no_grad():
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
            output = self.model(**inputs, output_hidden_states=True)
            hidden_states = output.hidden_states
            hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
            hidden_states = hidden_states.detach().cpu()
            # if hidden_states has shape: (hidden layers, sentense length, vocab size), then get the last token, and add a dim at batch dim
            if len(hidden_states.shape) == 3:
                hidden_states_last_token = hidden_states[chosen_layers, -1, :].squeeze().reshape(1, self.model.config.num_hidden_layers, -1)
            # (hidden layers, batch size, sentense length, vocab size) -> (batch size, hidden layers, vocab size)
            else:
                hidden_states_last_token = hidden_states[chosen_layers, : , -1, :].squeeze().permute(1, 0, 2)
        
        return hidden_states_last_token
