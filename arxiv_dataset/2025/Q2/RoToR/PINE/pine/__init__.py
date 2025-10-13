try:
    from .models.llama.tokenization_llama import LlamaTokenizerWithPS, LlamaTokenizerFastWithPS
    from .models.llama.modeling_llama_orig import LlamaForCausalLMWithPS
    from .models.llama.modeling_llama_rotor import LlamaForCausalLMWithPSRoToR
except Exception as e:
    print(f"Error loading Llama model: {e}")
try:
    from .models.qwen2.modeling_qwen2 import Qwen2ForCausalLMWithPS
    from .models.qwen2.modeling_qwen2_rotor import Qwen2ForCausalLMWithPSRoToR
    from .models.qwen2.tokenization_qwen2 import Qwen2TokenizerWithPS
except Exception as e:
    print(f"Error loading Qwen model: {e}")
