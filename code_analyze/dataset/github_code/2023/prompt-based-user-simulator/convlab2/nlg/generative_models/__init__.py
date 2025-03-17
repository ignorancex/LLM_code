from convlab2.nlg.generative_models.flanT5 import FLANT5Model
from convlab2.nlg.generative_models.llama import LLAMAModel
from convlab2.nlg.generative_models.auto_model import AutoModel
from convlab2.nlg.generative_models.azure_openai import AzureOpenAIModel

MODEL_ID_MODEL_CLASS_MAPPING = {
    'google/flan-t5-xl': FLANT5Model,
    'google/flan-t5-small': FLANT5Model,
    'decapoda-research/llama-7b-hf': LLAMAModel,
    'nomic-ai/gpt4all-j': AutoModel,
    'databricks/dolly-v2-7b': AutoModel,
    'azure_openai/gpt-35-turbo': AzureOpenAIModel,
}
