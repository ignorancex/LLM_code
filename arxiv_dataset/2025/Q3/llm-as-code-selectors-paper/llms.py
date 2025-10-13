import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_deepseek import ChatDeepSeek

from llms_chat_models import FireworksChatModel, FireworksDeepSeekR1, Gemma3ChatModel, HuggingFaceLocalChatModel, MaritacaChatModel

load_dotenv()

llm_dict = {}

# OpenAI models
llm_dict["gpt-4o-mini"] = ChatOpenAI(
                        model="gpt-4o-mini",
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )

llm_dict["gpt-4o"] = ChatOpenAI(
                        model="gpt-4o",
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )

llm_dict["gpt-4o-baseline"] = ChatOpenAI(
                        model="gpt-4o",
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )

llm_dict["gpt-4.5-preview"] = ChatOpenAI(
                        model="gpt-4.5-preview",
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )

llm_dict["o1-mini"] = ChatOpenAI(
                        model="o1-mini",
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )

llm_dict["o1"] = ChatOpenAI(
                        model="o1",
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )

llm_dict["o3-mini"] = ChatOpenAI(
                        model="o3-mini",
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )

llm_dict["gpt-4.1"] = ChatOpenAI(
                        model="gpt-4.1",
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )

llm_dict["gpt-4.1-mini"] = ChatOpenAI(
                        model="gpt-4.1-mini",
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )

llm_dict["gpt-4.1-nano"] = ChatOpenAI(
                        model="gpt-4.1-nano",
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )

llm_dict["o4-mini"] = ChatOpenAI(
                        model="o4-mini",
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )

llm_dict["o3"] = ChatOpenAI(
                        model="o3",
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )

# Google models
llm_dict['gemini-2.0-flash'] = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash",
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )

llm_dict['gemini-2.0-flash-lite'] = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash-lite",
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )

# Set rate limiter for gemini-2.0-pro-exp-02-05
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  
    check_every_n_seconds=0.1,  
    max_bucket_size=1,  
)

llm_dict['gemini-2.0-pro-exp-02-05'] = ChatGoogleGenerativeAI(
                        model="gemini-2.0-pro-exp-02-05",
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=10,
                        rate_limiter=rate_limiter,
                    )

llm_dict['gemini-2.5-pro-exp-03-25'] = ChatGoogleGenerativeAI(
                        model="gemini-2.5-pro-exp-03-25",
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=10,
                        rate_limiter=rate_limiter,
                    )

# Google open source models
llm_dict['gemma-3-4b-it'] = Gemma3ChatModel()

llm_dict['gemma-3-27b-it'] = ChatHuggingFace(
                        llm=HuggingFaceEndpoint(
                            repo_id="google/gemma-3-27b-it",
                            task="text-generation",
                            temperature=0,
                        )
                    )

llm_dict['gemma-2-27b-it'] = ChatHuggingFace(
                        llm=HuggingFaceEndpoint(
                            repo_id="google/gemma-2-27b-it",
                            task="text-generation",
                            temperature=0,
                        )
                    )

# Deepseek models
llm_dict['DeepSeek-V3'] = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llm_dict['DeepSeek-R1'] = FireworksDeepSeekR1(
    model="DeepSeek-R1",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
    )

llm_dict['DeepSeek-R1-Distill-Qwen-1.5B'] = HuggingFaceLocalChatModel(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
        max_tokens=4000,
    )

llm_dict['DeepSeek-R1-Distill-Qwen-7B'] = FireworksChatModel(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        custom_endpoint="accounts/almeida-va93-67c8dc/deployedModels/deepseek-r1-distill-qwen-7b-1b7c157a"
    )


# Meta open source models
llm_dict['Llama-3.3-70B-Instruct'] = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.3-70B-Instruct",
        task="text-generation",
        temperature=0,
    )
)

llm_dict['Llama-3.2-3B-Instruct'] = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        task="text-generation",
        temperature=0,
    )
)

llm_dict['Llama-3.2-1B-Instruct'] = HuggingFaceLocalChatModel(
        model="meta-llama/Llama-3.2-1B-Instruct",
        max_tokens=4000,
    )

llm_dict['Llama-3.1-405B-Instruct'] = FireworksChatModel(
        model="llama-v3p1-405b-instruct",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

llm_dict['Llama-3.1-70B-Instruct'] = FireworksChatModel(
        model="llama-v3p1-70b-instruct",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

llm_dict['Llama-3-70B-Instruct'] = FireworksChatModel(
        model="llama-v3-70b-instruct",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

llm_dict['Llama-4-Maverick-Instruct-Basic'] = FireworksChatModel(
        model="llama4-maverick-instruct-basic",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

llm_dict['Llama-4-Scout-Instruct-Basic'] = FireworksChatModel(
        model="llama4-scout-instruct-basic",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

# Maritaca AI API models
llm_dict['sabia-3'] = MaritacaChatModel(
        model="sabia-3",
        temperature=1,
        max_retries=10,
    )

llm_dict['sabiazinho-3'] = MaritacaChatModel(
        model="sabiazinho-3",
        temperature=1,
        max_retries=10,
    )

# Qwen 
llm_dict['QwQ-32B'] = FireworksChatModel(
        model="qwq-32b",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
