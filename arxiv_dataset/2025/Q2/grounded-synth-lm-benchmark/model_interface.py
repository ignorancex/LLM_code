import os
import httpx
import openai
import asyncio
from openai import AsyncOpenAI
import time
import prompts

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def translate_remote(remote:str) -> tuple[str, str]:
    if ":" in remote:
        remote, port_num = remote.split(":")
        port_num = int(port_num)
    else:
        port_num = 8443
    if remote == "sierra":
        url=f"https://pn131285.nist.gov:{port_num}/v1"
        api_key = "sk-no-key-required"
    elif remote == "oscar":
        url=f"https://pn131274.nist.gov:{port_num}/v1"
        api_key = "sk-no-key-required"
    elif remote == "papa":
        url=f"https://pn131275.nist.gov:{port_num}/v1"
        api_key = "sk-no-key-required"
    elif remote == "echo":
        url=f"https://pn125915.nist.gov:{port_num}/v1"
        api_key = "sk-no-key-required"
    elif remote == "foxtrot":
        url=f"https://pn125916.nist.gov:{port_num}/v1"
        api_key = "sk-no-key-required"
    elif remote == "redwing":
        url=f"https://redwing.nist.gov:{port_num}/v1"
        api_key = "sk-no-key-required"
    elif remote == "rchat":
        url=f"https://rchat.nist.gov/api"
        api_key = os.environ.get("RCHAT_API_KEY")
    elif remote == "openai":
        url = "https://api.openai.com/v1"
        api_key = os.environ.get("OPENAI_API_KEY")
    else:
        url=f"https://{remote}.nist.gov:{port_num}/v1"
        api_key = "sk-no-key-required"

        
    
    if api_key is None:
        raise ValueError("API key is not set for remote: %s" % remote)
    if url is None:
        raise ValueError("URL is not set for remote: %s" % remote)
    
    return url, api_key


    


class SglModelSync:
    def __init__(self, model="meta-llama/Llama-3.3-70B-Instruct", remote:str='sierra'):
        if model is None:
            print("Model is not set, using default model: meta-llama/Llama-3.3-70B-Instruct")
            model = "meta-llama/Llama-3.3-70B-Instruct"
        self.model = model
        self.remote = remote
    
        self.url, self.api_key = translate_remote(remote)

        
    
        self.client = openai.OpenAI(
            base_url=self.url, 
            api_key=self.api_key,
            http_client=httpx.Client(verify=False)
        )
    
    def process_all_batches(self, model_prompts):
        """Process all batches sequentially, one prompt at a time"""

        total_start_time = time.time()
        results = []
        
        # Process prompts one by one
        for i, prompt in enumerate(model_prompts):
            current_time = time.strftime("%H:%M:%S")
            print(f"  ({current_time}) remote request {i+1}/{len(model_prompts)}")
            
            # Process the request synchronously
            start_time = time.time()
            try:
                if self.model == "o1-mini" or self.model == "o1-preview":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=4096,
                        temperature=0.7,
                        stream=False
                    )
                elif self.model == 'o3-mini' or self.model == 'o4-mini':
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=4096,
                        stream=False,
                        temperature=0.7,
                        # service_tier="flex",
                    )
                else:
                    messages = [
                        {"role": "system", "content": prompts.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                    
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=2048,
                        temperature=0.7,
                        stream=False
                    )
                
                elapsed = time.time() - start_time
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                # print("********* <RESPONSE> *********")
                # print(response)
                # print("********* <CONTENT> *********")
                # print(response.choices[0].message.content)
                # print("********* </CONTENT> *********")
                # print("********* </RESPONSE> *********")
                
                result = {
                    "request_id": i,
                    "content": response.choices[0].message.content,
                    "error": None,
                    "elapsed_time": elapsed,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                }

                if response.choices[0].message.content.isspace():
                    result['error'] = "Empty response"
            except Exception as e:
                # optional handling here, but for now nothing to do
                raise e
            
            results.append(result)
        
        total_time = time.time() - total_start_time
        
        return results, total_time
    
    
    def generate(self, model_prompts: list[str]):
        print("Remote model generating %d prompts (synchronously)" % len(model_prompts))
        return self.process_all_batches(model_prompts)



class SglModelAsync:
    def __init__(self, model="meta-llama/Llama-3.3-70B-Instruct", remote:str='sierra'):
        if model is None:
            print("Model is not set, using default model: meta-llama/Llama-3.3-70B-Instruct")
            model = "meta-llama/Llama-3.3-70B-Instruct"
        self.model = model
        self.remote = remote
        self.url, self.api_key = translate_remote(remote)

        self.connection_parallelism = 64
        if 'openai' in self.url:
            print("OpenAI detected, setting connection parallelism to 4")
            self.connection_parallelism = 4

        print(f"Using model: {self.model} on remote: {self.remote} at {self.url}")

        self.client = openai.AsyncOpenAI(
            base_url=self.url, 
            api_key=self.api_key,
            http_client=httpx.AsyncClient(verify=False)
        )


        
    @staticmethod
    async def generate_text_async(model, client, prompt, request_id):
        start_time = time.time()
        try:
            
            if model == "o1-mini" or model == "o1-preview":
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=4096,
                    temperature=0.7,
                    stream=False
                )
            elif model == 'o3-mini' or model == 'o4-mini':
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=4096,
                    temperature=0.7,
                    stream=False,
                    # service_tier="flex",
                )
            else:
                messages = [
                    {"role": "system", "content": prompts.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
                
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.7,
                    stream=False
                )
            
            elapsed = time.time() - start_time
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            
            result = {
                "request_id": request_id,
                "content": response.choices[0].message.content,
                "error": None,
                "elapsed_time": elapsed,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                # "prompt": prompt
            }
            if response.choices[0].message.content.isspace():
                result['error'] = "Empty response"
            return result
        except Exception as e:
            # optional handling here, but for now nothing to do
            result = {
                "request_id": request_id,
                "content": None,
                "error": str(e),
                "elapsed_time": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
            # return result
            raise e
        

    async def process_all_batches(self, model_prompts):
        """Process all batches with a single client instance, in chunks of 64"""

        total_start_time = time.time()
        results = []
        
        # Process prompts in batches
        batch_size = self.connection_parallelism
        for i in range(0, len(model_prompts), batch_size):
            batch_prompts = model_prompts[i:i+batch_size]
            current_time = time.strftime("%H:%M:%S")
            print(f"  ({current_time}) remote batch {i//batch_size + 1}/{(len(model_prompts)-1)//batch_size + 1} ({len(batch_prompts)} prompts per batch)")
            
            # Create tasks for this batch with the shared client
            batch_tasks = [
                self.generate_text_async(self.model, self.client, prompt, i+j) 
                for j, prompt in enumerate(batch_prompts)
            ]

            # Execute this batch concurrently and gather results
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        total_time = time.time() - total_start_time
        
        return results, total_time
    
    
    def generate(self, model_prompts: list[str]):
        print("Remote model generating %d prompts (asynchronously)" % len(model_prompts))
        return asyncio.run(self.process_all_batches(model_prompts))
    

    


# Example usage
if __name__ == "__main__":
    async def main():
        model = SglModelAsync()
        import json
        # load the squadv2 dataset subset
        with open('squadv2_subset.json', 'r') as f:
            dataset = json.load(f)
        dataset = dataset[:32]

        contexts = [item['context'] for item in dataset]

        # Async usage
        results, total_time = await model.generate_async(contexts)
        
        # Or synchronous usage
        # results, total_time = model.generate(contexts)
        
        res = results[0]
        print(res)
        print(f"in total took: {total_time} seconds")
        print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")

    asyncio.run(main())
    
    