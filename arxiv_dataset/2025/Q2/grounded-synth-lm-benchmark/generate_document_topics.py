import time

from model_interface import SglModelAsync, SglModelSync
import prompts
import answer_parser



def build_topic_prompt(context:str):
    # By the end of the list, topics should require a very deep understanding of the content by a professional domain expert. 
    # prompt = prompts.TOPIC_EXTRACT_PROMPT.format(context=context)
    prompt = prompts.TOPIC_EXTRACT_PROMPT.format(context=context)
    
    return prompt

def extract_topics_per_context(contexts: list[str], remote:str, model:str = None, async_flag:bool=False) -> list[list[str]]:

    if async_flag:
        model = SglModelAsync(remote=remote, model=model)
    else:
        model = SglModelSync(remote=remote, model=model)

    start_time = time.time()
    prompts = [build_topic_prompt(c) for c in contexts]
    
    results, _ = model.generate(prompts)
    total_time = time.time() - start_time
    topic_extraction_responses = []

    total_input_tokens = sum([res['input_tokens'] for res in results])
    total_output_tokens = sum([res['output_tokens'] for res in results])
    total_tokens = sum([res['total_tokens'] for res in results])
    print(f"total input tokens: {total_input_tokens}")
    print(f"total output tokens: {total_output_tokens}")
    print(f"total tokens: {total_tokens}")
    print(f"total toks/sec: {total_tokens / total_time}")
    print(f"total output toks/sec: {total_output_tokens / total_time}")
    
    # Parse the topics from the model response
    topics_list = []
    for i in range(len(contexts)):
        if results[i]['error'] is not None:
            print(results[i]['error'])
            exit(1)
        
        vals = answer_parser.parse_topic_extraction(results[i]['content'])
        topic_extraction_responses.append(results[i]['content'])
        topics_list.append(vals)

    return topics_list, topic_extraction_responses





if __name__ == '__main__':
   
    start_time = time.time()

    dataset = './source_data/Current Solutions and Future Trends for Robotic Prosthetic Hands.txt'
    with open(dataset, 'r') as f:
        context = f.read()

    topics = extract_topics_per_context([context], 'sierra')
    print(topics)
    