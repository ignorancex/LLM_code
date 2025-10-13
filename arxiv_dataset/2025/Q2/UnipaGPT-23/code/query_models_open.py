import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, set_seed
from huggingface_hub import login
from peft import PeftModel
from tqdm import tqdm
import argparse
import json
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--llm_model', type=str, default='swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA')

    args = parser.parse_args()
    model_name = args.llm_model
    device = args.device

    model_name_short = {
        'meta-llama/Llama-2-7b-hf': 'Llama-2',
        'meta-llama/Llama-2-7b-chat-hf' : 'Llama-2-chat',
        'meta-llama/Meta-Llama-3-8B' : 'Llama-3',
        'meta-llama/Meta-Llama-3-8B-Instruct' : 'Llama-3-instruct',
        'sapienzanlp/Minerva-3B-base-v1.0' : 'Minerva-3B',
        'swap-uniba/LLaMAntino-2-7b-hf-ITA' : 'Llamantino-2',
        'swap-uniba/LLaMAntino-2-chat-7b-hf-ITA' : 'Llamantino-2-chat',
        'swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA' : 'Anita-3',
        'teelinsan/camoscio-7b-llama': 'Camoscio',
        'andreabac3/Fauno-Italian-LLM-7B' : 'Fauno'}

	path = 'unipa-gpt/' 	# path till main folder

    with open(path+'corpora/qa_retrieved.json', 'r') as f:
        fixed_qa_retrieved = json.load(f)

    behaviour = """Sono Unipa-GPT, chatbot e assistente virtuale dell'Università degli Studi di Palermo che risponde cordialmente e in forma colloquiale.\nAi saluti, rispondi salutando e presentandoti.\nRicordati che il rettore dell'Università è il professore Massimo Midiri.\nSe la domanda riguarda l'università degli studi di Palermo, rispondi in base alle informazioni e riporta i link ad esse associate;\nSe non sai rispondere alla domanda, rispondi dicendo che sei un'intelligenza artificiale che ha ancora molto da imparare e suggerisci di andare su https://www.unipa.it/, non inventare risposte.\n"""
   
    if model_name == 'andreabac3/Fauno-Italian-LLM-7B':
        template = f'[|AI|] {behaviour}\n'
        t2 = "{context}\n[|Umano|] {question}\n[|AI|] "
    elif model_name == 'meta-llama/Meta-Llama-3-8B-Instruct' or model_name == 'swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA' or model_name == 'meta-llama/Llama-2-7b-chat-hf' or model_name == 'swap-uniba/LLaMAntino-2-chat-7b-hf-ITA':
        template = {"role": "system", "content": behaviour}
        raw_t2 = {"role": "user", "content":  "Domanda: {question}\nInformazioni: {context}\n"}
    else:
        template = "### Istruzione:\n" + behaviour
        t2 = "### Domanda:\n{question}\n### Informazioni:\n{context}\n### Risposta:\n"


    if model_name == 'andreabac3/Fauno-Italian-LLM-7B' or model_name == 'teelinsan/camoscio-7b-llama':
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Llama-2-7b-hf',
            max_length=3000,
            #max_new_tokens=256,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, model_name)
    else:
        hf_auth=''
        login(token = hf_auth)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token = hf_auth)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            max_length=3000,
            trust_remote_code=True,
            token = hf_auth)
    
    model.to(device)
    model.eval()
    set_seed(17)

    model_answers = []

    os.makedirs(path+'results', exist_ok=True)
    with open(path+'results/qa_'+model_name_short[model_name]+'.txt', 'w') as f:
        f.write('')

    print(f'inference with {model_name_short[model_name]} model')

    for i in tqdm(range(len(fixed_qa_retrieved))):
        context = ''
        for doc in fixed_qa_retrieved[i][1]:
            context += doc['page_content'] + '\n'

        if model_name == 'swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA' or model_name == 'meta-llama/Meta-Llama-3-8B-Instruct' or model_name == 'meta-llama/Llama-2-7b-chat-hf' or model_name == 'swap-uniba/LLaMAntino-2-chat-7b-hf-ITA':
            print(fixed_qa_retrieved[i][0])
            t2 = copy.deepcopy(raw_t2)
            t2['content'] = t2['content'].format(question=fixed_qa_retrieved[i][0], context=context)
            query = t2['content']
            messages = [template, t2]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids=input_tokens['input_ids'].to(device)
            attention_mask=input_tokens['attention_mask'].to(device)
            generation_config = GenerationConfig(
                do_sample=True,
                max_new_tokens=256,
                temperature=0.001,
                top_p=0.92,
                top_k=0,
                repetition_penalty=1.5)
            response = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config)

            try:
                output = tokenizer.decode(response[0], skip_special_tokens=False).split(prompt)[1]
            except:
                try:
                    output = tokenizer.decode(response[0], skip_special_tokens=True).split(prompt)[1]
                except:
                    output = tokenizer.decode(response[0], skip_special_tokens=True)
                        
            print(output)
            with open(path+'results/qa_'+model_name_short[model_name]+'.txt', 'a') as f:
                f.write(query+'\n')
                f.write(output+'\n')
                f.write('-----------------------\n')

            model_answers.append((template, query, output))

        else:
            print(fixed_qa_retrieved[i][0])
            query = t2.format(question=fixed_qa_retrieved[i][0], context=context)
            textual_input = template + query
            input_tokens = tokenizer(textual_input, return_tensors='pt')
            input_ids=input_tokens['input_ids'].to(device)
            generation_config = GenerationConfig(
                do_sample=True,
                max_new_tokens=256,
                temperature=0.001,
                top_p=0.92,
                top_k=0,
                repetition_penalty=1.5)
            response = model.generate(
                input_ids=input_ids,
                generation_config=generation_config)

            try:
                output = tokenizer.decode(response[0], skip_special_tokens=True).split(prompt)[1]
            except:
                try:
                    output = tokenizer.decode(response[0], skip_special_tokens=False).split(prompt)[1]    
                except:
                    output = tokenizer.decode(response[0], skip_special_tokens=True)
                        
            print(output)
        
            with open(path+'results/qa_'+model_name_short[model_name]+'.txt', 'a') as f:
                f.write(query+'\n')
                f.write(output+'\n')
                f.write('-----------------------\n')

            model_answers.append((template, query, output))
    

    with open(path+'results/qa_'+model_name_short[model_name]+'.json', 'w') as f:
        json.dump(model_answers, f, indent=4)
