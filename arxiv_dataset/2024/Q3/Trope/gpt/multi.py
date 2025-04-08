import json
import openai, time
from tqdm import tqdm
import os
import ast


def multi(args, system_prompt):
    openai.api_key = args.openai_key
    data_path = args.data
    file_path = args.output
    os.makedirs(file_path, exist_ok = True)
    file = open(f"error.txt", "w")
    data = json.load(open(f"{data_path}/test.json"))
    for INDEX, (k, v) in enumerate(tqdm(data.items())):
        if os.path.exists(f"{file_path}/{k}.json"):
            continue
        print("article:",k)
        plot = json.load(open(f"{data_path}/synopses_segment/{k}.json"))['plot']
        user_content = 'Article:\n'
        for i in range(len(plot)):
            user_content += f"{i}. {plot[i]}\n"
        user_content += "Strictly select only the tropes related to the article from the TropeList mentioned above, and feel free to pick multiple tropes if they are relevant"

        prompt = [{"role": "system", "content":system_prompt},
                {"role": "user", "content":user_content}]
        cnt = 0
        exceed = False
        while True and not exceed:
            try:
                response = openai.ChatCompletion.create(
                    model= args.model,
                    messages= prompt, 
                    temperature= args.temperature,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                print(response)
                print(response.choices[0].message.content)
                try:
                    output = ast.literal_eval(response.choices[0].message.content)
                    with open(f"{file_path}/{k}.json", 'w') as fp:
                        json.dump( output , fp, indent=4) 
                except Exception as e: 
                    print("0. ", e)
                    file.write(f"{k}:\n")
                    file.write(f"{response.choices[0].message.content}\n")
                
                break
            except Exception as e:
                print(e) # Print out handled error
                if "Request too large for gpt-4" in str(e):
                    exceed = True
                if "This model's maximum context length is 4097 tokens." in str(e):
                    exceed = True
                if "This model's maximum context length is 8192 tokens." in str(e):
                    exceed = True
                cnt += 1
                print( "SLEEP: ", cnt)
                time.sleep(5)
    file.close()
