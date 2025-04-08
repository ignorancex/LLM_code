import json
import pandas as pd
import openai, time
from tqdm import tqdm
import os, ast

trope = ['Big Bad', 'Jerkass', 'Faux Affably Evil', 'Smug Snake', 'Abusive Parents', 'Would Hurt a Child', 'Action Girl', 'Reasonable Authority Figure', 'Papa Wolf', 'Deadpan Snarker', 'Determinator', 'Only Sane Man', 'Anti-Hero', 'Asshole Victim', 'Jerk with a Heart of Gold', 'Even Evil Has Standards', 'Affably Evil', 'Too Dumb to Live', 'Butt-Monkey', 'Ax-Crazy', 'Adorkable', 'Berserk Button', 'Ms. Fanservice', 'The Alcoholic', 'Disappeared Dad', 'Would Hit a Girl', 'Oh, Crap!', 'Driven to Suicide', 'Adult Fear', 'Not So Different', 'Heroic BSoD', 'Eye Scream', 'Gory Discretion Shot', 'Impaled with Extreme Prejudice', 'Off with His Head!', 'Disney Villain Death', 'Your Cheating Heart', 'Tempting Fate', 'Disproportionate Retribution', 'Badass Boast', 'Groin Attack', 'Roaring Rampage of Revenge', 'Big Damn Heroes', 'Heroic Sacrifice', "Screw This, I'm Outta Here!", 'Kick the Dog', 'Pet the Dog', 'Villainous Breakdown', 'Precision F-Strike', 'Cluster F-Bomb', 'Jerkass Has a Point', 'Idiot Ball', 'Batman Gambit', 'Police are Useless', 'The Dragon', 'Cool Car', 'Body Horror', 'The Reveal', 'Curb-Stomp Battle', 'Cassandra Truth', 'Blatant Lies', 'Crapsack World', 'Comically Missing the Point', 'Fanservice', 'Fan Disservice', 'Brick Joke', 'Hypocritical Humor', 'Does This Remind You of Anything?', 'Black Comedy', 'Irony', 'Exact Words', 'Stealth Pun', 'Bittersweet Ending', 'Karma Houdini', 'Downer Ending', 'Laser-Guided Karma', 'Earn Your Happy Ending', 'Karmic Death', 'Nice Job Breaking It, Hero!', 'My God, What Have I Done?', 'What the Hell, Hero?', 'Hope Spot', 'Heel Face Turn', 'Took a Level in Badass', "Chekhov's Gun", 'Foreshadowing', "Chekhov's Skill", "Chekhov's Gunman", 'Red Herring', 'Ironic Echo', 'Hoist by His Own Petard', 'Meaningful Echo', 'Freudian Excuse', 'Big \\"NO!\\"', '\\"The Reason You Suck\\" Speech']
# subset trope:
# trope = ['Kick the Dog','Big Bad','What the Hell, Hero?', "Chekhov's Gunman", 'Irony', 'Determinator', "Screw This, I'm Outta Here!",'Adorkable', 'Adult Fear', 'Too Dumb to Live', 'Deadpan Snarker', 'Would Hurt a Child','Eye Scream','Stealth Pun', 'Not So Different','Bittersweet Ending','Only Sane Man','Smug Snake','Red Herring','Blatant Lies']


def cot(args, system_prompt):
    openai.api_key = args.openai_key
    data_path = args.data
    file_path = args.output
    file = open(f"error.txt", "w")
    data = json.load(open(f"{data_path}/test.json"))
    os.makedirs(file_path, exist_ok = True)

    for INDEX, (k, v) in enumerate(tqdm(data.items())):
        print("article:",k)
        plot = json.load(open(f"{data_path}/synopses_segment/{k}.json"))['plot']
        
        data = []
        if os.path.exists(f"{file_path}/{k}.json"):
            data = json.load(open(f"{file_path}/{k}.json"))
        
        done = set()
        if len(data):
            for d in data:
                done.add(d['Trope'])

        exceed = False
        for trp in (trope):
            if trp in done:
                continue
            content = ""
            for i in range(len(plot)):
                content += f"{i}. {plot[i]}\n"

            user_content = f"Article:\n{content}\n"
            user_content += f'Is the trope "{trp}" in the article?\n'

            prompt = [{"role": "system", "content":system_prompt},
                    {"role": "user", "content":user_content}]
            cnt = 0
            while True and not exceed:
                try:
                    response = openai.ChatCompletion.create(
                    model= args.model,
                    messages= prompt, 
                    temperature= args.temperature,
                    frequency_penalty=0,
                    presence_penalty=0
                    )
                    print("idx:", INDEX)
                    print(response.choices[0].message.content)
                    try:
                        output = response.choices[0].message.content
                        output = ast.literal_eval(output)
                        data.append(output)
                        with open(f"{file_path}/{k}.json", 'w') as fp:
                            json.dump( [ trp for trp in data], fp, indent=4) 
                    except Exception as e: 
                        print("0. ", e)
                        file.write(f"{k}:\n")
                        file.write(f"{response.choices[0].message.content}\n")
                    break
                except Exception as e: 
                    print("1. ", e) # Print out handled error
                    if "This model's maximum context length is 4097 tokens." in str(e):
                        exceed = True
                    if "This model's maximum context length is 8192 tokens." in str(e):
                        exceed = True
                    cnt += 1
                    print( "SLEEP: ", cnt)
                    time.sleep(4)
    file.close()



def base(args, system_prompt):

    openai.api_key = args.openai_key
    data_path = args.data
    path = args.output
    file = open(f"error.txt", "w")
    data = json.load(open(f"{data_path}/test.json"))
    os.makedirs(path, exist_ok = True)

    for INDEX, (k, v) in enumerate(tqdm(data.items())):
        print("article:",k)
        if os.path.exists(f"{path}/{k}.txt"):
            continue
        plot = json.load(open(f"{data_path}/synopses/{k}.json"))['plot']
        total_cnt = 0
        file_content = ""
        if os.path.exists(f"{path}/{k}.txt"):
            with open(f'{path}/{k}.txt', 'r+') as file:
                file_content = file.read()
        done = set()

        if file_content != "":
            for line in file_content.split("\n"):
                i = line.split(".")
                if len(i) >= 2 and i[0].isnumeric() and ('yes' in i[1].lower() or 'no' in i[1].lower()):
                    if int(i[0]) < 95:
                        done.add(trope[int(i[0])])
        if len(done) == len(trope):
            break

        file = open(f"{path}/{k}.txt", "w")
        file.write(file_content)
        file.writelines("\n")

        system_prompt += f'article: {plot}\n'
        user_content = ""
        for j in range(len(trope)):
            trp = trope[j]
            if trp in done:
                continue
            user_content += f"{j}. Is the trope '{trp}' related to the article?\n"

        prompt = [{"role": "system", "content":system_prompt},
                {"role": "user", "content":user_content}]
        cnt = 0
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model= args.model,
                    messages= prompt, 
                    temperature= args.temperature,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                print(response.choices[0].message.content)
                file.writelines( response.choices[0].message.content+ "\n")
                break
            except Exception as e:
                print(e) # Print out handled error
                if "This model's maximum context length is 8192 tokens." in str(e):
                    break
                if "Request too large for gpt-4 in organization" in str(e):
                    break
                cnt += 1
                print(  "SLEEP: ", cnt)
                time.sleep(1)

        file.close()             


