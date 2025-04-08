import json
import pandas as pd
import openai, time
from tqdm import tqdm, trange
from collections import defaultdict
import os
import ast




openai.api_key = ""

attacks = {
    # Friday
    "Kick the Dog":['He and Jason are about to leave when they witness Kate and Glen arguing; when Glen shoves Kate to the ground, Rusty punches him in the face and flees with Kate and Jason.'],
#  AnAllDogsChristmasCarol
  "Big Bad":["During the tour around the facility, Walter sneaks into Kermit the Frog's office and discovers Statler and Waldorf selling the theatre to Tex Richman (Chris Cooper), an oil magnate, and his associates Bobo the Bear and Uncle Deadly. Once Statler and Waldorf leave, Walter learns of Tex's true intentions: to tear down the Muppet Studios and drill underneath for oil. Walter explains the situation to Gary and Mary, and the three track down Kermit at his mansion."],
    "Bittersweet Ending":['The film ends with Pauline being arrested after protesting and refusing to continue but before she can be put on trial, she is captured by the resurgent British Resistance and agrees to work for them as they fight to liberate the country with the help of arriving American troops.']}


_plot = {
"Big Bad":['MoonOverMiami', 'StarterFor10', 'Secretariat', 'CantHardlyWait', 'MarleyAndMe', 'Evita', 'WeWereSoldiers', 'TheChristmasToy', 'ShatteredGlass', 'StuckInLove', 'NightMoves', 'TheDevilsArithmetic', 'QuickChange', 'Ararat', 'WhistleDownTheWind', 'FindingNeverland', 'BlackSnakeMoan', 'Poseidon', 'SnowCake', 'BlackSea', 'HeadOfState', 'Twister', 'ItHappenedHere', 'TheIncredibleShrinkingMan', 'TheAmericanPresident', 'W', 'Goodfellas', 'RabbitFire', 'CoolRunnings', 'HeartsInAtlantis'],   
    "Blatant Lies":[ 'CantHardlyWait' ,'TheShallows', 'FatalAttraction', 'JesusChristVampireHunter', 'AnAllDogsChristmasCarol', 'HigherLearning', 'CloakAndDagger', 'WhistleDownTheWind', 'TheAngelsShare', 'BartonFink', 'DeadAndDeader', 'NightMoves', 'QuickChange', 'BlackSnakeMoan', 'FromBeyond', 'Sleepwalkers', 'TheAdventuresOfPlutoNash', 'StreetsOfFire', 'FindingNeverland', 'TheTenCommandments', 'CoolRunnings', 'Evita', 'WeWereSoldiers', 'MissionImpossibleII', 'Poseidon', 'Goodfellas', 'NancyDrew', 'InvasionOfTheBodySnatchers', 'SnowCake', 'TheIncredibleShrinkingMan'],
    "Kick the Dog":['CloakAndDagger', 'MarleyAndMe', 'FuneralInBerlin', 'RabbitFire', 'LakePlacid3', 'Horsemen', 'Ararat', 'CoolRunnings', 'Poseidon', 'Evita', 'NightMoves', 'ScaryMovie', 'ShatteredGlass', 'HeadOfState', 'WeWereSoldiers', 'SnowCake', 'LittleRuralRidingHood', 'WhistleDownTheWind', 'TheAmericanPresident', 'TheIncredibleShrinkingMan', 'W', 'TheChristmasToy', 'DeadSpaceDownfall', 'FromBeyond', 'BartonFink', 'QuickChange', 'FindingNeverland', 'Twister', 'InvasionOfTheBodySnatchers', 'NancyDrew']
}


system_prompt = '''You are a trope detector, tasked with identifying the presence or absence of a specific trope in an article. You will be provided with an article and a trope to detect. 
Your task is to generate a JSON object with the following keys:
Trope: the name of the trope provided
Definition: Concisely and briefly explain the meaning of the trope
Thought: The reason why the trope is depicted in the plot with reasoning, evidence, and relevant paragraphs
Answer: "yes" if the trope is in the article, and "no" if it is not.
Here is an example provided:
Article:
0, "Le Boucher" (The Butcher) starts with a wedding ceremony in which Paul, a war veteran , encounters Helene, a school teacher.
1, Paul has taken his father's business and owns a butchery in the village.
2, As the ceremony progressed, both seemed to enjoy each other's company while they ate and chatted merrily.
3, That was a happy beginning.
4, Afterwards, the general atmosphere of the film changed.
5, We could feel an eeriness, a sort of awkward silence and emptiness as they walked out of the celebrations.
6, Paul accompanied Helene home and they got separated with a sense of sadness, as if they wanted to stay together.
7, The next day, Paul visited Helene in her classroom and brought a piece of meat, well wrapped, like a flower bouquet.
8, He promised to bring more.
9, Helene enjoyed Popaul's company and could do anything for him.
10, Their relationship deepened, until some unexplained murders started to occur in the village...
Query: Is the trope "Chekhov's Gun" in the article?
Answer:
{
    "Trope": "Chekhov's Gun",
    "Definition": "Chekhov's Gun is a storytelling rule that says if you introduce something in a story, it should have a purpose and be used later.",
    "Thought": [
        {
            "Reasoning": "From paragraph 7, the meat, initially presented as a gift, might hold a crucial role or link to the unexplained murders mentioned in paragraph 10.”,
            "Evidence":  Paul surprises Helene with a meat gift, promising more. Their relationship deepens while unexplained murders disturb the village peace.
            "Relevant Paragraphs": "7, 10"
        }
       ],
    "Answer": yes
}
'''
attack_trp = [ "Big Bad", "Blatant Lies", "Kick the Dog"]
file = open(f"error.txt", "w")
file_path = "output"
gt = json.load(open("../test.json"))
os.makedirs(file_path, exist_ok = True)

for trp in attack_trp:
    for index, (k,v) in enumerate(gt.items()):
        if k not in _plot[trp]:
            continue
        print("article", k)
        data = []
        if os.path.exists(f"{file_path}/{k}.json"):
            data = json.load(open(f"{file_path}/{k}.json"))
        
        done = set()
        if len(data):
            for d in data:
                done.add(d['Trope'])
        if trp in done:
            continue
        content = ""
        attack = attacks[trp]
        plot = json.load(open(f"../synopses_segment/{k}.json"))['plot']
        attack_paragraph = []

        for i in range(len(plot)):
            content += f"{i}. {plot[i]}\n"
        for i in range(len(attack)):
            attack_paragraph.append(str(i+len(plot)))
            content += f"{i+len(plot)}. {attack[i]}\n"
        attack_paragraph = ', '.join(attack_paragraph)

        user_content = f"Article:\n{content}\n"
        user_content += f"Is the trope '{trp}' related to the article?\n"
        prompt = [{"role": "system", "content":system_prompt},
                {"role": "user", "content":user_content}]
        cnt = 0
        exceed = False
        while True and not exceed:
            try:
                response = openai.ChatCompletion.create(
                    model='gpt-4',
                    messages= prompt, 
                    temperature=0,
                )
                print(response)
                print(response.choices[0].message.content)
                try:
                    output = response.choices[0].message.content
                    output = ast.literal_eval(output)
                    output['Attack_Paragraphs'] = attack_paragraph
                    data.append(output)
                    with open(f"{file_path}/{k}.json", 'w') as fp:
                        json.dump( [ trp for trp in data], fp, indent=4) 
                except Exception as e: 
                    print("0. ", e)
                    file.write(f"{k}:\n")
                    file.write(f"{response.choices[0].message.content}\n")
                    file.write(f'"Attack_Paragraphs":{attack_paragraph}')
                break
            except Exception as e: 
                print("1. ", e) # Print out handled error
                # if "This model's maximum context length is 4097 tokens." in str(e):
                if "This model's maximum context length is 8192 tokens." in str(e):
                    exceed = True
                cnt += 1
                print( "SLEEP: ", cnt)
                time.sleep(1)