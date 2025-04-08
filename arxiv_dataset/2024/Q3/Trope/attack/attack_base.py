import json
import openai, time
from tqdm import tqdm
import os

openai.api_key = ""
data_path = "../data"

data = json.load(open(f"{data_path}/test.json"))
path = "output"
os.makedirs(path, exist_ok = True)
trope = ['Kick the Dog','Big Bad','What the Hell, Hero?', "Chekhov's Gunman", 'Irony', 'Determinator', "Screw This, I'm Outta Here!",'Adorkable', 'Adult Fear', 'Too Dumb to Live', 'Deadpan Snarker', 'Would Hurt a Child','Eye Scream','Stealth Pun', 'Not So Different','Bittersweet Ending','Only Sane Man','Smug Snake','Red Herring','Blatant Lies', 'Freudian Excuse', 'Abusive Parents', 'Disappeared Dad', 'Cluster F-Bomb', 'Off with His Head!', 'Anti-Hero', 'The Alcoholic', 'The Reveal' , 'Fan Disservice' , 'Crapsack World']

attacks = {
    # Plot: Friday
    "Kick the Dog":['He and Jason are about to leave when they witness Kate and Glen arguing; when Glen shoves Kate to the ground, Rusty punches him in the face and flees with Kate and Jason.'],
    # Plot: AnAllDogsChristmasCarol
    "Big Bad":["During the tour around the facility, Walter sneaks into Kermit the Frog's office and discovers Statler and Waldorf selling the theatre to Tex Richman (Chris Cooper), an oil magnate, and his associates Bobo the Bear and Uncle Deadly. Once Statler and Waldorf leave, Walter learns of Tex's true intentions: to tear down the Muppet Studios and drill underneath for oil. Walter explains the situation to Gary and Mary, and the three track down Kermit at his mansion."],
    # Plot: TheBirds
    "Blatant Lies":['Not wanting Mitch to realize the lengths to which she went to get his attention, she lies and tells him that her primary reason for coming to Bodega Bay was to visit Annie, a friend of hers from school.']
}

_plot = {
    "Big Bad":['MoonOverMiami', 'StarterFor10', 'Secretariat', 'CantHardlyWait', 'MarleyAndMe', 'Evita', 'WeWereSoldiers', 'TheChristmasToy', 'ShatteredGlass', 'StuckInLove', 'NightMoves', 'TheDevilsArithmetic', 'QuickChange', 'Ararat', 'WhistleDownTheWind', 'FindingNeverland', 'BlackSnakeMoan', 'Poseidon', 'SnowCake', 'BlackSea', 'HeadOfState', 'Twister', 'ItHappenedHere', 'TheIncredibleShrinkingMan', 'TheAmericanPresident', 'W', 'Goodfellas', 'RabbitFire', 'CoolRunnings', 'HeartsInAtlantis'],   
    "Blatant Lies":['CloakAndDagger', 'StarterFor10', 'DeadAndDeader', 'JesusChristVampireHunter', 'CantHardlyWait', 'PoolhallJunkies', 'QuickChange', 'AnAllDogsChristmasCarol', 'Stagecoach', 'TheAdventuresOfPlutoNash', 'TheDevilsArithmetic', 'NancyDrew', 'LakePlacid3', 'MissionImpossibleII', 'MarleyAndMe', 'DeadSpaceDownfall', 'CoolRunnings', 'SmallTimeCrooks', 'Popeye', 'StarTrekTheWrathOfKhan', 'ANightmareOnElmStreet3DreamWarriors', 'WeWereSoldiers', 'SnowCake', 'StarTrekGenerations', 'Anaconda', 'TheIncredibleShrinkingMan', 'Secretariat', 'LittleRuralRidingHood', 'TheAngelsShare', 'AvatarTheLastAirbender'],    "Kick the Dog":['CloakAndDagger', 'MarleyAndMe', 'FuneralInBerlin', 'RabbitFire', 'LakePlacid3', 'Horsemen', 'Ararat', 'CoolRunnings', 'Poseidon', 'Evita', 'NightMoves', 'ScaryMovie', 'ShatteredGlass', 'HeadOfState', 'WeWereSoldiers', 'SnowCake', 'LittleRuralRidingHood', 'WhistleDownTheWind', 'TheAmericanPresident', 'TheIncredibleShrinkingMan', 'W', 'TheChristmasToy', 'DeadSpaceDownfall', 'FromBeyond', 'BartonFink', 'QuickChange', 'FindingNeverland', 'Twister', 'InvasionOfTheBodySnatchers', 'NancyDrew'],
    "Kick the Dog":['CloakAndDagger', 'MarleyAndMe', 'FuneralInBerlin', 'RabbitFire', 'LakePlacid3', 'Horsemen', 'Ararat', 'CoolRunnings', 'Poseidon', 'Evita', 'NightMoves', 'ScaryMovie', 'ShatteredGlass', 'HeadOfState', 'WeWereSoldiers', 'SnowCake', 'LittleRuralRidingHood', 'WhistleDownTheWind', 'TheAmericanPresident', 'TheIncredibleShrinkingMan', 'W', 'TheChristmasToy', 'DeadSpaceDownfall', 'FromBeyond', 'BartonFink', 'QuickChange', 'FindingNeverland', 'Twister', 'InvasionOfTheBodySnatchers', 'NancyDrew']
}

attack_trp = [ "Big Bad", "Blatant Lies", "Kick the Dog"]

for trp in attack_trp:
    for INDEX, (k, v) in enumerate(tqdm(data.items())):
        if k not in _plot[trp]:
            continue

        plot = json.load(open(f"{data_path}/synopses/{k}.json"))['plot']

        for sentence in attacks[trp]:
            plot += sentence

        total_cnt = 0

        while True:
            total_cnt += 1
            if total_cnt > 1:
                break
            file_content = ""
            if os.path.exists(f"{path}/{k}.txt"):
                with open(f'{path}/{k}.txt', 'r+') as file:
                    # Read the content of the input file
                    file_content = file.read()
            done = set()

            if file_content != "":
                for line in file_content.split("\n"):
                    i = line.split(".")
                    if len(i) >= 2 and i[0].isnumeric() and ('yes' in i[1].lower() or 'no' in i[1].lower()):
                        done.add(trope[int(i[0])])
            if len(done) == len(trope):
                break

            file = open(f"{path}/{k}.txt", "w")
            file.write(file_content)
            file.writelines("\n")

            system_prompt = '''You are a trope detector, given a trope, answer 'yes' if the trope is relevant to the article, 'no' otherise. And provide a brief explanation for your answer.\n'''
            system_prompt += f'article: {plot}\n'
            user_content = ""

            for j in range(len(trope)):
                if trope[j] != trp:
                    continue
                user_content += f"Is the trope '{trp}' related to the article?\n"

            prompt = [{"role": "system", "content":system_prompt},
                    {"role": "user", "content":user_content}]
            cnt = 0
            while True:
                try:
                    response = openai.ChatCompletion.create(
                        model='gpt-4',
                        messages= prompt, 
                        temperature=0,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    # print(response)
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