# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire
import math
import json
import pandas as pd
from tqdm import tqdm
import os
from llama.tokenizer import Tokenizer

from llama import Llama, Dialog


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    data = json.load(open("../data/test.json"))
    _trope = ['Big Bad', 'Jerkass', 'Faux Affably Evil', 'Smug Snake', 'Abusive Parents', 'Would Hurt a Child', 'Action Girl', 'Reasonable Authority Figure', 'Papa Wolf', 'Deadpan Snarker', 'Determinator', 'Only Sane Man', 'Anti-Hero', 'Asshole Victim', 'Jerk with a Heart of Gold', 'Even Evil Has Standards', 'Affably Evil', 'Too Dumb to Live', 'Butt-Monkey', 'Ax-Crazy', 'Adorkable', 'Berserk Button', 'Ms. Fanservice', 'The Alcoholic', 'Disappeared Dad', 'Would Hit a Girl', 'Oh, Crap!', 'Driven to Suicide', 'Adult Fear', 'Not So Different', 'Heroic BSoD', 'Big \\"NO!\\"', 'Eye Scream', 'Gory Discretion Shot', 'Impaled with Extreme Prejudice', 'Off with His Head!', 'Disney Villain Death', 'Your Cheating Heart', '\\"The Reason You Suck\\" Speech', 'Tempting Fate', 'Disproportionate Retribution', 'Badass Boast', 'Groin Attack', 'Roaring Rampage of Revenge', 'Big Damn Heroes', 'Heroic Sacrifice', "Screw This, I'm Outta Here!", 'Kick the Dog', 'Pet the Dog', 'Villainous Breakdown', 'Precision F-Strike', 'Cluster F-Bomb', 'Jerkass Has a Point', 'Idiot Ball', 'Batman Gambit', 'Police are Useless', 'The Dragon', 'Cool Car', 'Body Horror', 'The Reveal', 'Curb-Stomp Battle', 'Cassandra Truth', 'Blatant Lies', 'Crapsack World', 'Comically Missing the Point', 'Fanservice', 'Fan Disservice', 'Brick Joke', 'Hypocritical Humor', 'Does This Remind You of Anything?', 'Black Comedy', 'Irony', 'Exact Words', 'Stealth Pun', 'Bittersweet Ending', 'Karma Houdini', 'Downer Ending', 'Laser-Guided Karma', 'Earn Your Happy Ending', 'Karmic Death', 'Nice Job Breaking It, Hero!', 'My God, What Have I Done?', 'What the Hell, Hero?', 'Hope Spot', 'Heel Face Turn', 'Took a Level in Badass', "Chekhov's Gun", 'Foreshadowing', "Chekhov's Skill", "Chekhov's Gunman", 'Red Herring', 'Ironic Echo', 'Hoist by His Own Petard', 'Meaningful Echo', 'Freudian Excuse']
    tokenizer = Tokenizer(model_path="tokenizer.model")
    per_batch = 3
    times = math.ceil(len(_trope)/per_batch)
    path = "output"
    os.makedirs(path, exist_ok = True)
    for k, v in tqdm(data.items()):
        if os.path.exists(f"{path}/{k}.txt"):
            continue

        file = open(f"{path}/{k}.txt", "w")
        plot = json.load(open(f"../data/synopses/{k}.json"))['plot']  
        tmp = tokenizer.encode(plot,bos=True, eos=True)

        plot = tokenizer.decode(tmp[:1800])
        system_prompt = '''You are a trope detector, given a trope, answer 'yes' if the trope is relevant to the article, 'no' otherise. And provide a brief explanation for your answer.\n'''
        system_prompt += f'article: {plot}\n'
        for epoch in range(times):
            trope = _trope[epoch*per_batch:min((epoch+1)*per_batch, len(_trope))]
            dialogs: List[Dialog] = [
            ]

            for j in range(len(trope)):
                trp = trope[j]
                user_content = f"Is the trope '{trp}' related to the article?\n"
                prompt = [{"role": "system", "content":system_prompt},

                        {"role": "user", "content":user_content}]
                dialogs.append(prompt)

            results = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            for i, (dialog, result) in enumerate(zip(dialogs, results)):
                # for msg in dialog:
                #     print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                # print(
                #     f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                # )
                # print("\n==================================\n")
                file.writelines( f"{i+(epoch*per_batch)}).{result['generation']['content']}\n")
        file.close()  

if __name__ == "__main__":
    fire.Fire(main)
