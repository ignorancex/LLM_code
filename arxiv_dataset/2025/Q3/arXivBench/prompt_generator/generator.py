from openai import OpenAI
import re
from tqdm import tqdm
import pandas as pd 
import os

class PromptGenerator:
    def __init__(self, openai_client):
        assert(openai_client != None)
        self.client = openai_client

    def generate_via_loop(self, subject, num_prompts, step_size=20, enable_tqdm=False, outputfile_path=None):
        all_prompts = set()
        while len(all_prompts) < num_prompts:
            print("start a new loop:")
            for i in tqdm(range(len(all_prompts), num_prompts, step_size), disable=(not enable_tqdm)):
                if i + step_size > num_prompts:
                    batch_size = num_prompts - i
                else:
                    batch_size = step_size

                generated_prompts = self.generate(subject=subject, num_prompts=batch_size)

                if outputfile_path is not None:

                    filtered_prompts = [x for x in generated_prompts if x not in all_prompts]
                    filtered_prompts = filtered_prompts[:batch_size]

                    df = pd.DataFrame({"prompt": filtered_prompts})
                    df.to_csv(outputfile_path, index=False, header=(not os.path.exists(outputfile_path)), mode='a')
                    all_prompts.update(filtered_prompts)

                if len(all_prompts) >= num_prompts: 
                    break

                if i % 20 == 0:
                    print("num generated prompts:", len(all_prompts))


                


        return list(all_prompts)[:num_prompts]



    def generate(self, subject, num_prompts):
        prompt = f"""Generate {num_prompts} well-formed prompt(s) that can be used to request assistance from the LLM 
        in identifying relevant papers on {subject} topics with associated arXiv links.
        Place these generated prompts inside [[ ]] like [[here is prompt]].
        Do not include any additional commentary or explanations.
        """

        response = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {  
                "role": "system", 
                "content": "You are a helpful prompt generator."

            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1.0,
        )

        prompts = response.choices[0].message.content
        prompt_pattern = r"\[\[.+\]\]"

        all_matches = re.findall(prompt_pattern, prompts)
        all_matches = [line.replace('[', "").replace(']', "").strip('"').strip() for line in all_matches]

        return list(set(all_matches))
