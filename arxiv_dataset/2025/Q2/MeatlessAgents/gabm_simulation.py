import asyncio
from itertools import product
import json
import pandas as pd
from tqdm.auto import tqdm

from autogen_agentchat.teams import RoundRobinGroupChat
from local_vllm_model import LocalOpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination

from custom_agents import MyAgent

model_client=LocalOpenAIChatCompletionClient(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_key="token-abc123",
    base_url="http://localhost:8080/v1",
    model_capabilities={
        "vision": False, # Replace with True if the model has vision capabilities.
        "function_calling": False, # Replace with True if the model has function calling capabilities.
        "json_output": True,  # Replace with True if the model has JSON output capabilities.
    },
    max_response_tokens=15000
)

async def run_simulation(persona_persuader:dict, persona_target:dict, context_dict:dict, seed:int|None, debate_id:int,
                         semaphore:asyncio.Semaphore, pbar:tqdm, max_rounds:int, run_name:str) -> None:

    async with semaphore:

        context = context_dict['context']


        personaA = persona_target
        profileA = f"""You are simulating an individual with specific traits, values, and motivations for consuming meat. Engage authentically and stay true to your persona's profile throughout the conversation.
Reflect on arguments that impact you, explain why they are persuasive or not, and challenge your partner's arguments where appropriate. If anything is unclear, ask for clarification.
You are a {personaA['education']}, {personaA['gender']}, {personaA['age']} adult.
You live in the {personaA['urbanization']} and have a {personaA['income_level']} income level."""
        
        # optional profile variables to save computation
        if personaA['value_1'] is not None and personaA['value_2'] is not None and personaA['value_3'] is not None:
             profileA += f"Your core values are {personaA['value_1']}, {personaA['value_2']}, and {personaA['value_3']}."
        if personaA['openness'] is not None and personaA['conscientiousness'] is not None:
             profileA += f"You are {personaA['openness']} to new experiences and are a {personaA['conscientiousness']} person."
        
        profileA += f"The context is: {context} The setting should influence your willingness to engage, balancing social and cultural considerations."
        
        personaB = persona_persuader
        profileB = f"""You aim to persuade agent_A to reduce or eliminate their meat consumption. Tailor your arguments to the scenario and agent_A's traits, using a relatable and empathetic approach.
Your goal is to encourage incremental progress without creating resistance or discomfort that would lead to social costs on your behalf.
Your diet is: {personaB['diet']}
The context is: {context} The setting should inform your arguments, emphasizing convenience, social norms, and relatable changes."""

        reflection_task = """
You are {name}.
{profile}

Based on the following transcript, take a moment for internal reflection. Ask yourself:
1. What emotions did the statement evoke?
2. Which of your core values does the statement challenge or support?
3. What questions or uncertainties do you have about the statement?
{transcript}

Now individually reflect each of the 3 points about emotions, core values, and open questions, and answer all of them.
Do not respond to the other agent and do not describe any of your behavior, just answer the reflection questions.
"""

        response_task = """
You are {name}.
{profile}

Based on the following transcript and your previous reflection, formulate your response to the other agent.
Responses should be thoughtful and authentic, considering the context and your persona's values.
"""
        response_task += "Provide only the response you would give to the other agent. DO NOT add notes about your behavior.\n" # had to add this to improve adherence
        response_task += "Keep your response as short as it would typically occur in a conversation in this context.\n"
        
        # make the messages shorter and avoid multiple thoughts in a single statement
        response_task += "Only provide a single thought or argument to keep the conversation going smoothly. Do not write multiple paragraphs!"

        response_task_agentB = response_task
        response_task_agentB += "Remember that you aim to persuade agent_A to reduce or eliminate their meat consumption. Your goal is to encourage incremental progress without creating resistance or discomfort that would lead to social costs on your behalf.\n"

        questionnaire_data = json.load(open('questionnaires/questionnaire_separate.json'))['questionnaire']

        ## Agent to be persuaded
        agentA = MyAgent(name='agent_A',
                            profile=profileA,
                            reflection_task=reflection_task,
                            response_task=response_task,
                            questionnaire_data = questionnaire_data,
                            model_client=model_client,
                            pbar=pbar,
                            description='agent to be persuaded')

        ## Agent that is persuading
        agentB = MyAgent(name='agent_B',
                            profile=profileB,
                            reflection_task=reflection_task,
                            response_task=response_task_agentB,
                            model_client=model_client,
                            pbar=pbar,
                            description='agent that is persuading')

        ## The group chat
        # TODO: test much longer conversations to see if things mostly stay the same
        team = RoundRobinGroupChat([agentA, agentB], termination_condition=MaxMessageTermination(max_rounds))

        ## Create an opener
        opening_task = "Begin the conversation with an opening statement. " + context_dict['opening_task']
        opening_response = await agentB._create_opener(task=opening_task)

        ## Run the conversation
        await team.run(task=opening_response)

        ## Save the results
        message_history = [{msg.source: msg.content} for msg in agentA._messsage_history]
        json.dump(message_history, open(f'results/transcripts/{run_name}/debate{debate_id}_messages.json', mode='w'), indent = 2)
        
        agentA_thoughts = [{'agentA_thoughts': msg.content} for msg in agentA._thoughts]
        json.dump(agentA_thoughts, open(f'results/transcripts/{run_name}/debate{debate_id}_agentA_thoughts.json', mode='w'), indent = 2)
        
        agentB_thoughts = [{'agentB_thoughts': msg.content} for msg in agentB._thoughts]
        json.dump(agentB_thoughts, open(f'results/transcripts/{run_name}/debate{debate_id}_agentB_thoughts.json', mode='w'), indent = 2)

        json.dump(agentA._questionnaire_responses, open(f'results/transcripts/{run_name}/debate{debate_id}_questionnaire_responses.json', mode='w'), indent = 2)
        pd.DataFrame(agentA._questionnaire_responses).groupby(['round', 'repetition']).first().to_csv(f'results/transcripts/{run_name}/debate{debate_id}_questionnaire_responses.csv', index=True)

        settings = {'target': persona_target, 'persuader': persona_persuader, 'context': context_dict['context_id'], 'seed': 'N/A'}
        json.dump(settings, open(f'results/transcripts/{run_name}/debate{debate_id}_settings.json', mode='w'), indent = 2)


### The Simulation Setup

## Extreme Personas
personas_target = [
    # extreme persona 1 – female
    {'education': 'well-educated',
     'gender': 'female',
     'age': 'younger',
     'urbanization': 'city',
     'income_level': 'high',
     'value_1': 'self-transcendence',
     'value_2': 'openness to change',
     'value_3': 'empathy toward animals',
     'openness': 'open',
     'conscientiousness': 'conscientious'},

    # extreme persona 2 – male
    {'education': 'less well-educated',
     'gender': 'male',
     'age': 'older',
     'urbanization': 'countryside',
     'income_level': 'low',
     'value_1': 'self-enhancement',
     'value_2': 'conservation',
     'value_3': 'discouraging empathy toward animals',
     'openness': 'less open',
     'conscientiousness': 'less conscientious'},
]

personas_persuader = [
    {'diet': 'Flexitarian (occasional meat consumption but primarily plant-based)'},
]

# contexts + opening tasks
contexts = [
    {
        'context_id': 'canteen',
        'context': 'The discussion occurs during lunch in the canteen.',
        'opening_task': "Comment on your colleague's choice of meal in the canteen and how it compares to your usual preferences."
    },
    {
        'context_id': 'social_media',
        'context': 'The discussion occurs as a private reply to an Instagram story.', # making sure that this is as precise as necessary
        'opening_task': "Comment on the food picture that the other person just posted and how it compares to your usual food preferences."
    },
]

run_name = 'extreme_run_70B'

simulations = list(enumerate(product(range(100), contexts, personas_persuader, personas_target))) # iterates target first, then persuader, context, seed

max_rounds = 10 # how many messages to exchange in each discussion, A & B combined

async def main():

    semaphore = asyncio.Semaphore(8) # maximal number of concurrent tasks
    total_simulations = len(simulations) * max_rounds

    # Initialize tqdm progress bar
    with tqdm(total=total_simulations, desc="Rounds Simulated") as pbar:
        tasks = [
            run_simulation(persona_persuader, persona_target, context, seed, debate_id, semaphore, pbar, max_rounds, run_name)
            for debate_id, (seed, context, persona_persuader, persona_target) in simulations
        ]
        await asyncio.gather(*tasks)  # Run tasks concurrently

# Run the async event loop
asyncio.run(main())