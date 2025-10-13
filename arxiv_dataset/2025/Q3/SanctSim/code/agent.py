
# agent.py

"""
This module defines the Agent class representing participants in the experiment.

Agents use Large Language Models (LLMs) via API calls to make their decisions.

Agents will:

- Choose institutions based on prompts sent to the LLM.

- Decide on contributions by generating prompts and parsing the LLM's responses.

- Assign punishments or rewards in the Sanctioning Institution (SI) based on LLM output.

The agent's internal monologue or reasoning is captured and stored under attributes:

- 'institution_reasoning' for institution choice

- 'contribution_reasoning' for contribution decision

- 'punishment_reasoning' for punishment and reward assignments

Dependencies:

- Requires an API client (AzureOpenAIClient, OpenRouterClient, KlusterAIClient, or OpenAIClient) for LLM interaction.

- Ensure that the API keys and configurations are properly set up.

Note:

- Be mindful of API rate limits and costs.

"""

import random
import parameters
import os
import json
import re  # For parsing responses

# Assuming the API clients are imported correctly
from azure_openai_client import AzureOpenAIClient
from openrouter_client import OpenRouterClient
from kluster_ai_client import KlusterAIClient
from openai_client import OpenAIClient

class Agent:
    def __init__(self, agent_id, api_client, summary_api_client=None):
        """
        Initialize an Agent.

        Parameters:
        - agent_id (int): Unique identifier for the agent.
        - api_client: The main API client used for agent's decision-making LLM calls.
        - summary_api_client (optional): API client used for isolated LLM calls (e.g., deanonymization).
          If not provided, the main api_client is used.
        """
        self.agent_id = agent_id
        self.api_client = api_client
        self.summary_api_client = summary_api_client if summary_api_client else api_client

        self.institution_choice = None  # 'SI' or 'SFI'
        self.contribution = 0
        self.cumulative_payoff = parameters.INITIAL_TOKENS  # Total accumulated payoff
        self.round_payoff = 0  # Payoff for the current round
        self.history = []
        self.current_group = None  # Reference to the institution/group the agent is currently in

        # Round-specific attributes
        self.received_punishments = 0
        self.received_rewards = 0
        self.assigned_punishments = {}  # Dict of agent_id: tokens assigned
        self.assigned_rewards = {}  # Dict of agent_id: tokens assigned

        # Additional attributes for LLM interaction
        self.round_number = 0  # Current round number

        # Add 'strategy' attribute for compatibility
        self.strategy = 'LLM'  # All agents are using LLM-based decision-making

        # Attributes to store reasoning
        self.institution_reasoning = ''
        self.contribution_reasoning = ''
        self.punishment_reasoning = ''

        # Attribute to store anonymous data history
        self.anonymous_data_history = []  # List of dicts storing data for previous rounds
        self.current_round_anonymous_data = None  # Data collected in the current round

        # Add attribute to store mapping of anonymized IDs to actual agent IDs
        self.anonymized_id_mapping = {}  # Mapping of anonymized agent numbers to actual agent IDs for the current prompt

        # For deanonymized reasoning
        self.deanonymized_punishment_reasoning = ''  # Deanonymized version of punishment reasoning

    def choose_institution(self, round_number):
        """
        Decide whether to join the Sanctioning Institution (SI) or the Sanction-Free Institution (SFI)
        by generating a prompt and sending it to the LLM.
        """
        self.round_number = round_number  # Update current round

        # Construct the prompt
        prompt = self.construct_institution_choice_prompt(round_number)

        # Send the prompt to the LLM via the API client
        response = self.api_client.send_request(
            model_name=self.api_client.deployment_name,  # Adjust as needed
            prompt=prompt
        )

        # Parse the response
        self.parse_institution_choice_response(response)

    def construct_institution_choice_prompt(self, round_number):
        """
        Construct a prompt that will be sent to the LLM to decide between SI and SFI.
        """
        past_actions = self.get_past_actions_string()
        cumulative_payoff = self.cumulative_payoff  # Correctly reference cumulative_payoff

        # Use parameters from parameters.py
        initial_tokens = parameters.INITIAL_TOKENS
        endowment_stage1 = parameters.ENDOWMENT_STAGE_1
        endowment_stage2 = parameters.ENDOWMENT_STAGE_2
        punishment_cost = parameters.PUNISHMENT_COST
        punishment_effect = parameters.PUNISHMENT_EFFECT
        reward_cost = parameters.REWARD_COST
        reward_effect = parameters.REWARD_EFFECT
        max_punishment_tokens = parameters.MAX_PUNISHMENT_TOKENS
        public_good_multiplier = parameters.PUBLIC_GOOD_MULTIPLIER

        # Construct the prompt based on the original instructions
        prompt = f"""
You are participating in a public goods game in Round {round_number}. You will interact only with participants in your institution. At the beginning of the experiment, you were assigned {initial_tokens} tokens in your experimental account.

Each round consists of 2 stages.

**Stage 1: Group Choice and Contribution to the Project**

(i) **Group Choice:**

In Stage 1, you decide which group you want to join. There are two different groups that can be joined:

1. **Group A (Sanction-Free Institution - SFI):**

- No possibility to impose sanctions or rewards on other group members

2. **Group B (Sanctioning Institution - SI):**

- Possibility to impose sanctions or rewards on other group members

(ii) **Contribution to the Project:**

In Stage 1 of each round, each group member is endowed with {endowment_stage1} tokens. You have to decide how many of these {endowment_stage1} tokens you are going to contribute to the project. The remaining tokens will be kept in your private account.

**Calculation of your payoff in Stage 1:**

Your payoff in Stage 1 consists of two components:

- Tokens you have kept: {endowment_stage1} – your contribution to the project

- Earnings from the project: ({public_good_multiplier} × sum of the contributions of all group members) / number of group members

Thus, your payoff in Stage 1 amounts to:

{endowment_stage1} – your contribution to the project + ({public_good_multiplier} × sum of the contributions of all group members) / number of group members

**Stage 2: Assignment of Tokens (Only in SI)**

If you choose SI, in Stage 2 you can increase or reduce the payoff of a group member or keep it unchanged. In each round, each participant in SI receives additional {endowment_stage2} tokens in Stage 2. You have to decide how many of these {endowment_stage2} tokens you are going to assign to other group members. The remaining tokens are kept in your private account. You can assign up to {max_punishment_tokens} tokens in total.

- Each positive token that you assign to a group member increases their payoff by {reward_effect} token(s) and costs you {reward_cost} token(s).

- Each negative token that you assign to a group member reduces their payoff by {punishment_effect} token(s) and costs you {punishment_cost} token(s).

- If you assign 0 tokens to a group member, their payoff won’t change.

**Calculation of your payoff in Stage 2:**

Your payoff in Stage 2 consists of:

- Tokens you kept in your private account: {endowment_stage2} – sum of the tokens that you have assigned to other group members

- Increased by the number of positive tokens you have received from other group members × {reward_effect}

- Diminished by the number of negative tokens you have received from other group members × {punishment_effect}

Thus, your payoff in Stage 2 amounts to:

{endowment_stage2} – sum of the tokens that you assigned to other group members + (number of positive tokens you have received × {reward_effect}) – (number of negative tokens you have received × {punishment_effect})

**Your Total Payoff in the Round:**

Your payoff from Stage 1 + Your payoff from Stage 2

**Current situation**

You are in the institution selection stage in Round {round_number}.

**Your Cumulative Payoff So Far:** {cumulative_payoff}

**Your Past Actions and Outcomes:**

{past_actions}
"""

        # Include anonymous data from the last DISPLAY_PAST_ACTIONS rounds
        if self.anonymous_data_history:
            prompt += f"\n\nAnonymous Data from Previous Rounds (up to last {parameters.DISPLAY_PAST_ACTIONS} rounds):"
            for idx, round_data in enumerate(self.anonymous_data_history[-parameters.DISPLAY_PAST_ACTIONS:]):
                round_num = round_data['round_number']
                anonymous_data_list = round_data['anonymous_data']
                anonymous_data_str = "\n".join(
                    [f"Agent {i+1}: Institution: {entry.get('institution_choice', 'Unknown')}, "
                     f"Contributed {entry['contribution']} tokens, "
                     f"Assigned Punishments: {entry.get('assigned_punishments', 0)}, "
                     f"Assigned Rewards: {entry.get('assigned_rewards', 0)}, "
                     f"Received Punishments: {entry.get('received_punishments', 0)}, "
                     f"Received Rewards: {entry.get('received_rewards', 0)}, "
                     f"Stage 1 Payoff: {entry.get('stage1_payoff', 0):.2f}, "
                     f"Stage 2 Payoff: {entry.get('stage2_payoff', 0):.2f}, "
                     f"Total Round Payoff: {entry.get('total_round_payoff', 0):.2f}"
                     for i, entry in enumerate(anonymous_data_list)]
                )
                prompt += f"\n\nRound {round_num}:\n{anonymous_data_str}"
            prompt += "\n\nAnalyze the contributions and outcomes of other agents over previous rounds. Consider their institution choices when deciding your strategy."
        else:
            prompt += "\n\nNo data about other agents is available from previous rounds."

        prompt += "\n\nDecide which institution to join. Reason deeply about the best strategy to follow moving forward."

        prompt += """
**Respond in the following JSON format:**

```json
{
  "reasoning": "Your reasoning here.",
  "institution_choice": "SI" or "SFI" 
}
```
"""

        return prompt.strip()

    def parse_institution_choice_response(self, response):
        """
        Parse the LLM's response to extract the institution choice and reasoning.
        """
        try:
            # Extract JSON content from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            institution_choice = data.get('institution_choice', '').upper()
            reasoning = data.get('reasoning', '')
            if institution_choice in ['SI', 'SFI']:
                self.institution_choice = institution_choice
            else:
                self.institution_choice = 'SFI'  # Default to 'SFI' if unclear
            self.institution_reasoning = reasoning
        except Exception as e:
            # Default choice and empty reasoning if parsing fails
            self.institution_choice = 'SFI'
            self.institution_reasoning = ''

    def decide_contribution(self, group_state):
        """
        Decide how much to contribute to the public good by generating a prompt and sending it to the LLM.
        """
        prompt = self.construct_contribution_prompt(group_state)
        response = self.api_client.send_request(
            model_name=self.api_client.deployment_name,
            prompt=prompt
        )
        contribution = self.parse_contribution_response(response)
        self.contribution = contribution

    def construct_contribution_prompt(self, group_state):
        """
        Construct a prompt for the LLM to decide the contribution amount.
        """
        cumulative_payoff = self.cumulative_payoff  # Correctly reference cumulative_payoff
        past_actions = self.get_past_actions_string()
        institution = self.institution_choice
        round_number = self.round_number

        # Use parameters from parameters.py
        endowment_stage1 = parameters.ENDOWMENT_STAGE_1
        public_good_multiplier = parameters.PUBLIC_GOOD_MULTIPLIER
        min_contribution = parameters.MIN_CONTRIBUTION
        max_contribution = parameters.MAX_CONTRIBUTION
        initial_tokens = parameters.INITIAL_TOKENS
        endowment_stage2 = parameters.ENDOWMENT_STAGE_2
        punishment_cost = parameters.PUNISHMENT_COST
        punishment_effect = parameters.PUNISHMENT_EFFECT
        reward_cost = parameters.REWARD_COST
        reward_effect = parameters.REWARD_EFFECT
        max_punishment_tokens = parameters.MAX_PUNISHMENT_TOKENS

        prompt = f"""
You are participating in a public goods game in Round {round_number}. You will interact only with participants in your institution. At the beginning of the experiment, you were assigned {initial_tokens} tokens in your experimental account.

Each round consists of 2 stages.

**Stage 1: Group Choice and Contribution to the Project**

(i) **Group Choice:**

In Stage 1, you decide which group you want to join. There are two different groups that can be joined:

1. **Group A (Sanction-Free Institution - SFI):**

- No possibility to impose sanctions or rewards on other group members

2. **Group B (Sanctioning Institution - SI):**

- Possibility to impose sanctions or rewards on other group members

(ii) **Contribution to the Project:**

In Stage 1 of each round, each group member is endowed with {endowment_stage1} tokens. You have to decide how many of these {endowment_stage1} tokens you are going to contribute to the project. The remaining tokens will be kept in your private account.

**Calculation of your payoff in Stage 1:**

Your payoff in Stage 1 consists of two components:

- Tokens you have kept: {endowment_stage1} – your contribution to the project

- Earnings from the project: ({public_good_multiplier} × sum of the contributions of all group members) / number of group members

Thus, your payoff in Stage 1 amounts to:

{endowment_stage1} – your contribution to the project + ({public_good_multiplier} × sum of the contributions of all group members) / number of group members

**Stage 2: Assignment of Tokens (Only in SI)**

If you choose SI, in Stage 2 you can increase or reduce the payoff of a group member or keep it unchanged. In each round, each participant in SI receives additional {endowment_stage2} tokens in Stage 2. You have to decide how many of these {endowment_stage2} tokens you are going to assign to other group members. The remaining tokens are kept in your private account. You can assign up to {max_punishment_tokens} tokens in total.

- Each positive token that you assign to a group member increases their payoff by {reward_effect} token(s) and costs you {reward_cost} token(s).

- Each negative token that you assign to a group member reduces their payoff by {punishment_effect} token(s) and costs you {punishment_cost} token(s).

- If you assign 0 tokens to a group member, their payoff won’t change.

**Calculation of your payoff in Stage 2:**

Your payoff in Stage 2 consists of:

- Tokens you kept in your private account: {endowment_stage2} – sum of the tokens that you have assigned to other group members

- Increased by the number of positive tokens you have received from other group members × {reward_effect}

- Diminished by the number of negative tokens you have received from other group members × {punishment_effect}

Thus, your payoff in Stage 2 amounts to:

{endowment_stage2} – sum of the tokens that you assigned to other group members + (number of positive tokens you have received × {reward_effect}) – (number of negative tokens you have received × {punishment_effect})

**Your Total Payoff in the Round:**

Your payoff from Stage 1 + Your payoff from Stage 2

**Current situation**

You are in Stage 1, in the {institution} in Round {round_number}.

**Your Cumulative Payoff So Far:** {cumulative_payoff}

**Your Past Actions and Outcomes:**

{past_actions}
"""

        # Include anonymous data from previous rounds
        if self.anonymous_data_history:
            prompt += f"\n\nAnonymous Data from Previous Rounds (up to last {parameters.DISPLAY_PAST_ACTIONS} rounds):"
            for idx, round_data in enumerate(self.anonymous_data_history[-parameters.DISPLAY_PAST_ACTIONS:]):
                round_num = round_data['round_number']
                anonymous_data_list = round_data['anonymous_data']
                anonymous_data_str = "\n".join(
                    [f"Agent {i+1}: Institution: {entry.get('institution_choice', 'Unknown')}, "
                     f"Contributed {entry['contribution']} tokens, "
                     f"Assigned Punishments: {entry.get('assigned_punishments', 0)}, "
                     f"Assigned Rewards: {entry.get('assigned_rewards', 0)}, "
                     f"Received Punishments: {entry.get('received_punishments', 0)}, "
                     f"Received Rewards: {entry.get('received_rewards', 0)}, "
                     f"Stage 1 Payoff: {entry.get('stage1_payoff', 0):.2f}, "
                     f"Stage 2 Payoff: {entry.get('stage2_payoff', 0):.2f}, "
                     f"Total Round Payoff: {entry.get('total_round_payoff', 0):.2f}"
                     for i, entry in enumerate(anonymous_data_list)]
                )
                prompt += f"\n\nRound {round_num}:\n{anonymous_data_str}"
            prompt += "\n\nAnalyze the contributions and outcomes of other agents over previous rounds. Consider their institution choices when deciding your contribution."
        else:
            prompt += "\n\nNo data about other agents is available from previous rounds."

        prompt += f"\n\nDecide how many tokens (between {min_contribution} and {max_contribution}) you will contribute to the project. Provide your contribution amount and a brief reasoning."

        prompt += """
**Respond in the following JSON format:**

```json
{
  "reasoning": "Your reasoning here.",
  "contribution": amount
}
```
"""

        return prompt.strip()

    def parse_contribution_response(self, response):
        """
        Parse the LLM's response to extract the contribution amount and reasoning.
        """
        try:
            # Extract JSON content from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            contribution = int(data.get('contribution', parameters.ENDOWMENT_STAGE_1 // 2))
            # Ensure contribution is within allowed limits
            contribution = max(parameters.MIN_CONTRIBUTION, min(contribution, parameters.MAX_CONTRIBUTION))
            reasoning = data.get('reasoning', '')
            self.contribution_reasoning = reasoning
            return contribution
        except Exception as e:
            # Default contribution and empty reasoning if parsing fails
            self.contribution_reasoning = ''
            return parameters.ENDOWMENT_STAGE_1 // 2

    def assign_punishment(self, group_state):
        """
        Decide on assigning punishments or rewards via the LLM.
        """
        prompt = self.construct_punishment_prompt(group_state)
        response = self.api_client.send_request(
            model_name=self.api_client.deployment_name,
            prompt=prompt
        )

        # save prompt and response to json 
        with open(f'info_{self.agent_id}.json', 'w') as f:
            json.dump({'prompt': prompt, 'response': response}, f)


        punishment_allocations, reward_allocations = self.parse_punishment_response(response, group_state)
        self.assigned_punishments = punishment_allocations
        self.assigned_rewards = reward_allocations
        return punishment_allocations, reward_allocations

    def construct_punishment_prompt(self, group_state):
        """
        Construct a prompt for the LLM to decide on punishment and reward allocations.
        """
        cumulative_payoff = self.cumulative_payoff  # Correctly reference cumulative_payoff

        # Use parameters from parameters.py
        initial_tokens = parameters.INITIAL_TOKENS
        endowment_stage1 = parameters.ENDOWMENT_STAGE_1
        endowment_stage2 = parameters.ENDOWMENT_STAGE_2
        punishment_cost = parameters.PUNISHMENT_COST
        punishment_effect = parameters.PUNISHMENT_EFFECT
        reward_cost = parameters.REWARD_COST
        reward_effect = parameters.REWARD_EFFECT
        max_punishment_tokens = parameters.MAX_PUNISHMENT_TOKENS
        public_good_multiplier = parameters.PUBLIC_GOOD_MULTIPLIER
        past_actions = self.get_past_actions_string()
        institution = self.institution_choice
        round_number = self.round_number

        # Prepare the contributions of other agents

        # Map agent numbers to match anonymous data
        num_agents = len(group_state['members'])

        # Create a fresh mapping for this prompt
        self.anonymized_id_mapping = {}

        # Include current round contributions
        contributions_lines = []
        for i, agent in enumerate(group_state['members']):
            anonymized_number = i + 1
            self.anonymized_id_mapping[anonymized_number] = agent.agent_id
            contributions_lines.append(
                f"Agent {anonymized_number}: Contributed {agent.contribution} tokens"
            )
        contributions_str = "\n".join(contributions_lines)

        prompt = f"""
You are participating in a public goods game in Round {round_number}. You will interact only with participants in your institution. At the beginning of the experiment, you were assigned {initial_tokens} tokens in your experimental account.

Each round consists of 2 stages.

**Stage 1: Group Choice and Contribution to the Project**

(i) **Group Choice:**

In Stage 1, you decide which group you want to join. There are two different groups that can be joined:

1. **Group A (Sanction-Free Institution - SFI):**

- No possibility to impose sanctions or rewards on other group members

2. **Group B (Sanctioning Institution - SI):**

- Possibility to impose sanctions or rewards on other group members

(ii) **Contribution to the Project:**

In Stage 1 of each round, each group member is endowed with {endowment_stage1} tokens. You have to decide how many of these {endowment_stage1} tokens you are going to contribute to the project. The remaining tokens will be kept in your private account.

**Calculation of your payoff in Stage 1:**

Your payoff in Stage 1 consists of two components:

- Tokens you have kept: {endowment_stage1} – your contribution to the project

- Earnings from the project: ({public_good_multiplier} × sum of the contributions of all group members) / number of group members

Thus, your payoff in Stage 1 amounts to:

{endowment_stage1} – your contribution to the project + ({public_good_multiplier} × sum of the contributions of all group members) / number of group members

**Stage 2: Assignment of Tokens (Only in SI)**

If you choose SI, in Stage 2 you can increase or reduce the payoff of a group member or keep it unchanged. In each round, each participant in SI receives additional {endowment_stage2} tokens in Stage 2. You have to decide how many of these {endowment_stage2} tokens you are going to assign to other group members. The remaining tokens are kept in your private account. You can assign up to {max_punishment_tokens} tokens in total.

- Each positive token that you assign to a group member increases their payoff by {reward_effect} token(s) and costs you {reward_cost} token(s).

- Each negative token that you assign to a group member reduces their payoff by {punishment_effect} token(s) and costs you {punishment_cost} token(s).

- If you assign 0 tokens to a group member, their payoff won’t change.

**Calculation of your payoff in Stage 2:**

Your payoff in Stage 2 consists of:

- Tokens you kept in your private account: {endowment_stage2} – sum of the tokens that you have assigned to other group members

- Increased by the number of positive tokens you have received from other group members × {reward_effect}

- Diminished by the number of negative tokens you have received from other group members × {punishment_effect}

Thus, your payoff in Stage 2 amounts to:

{endowment_stage2} – sum of the tokens that you assigned to other group members + (number of positive tokens you have received × {reward_effect}) – (number of negative tokens you have received × {punishment_effect})

**Your Total Payoff in the Round:**

Your payoff from Stage 1 + Your payoff from Stage 2

**Current situation**

You are in Stage 2, in the {institution} in Round {round_number}.

**Your Cumulative Payoff So Far:** {cumulative_payoff}

**Your Past Actions and Outcomes:**

{past_actions}

**Contributions of Other Agents in Your Institution:**

{contributions_str}
"""

        # Include anonymous data from previous rounds
        if self.anonymous_data_history:
            prompt += f"\n\nAnonymous Data from Previous Rounds (up to last {parameters.DISPLAY_PAST_ACTIONS} rounds):"
            for idx, round_data in enumerate(self.anonymous_data_history[-parameters.DISPLAY_PAST_ACTIONS:]):
                round_num = round_data['round_number']
                anonymous_data_list = round_data['anonymous_data']
                anonymous_data_str = "\n".join(
                    [f"Agent {i+1}: Institution: {entry.get('institution_choice', 'Unknown')}, "
                     f"Contributed {entry['contribution']} tokens, "
                     f"Assigned Punishments: {entry.get('assigned_punishments', 0)}, "
                     f"Assigned Rewards: {entry.get('assigned_rewards', 0)}, "
                     f"Received Punishments: {entry.get('received_punishments', 0)}, "
                     f"Received Rewards: {entry.get('received_rewards', 0)}, "
                     f"Stage 1 Payoff: {entry.get('stage1_payoff', 0):.2f}, "
                     f"Stage 2 Payoff: {entry.get('stage2_payoff', 0):.2f}, "
                     f"Total Round Payoff: {entry.get('total_round_payoff', 0):.2f}"
                     for i, entry in enumerate(anonymous_data_list)]
                )
                prompt += f"\n\nRound {round_num}:\n{anonymous_data_str}"
            prompt += "\n\nReview the anonymous data of other agents' contributions and outcomes. Consider their institution choices. Decide how to allocate your punishment and reward tokens based on their behavior."
        else:
            prompt += "\n\nNo data about other agents is available from previous rounds."

        prompt += """
Decide how many punishment or reward tokens to allocate to each agent.

**Respond in the following JSON format:**

```json
{
  "reasoning": "Your reasoning here.",
  "punishments": {"agent_number": tokens, ...},
  "rewards": {"agent_number": tokens, ...}
}
```
Where agent_number is an integer between double quotes for correct parsing. Ensure your JSON response strictly adheres to the JSON standard by enclosing all keys in unescaped double quotes, ensuring proper syntax, and avoiding additional or unnecessary escape characters.
"""

        return prompt.strip()

    def parse_punishment_response(self, response, group_state):
        """
        Parse the LLM's response to extract punishment and reward allocations and reasoning.
        This version detects and removes extra escaped quotes from JSON keys.
        """
        try:
            # Extract JSON content from the response.
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            punishments = data.get('punishments', {})
            rewards = data.get('rewards', {})
            reasoning = data.get('reasoning', '')
            self.punishment_reasoning = reasoning

            # Deanonymize the reasoning for logging purposes
            self.deanonymized_punishment_reasoning = self.deanonymize_reasoning(reasoning)

            # Helper function to clean keys by removing extra quotes.
            def clean_agent_key(key):
                key = key.strip()
                # Remove leading and trailing quotes if present
                if key.startswith('"') and key.endswith('"'):
                    key = key[1:-1]
                return key

            punishment_allocations = {}
            reward_allocations = {}
            tokens_remaining = parameters.ENDOWMENT_STAGE_2

            # Get the number of valid agents (indices)
            num_agents = len(group_state['members'])

            # Map from anonymized numbers to agent instances
            anonymized_number_to_agent = {}
            for i, agent in enumerate(group_state['members']):
                anonymized_number = i + 1
                anonymized_number_to_agent[str(anonymized_number)] = agent

            # Process punishments
            for agent_num_str, tokens in punishments.items():
                clean_key = clean_agent_key(agent_num_str)
                try:
                    agent_num = int(clean_key)
                except ValueError:
                    # Skip invalid keys that cannot be converted to an integer.
                    continue
                if 1 <= agent_num <= num_agents:
                    tokens = int(tokens)
                    tokens = max(0, min(tokens, tokens_remaining))
                    # Map back to the corresponding agent's ID.
                    agent_id = anonymized_number_to_agent[str(agent_num)].agent_id
                    punishment_allocations[agent_id] = tokens
                    tokens_remaining -= tokens
                    if tokens_remaining <= 0:
                        break

            # Process rewards similarly.
            for agent_num_str, tokens in rewards.items():
                clean_key = clean_agent_key(agent_num_str)
                try:
                    agent_num = int(clean_key)
                except ValueError:
                    continue
                if 1 <= agent_num <= num_agents:
                    tokens = int(tokens)
                    tokens = max(0, min(tokens, tokens_remaining))
                    agent_id = anonymized_number_to_agent[str(agent_num)].agent_id
                    reward_allocations[agent_id] = tokens
                    tokens_remaining -= tokens
                    if tokens_remaining <= 0:
                        break

            return punishment_allocations, reward_allocations
        except Exception as e:
            print(f"Error parsing punishment response")
            # In case of parsing error, reset reasoning and return empty allocations.
            self.punishment_reasoning = ''
            self.deanonymized_punishment_reasoning = ''
            return {}, {}

    def deanonymize_reasoning(self, reasoning):
        """
        Replace anonymized agent numbers in the reasoning with actual agent IDs.
        """
        if not self.anonymized_id_mapping:
            return reasoning  # No mapping available, return original reasoning

        # Prepare a mapping from anonymized agent names to actual names
        # For example, map "Agent 1" to "Agent_ID_0"
        replacement_dict = {f"Agent {k}": f"Agent_ID_{v}" for k, v in self.anonymized_id_mapping.items()}

        # Use an LLM to rewrite the reasoning with replacements
        deanonymized_reasoning = self.rewrite_reasoning_with_replacements(reasoning, replacement_dict)

        return deanonymized_reasoning

    def rewrite_reasoning_with_replacements(self, reasoning, replacement_dict):
        """
        Use an LLM to rewrite the reasoning, replacing anonymized agent names
        with actual agent IDs.
        This method should not expose agent reasoning to the LLM used in the simulation,
        to avoid influencing agent behavior.
        """

        # Construct a prompt for the isolated LLM
        mapping_str = "\n".join([f"'{k}': '{v}'" for k, v in replacement_dict.items()])
        prompt = f"""
The following text refers to agents using anonymized identifiers (e.g., 'Agent 1', 'Agent 2'). Please replace the anonymized agent names with their corresponding actual agent IDs as per the mapping provided, without altering any other content.

Mapping:

{mapping_str}

Text:

{reasoning}

Rewritten Text:
"""

        # Use the summary_api_client for the isolated request
        response = self.summary_api_client.send_request(
            model_name=self.summary_api_client.deployment_name,
            prompt=prompt
        )

        # Extract the rewritten text from the response
        rewritten_text = response.strip()

        return rewritten_text

    def get_past_actions_string(self):
        """
        Helper method to format past actions for prompts.

        Returns:
        str: A formatted string of past actions and outcomes.
        """
        actions = []
        for entry in self.history[-parameters.DISPLAY_PAST_ACTIONS:]:
            round_num = entry.get('round_number', '')
            institution = entry.get('institution_choice', '')
            institution_reasoning = entry.get('institution_reasoning', '')
            contribution = entry.get('contribution', '')
            contribution_reasoning = entry.get('contribution_reasoning', '')
            stage1_payoff = entry.get('stage1_payoff', '')
            stage2_payoff = entry.get('stage2_payoff', '')
            total_round_payoff = entry.get('payoff', '')
            received_punishments = entry.get('received_punishments', 0)
            received_rewards = entry.get('received_rewards', 0)
            assigned_punishments = entry.get('assigned_punishments', 0)
            assigned_rewards = entry.get('assigned_rewards', 0)
            punishment_reasoning = entry.get('punishment_reasoning', '')
            rank = entry.get('rank', '')

            # Calculate number of punishment and reward tokens received
            punishment_effect = parameters.PUNISHMENT_EFFECT
            reward_effect = parameters.REWARD_EFFECT

            punishment_tokens_received = int(received_punishments / punishment_effect) if received_punishments > 0 else 0
            reward_tokens_received = int(received_rewards / reward_effect) if received_rewards > 0 else 0

            # Build the action string
            action_str = (
                f"Round {round_num}: Institution: {institution}, "
                f"institution_reasoning: '{institution_reasoning}', "
                f"Contribution: {contribution}, contribution_reasoning: '{contribution_reasoning}', "
                f"Stage 1 Payoff: {stage1_payoff}, Stage 2 Payoff: {stage2_payoff}, Total Round Payoff: {total_round_payoff}, "
                f"Received {punishment_tokens_received} punishment token(s) (total effect: -{received_punishments} tokens), "
                f"Received {reward_tokens_received} reward token(s) (total effect: +{received_rewards} tokens), "
                f"Assigned Punishments: {assigned_punishments}, Assigned Rewards: {assigned_rewards}, "
                f"Punishment Reasoning: '{punishment_reasoning}', "
                f"Rank: {rank}"
            )
            actions.append(action_str)

        if actions:
            action_str = "\n".join(actions)
        else:
            action_str = "No past actions."
        return action_str

    def update_payoff(self, amount):
        """
        Update the agent's payoffs.
        Args:
        amount (float): The total payoff for the current round.
        """
        self.round_payoff = amount  # Current round's payoff
        self.cumulative_payoff += amount  # Aggregate cumulative payoff

    def update_history(self, round_data):
        """
        Record the actions and outcomes of the round.
        Args:
        round_data (dict): Contains information about the round.
        """
        self.history.append(round_data)

    def reset_for_new_round(self):
        """
        Reset variables that are specific to a round.
        """
        # Move current round data to anonymous data history before resetting
        if hasattr(self, 'current_round_anonymous_data') and self.current_round_anonymous_data is not None:
            round_data = {
                'round_number': self.round_number,
                'anonymous_data': self.current_round_anonymous_data
            }
            self.anonymous_data_history.append(round_data)
            # Ensure the history does not exceed DISPLAY_PAST_ACTIONS
            if len(self.anonymous_data_history) > parameters.DISPLAY_PAST_ACTIONS:
                self.anonymous_data_history.pop(0)
            self.current_round_anonymous_data = None

        # Reset other attributes
        self.contribution = 0
        self.received_punishments = 0
        self.received_rewards = 0
        self.assigned_punishments = {}
        self.assigned_rewards = {}
        self.current_group = None
        self.round_payoff = 0  # Reset current round's payoff

        # Reset reasoning attributes
        self.institution_reasoning = ''
        self.contribution_reasoning = ''
        self.punishment_reasoning = ''
        self.deanonymized_punishment_reasoning = ''
        self.anonymized_id_mapping = {}

    def receive_punishment(self, amount):
        """
        Record the amount of punishment received.
        Args:
        amount (float): The total punishment effect received.
        """
        self.received_punishments += amount

    def receive_reward(self, amount):
        """
        Record the amount of reward received.
        Args:
        amount (float): The total reward effect received.
        """
        self.received_rewards += amount

    def get_stage1_payoff(self, group_size, total_group_contribution):
        """
        Calculate the payoff from Stage 1.
        Args:
        group_size (int): The number of members in the group.
        total_group_contribution (float): The sum of contributions in the group.
        Returns:
        float: The payoff from Stage 1.
        """
        # Tokens kept by agent
        tokens_kept = parameters.ENDOWMENT_STAGE_1 - self.contribution
        # Earnings from the public good
        if group_size > 0:
            earnings_from_public_good = (parameters.PUBLIC_GOOD_MULTIPLIER * total_group_contribution) / group_size
        else:
            earnings_from_public_good = 0
        stage1_payoff = tokens_kept + earnings_from_public_good
        return stage1_payoff

    def get_stage2_payoff(self):
        """
        Calculate the net payoff from Stage 2, after considering assigned punishments and rewards.
        Returns:
        float: The payoff from Stage 2.
        """
        # Tokens used for assigning punishments and rewards
        tokens_spent = (
            sum(self.assigned_punishments.values()) * parameters.PUNISHMENT_COST +
            sum(self.assigned_rewards.values()) * parameters.REWARD_COST
        )
        # Tokens remaining from the initial Stage 2 endowment
        tokens_remaining = parameters.ENDOWMENT_STAGE_2 - tokens_spent

        # Effects of punishments and rewards received
        punishment_effect = self.received_punishments  # Already includes the punishment effect
        reward_effect = self.received_rewards  # Already includes the reward effect

        # Net payoff
        stage2_payoff = tokens_remaining + reward_effect - punishment_effect
        return stage2_payoff

    def __repr__(self):
        return f"Agent({self.agent_id}, Cumulative Payoff: {self.cumulative_payoff})"
