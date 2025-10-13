# institution.py

"""
This module defines the Institution classes for the Sanctioning Institution (SI)
and the Sanction-Free Institution (SFI).

Both institutions manage the grouping of agents and their interactions within the institution.
- The Sanctioning Institution allows agents to assign punishments and rewards after contributions.
- The Sanction-Free Institution does not have mechanisms for punishment or reward.
"""

import parameters

class Institution:
    def __init__(self):
        self.members = []
        self.total_contribution = 0
        self.group_size = 0
        self.round_number = 0
        self.public_good_share = {}
        self.stage1_payoffs = {}
        self.anonymized_contributions = []  # For agents to see contributions without identifying individuals

    def add_member(self, agent):
        """
        Adds an agent to the institution's member list.
        """
        self.members.append(agent)

    def reset_institution(self, round_number):
        """
        Resets institution-specific variables for a new round.
        """
        self.members = []
        self.total_contribution = 0
        self.group_size = 0
        self.round_number = round_number
        self.public_good_share = {}
        self.stage1_payoffs = {}
        self.anonymized_contributions = []

    def collect_contributions(self):
        """
        Collect contributions from members and calculate the total contribution.
        """
        self.total_contribution = 0
        self.group_size = len(self.members)
        self.anonymized_contributions = []

        for agent in self.members:
            agent.decide_contribution(self.get_group_state(agent))
            self.total_contribution += agent.contribution
            self.anonymized_contributions.append(agent.contribution)
            print(f"Agent {agent.agent_id} contributed {agent.contribution} tokens")

    def distribute_public_goods(self):
        """
        Calculate and distribute the share of the public good to each member.
        """
        if self.group_size > 0:
            public_good_earning = (parameters.PUBLIC_GOOD_MULTIPLIER * self.total_contribution) / self.group_size
        else:
            public_good_earning = 0

        for agent in self.members:
            stage1_payoff = agent.get_stage1_payoff(
                group_size=self.group_size,
                total_group_contribution=self.total_contribution
            )
            # Update the agent's payoff; Stage 1 payoff is added after Stage 2
            self.stage1_payoffs[agent.agent_id] = stage1_payoff
            # The payoff will be updated later after Stage 2 in the environment
            print(f"Agent {agent.agent_id} earned {stage1_payoff} tokens in Stage 1") 

    def get_group_state(self, requesting_agent):
        """
        Provides information about the group that may be used by the agent for decision-making.

        Args:
            requesting_agent (Agent): The agent requesting the information.

        Returns:
            dict: Information about the group.
        """
        # For anonymity, agents receive only aggregated or anonymized information
        group_state = {
            'members': [agent for agent in self.members],  # Agents may need references to assign punishments
            'anonymized_contributions': self.anonymized_contributions.copy(),
            'group_size': self.group_size,
            'round_number': self.round_number,
            # Additional information can be added as needed
        }
        return group_state

    def get_average_contribution(self):
        if self.group_size > 0:
            return self.total_contribution / self.group_size
        else:
            return 0.0


class SanctioningInstitution(Institution):
    def __init__(self):
        super().__init__()
        self.punishment_matrix = {}  # Records punishments assigned between agents
        self.reward_matrix = {}      # Records rewards assigned between agents

    def reset_institution(self, round_number):
        super().reset_institution(round_number)
        self.punishment_matrix = {}
        self.reward_matrix = {}

    def handle_punishments_and_rewards(self):
        """
        Agents assign punishments and rewards after contributions are revealed.
        """
        # Each agent in SI can assign punishments and rewards
        for agent in self.members:
            group_state = self.get_group_state(agent)
            punishment_allocations, reward_allocations = agent.assign_punishment(group_state)

            # Record punishments
            for punished_agent_id, tokens in punishment_allocations.items():
                if punished_agent_id not in self.punishment_matrix:
                    self.punishment_matrix[punished_agent_id] = {}
                self.punishment_matrix[punished_agent_id][agent.agent_id] = tokens

            # Record rewards
            for rewarded_agent_id, tokens in reward_allocations.items():
                if rewarded_agent_id not in self.reward_matrix:
                    self.reward_matrix[rewarded_agent_id] = {}
                self.reward_matrix[rewarded_agent_id][agent.agent_id] = tokens

            print(f"Agent {agent.agent_id} assigned punishments: {punishment_allocations}")

    def apply_punishments_and_rewards(self):
        """
        Apply the punishments and rewards to each agent's payoff and record received amounts.
        """
        for agent in self.members:
            # Total punishments received
            total_punishment_tokens = 0
            if agent.agent_id in self.punishment_matrix:
                total_punishment_tokens = sum(self.punishment_matrix[agent.agent_id].values())

            total_punishment_effect = total_punishment_tokens * parameters.PUNISHMENT_EFFECT
            agent.receive_punishment(total_punishment_effect)

            # Total rewards received
            total_reward_tokens = 0
            if agent.agent_id in self.reward_matrix:
                total_reward_tokens = sum(self.reward_matrix[agent.agent_id].values())

            total_reward_effect = total_reward_tokens * parameters.REWARD_EFFECT
            agent.receive_reward(total_reward_effect)

            # Update the agent's payoff for tokens spent on punishments/rewards
            stage2_payoff = agent.get_stage2_payoff()
            # The agent's total payoff will be updated in the environment after combining Stage 1 and Stage 2

    def get_group_state(self, requesting_agent):
        """
        In SI, agents can see contributions of other members and need to assign punishments/rewards.

        Returns:
            dict: Information including references to other agents.
        """
        group_state = super().get_group_state(requesting_agent)
        # We can exclude the requesting agent from the list of members when assigning punishments
        group_state['members'] = [agent for agent in self.members if agent.agent_id != requesting_agent.agent_id]

        return group_state


class SanctionFreeInstitution(Institution):
    def __init__(self):
        super().__init__()

    def handle_punishments_and_rewards(self):
        """
        In SFI, there are no punishments or rewards.
        """
        pass

    def apply_punishments_and_rewards(self):
        """
        In SFI, there are no punishments or rewards to apply.
        """
        pass

    def get_group_state(self, requesting_agent):
        """
        In SFI, agents may receive less information due to the lack of punishments/rewards.

        Returns:
            dict: Anonymized information about the group.
        """
        group_state = super().get_group_state(requesting_agent)
        # Agents in SFI may only receive limited information
        return group_state