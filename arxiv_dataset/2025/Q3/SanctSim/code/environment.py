# environment.py

"""
This module defines the Environment class that manages the overall game environment.
It coordinates the interactions between agents and institutions, resolves actions and payoffs,
and handles the progression of rounds.

The Environment class uses the Agent and Institution classes to simulate the experiment.
"""

import random
import parameters
from agent import Agent
from institution import SanctioningInstitution, SanctionFreeInstitution

class Environment:
    def __init__(self, agents):
        self.agents = agents # List of Agent instances
        self.current_round = 0
        self.history = [] # Record of each round's outcomes
        self.si = SanctioningInstitution()
        self.sfi = SanctionFreeInstitution()
        self.results = [] # To store results for analysis
        # Optional: Set the random seed for reproducibility
        random.seed(parameters.SEED)

    def run_simulation(self):
        """
        Runs the simulation for the specified number of rounds.
        """
        for round_number in range(1, parameters.NUM_ROUNDS + 1):
            self.current_round = round_number
            print(f"\nStarting Round {self.current_round}")
            self.run_round()

    def run_round(self):
        """
        Executes a single round of the simulation.
        """
        # Reset institutions for the new round
        self.si.reset_institution(round_number=self.current_round)
        self.sfi.reset_institution(round_number=self.current_round)

        # Agents choose institutions
        for agent in self.agents:
            print(f"Agent {agent} started")
            # Reset agent's round-specific attributes
            agent.reset_for_new_round()
            agent.choose_institution(self.current_round)

            # Add agent to the chosen institution
            if agent.institution_choice == 'SI':
                self.si.add_member(agent)
                agent.current_group = self.si
            else:
                self.sfi.add_member(agent)
                agent.current_group = self.sfi
            print(f"Agent {agent} finished institution choice")

        # Collect contributions in each institution
        self.si.collect_contributions()
        self.sfi.collect_contributions()

        # Distribute public goods in each institution
        self.si.distribute_public_goods()
        self.sfi.distribute_public_goods()

        # Handle punishments and rewards in SI
        self.si.handle_punishments_and_rewards()
        self.si.apply_punishments_and_rewards()

        # Calculate payoffs and update agents
        self.calculate_payoffs()

        # Update history and agents' personal histories
        self.record_round_history()

    def calculate_payoffs(self):
        """
        Calculates the total payoffs for all agents after considering contributions,
        punishments, rewards, and updates their cumulative payoffs.
        """
        # Update payoffs for agents in SI
        for agent in self.si.members:
            stage1_payoff = self.si.stage1_payoffs.get(agent.agent_id, 0)
            stage2_payoff = agent.get_stage2_payoff()
            total_round_payoff = stage1_payoff + stage2_payoff
            agent.update_payoff(total_round_payoff)
            # Optional: Print or log agent's payoff details for debugging
            if parameters.VERBOSE:
                print(f"Agent {agent.agent_id} in SI - Stage 1 Payoff: {stage1_payoff}, "
                      f"Stage 2 Payoff: {stage2_payoff}, Total Round Payoff: {total_round_payoff}")

        # Update payoffs for agents in SFI
        for agent in self.sfi.members:
            stage1_payoff = self.sfi.stage1_payoffs.get(agent.agent_id, 0)
            # In SFI, Stage 2 payoff is just the tokens remaining from Stage 2 endowment
            stage2_payoff = parameters.ENDOWMENT_STAGE_2
            total_round_payoff = stage1_payoff + stage2_payoff
            agent.update_payoff(total_round_payoff)
            if parameters.VERBOSE:
                print(f"Agent {agent.agent_id} in SFI - Stage 1 Payoff: {stage1_payoff}, "
                      f"Stage 2 Payoff: {stage2_payoff}, Total Round Payoff: {total_round_payoff}")

    def record_round_history(self):
        """
        Records the outcome of the round for analysis and provides feedback to agents.
        """
        # Collect data for the round
        round_data = {
            'round_number': self.current_round,
            'si_members': [agent.agent_id for agent in self.si.members],
            'sfi_members': [agent.agent_id for agent in self.sfi.members],
            'si_total_contribution': self.si.total_contribution,
            'sfi_total_contribution': self.sfi.total_contribution,
            'si_avg_contribution': self.si.get_average_contribution(),
            'sfi_avg_contribution': self.sfi.get_average_contribution(),
            'agents': {}
        }

        # Calculate cumulative payoffs for ranking
        agent_cumulative_payoffs = {
            agent.agent_id: agent.cumulative_payoff for agent in self.agents
        }
        # Calculate ranks based on cumulative payoff
        sorted_payoffs = sorted(
            agent_cumulative_payoffs.items(), key=lambda x: x[1], reverse=True
        )
        agent_ranks = {
            agent_id: rank + 1 for rank, (agent_id, _) in enumerate(sorted_payoffs)
        }

        total_agents = len(self.agents)

        # Record detailed agent data and provide feedback
        for agent in self.agents:
            # Retrieve stage payoffs
            if agent.institution_choice == 'SI':
                stage1_payoff = self.si.stage1_payoffs.get(agent.agent_id, 0)
                stage2_payoff = agent.get_stage2_payoff()
            else:
                stage1_payoff = self.sfi.stage1_payoffs.get(agent.agent_id, 0)
                stage2_payoff = parameters.ENDOWMENT_STAGE_2

            # Total round payoff
            total_round_payoff = stage1_payoff + stage2_payoff

            # Retrieve rank
            rank = agent_ranks[agent.agent_id]

            # Prepare agent data with detailed assigned punishments and rewards
            agent_data = {
                'institution_choice': agent.institution_choice,
                'institution_reasoning': agent.institution_reasoning,
                'contribution': agent.contribution,
                'contribution_reasoning': agent.contribution_reasoning,
                'stage1_payoff': stage1_payoff,
                'stage2_payoff': stage2_payoff,
                'payoff': total_round_payoff, # Current round's total payoff
                'cumulative_payoff': agent.cumulative_payoff, # Cumulative payoff
                'strategy': agent.strategy,
                'received_punishments': agent.received_punishments,
                'received_rewards': agent.received_rewards,
                # Keep the breakdown of assigned punishments and rewards
                'assigned_punishments': agent.assigned_punishments,
                'assigned_rewards': agent.assigned_rewards,
                'punishment_reasoning': agent.punishment_reasoning,
                'deanonymized_punishment_reasoning': agent.deanonymized_punishment_reasoning,
                'rank': f"{rank} out of {total_agents}"
            }

            # Add the agent data to the round data
            round_data['agents'][agent.agent_id] = agent_data

            # Prepare feedback for the agent
            avg_contribution_institution = (
                self.si.get_average_contribution()
                if agent.institution_choice == 'SI'
                else self.sfi.get_average_contribution()
            )

            feedback = agent_data.copy()
            feedback['round_number'] = self.current_round
            feedback['avg_payoff_SI'] = (
                sum([a.round_payoff for a in self.si.members]) / len(self.si.members)
                if self.si.members
                else 0
            )
            feedback['avg_payoff_SFI'] = (
                sum([a.round_payoff for a in self.sfi.members]) / len(self.sfi.members)
                if self.sfi.members
                else 0
            )
            feedback['avg_contribution_institution'] = avg_contribution_institution

            # Update agent's personal history
            agent.update_history(feedback)

        # Build anonymous data for each agent
        for agent in self.agents:
            anonymous_data_list = []
            for other_agent in self.agents:
                if other_agent.agent_id != agent.agent_id:
                    # Get other agent's data from round_data
                    other_agent_data = round_data['agents'][other_agent.agent_id]
                    # Extract the data needed
                    anonymous_entry = {
                        'institution_choice': other_agent_data['institution_choice'],
                        'contribution': other_agent_data['contribution'],
                        'received_punishments': other_agent_data['received_punishments'],
                        'received_rewards': other_agent_data['received_rewards'],
                        'stage1_payoff': other_agent_data['stage1_payoff'],
                        'stage2_payoff': other_agent_data['stage2_payoff'],
                        'total_round_payoff': other_agent_data['payoff']
                    }
                    # Note: We do not include the detailed assigned punishments and rewards in the anonymous data
                    # to maintain anonymity and prevent deanonymization.
                    anonymous_data_list.append(anonymous_entry)
            # Set the agent's current_round_anonymous_data
            agent.current_round_anonymous_data = anonymous_data_list

        # Append round data to environment history
        self.history.append(round_data)

        # Optionally, save data for analysis
        self.results.append(round_data)

    def save_results_to_filename(self, filename, model_name, num_agents, num_rounds):
        """
        Saves the simulation results to a specified file with parameters in the filename.
        """
        if parameters.SAVE_RESULTS:
            import json

            # Construct dynamic filename
            base_filename = "simulation_results"
            model_component = f"_{model_name.replace('/', '_')}" if model_name else "" # Replace '/' for filename safety
            agents_component = f"_{num_agents}agents"
            rounds_component = f"_{num_rounds}rounds"
            full_filename = f"{base_filename}{model_component}{agents_component}{rounds_component}.json"

            # If a filename is provided via command line, prioritize it, otherwise use the constructed name
            final_filename = filename if filename else full_filename

            with open(final_filename, 'w') as f:
                json.dump(self.results, f, indent=4)
            print(f"\nSimulation results saved to '{final_filename}'.")