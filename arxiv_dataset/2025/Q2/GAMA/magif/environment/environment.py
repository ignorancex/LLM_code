import json
import os
from datetime import datetime
from typing import List, Tuple
from magif.agent.agent import Agent
from magif.utils.setup_logger import logger
from magif.utils.utils import set_default


class Environment:
	"""
	A class to manage a game-theoretic tournament using agents, rounds, and target payoffs.

	Attributes:
		agent_pool (AgentPool): The pool of agents participating in the tournament.
		num_rounds (int): The number of rounds in the tournament.
		match_maker (Callable[[list], list[tuple]]): A function that
                generates match pairings based on valid and invalid agents.
		target_payoffs (list[float]): Optional target payoffs for specific tournament outcomes.
	"""
	def __init__(self, agent_pool, num_rounds, match_maker, target_payoffs=None):
		"""
		Initializes the Tournament with a pool of agents, a specified number of rounds,
		and optional target payoffs.

		Args:
			agent_pool (AgentPool): The pool of agents participating in the tournament.
			num_rounds (int): The number of rounds to be played in the tournament.
			match_maker (Callable[[list], list[tuple]]): A function that
                generates match pairings based on valid and invalid agents.
			target_payoffs (list[float], optional): A list of target payoffs to guide
				tournament objectives. Defaults to an empty list if not provided.
		"""
		self.agent_pool = agent_pool
		self.num_rounds = num_rounds
		self.match_maker = match_maker
		self.target_payoffs = target_payoffs if target_payoffs else []

	def play_tournament(self) -> None:
		"""
		Run the tournament where agents play against each other.
		Raises a ValueError if agents have not been created.
		"""
		# Step 1: Validate that agents have been created
		if not self.agent_pool:
			raise ValueError("Agents must be created before playing the tournament.")

		# Step 2: Generate agent pairs for the tournament
		agent_pairs = self.match_maker(self.agent_pool.valid_agents)

		# Step 3: Conduct matches between agent pairs
		self._play_matches(agent_pairs)

	def _play_matches(self, agent_pairs: List[Tuple[Agent, Agent]]) -> None:
		"""
		Play the specified number of rounds between each pair of agents.

		Args:
			agent_pairs (List[Tuple[Agent, Agent]]): List of tuples representing pairs of agents.
		"""
		for agent1, agent2 in agent_pairs:
			valid_pair = self._play_match(agent1, agent2)
			if not valid_pair:
				logger.debug(
					f"Agent {agent1.name} or {agent2.name} not valid. Excluding the pair from the tournament.")
				self.agent_pool.move_agent(agent1)
				self.agent_pool.move_agent(agent2)

	def _play_match(self, agent1: Agent, agent2: Agent) -> bool:
		"""
		Play a match between two agents for multiple rounds.

		Args:
			agent1 (Agent): The first agent.
			agent2 (Agent): The second agent.

		Returns:
			bool: True if both agents are valid throughout the match, False otherwise.
		"""
		for round_num in range(self.num_rounds):
			logger.info(
				f"\nAgent {agent1.name} with {agent1.strategy_name} vs {agent2.name} with {agent2.strategy_name}, Round {round_num}.")

			# Get moves from both agents
			move_agent_1, move_agent_2 = agent1.mind.act(), agent2.mind.act()
			if not (move_agent_1 and move_agent_2):
				return False

			agent1.mind.observe(move_agent_2)
			agent2.mind.observe(move_agent_1)

			# Update payoffs based on the opponents' moves
			updated_1 = agent1.mind.think()
			updated_2 = agent2.mind.think()
			if not (updated_1 and updated_2):
				return False

		return True

	def get_winners(self) -> List[Agent]:
		"""
		Determine the winners of the tournament.

		If target payoffs are specified, winners are agents who achieved their target payoffs.
		Otherwise, winners are agents with the highest overall payoff.

		Returns:
			List[Agent]: A list of agents who are the winners.
		"""
		if self.target_payoffs:
			return self._get_winners_by_target_payoff()
		else:
			return self._get_winners_by_highest_payoff()

	def _get_winners_by_target_payoff(self) -> List[Agent]:
		"""
		Get the agents who achieved their target payoffs.

		Returns:
			List[Agent]: A list of agents who met their target payoffs.
		"""
		return [
			agent for i, agent in enumerate(self.agent_pool.valid_agents)
			if agent.mind.get_total_payoff() == self.target_payoffs[i]
		]

	def _get_winners_by_highest_payoff(self) -> List[Agent]:
		"""
		Get the agents with the highest total payoff.

		Returns:
			List[Agent]: A list of agents with the highest payoff.
		"""
		max_payoff = max(agent.mind.get_total_payoff() for agent in self.agent_pool.valid_agents)
		return [agent for agent in self.agent_pool.valid_agents if agent.mind.get_total_payoff(log=False) == max_payoff]

	def log_tournament(
			self,
			experiment_dir: str,
			tournament_name: str = "tournament"
	) -> bool:
		"""
		Logs the details of a tournament, including its configuration and agents' information.

		Args:
			experiment_dir (str): The directory where the tournament logs will be saved.
			tournament_name (str): The name of the tournament (default is "tournament").
		"""
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		tournament_dir = os.path.join(experiment_dir, f"{tournament_name}_{timestamp}")
		os.makedirs(tournament_dir, exist_ok=True)

		# Log tournament info
		tournament_info = {
			"num_agents": len(self.agent_pool.valid_agents)+len(self.agent_pool.invalid_agents),
			"num_rounds": self.num_rounds,
			"target_payoffs": self.target_payoffs,
			"winners_payoffs": [(winner.name, winner.strategy_name, winner.mind.get_total_payoff()) for winner in
								self.get_winners()]
		}
		with open(os.path.join(tournament_dir, "tournament_info.json"), "w") as f:
			json.dump(tournament_info, f, indent=2, default=set_default)

		# Log each agent's info
		agents = self.agent_pool.valid_agents + self.agent_pool.invalid_agents
		for agent in agents:
			agent.save(tournament_dir)

		return True, tournament_dir