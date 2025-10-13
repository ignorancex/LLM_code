import unittest
import itertools
from magif.environment.environment import Environment
from magif.agent.agent import Agent
from magif.environment.agent_pool import AgentPool
import logging
from magif.utils.setup_logger import logger
from magif.utils.utils import generate_agent_name, Mode, normalize_path
from magif.utils.data_object import DataObject


class TestTournament(unittest.TestCase):
	def setUp(self):
		"""
		Set up the testing environment by preparing agent JSON data and initializing required variables.
		"""
		# Path to test configuration or JSON files
		logging.info('Test')
		logger.setLevel(level=logging.INFO)
		self.agent_json_path = normalize_path("unit_tests/LOGS/agent_Jekuti.json")
		self.logdir = normalize_path("unit_tests/LOGS")

		# Number of rounds and dummy matchmaker
		self.num_rounds = 4
		self.match_maker = lambda agents: list(itertools.combinations_with_replacement(agents, 2))

		# Initialize agent pool
		self.agent_pool = AgentPool()

	def _load_agents_from_json(self):
		"""
		Helper method to load agents from JSON files and add them to the agent pool.
		"""
		for i in range(3):
			agent = Agent(agent_json=self.agent_json_path, autoformalization_on=False)
			agent.name = generate_agent_name(3)
			self.agent_pool.add_agent(agent)

	def test_tournament_round_robin_play(self):
		"""
		Test that a tournament runs correctly with agents loaded from JSON files.
		"""
		logger.info("Testing tournament play with agents loaded from JSON in round robin mode.")

		# Load agent data from JSON files
		self._load_agents_from_json()

		tournament = Environment(
			agent_pool=self.agent_pool,
			num_rounds=self.num_rounds,
			match_maker=self.match_maker,
			target_payoffs=[0] * 16
		)

		# Run the tournament
		tournament.play_tournament()

		# Validate results
		winners = tournament.get_winners()
		self.assertTrue(len(winners) == 3, "Tournament must have at least one winner.")
		self.agent_pool.clean_agents()

	def test_tournament_logging(self):
		"""
		Test that tournament results are logged correctly.
		"""
		logger.info("Testing tournament logging with agents loaded from JSON.")

		self._load_agents_from_json()
		tournament_name = "test_tournament"
		tournament = Environment(
			agent_pool=self.agent_pool,
			num_rounds=self.num_rounds,
			match_maker=self.match_maker
		)

		# Run the tournament
		tournament.play_tournament()

		# Log the results
		tournament_dir = self.logdir
		success, _ = tournament.log_tournament(tournament_dir, tournament_name)
		self.assertEqual(True, success)
		self.agent_pool.clean_agents()

	def test_tournament_clones_play(self):
		"""
		Test that a tournament runs correctly with agents loaded from JSON files.
		"""
		logger.info("Testing tournament play with agents loaded from JSON in clones mode.")

		# Load agent data from JSON files
		for i in range(3):
			agent = Agent(agent_json=self.agent_json_path, autoformalization_on=False)
			agent.name = generate_agent_name(3)
			game_data = DataObject(rules_path=normalize_path("unit_tests/DATA/MISC/bos_agent.pl"), mode=Mode.RULES_PATH)
			agent.set_game(game_data)
			self.agent_pool.add_agent(agent)

		# Record original number of agents
		agents_num = len(self.agent_pool.valid_agents)

		# Add copies of an agent with tat-for-tit strategy
		for i in range(agents_num):
			agent = self.agent_pool.valid_agents[i]
			clone = Agent(agent_json=self.agent_json_path, autoformalization_on=False)
			clone.name = generate_agent_name(3)

			game_data = DataObject(rules_string=agent.game.game_rules, mode=Mode.RULES_STRING)
			strategy_data = DataObject(rules_path=normalize_path("DATA/STRATEGIES/anti-tit-for-tat.pl"), mode=Mode.RULES_PATH)
			clone.set_game(game_data)
			clone.set_strategy(strategy_data)

			self.agent_pool.add_agent(clone)

		# Create matching: (original_agent, clone)
		self.match_maker = lambda agents: [(agents[i], agents[i+agents_num]) for i in range(agents_num)]

		tournament = Environment(
			agent_pool=self.agent_pool,
			num_rounds=self.num_rounds,
			match_maker=self.match_maker,
			target_payoffs=[3.0, 3.0, 3.0]
		)

		# Run the tournament
		tournament.play_tournament()

		# Remove clones for the evaluation
		self.agent_pool.truncate_pool(agents_num)

		# Validate results
		winners = tournament.get_winners()

		# Log the results
		tournament_dir = self.logdir
		tournament.log_tournament(tournament_dir, "bs_test_tournament")

		self.assertTrue(len(winners) == 3, "Tournament must have at least one winner.")
		self.agent_pool.clean_agents()

	def test_agent_pool_integrity(self):
		"""
		Test the integrity of the agent pool after the tournament.
		"""
		logger.info("Testing agent pool integrity after tournament.")

		# Load agent data from JSON files
		self._load_agents_from_json()
		tournament = Environment(
			agent_pool=self.agent_pool,
			num_rounds=self.num_rounds,
			match_maker=self.match_maker
		)

		# Run the tournament
		tournament.play_tournament()

		# Validate agent pool
		self.assertTrue(
			len(self.agent_pool.valid_agents) > 0,
			"Agent pool should have valid agents after the tournament."
		)
		self.assertTrue(
			all(isinstance(agent, Agent) for agent in self.agent_pool.valid_agents),
			"All valid agents should be instances of the Agent class."
		)
		self.agent_pool.clean_agents()


if __name__ == "__main__":
	unittest.main()
