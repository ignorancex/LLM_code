import configparser
import logging
from magif.agent.agent import Agent
from magif.utils.data_object import DataObject
from magif.utils.utils import AgentStatus, Mode, read_file, normalize_path
from magif.utils.setup_logger import logger
import os
import unittest


class TestAgentCreation(unittest.TestCase):
	def setUp(self):
		# Step 1: Read configuration
		logging.debug('Test')
		self.config = configparser.ConfigParser()
		self.config.read(normalize_path("unit_tests/CONFIG/test_config.ini"))

		# Step 2: Extract configuration parameters
		self.GAME_DIR = normalize_path(self.config.get("Paths", "GAME_DIR"))
		self.OUT_DIR = self.config.get("Paths", "OUT_DIR")
		if not os.path.exists(self.OUT_DIR):
			os.makedirs(self.OUT_DIR)

		self.game_template_path = normalize_path(self.config.get("Paths", "GAME_TEMPLATE_PATH"))
		self.strategy_template_path = normalize_path(self.config.get("Paths", "STRATEGY_TEMPLATE_PATH"))
		self.feedback_template_path = normalize_path(self.config.get("Paths", "FEEDBACK_TEMPLATE_PATH"))
		self.game_path = normalize_path(self.config.get("Paths", "GAME_PATH"))
		self.strategy_path = normalize_path(self.config.get("Paths", "STRATEGY_PATH"))
		self.game_rules = read_file(self.game_path)
		self.strategy_rules = read_file(self.strategy_path)

	def test_game_autofomalization(self):
		"""Test that autoformalization of games rules works correctly."""
		logger.debug("Test that autoformalization of games rules works correctly.")
		game_description = read_file(os.path.join(self.GAME_DIR, "bs_canonic_numbers.txt"))
		prompt = read_file(self.game_template_path).format(game_description=game_description)
		game_data = DataObject(nl_description=game_description, instruction_prompt=prompt,
							   feedback_prompt=read_file(self.feedback_template_path), mode=Mode.AUTOFORMALIZATION)
		strategy_data = DataObject(rules_path=self.strategy_path, mode=Mode.RULES_PATH)

		agent = Agent(game_data, strategy_data, max_attempts=5)
		self.assertEqual(AgentStatus.CORRECT, agent.status)
		agent.release_solver()

	def test_game_path(self):
		"""Test that loading games rules from a path correctly."""
		logger.debug("Test that loading games rules from a path works correctly.")
		game_data = DataObject(rules_path=self.game_path, mode=Mode.RULES_PATH)
		strategy_data = DataObject(rules_path=self.strategy_path, mode=Mode.RULES_PATH)

		agent = Agent(game_data, strategy_data, autoformalization_on=False)
		self.assertEqual(AgentStatus.CORRECT, agent.status)
		agent.release_solver()

	def test_game_string(self):
		"""Test that loading games rules from a string works correctly."""
		logger.debug("Test that loading games rules from a string works correctly.")
		game_data = DataObject(rules_string=self.game_rules, mode=Mode.RULES_STRING)
		strategy_data = DataObject(rules_path=self.strategy_path, mode=Mode.RULES_PATH)

		agent = Agent(game_data, strategy_data, autoformalization_on=False)
		self.assertEqual(AgentStatus.CORRECT, agent.status)
		agent.release_solver()

	def test_strategy_string(self):
		"""Test that loading strategy rules from a string works correctly."""
		logger.debug("Test that loading strategy rules from a string works correctly.")
		game_data = DataObject(rules_path=self.game_path, mode=Mode.RULES_PATH)
		strategy_data = DataObject(rules_string=self.strategy_rules, mode=Mode.RULES_STRING)

		agent = Agent(game_data, strategy_data, autoformalization_on=False)
		self.assertEqual(AgentStatus.CORRECT, agent.status)
		agent.release_solver()

	def test_strategy_autofomalization(self):
		"""Test that autoformalization of strategy rules works correctly."""
		logger.debug("Test that autoformalization of strategy rules works correctly.")
		strategy_description_path = normalize_path(self.config.get("Paths", "STRATEGY_DESCRIPTION"))
		strategy_description = read_file(strategy_description_path)
		prompt = read_file(self.strategy_template_path).format(strategy_description=strategy_description)
		strategy_data = DataObject(nl_description=strategy_description, instruction_prompt=prompt,
								   feedback_prompt=read_file(self.feedback_template_path), mode=Mode.AUTOFORMALIZATION)
		game_data = DataObject(rules_path=self.game_path, mode=Mode.RULES_PATH)

		agent = Agent(game_data, strategy_data, max_attempts=5)
		self.assertEqual(AgentStatus.CORRECT, agent.status)
		agent.release_solver()

	def test_json(self):
		"""Test that loading agent from json works correctly."""
		logger.debug("Test that loading agent from json works correctly.")

		agent_json = normalize_path(self.config.get("Paths", "AGENT_JSON"))
		agent = Agent(agent_json=agent_json, autoformalization_on=False)

		self.assertEqual(AgentStatus.CORRECT, agent.status)
		agent.release_solver()

	def test_json_and_game_path(self):
		"""Test that loading agent from json and game rules from a path works correctly."""
		logger.debug("Test that loading agent from json and game rules from a path works correctly.")

		agent_json = normalize_path(self.config.get("Paths", "AGENT_JSON"))
		agent = Agent(agent_json=agent_json, autoformalization_on=False)
		game_data = DataObject(rules_path=self.game_path, mode=Mode.RULES_PATH)
		agent.set_game(game_data)

		self.assertEqual(AgentStatus.CORRECT, agent.status)
		agent.release_solver()

	def test_json_and_game_path_and_strategy_path(self):
		"""Test that loading agent from json and game and strategy rules from a path works correctly."""
		logger.debug("Test that loading agent from json and game and strategy rules from a path works correctly.")

		agent_json = normalize_path(self.config.get("Paths", "AGENT_JSON"))
		agent = Agent(agent_json=agent_json, autoformalization_on=False)
		game_data = DataObject(rules_path=self.game_path, mode=Mode.RULES_PATH)
		agent.set_game(game_data)
		strategy_data = DataObject(rules_path=self.strategy_path, mode=Mode.RULES_PATH)
		agent.set_strategy(strategy_data)

		self.assertEqual(AgentStatus.CORRECT, agent.status)
		agent.release_solver()

	def test_json_and_game_autoformalization(self):
		"""Test that loading agent from json and game autoformalization works correctly."""
		logger.debug("Test that loading agent from json and game autoformalization works correctly.")

		agent_json = normalize_path(self.config.get("Paths", "AGENT_JSON"))
		agent = Agent(agent_json=agent_json, max_attempts=5)

		game_description = read_file(os.path.join(self.GAME_DIR, "bs_canonic_numbers.txt"))
		prompt = read_file(self.game_template_path).format(game_description=game_description)
		game_data = DataObject(nl_description=game_description, instruction_prompt=prompt,
							   feedback_prompt=read_file(self.feedback_template_path), mode=Mode.AUTOFORMALIZATION)
		agent.set_game(game_data)

		self.assertEqual(AgentStatus.CORRECT, agent.status)
		agent.release_solver()

	def test_json_and_strategy_autoformalization(self):
		"""Test that loading agent from json and strategy autoformalization works correctly."""
		logger.debug("Test that loading agent from json and strategy autoformalization works correctly.")

		agent_json = normalize_path(self.config.get("Paths", "AGENT_JSON"))
		agent = Agent(agent_json=agent_json)
		strategy_description_path = normalize_path(self.config.get("Paths", "STRATEGY_DESCRIPTION"))
		strategy_description = read_file(strategy_description_path)
		prompt = read_file(self.strategy_template_path).format(strategy_description=strategy_description)
		strategy_data = DataObject(nl_description=strategy_description, instruction_prompt=prompt,
								   feedback_prompt=read_file(self.feedback_template_path), mode=Mode.AUTOFORMALIZATION)
		agent.set_strategy(strategy_data)

		self.assertEqual(AgentStatus.CORRECT, agent.status)
		agent.release_solver()


if __name__ == "__main__":
	unittest.main()
