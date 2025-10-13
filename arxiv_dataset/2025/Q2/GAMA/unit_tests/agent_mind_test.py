import unittest
import os
import logging
from configparser import ConfigParser
from magif.agent.agent import Agent
from magif.utils.utils import normalize_path
import json


class TestAgentMindFunctions(unittest.TestCase):
	def setUp(self):
		# Step 1: Read configuration
		logging.debug('Setting up TestAgentMindFunctions')
		self.config = ConfigParser()
		self.config.read(normalize_path("unit_tests/CONFIG/test_config.ini"))

		# Step 2: Extract configuration parameters
		self.GAME_DIR = normalize_path(self.config.get("Paths", "GAME_DIR"))
		self.OUT_DIR = self.config.get("Paths", "OUT_DIR")
		if not os.path.exists(self.OUT_DIR):
			os.makedirs(self.OUT_DIR)

		self.agent_json = normalize_path(self.config.get("Paths", "AGENT_JSON"))
		self.agent = Agent(agent_json=self.agent_json, autoformalization_on=False)  # Initialize Agent

	def tearDown(self):
		self.agent.release_solver()
		del self.agent

	def test_act(self):
		"""Test the agent's ability to make a move using the `act` method."""
		logging.debug("Testing the `act` method.")
		move = self.agent.mind.act()  # Use the Mind instance within Agent
		self.assertIsNotNone(move, "The agent should successfully make a move.")
		self.assertEqual("Move1", move, "The agent should successfully make a move.")
		self.assertIn(move, self.agent.memory.moves, "The move should be stored in the agent's memory.")

	def test_perceive(self):
		"""Test the agent's ability to perceive an opponent's move using the `perceive` method."""
		logging.debug("Testing the `perceive` method.")
		opponent_move = "Move2"
		self.agent.mind.observe(opponent_move)  # Use the Mind instance within Agent
		self.assertIn(opponent_move, self.agent.memory.opponent_moves,
					  "The opponent's move should be stored in the agent's memory.")

	def test_revise(self):
		"""Test the agent's ability to revise its state using the `revise` method."""
		logging.debug("Testing the `revise` method.")
		# Set up initial moves to enable revision
		self.agent.memory.moves.append("Move1")
		opponent_move = "Move2"
		self.agent.memory.opponent_moves.append(opponent_move)
		self.agent.game.game_players = ["player1", "player2"]

		success = self.agent.mind.think()  # Use the Mind instance within Agent
		payoff = self.agent.memory.payoffs[-1]
		result = self.agent.solver.engine.query(f"holds(last_move({self.agent.game.game_players[1]}, {opponent_move}), s0).", 1)
		last_move = result.data

		self.assertTrue(success, "The revise method should successfully update the solver and state.")
		self.assertEqual(1, payoff, "Payoff should be equal to 1")
		self.assertEqual(opponent_move, last_move[0], f"Opponent move should be {opponent_move}.")
		self.assertGreater(len(self.agent.memory.payoffs), 0,
						   "The payoff should be calculated and added to the agent's memory.")

	def test_memory_save(self):
		"""Test the agent's ability to save memory to a JSON file after acting, perceiving, and revising."""
		logging.debug("Testing saving memory to a JSON file.")

		# Set up opponent moves
		opponent_moves = ["Move1", "Move1", "Move1", "Move1"]

		# Perform act, perceive, revise loop four times
		for opponent_move in opponent_moves:
			move = self.agent.mind.act()  # Agent makes a move
			self.assertIsNotNone(move, "The agent should successfully make a move.")
			self.agent.mind.observe(opponent_move)  # Agent perceives opponent's move
			success = self.agent.mind.think()  # Agent revises based on moves
			self.assertTrue(success, "The revise method should successfully update the solver and state.")

		# Save memory to a temporary JSON file
		self.agent.save("./")

		# Load the JSON file and check the "payoffs" field
		with open("agent_Jekuti.json", "r") as file:
			memory_data = json.load(file)

		# Assert that "payoffs" contains four entries, all equal to 1
		self.assertIn("payoffs", memory_data, "The memory JSON should contain the 'payoffs' field.")
		self.assertEqual(len(memory_data["payoffs"]), 4, "The 'payoffs' field should contain four entries.")
		self.assertTrue(all(payoff == 0 for payoff in memory_data["payoffs"]),
						"All payoffs should be equal to 0.")


if __name__ == "__main__":
	unittest.main()
