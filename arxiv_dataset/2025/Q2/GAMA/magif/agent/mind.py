from magif.utils.setup_logger import logger
from magif.utils.utils import AgentStatus
from typing import Optional


class Mind:
	"""
    Represents the decision-making and internal logic of an Agent.

    The agent:
    1. Acts, by selecting a move in a given round.
    2. Perceives the opponent move and adds it to the memory.
    3. Revises by calculating the payoff based on all players' moves in the given round, and updates the last opponent
    move in the solver, which may influence the strategy in the next round.

    Attributes:
        agent (Agent): The agent instance to manage the decision-making and memory.
    """

	def __init__(self,
				 agent):
		"""
		Initializes the Mind with an agent.

		Args:
			agent (Agent): The agent instance.
		"""
		self.agent = agent

	def observe(self, opponent_move):
		"""
		Observe the opponent move in the current round and add it to memory.

		Args:
			opponent_move (str): The move made by the opponent in the current round.
		"""
		if not self.agent.solver:
			logger.debug(f"Agent {self.agent.name} cannot update payoff due to an uninitialized solver.")
			self.agent.status = AgentStatus.RUNTIME_ERROR
			return False

		# Log the opponent's move
		self.agent.memory.opponent_moves.append(opponent_move)

	def think(self):
		"""
		Revise the agent's state based on the most recent moves and update its solver.

		This method performs the following steps:
		1. Validates that the agent's memory of moves and game players is initialized and sufficiently populated.
		2. Calculates the agent's payoff for the last move using the solver.
		3. Updates the solver state with the opponent's last move.
		4. Logs the payoff and the opponent's last move for future reference.

		Returns:
		    Optional[bool]:
		        - `True` if the payoff was successfully calculated and the solver state was updated.
		        - `False` if the payoff calculation or solver update failed.
		        - `None` if the memory or game player data is not initialized or too short.
		"""
		if not self.agent.memory.moves or not self.agent.memory.opponent_moves or not self.agent.game.game_players:
			self.agent.status = AgentStatus.RUNTIME_ERROR
			logger.debug(f"Memory of moves or player names not initialised!")
			return None
		if len(self.agent.memory.moves) < 1 or len(self.agent.memory.opponent_moves) < 1 or len(
				self.agent.game.game_players) < 2:
			self.agent.status = AgentStatus.RUNTIME_ERROR
			logger.debug(f"Memory of moves or player names not too short!")
			return None

		# Step 2: Calculate payoff using the solver
		opponent_move = self.agent.memory.opponent_moves[-1]
		opponent_name = self.agent.game.game_players[1]
		payoff_success, payoff = self.agent.solver.calculate_payoff(self.agent.game.game_players[0], opponent_name,
													self.agent.memory.moves[-1], opponent_move)
		if not payoff_success:
			# TODO re-formalize
			self.agent.status = AgentStatus.RUNTIME_ERROR
			logger.debug(f"Payoff not calculated!")
			return False

		# Step 3: Update the solver state with the opponent's last move
		update_success, _ = self.agent.solver.update_opponent_last_move(opponent_name, opponent_move)
		if not update_success:
			#TODO re-formalize
			self.agent.status = AgentStatus.RUNTIME_ERROR
			logger.debug(f"Opponent's last move not updated!")
			return False

		# Step 4: Log the successful update and store the payoff
		self.agent.memory.payoffs.append(payoff)
		logger.info(f"Agent {self.agent.name} received payoff: {payoff} and logged opponent's move: {opponent_move}")
		return True

	def act(self):
		"""
        The agent makes a move in the tournament.

        Uses the solver to determine the next move based on the current game state.
        If a move is successfully selected, it is stored in the agent's move history.

        Returns:
        	Optional[str]: The move selected by the agent, or None if no valid move is made.
        """
		if not self.agent.solver:
			logger.debug(f"Agent {self.agent.name} is unable to play due to an uninitialized solver.")
			self.agent.status = AgentStatus.RUNTIME_ERROR
			return None

		if not self.agent.game.game_players:
			logger.debug(f"Agent {self.agent.name} is unable to play due to lack of players names.")
			self.agent.status = AgentStatus.RUNTIME_ERROR
			return None

		agent_name = self.agent.game.game_players[0]
		logger.debug(f"Agent {self.agent.name} with strategy {self.agent.strategy_name} is making a move.")

		# Step 1: Attempt to get a move using the solver
		success, move = self.agent.solver.select_move(agent_name)
		if success:
			#TODO re-formalize
			self.agent.memory.moves.append(move)
			logger.info(f"Agent {self.agent.name} with strategy {self.agent.strategy_name} made move: {move}")
			return move

		# If no move is selected, log the error and update status
		logger.debug(f"Agent {self.agent.name} did not select a move!")
		self.agent.status = AgentStatus.RUNTIME_ERROR
		return None

	def get_total_payoff(self, log=True) -> float:
		"""
		Get the total payoff accumulated by the agent.

		Returns:
			float: The total sum of payoffs.
		"""
		total = sum(self.agent.memory.payoffs)
		if log:
			logger.info(f"Total payoff for agent {self.agent.name} with strategy {self.agent.strategy_name}: {total}")
		return total
