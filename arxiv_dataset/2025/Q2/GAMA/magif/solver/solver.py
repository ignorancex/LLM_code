from magif.solver.engine import PrologEngine
from magif.solver.prolog_validator import PrologValidator
from magif.solver.game_logic import GameSolver
from magif.solver.solver_utils import file_writer
from swiplserver import PrologMQI
from typing import Any, Optional, Tuple

class Solver:
    """
    Coordinates Prolog engine interaction, code validation, and game logic operations.
    """

    def __init__(self, solver_string: str, game_string: str = None, strategy_string: str = None, logger=None):
        """
        Initialize the Solver with provided Prolog code segments.

        Args:
            solver_string (str): Prolog code for the solver logic.
            game_string (str): Prolog rules for the game logic.
            strategy_string (str): Prolog strategy logic.
        """
        # Initialize validity and trace attributes.
        self.valid: bool = False
        self.trace: Optional[str] = None

        self.solver_string = solver_string
        self.game_string = game_string
        self.strategy_string = strategy_string

        self.engine = PrologEngine(thread_creator=PrologMQI().create_thread)
        self.validator = PrologValidator(self.engine, file_writer)
        self.game_solver = GameSolver(self.engine)

        self._validate_all()

    def _validate_all(self):
        """
        Validate the solver, game, and strategy Prolog code (if provided).
        Sets self.valid and self.trace accordingly.
        """
        components = [("solver", self.solver_string), ("game", self.game_string), ("strategy", self.strategy_string)]
        all_valid = True
        for label, code in components:
            if code:
                result = self.validator.validate(code)
                if not result.is_valid:
                    all_valid = False
                    self.trace = result.trace
        self.valid = all_valid

    def validate(self, rules):
        result = self.validator.validate(rules)
        return result.is_valid, result.trace

    def release(self):
        """
        Release resources by stopping the Prolog engine.
        """
        self.engine.stop()

    def get_params(self):
        """
		Retrieves non-None parameters from the solver, game, and strategy strings.

		Returns:
			List[str]: A list containing the non-None values of `solver_string`, `game_string`, and `strategy_string`.
		"""
        # Filter out `None` values and return the remaining strings.
        return [item for item in [self.solver_string, self.game_string, self.strategy_string] if item is not None]

    def consult(self, file_path):
        self.engine.consult(file_path)

    def get_player_names(self):
        """
        Retrieve the list of player names from the game state.

        Returns:
            Tuple[bool, Any]: (True, list of players) or (False, error message).
        """
        return self.game_solver.get_player_names()

    def get_possible_moves(self):
        """
        Get all valid possible moves from the current game state.

        Returns:
            QueryResult: Result containing possible moves or error.
        """
        return self.game_solver.get_possible_moves()

    def get_default_move(self, player_name):
        """
        Retrieve the default move for a given player.

        Args:
            player_name (str): The player's name.

        Returns:
            QueryResult: Result with default move or error.
        """
        return self.game_solver.get_default_move(player_name)

    def update_default_move(self, move):
        """
        Update the default move in the game environment.

        Args:
            move (str): Move to set as the new default.

        Returns:
            QueryResult: Success of update.
        """
        return self.game_solver.update_default_move(move)

    def calculate_payoff(self, player, opponent, move_p, move_o):
        """
        Calculate payoff based on two players' moves.

        Args:
            player (str): Player's name.
            opponent (str): Opponent's name.
            move_p (str): Player's move.
            move_o (str): Opponent's move.

        Returns:
            QueryResult: Payoff result or error.
        """
        return self.game_solver.calculate_payoff(player, opponent, move_p, move_o)

    def select_move(self, agent_name: str) -> Tuple[bool, Any]:
        """
        Select a move for the given agent using the game logic.

        Args:
            agent_name (str): Name of the agent.

        Returns:
            Tuple[bool, Any]: (True, selected move) or (False, error message).
        """
        return self.game_solver.select_move(agent_name)

    def update_opponent_last_move(self, opponent_name: str, opponent_move: str) -> Tuple[bool, Any]:
        """
        Update the internal game state with the opponent's most recent move.

        Args:
            opponent_name (str): Name of the opponent.
            opponent_move (str): The move made.

        Returns:
            Tuple[bool, Any]: (True, confirmation) or (False, error message).
        """
        return self.game_solver.update_opponent_last_move(opponent_name, opponent_move)