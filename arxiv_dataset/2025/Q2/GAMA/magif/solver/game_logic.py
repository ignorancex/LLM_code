from magif.solver.engine import PrologEngine
from typing import Any, Tuple


class GameSolver:
    """
    Contains domain-specific Prolog logic for game operations such as moves, player names, and payoffs.
    """

    def __init__(self, engine: PrologEngine):
        """
        Initialize the GameSolver with a given Prolog engine.

        Args:
            engine (PrologEngine): The engine used to query the Prolog environment.
        """
        self.engine = engine

    def get_possible_moves(self) -> Tuple[bool, Any]:
        """
        Retrieve all possible moves from the current game state.

        Returns:
            Tuple[bool, Any]: (True, list of moves) or (False, error message).
        """
        result = self.engine.query("possible(move(_,X), s0).")
        return result.success, result.data if result.success else result.error

    def get_player_names(self) -> Tuple[bool, Any]:
        """
        Retrieve the list of player names from the initial game state.

        Returns:
            Tuple[bool, Any]: (True, list of names) or (False, error message).
        """
        result = self.engine.query("holds(player(N), s0).")
        return result.success, result.data if result.success else result.error

    def get_default_move(self, player_name: str) -> Tuple[bool, Any]:
        """
        Retrieve the default move for a specific player.

        Args:
            player_name (str): The player's name.

        Returns:
            Tuple[bool, Any]: (True, list with one default move) or (False, error message).
        """
        result = self.engine.query(f"initially(default_move({player_name}, X), s0).", 1)
        return result.success, result.data if result.success else result.error

    def update_default_move(self, move: str) -> Tuple[bool, Any]:
        """
        Set a new default move.

        Args:
            move (str): The move to be set as the default.

        Returns:
            Tuple[bool, Any]: (True, confirmation) or (False, error message).
        """
        result = self.engine.query(f"initialise(default_move(_, '{move}'), s0).")
        return result.success, result.data if result.success else result.error

    def select_move(self, agent_name: str) -> Tuple[bool, Any]:
        """
        Use the solver's logic to select a move for the specified agent.

        Args:
            agent_name (str): The name of the agent.

        Returns:
            Tuple[bool, Any]: (True, selected move) or (False, error message).
        """
        result = self.engine.query(f"select({agent_name}, _, s0, M).",1)
        return result.success, result.data[0] if result.success else result.error

    def calculate_payoff(
        self, player: str, opponent: str, move_p: str, move_o: str
    ) -> Tuple[bool, Any]:
        """
        Calculate the payoff for a player given both players' moves.

        Args:
            player (str): Name of the player.
            opponent (str): Name of the opponent.
            move_p (str): Move made by the player.
            move_o (str): Move made by the opponent.

        Returns:
            Tuple[bool, Any]: (True, payoff as float) or (False, error message).
        """
        query = (
            f"finally(goal({player}, U), "
            f"do(move({player}, '{move_p}'), "
            f"do(move({opponent}, '{move_o}'), s0)))."
        )
        result = self.engine.query(query, 1)
        return result.success, result.data[0] if result.success else result.error

    def update_opponent_last_move(self, opponent_name: str, opponent_move: str) -> Tuple[bool, Any]:
        """
        Update the game state with the opponent's last move.

        Args:
            opponent_name (str): Name of the opponent.
            opponent_move (str): Move made by the opponent.

        Returns:
            Tuple[bool, Any]: (True, confirmation) or (False, error message).
        """
        query = f"initialise(last_move({opponent_name}, '{opponent_move}'), s0)."
        result = self.engine.query(query)
        return result.success, result.data if result.success else result.error

