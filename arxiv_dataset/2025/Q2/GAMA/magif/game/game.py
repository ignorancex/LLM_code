from typing import List, Optional


class Game:
	"""
	Represents a game with natural language descriptions, Prolog rules, possible moves, and player names.

	Attributes:
	    game_string (Optional[str]): A natural language description of the game.
	    strategy_string (Optional[str]): A natural language description of the strategy.
	    game_rules (Optional[str]): A Prolog program describing the rules of the game.
	    strategy_rules (Optional[str]): A Prolog program describing the strategy used in the game.
	    strategy_name (str): The name of the strategy (default is "unnamed_strategy").
	    game_moves (List[str]): A list of possible moves in the game.
	    game_players (List[str]): A list of players participating in the game.
	    default_move (Optional[str]): The default move for the game (if applicable).
	"""

	def __init__(self, game_string: Optional[str] = None, strategy_string: Optional[str] = None, game_rules: Optional[str] = None, strategy_rules: Optional[str] = None, game_moves: Optional[List[str]] = None, game_players: Optional[List[str]] = None):
		"""
		Initializes the Game object with optional parameters for game descriptions, rules, moves, and players.

		Args:
		    game_string (Optional[str]): A natural language description of the game.
		    strategy_string (Optional[str]): A natural language description of the strategy.
		    game_rules (Optional[str]): A Prolog program describing the rules of the game.
		    strategy_rules (Optional[str]): A Prolog program describing the strategy used in the game.
		    game_moves (Optional[List[str]]): A list of possible moves in the game (default: an empty list).
		    game_players (Optional[List[str]]): A list of players participating in the game (default: an empty list).
		"""
		# Descriptions of the game and strategy in natural language.
		self.game_string: Optional[str] = game_string
		self.strategy_string: Optional[str] = strategy_string

		# Prolog rules for the game and strategy.
		self.game_rules: Optional[str] = game_rules
		self.strategy_rules: Optional[str] = strategy_rules

		# Default name for the strategy if none is provided.
		self.strategy_name = "unnamed_strategy"

		# Initialize moves and players.
		self.game_moves: List[str] = game_moves if game_moves else []
		self.game_players: List[str] = []
		self.default_move = None

	def set_possible_moves(self, moves: List[str]) -> None:
		"""
		Set the possible moves for the game.

		Args:
			moves (List[str]): A list of valid moves.
		"""
		if not isinstance(moves, list):
			raise ValueError("Moves should be a list of strings.")
		if not all(isinstance(move, str) for move in moves):
			if not all(isinstance(move, int) for move in moves):
				raise ValueError("Moves should be a list of strings.")
			else:
				moves = [str(m) for m in moves]
		self.game_moves = moves

	def get_possible_moves(self) -> List[str]:
		"""
		Get the list of possible moves for the game.

		Returns:
			List[str]: A list of possible moves.
		"""
		return self.game_moves

	def add_possible_move(self, move: str) -> None:
		"""
		Add a single move to the list of possible moves.

		Args:
			move (str): A move to be added.
		"""
		if move not in self.game_moves:
			self.game_moves.append(move)

	def set_players(self, players: List[str]) -> None:
		"""
		Set the list of players for the game.

		Args:
			players (List[str]): A list of player names.
		"""
		if not isinstance(players, list) or not all(isinstance(player, str) for player in players):
			raise ValueError("Players should be a list of strings.")
		self.game_players = players

	def get_players(self) -> List[str]:
		"""
		Get the list of players for the game.

		Returns:
			List[str]: A list of player names.
		"""
		return self.game_players

	def add_player(self, player: str) -> None:
		"""
		Add a single player to the list of players.

		Args:
			player (str): A player name to be added.
		"""
		if player not in self.game_players:
			self.game_players.append(player)

	def set_game_rules(self, rules: str) -> None:
		"""
		Set the rules for the game.

		Args:
			rules (str): A Prolog program describing the rules of the game.
		"""
		if not isinstance(rules, str):
			raise ValueError("Game rules should be a string.")
		self.game_rules = rules

	def get_game_rules(self) -> Optional[str]:
		"""
		Get the rules of the game.

		Returns:
			Optional[str]: A string with the game rules, or None if not set.
		"""
		return self.game_rules

	def clear_rules(self) -> None:
		"""
		Clear the game rules.
		"""
		self.game_rules = None

	def __repr__(self) -> str:
		"""
		Return a string representation of the Game object.

		Returns:
			str: A string representation of the Game.
		"""
		return (f"Game(description={self.game_string[:30]}..., "
				f"rules={'set' if self.game_rules else 'not set'}, "
				f"moves={len(self.game_moves)}, players={len(self.game_players)})")
