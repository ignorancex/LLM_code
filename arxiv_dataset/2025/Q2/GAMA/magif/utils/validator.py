import os
import json
import pandas as pd
from magif.solver.solver import Solver
from magif.utils.utils import read_file, normalize_path
import logging
import re


class Validator:
	"""
	A class to validate auto-formalized games.
	"""

	def __init__(self, agents_dir: str, matrices_file: str, validators_dir: str, constraints_only=False):
		"""
		Initializes the Validator with the given directory and file paths.

		Args:
			agents_dir (str): Path to the directory containing agent files.
			matrices_file (str): Path to the file containing matrix data.
			validators_dir (str): Path to the directory containing validators.
		"""
		self.agents_dir = agents_dir
		self.matrices = None
		self.target_payoffs = None
		if matrices_file:
			matrices_file = normalize_path(matrices_file)
			with open(matrices_file, 'r') as file:
				matrices = json.load(file)
			self.matrices = matrices
			self.target_payoffs = self.process_json_to_dataframe(matrices_file)
		self.validators = self.get_validators(normalize_path(validators_dir))
		self.constraints_only = constraints_only
		self.solver_path = normalize_path("magif/solver/solver.pl")  # game-independent part of the solver
		self.strategy = normalize_path("DATA/STRATEGIES/tit-for-tat.pl")  # strategy
		self.general_agent_file = normalize_path("DATA/MISC/general_agent.pl")
		self.solver = None
		self.result_headers = ['filename', 'agent_name', 'status', 'tournament', 'constraints', 'final', 'synt_correct', 'run_correct', 'sem_correct', 'attempts', 'trace']
		self.results = []
		self.actions = {'bs': [('F', 'F'), ('O', 'O'), ('O', 'F'), ('F', 'O')],
						'pd': [('C', 'C'), ('D', 'C'), ('C', 'D'), ('D', 'D')],
						'mp': [('H', 'H'), ('T', 'H'), ('T', 'T'), ('H', 'T')],
						'sh': [('S', 'S'), ('S', 'H'), ('H', 'S'), ('H', 'H')],
						'hd': [('S', 'S'), ('D', 'S'), ('S', 'D'), ('D', 'D')]}
		self.action_sequence = {'bs': [('O', 'O'), ('O', 'F'), ('F', 'F'), ('F', 'O')],
								'pd': [('C', 'C'), ('C', 'D'), ('D', 'D'), ('D', 'C')],
								'mp': [('H', 'H'), ('H', 'T'), ('T', 'T'), ('T', 'H')],
								'sh': [('S', 'S'), ('S', 'H'), ('H', 'H'), ('H', 'S')],
								'hd': [('S', 'S'), ('S', 'D'), ('D', 'D'), ('D', 'S')]}
		self.logger = logging.getLogger('Validator')
		self.logger.setLevel(level=logging.DEBUG)

	def process_json_to_dataframe(self, json_path):
		"""
		Processes a JSON file to extract game file names and calculate payoffs.

		Args:
			json_path (str): Path to the JSON file.

		Returns:
			pd.DataFrame: DataFrame with columns 'Game File' and 'Payoff'.
		"""
		with open(json_path, 'r') as file:
			data = json.load(file)

		records = []
		for game_file, tuples in data.items():
			payoff = sum(pair[0] for pair in tuples)
			records.append({'Game File': game_file, 'Payoff': payoff})

		return pd.DataFrame(records)

	def get_validators(self, validators_dir):
		validators_list = list(os.listdir(validators_dir))
		validators = {filename.replace(".pl", ""): os.path.join(validators_dir, filename) for filename in
					  validators_list}
		return validators

	def shift_right(self, lst, positions):
		"""
		Shifts the elements of a list to the right by the specified number of positions.
		:param lst: List to shift
		:param positions: Number of positions to shift
		:return: The shifted list
		"""
		# Ensure the shift amount doesn't exceed the list's length
		positions = positions % len(lst)
		# Rearrange the list by slicing
		return lst[-positions:] + lst[:-positions]

	def generate_payoff_array(self, filename, variables=False, shift=0):
		# Extract game type from filename
		game_type_match = re.match(r"^([a-z]+)_", filename)
		if not game_type_match:
			raise ValueError("Invalid filename format. Game type not found.")

		game_type = game_type_match.group(1)

		# Check if game type exists in actions
		if game_type not in self.actions:
			raise ValueError(f"Unknown game type '{game_type}'.")

		# Retrieve actions and matrix
		if variables:
			actions = self.action_sequence[game_type]
			if shift != 0:
				actions = self.shift_right(actions, shift)
			matrix = [('X', '_')] * 4

		else:
			actions = self.actions[game_type]
			if filename not in self.matrices:
				raise ValueError(f"Matrix for filename '{filename}' not found.")

			matrix = self.matrices[filename]

		# Generate payoff strings
		payoff_array = []
		for (action1, action2), (payoff1, payoff2) in zip(actions, matrix):
			if variables:
				payoff_str = f"payoff('{action1}', '{action2}', {payoff1}, {payoff2})."
			else:
				payoff_str = f"assertz(payoff('{action1}', '{action2}', {payoff1}, {payoff2}))."
			payoff_array.append(payoff_str)

		return payoff_array

	def compare_sequences(self, actual, target):
		"""
		Compare whether a lists of payoffs contain the same values in the same order.
		:param actual: List of floats
		:param target: List of integers
		:return: True if they contain the same values in the same order, False otherwise.
		"""
		# Check if both lists have the same length
		if len(actual) != len(target):
			return False

		# Compare element by element after rounding the floats
		for float_value, int_value in zip(actual, target):
			if round(float_value) != int_value:
				return False

		return True

	def compare_payoff_sequence(self, filename, actual_sequence, shift=0):
		payoff_matrix_variables = self.generate_payoff_array(filename, variables=True, shift=shift)
		target_sequence = []
		for predicate in payoff_matrix_variables:
			result = self.solver.engine.query(predicate)
			target_sequence += result.data
		same = self.compare_sequences(actual_sequence, target_sequence)
		return same

	def fill_numbers(self, matrix, game_type):
		ms = sorted(matrix, reverse=True)
		if game_type in ['pd', 'sh', 'hd']:
			return f"{game_type}({ms[0]},{ms[1]},{ms[2]},{ms[3]},C,D)."
		if game_type in ['mp']:
			if len(ms) == 4:
				return f"{game_type}({ms[0]},{ms[1]},{ms[2]},{ms[3]},H,T)."
			if len(ms) == 2:
				return f"{game_type}({ms[0]},{ms[0]},{ms[1]},{ms[1]},H,T)."
		if game_type in ['bs']:
			return f"{game_type}({ms[0]},{ms[1]},{ms[2]},F,O)."

	def check_constraints(self, game_type, game_rules):
		try:
			validator = self.validators[game_type]
			self.solver = Solver(read_file(self.solver_path), game_rules, read_file(self.strategy))
			self.solver.consult(validator)
			self.solver.consult(normalize_path("DATA/MISC/unique_payoffs.pl"))
			result = self.solver.engine.query("list_unique_payoffs(X).", 1)
			matrix = result.data[0]
			predicate = self.fill_numbers(matrix, game_type)
			result = self.solver.engine.query(predicate)
			values = result.data
		except Exception as e:
			print(f"Could not validate constraints: {e}")
			return False

		if values:
			return True
		else:
			return False

	def validate_all(self):
		"""
		Validates autoformalized code.

		Returns:
			pd.DataFrame: Validation results.
		Raises:
			ValueError: If any validation fails.
		"""
		validator_types = set(self.validators.keys())

		for i, agent_dir in enumerate(os.listdir(self.agents_dir)):
			self.logger.debug(f"Instance {i} {agent_dir}")
			game_type = agent_dir[:2]

			if game_type not in validator_types:
				raise ValueError(f"{game_type} not in validators!")

			agent_path = os.listdir(os.path.join(self.agents_dir, agent_dir))

			for agent in agent_path:
				if "agent" not in agent:
					continue  # Skip non-agent files

				filename = '_'.join(agent_dir.split('_')[:3]) + '.txt'
				name = agent[6:-5]
				result_row = [filename, name]
				tournament_status = True

				with open(os.path.join(self.agents_dir, agent_dir, agent), 'r') as file:
					self.logger.debug(f"Validating agent: {name}")
					data = json.load(file)
					status = data.get('status', 'unknown')
					result_row.append(status)

					synt_correct = status != "syntactic_error"
					if status != 'correct':
						self.logger.debug(f"Agent {name} is {status}")
						result_row += [False, False, False, synt_correct, False, False]
						self.results.append(result_row)
						continue

					# If syntactically correct, validate further
					if not self.constraints_only and self.target_payoffs is not None:
						target_payoff = self._get_target_payoff(filename)
						total_payoff = data.get('total_payoff')

						if total_payoff != target_payoff:
							self.logger.debug(f"Agent {name} did not achieve target payoff")
							tournament_status = False
						else:
							tournament_status = self._validate_sequence(filename, data.get('payoffs'))

					elif not self.constraints_only:
						tournament_status = False

					result_row.append(tournament_status)

					# Validate constraints
					game_rules = data.get('game_rules', [])
					constraint_status = self.check_constraints(game_type, game_rules)
					self.logger.debug(f"Agent {name} satisfies constraints")
					result_row.append(constraint_status)

					combined_status = tournament_status and constraint_status
					result_row.append(combined_status)

					run_correct = status == "correct"
					if not self.constraints_only and self.target_payoffs is not None:
						sem_correct = combined_status
					elif not self.constraints_only:
						sem_correct = "N/A"
					else:
						sem_correct = constraint_status

					result_row += [synt_correct, run_correct, sem_correct]
					result_row += [max(1, data.get('attempts', 1)), data.get('trace_messages', [])]

					self.results.append(result_row)

		return pd.DataFrame(self.results, columns=self.result_headers)

	def _get_target_payoff(self, filename):
		return self.target_payoffs.loc[
			self.target_payoffs['Game File'] == filename, 'Payoff'].values[0]

	def _validate_sequence(self, filename, actual_sequence):
		self.solver = Solver(
			read_file(self.solver_path),
			read_file(self.general_agent_file),
			read_file(self.strategy)
		)
		for predicate in self.generate_payoff_array(filename):
			self.solver.engine.query(predicate)

		if self.compare_payoff_sequence(filename, actual_sequence):
			return True
		return self.compare_payoff_sequence(filename, actual_sequence, shift=2)
