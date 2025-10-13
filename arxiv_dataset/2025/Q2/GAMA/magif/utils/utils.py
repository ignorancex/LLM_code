import os
from pathlib import Path
import random
import re
from magif.utils.setup_logger import logger
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from enum import Enum


class AgentStatus(Enum):
	CORRECT = "correct"
	INITIALIZING = "initializing"
	SYNTACTIC_ERROR = "syntactic_error"
	MISSING_PREDICATES = "missing_predicates"
	INSTRUCTION_ERROR = "instruction_following_error"
	RUNTIME_ERROR = "runtime_error"


class Mode(Enum):
    RULES_STRING = "rules_string"
    RULES_PATH = "rules_path"
    AUTOFORMALIZATION = "autoformalization"


def normalize_path(path: str) -> str:
    """
    Converts a relative path to an absolute path using the project's root directory.
    """
    base_dir = Path(__file__).resolve().parents[2]
    path = Path(path)

    return str(path.resolve()) if path.is_absolute() else str((base_dir / path).resolve())

def generate_syllable() -> str:
	"""
	Generates a random syllable using a combination of consonants and vowels.

	Returns:
		str: A randomly generated syllable.
	"""
	consonants = "bcdfghjklmnpqrstvwxyz"
	vowels = "aeiou"

	# Pick a random consonant and vowel
	consonant = random.choice(consonants)
	vowel = random.choice(vowels)

	return consonant + vowel


def generate_agent_name(num_syllables: int = 2) -> str:
	"""
	Generates a random agent name by combining a specified number of syllables.

	The generated name is capitalized, with the first letter in uppercase.

	Args:
		num_syllables (int): The number of syllables to use in the name (default is 2).

	Returns:
		str: A randomly generated agent name.

	Raises:
		ValueError: If the number of syllables is less than 1.
	"""
	name = ''.join(generate_syllable() for _ in range(num_syllables))
	return name.capitalize()  # Capitalize the first letter of the name


def read_file(filename: str) -> Optional[str]:
	"""
	Reads the content of a file and returns it as a string.

	Args:
		filename (str): The path to the file to be read.

	Returns:
		Optional[str]: The content of the file as a string if successful, or None if an error occurs.

	Raises:
		FileNotFoundError: If the specified file does not exist.
		IOError: If an error occurs while reading the file.
	"""
	try:
		with open(filename) as f:
			s = f.read()
			return s
	except FileNotFoundError:
		logger.error(f"The file {filename} was not found.")
		return None
	except IOError:
		logger.error(f"An error occurred while reading the file {filename}.")
		return None


def parse_axioms(response: str) -> str:
	"""
	Parses game axioms from the given response and extracts the content enclosed within '@' symbols.

	This function looks for content enclosed between two '@' symbols in the given response.
	If the content is found, it extracts and returns it. If no such content is found, a `ValueError` is raised.

	Args:
		response (str): The response string containing the game axioms to be parsed.

	Returns:
		str: A string containing the extracted game axioms.

	Raises:
		ValueError: If no content matching the pattern is found in the response.
	"""
	pattern = r'(?m)^@([^@]+)@'
	match = re.search(pattern, response)
	if match is None:
		raise ValueError(f"No match found for pattern: {pattern}")
	game_axioms = match.group(1)

	return game_axioms


def parse_trace(log: str) -> List[Dict[str, any]]:
	"""
	Parses a log string to extract warning and error messages, along with their line numbers.

	This function uses regular expressions to identify warnings and errors in the log. It then extracts
	the line numbers and messages, formats them, and returns a list of parsed entries.

	Args:
		log (str): The log string containing warnings and errors.

	Returns:
		List[Dict[str, any]]: A list of dictionaries, where each dictionary represents a parsed entry with:
							  - 'type': The type of the entry ('Warning' or 'Error').
							  - 'line': The line number where the issue occurred.
							  - 'message': The extracted warning or error message.
	"""
	parsed_entries = []

	# Regex patterns for warnings and errors
	warning_start_pattern = r"Prolog: Warning: .*:(\d+):"
	warning_message_pattern = r"Prolog: Warning:\s+(.*)"
	error_pattern = r"Prolog: ERROR: .*:(\d+):(\d+): (.*)"

	# Split the log into lines
	lines = log.splitlines()

	current_warning = None

	# Process each line
	for line in lines:
		# Match the start of a warning
		warning_start_match = re.match(warning_start_pattern, line)
		if warning_start_match:
			if current_warning:  # Append the previous warning if one exists
				parsed_entries.append(current_warning)
			line_number = warning_start_match.group(1)
			current_warning = {
				'type': 'Warning',
				'line': int(line_number),
				'message': ''
			}
			continue

		# Match warning messages (continuation lines)
		if current_warning:
			warning_message_match = re.match(warning_message_pattern, line)
			if warning_message_match:
				current_warning['message'] += warning_message_match.group(1).strip() + " "

		# Match errors (if applicable)
		error_match = re.match(error_pattern, line)
		if error_match:
			line_number = error_match.group(1)
			column_number = error_match.group(2)
			message = error_match.group(3).strip()
			parsed_entries.append({
				'type': 'Error',
				'line': int(line_number),
				'column': int(column_number),
				'message': re.sub(r"/[^:]+:", "line ", message)
			})

	# Append the last warning if it exists
	if current_warning:
		parsed_entries.append(current_warning)

	# Clean up the messages by removing extra spaces
	for entry in parsed_entries:
		entry['message'] = entry['message'].strip()

	return parsed_entries


def process_trace(trace: str, full_solver: str) -> List[Dict[str, Any]]:
	"""
	Processes a trace log to extract error/warning messages and associates them with their corresponding lines in the solver code.

	Args:
		trace (str): The trace log containing warnings and errors.
		full_solver (str): The full solver code as a string, split by lines.

	Returns:
		List[Dict[str, Any]]: A list of dictionaries where each entry contains:
			- 'type': The type of message ('Warning' or 'Error').
			- 'line': The line number where the issue occurred.
			- 'message': The extracted warning or error message.
			- 'line_content': The actual content of the corresponding line in the solver code.
	"""
	messages = parse_trace(trace)
	solver_lines = full_solver.split('\n')
	for message in messages:
		line = solver_lines[int(message['line']) - 1]
		message['line_content'] = line
	lines = process_trace_messages(messages, full_solver)
	return lines


def process_trace_messages(messages: List[Dict[str, str]], solver: str) -> str:
	"""
	Processes trace messages to generate a report of lines that produced warnings or errors.

	This function checks if each message's line content is present in the solver code and, if so,
	formats a report indicating which lines caused warnings or errors.

	Args:
		messages (List[Dict[str, str]]): A list of dictionaries containing trace messages with:
			- 'type': The type of message ('Warning' or 'Error').
			- 'line_content': The content of the line that caused the issue.
			- 'message': The detailed warning or error message.
		solver (str): The full solver code as a single string.

	Returns:
		str: A formatted string listing the lines in the solver code that produced warnings or errors.
	"""
	lines_to_correct = ""
	for message in messages:
		line_content = message['line_content']
		if line_content in solver:
			lines_to_correct += f"Line: {line_content} produced {message['type']}: {message['message']}\n"
	return lines_to_correct


def set_default(obj: Any) -> Any:
	"""
	Helper function for handling non-serializable objects during JSON serialization.

	This function converts a set to a list for JSON serialization. If the object is not
	a set, it raises a TypeError.

	Args:
		obj (Any): The object to serialize.

	Returns:
		Any: The object converted to a serializable format if possible.

	Raises:
		TypeError: If the object is not serializable (i.e., not a set).
	"""
	if isinstance(obj, set):
		return list(obj)
	raise TypeError
