from dataclasses import dataclass
import io
import os
import logging
from typing import Tuple
from magif.solver.engine import PrologEngine
from magif.utils.setup_logger import logger

@dataclass
class ValidationResult:
	"""
	Represents the outcome of a validation check.

	Attributes:
		is_valid (bool): Whether the validation passed.
		trace (str): Optional diagnostic message or trace.
	"""
	is_valid: bool
	trace: str = ""


class PrologValidator:
	"""
	Handles validation of Prolog code and predicate presence in the environment.
	"""

	def __init__(self, engine: PrologEngine, file_writer):
		self.engine = engine
		self.file_writer = file_writer

	def validate(self, code: str, predicates: Tuple[str, ...] = ()) -> ValidationResult:
		"""
		Validate that the code consults correctly and required predicates exist.
		Also captures Prolog errors from log output.

		Args:
			code (str): The Prolog code to validate.
			predicates (Tuple[str]): Expected predicates to be present.

		Returns:
			ValidationResult: Result including success status and trace.
		"""
		# Set up log capturing
		log_capture, log_handler = self._setup_logging()

		temp_file = self.file_writer(code)
		is_valid = True
		trace = ""

		try:
			# Consult Prolog file
			if not self.engine.consult(temp_file):
				logger.error(f"Failed to consult Prolog file: {temp_file}")
				is_valid = False

			# Check predicates if consulted successfully
			if is_valid:
				for predicate in predicates:
					result = self.engine.query(f"current_predicate({predicate}).")
					if not result.success or not result.data:
						logger.error(f"Missing predicate: {predicate}")
						trace = f"Missing predicate: {predicate}"
						is_valid = False
						break

			# Check log output for additional errors
			if is_valid:
				trace = self._check_logs_for_errors(log_capture)
				if trace:
					is_valid = False

		except Exception as e:
			is_valid = False
			trace = str(e)
			logger.error(f"Validation error: {trace}")

		# Cleanup
		self._cleanup_logging(log_handler)
		if os.path.exists(temp_file):
			os.remove(temp_file)

		return ValidationResult(is_valid, trace)

	def _setup_logging(self) -> Tuple[io.StringIO, logging.Handler]:
		"""
		Redirect critical logs from the swiplserver module to an in-memory stream.

		Returns:
			Tuple[io.StringIO, logging.Handler]: Log capture stream and handler.
		"""
		log_stream = io.StringIO()
		handler = logging.StreamHandler(log_stream)
		handler.setLevel(logging.CRITICAL)
		logging.getLogger('swiplserver').addHandler(handler)
		return log_stream, handler

	def _check_logs_for_errors(self, log_capture_string: io.StringIO) -> str:
		"""
		Check captured logs for any critical errors.

		Args:
			log_capture_string (io.StringIO): The log capture object.
		"""
		log_contents = log_capture_string.getvalue()
		if log_contents:
			if len(log_contents) > 0:
				self.valid = False
				self.trace = log_contents.strip()
				logger.error(f"Prolog error from logs: {self.trace}")

	def _cleanup_logging(self, handler: logging.Handler):
		"""
		Remove the log handler to stop capturing logs.

		Args:
			handler (logging.Handler): The handler to remove.
		"""
		logging.getLogger('swiplserver').removeHandler(handler)
