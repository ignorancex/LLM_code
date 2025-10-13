import unittest
import logging
from magif.utils.utils import normalize_path
from magif.utils.validator import Validator

class ValidatorTournament(unittest.TestCase):
	def setUp(self):
		"""
		Set up the testing environment by preparing agent JSON data and initializing required variables.
		"""
		# Path to test configuration or JSON files
		logging.debug('Test')
		self.agents_directory = normalize_path("unit_tests/TEST_RESULTS")
		self.matrices_filepath = normalize_path("unit_tests/DATA/MISC/matrices.json")
		self.validators_dir = normalize_path("DATA/EVAL")

	def test_validator_matrices(self):
		"""
		Test that a validator runs correctly with payoffs loaded from JSON files.
		"""
		validator = Validator(self.agents_directory, self.matrices_filepath, self.validators_dir)
		df = validator.validate_all()
		print(df)
		self.assertIsNotNone(df)

	def test_validator_no_matrices(self):
		"""
		Test that a validator runs correctly without payoffs loaded from JSON files.
		"""
		validator = Validator(self.agents_directory, None, self.validators_dir)
		df = validator.validate_all()
		print(df)
		self.assertIsNotNone(df)


if __name__ == "__main__":
	unittest.main()
