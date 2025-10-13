from magif.utils.base_lm import BaseLM
from lms.gpt4 import GPT4
from typing import Optional
from magif.utils.utils import AgentStatus


class Autoformalizer:
	"""
	Handles the process of autoformalizing game rules and strategies using an LM (Language Model).

	This class is responsible for:
	- Generating formalized rules based on user-provided prompts.
	- Iteratively refining rules using feedback until they are syntactically correct or attempts are exhausted.
	- Maintaining a history of trace messages for creating the feedback prompt.

	Attributes:
	    llm (BaseLM): The language model used for autoformalization.
	    attempts (int): The number of autoformalization attempts made so far.
	    max_attempts (int): The maximum number of attempts allowed for autoformalization.
	    instruction_prompt (str): The initial prompt provided to the LM.
	    feedback_prompt (str): The feedback prompt for refining rules based on errors.
	    trace_messages (list): A list of trace messages collected during the autoformalization process.
	"""

	def __init__(self,
				 llm: Optional[BaseLM] = GPT4,
				 max_attempts: Optional[int] = 1):
		"""
		Initializes the Autoformalizer with a language model and maximum attempts.

		Args:
		    llm (Optional[BaseLM]): The language model used for autoformalization (default: GPT4).
		    max_attempts (Optional[int]): The maximum number of attempts allowed (default: 1).
		"""
		self.llm = llm(save_history=True)
		self.attempts = 0
		self.max_attempts = max_attempts
		self.instruction_prompt = None
		self.feedback_prompt = None
		self.trace_messages = []

	def set_instruction_prompt(self, prompt: str) -> None:
		"""
		Sets the initial instruction prompt for autoformalization.

		Args:
		    prompt (str): The instruction prompt for the language model.
		"""
		self.instruction_prompt = prompt

	def set_feedback_prompt(self, prompt: str) -> None:
		"""
		Sets the feedback prompt for refining rules based on errors.

		Args:
		    prompt (str): The feedback prompt for the language model.
		"""
		self.feedback_prompt = prompt

	def autoformalize(self, agent, parser, trace_processor, clear_context=True):
		"""
		Performs the autoformalization process to generate syntactically correct game rules.

		Args:
		    agent (Agent): An agent with a solver instance to validate the generated rules.
		    parser (function): A function to parse the LLM's response into formalized rules.
		    trace_processor (function): A function to process error traces from the solver.

		Returns:
		    tuple: A tuple containing the final rules (str) and the status (AgentStatus).

		Raises:
		    RuntimeError: If the status is unknown during the autoformalization loop.
		"""
		# Clear the LLM's context and reset attempts.
		if clear_context:
			self.llm.clear_context()
		self.attempts = 0
		status = "initial"
		lines = None
		rules = None

		# Iteratively refine rules until success or attempts are exhausted.
		while self.attempts < self.max_attempts:
			self.attempts += 1

			# Use the instruction prompt for the first attempt; feedback prompt for subsequent attempts.
			if self.attempts == 1:
				prompt = self.instruction_prompt
			else:
				if status == AgentStatus.INSTRUCTION_ERROR:
					prompt = "Follow the rules of marking the beginning and the end of the code."
				elif status == AgentStatus.SYNTACTIC_ERROR:
					# Reload the solver to reset its state and refine the rules based on feedback.
					agent.reload_solver()
					prompt = self.feedback_prompt.format(code=rules, messages=lines)
				else:
					raise RuntimeError(f"Unknown status {status}")

			# Query the language model with the generated prompt.
			response = self.llm.prompt(prompt)

			try:
				# Parse the response into formalized rules.
				rules = parser(response)
			except ValueError:
				# Handle parsing errors by marking the status as an instruction error.
				status = AgentStatus.INSTRUCTION_ERROR
				continue

			# Validate the generated rules using the solver.
			correct, trace = agent.solver.validate(rules)
			if correct:
				status = AgentStatus.CORRECT
				break
			else:
				status = AgentStatus.SYNTACTIC_ERROR
				lines = trace_processor(trace, rules)
				self.trace_messages.append(lines)

		return rules, status
