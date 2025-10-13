from magif.utils.base_lm import BaseLM
from magif.utils.setup_logger import logger
from openai import OpenAI
from typing import List, Optional, Dict


class GPT4(BaseLM):
	"""
	GPT-4 class for managing interactions specific to the GPT-4 model.
	"""

	def __init__(
			self,
			save_history: bool = False,
			temperature: float = 1.0,
			model: str = "gpt-4o",
			context: Optional[str] = None
	) -> None:
		"""
		Initialize the GPT-4 model.

		Args:
			save_history (bool): Whether to retain the conversation history for subsequent prompts.
			temperature (float): GPT's temperature parameter for controlling response randomness.
			model (str): The GPT model name (e.g., "gpt-4o").
			context (Optional[str]): Initial context message content.
		"""
		super().__init__()
		self.client = OpenAI()
		self._save_history = save_history
		self.temperature = temperature
		self.model = model
		self._context = context
		self.messages: List[Dict[str, str]] = []

		# Initialize messages based on context
		self.__set_messages()

	@property
	def save_history(self) -> bool:
		"""Indicates whether conversation history should be saved."""
		return self._save_history

	@property
	def context(self) -> Optional[str]:
		"""Returns the current context message."""
		return self._context

	def prompt(self, instruction: str, max_tokens: int = 2048) -> str:
		"""
		Prompt the GPT-4 model with an instruction and return the response.

		Args:
			instruction (str): The instruction to prompt the language model.
			max_tokens (int): Maximum number of tokens to generate in the response.

		Returns:
			str: The response from the GPT-4 model.
		"""
		logger.debug(f"Prompting instruction: {instruction}")

		# Prepare the message for the current prompt
		user_message = {"role": "user", "content": instruction}
		if not self.save_history:
			self.__set_messages()  # Reset messages if history is not saved
		self.messages.append(user_message)

		# Generate response from GPT-4
		try:
			response = self.client.chat.completions.create(
				model=self.model,
				messages=self.messages,
				max_tokens=max_tokens,
				temperature=self.temperature
			)
			content = response.choices[0].message.content
			logger.debug(f"Received response: {content}")

			# Add the response to history if saving is enabled
			if self.save_history:
				self.add_response(content)

			return content
		except Exception as e:
			logger.error(f"Error while prompting GPT-4: {e}")
			return "An error occurred while generating the response."

	def add_response(self, response: str) -> None:
		"""
		Add a response to the conversation history.

		Args:
			response (str): The response content to be added.
		"""
		wrapped_response = {"role": "assistant", "content": response}
		self.messages.append(wrapped_response)

	def __set_messages(self) -> None:
		"""
		Initialize or reset the conversation messages based on the context.
		"""
		if self._context:
			self.messages = [{"role": "system", "content": self._context}]
		else:
			self.messages = []

	def clear_context(self) -> None:
		"""
		Clear the context of the conversation, resetting the message history.
		"""
		self.__set_messages()
		logger.debug("Conversation context has been cleared.")

	def get_name(self) -> str:
		"""
		Get the name of the current GPT model.

		Returns:
			str: The name of the GPT model.
		"""
		return self.model
