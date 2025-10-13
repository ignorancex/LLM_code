from typing import Optional
from magif.utils.utils import Mode


class DataObject:
	"""
	Represents input game data, including natural language descriptions, Prolog rules, and prompts.

	This class encapsulates various forms of input data that can be used to describe a game,
	such as natural language descriptions, Prolog rules as strings or files, and prompts
	for autoformalization.

	Attributes:
	    mode (Mode): The mode of input data processing. Defaults to `Mode.AUTOFORMALIZATION`.
	    nl_description (Optional[str]): A natural language description of the game (default: None).
	    rules_string (Optional[str]): The Prolog rules as a string (default: None).
	    rules_path (Optional[str]): The file path to a Prolog rules file (default: None).
	    prompt (Optional[str]): The initial instruction prompt for autoformalization (default: None).
	    feedback_prompt (Optional[str]): The feedback prompt for refining rules during autoformalization (default: None).
	    name (Optional[str]): The name of the strategy/game (default: None).
	"""

	def __init__(self,
				 mode: Mode=Mode.AUTOFORMALIZATION,
				 nl_description: Optional[str] = None,
				 rules_string: Optional[str] = None,
				 rules_path: Optional[str] = None,
				 instruction_prompt: Optional[str] = None,
				 feedback_prompt: Optional[str] = None,
				 name: Optional[str] = None):
		self.mode = mode
		self.nl_description = nl_description
		self.rules_string = rules_string
		self.rules_path = rules_path
		self.prompt = instruction_prompt
		self.feedback_prompt = feedback_prompt
		self.name = name
