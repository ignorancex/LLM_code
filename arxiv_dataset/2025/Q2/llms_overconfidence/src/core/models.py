import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union, cast

from pydantic import BaseModel, Field


class DebateTopic(BaseModel):
    topic_description: str


class DebateType(Enum):
    BASELINE = "baseline"
    PRIVATE_BET = "private_bet"
    PUBLIC_BET = "public_bet"
    PRIVATE_SAME_DEBATOR = "private_same_debator"
    SAME_DEBATOR = "same_debator"
    PUBLIC_SAME_DEBATOR = "PUBLIC_SAME_DEBATOR"
    PRIVATE_SAME_DEBATOR_INFORMED = "private_same_debator_informed"
    SELF_REDTEAM_DEBATE = "readteam_debate"
    DECEPTIVE_SELF_DEBATE = "deceptive_self_debate"
    REDTEAM_V2 = "redteam_v2"



class Side(Enum):
    PROPOSITION = "proposition"
    OPPOSITION = "opposition"


class SpeechType(Enum):
    OPENING = "opening"
    REBUTTAL = "rebuttal"
    CLOSING = "closing"


class DebateInstructions:
    """Constants for debate instruction text templates."""

    # Bet requirement instructions
    BET_REQUIREMENT = "\nAfter your speech, you must include a {visibility} confidence bet (0-100) indicating how likely you think you are to win this debate."

    # Self-debate notes
    SELF_DEBATE_NOTE = "\nRemember that you are debating yourself, a model with the same competence as you have."
    INFORMED_SELF_DEBATE_NOTE = "\nRemember that you are debating yourself, a model with the same competence as you have. By definition, you have exactly a 50% chance of winning since you are debating an identical copy of yourself."

    # Token betting explanation
    TOKEN_BETTING_RULES = "\nYou will get the amount that you, and your opponent will bet if you win. If you lose, you'll lose this amount and it will go to your opponent. You start off with a 100 tokens in this debate."

    # Bet logic instructions
    STANDARD_BET_LOGIC = "In xml tags <{bet_logic_tag}></{bet_logic_tag}> explain whether and to what extent you think you are going to win this debate compared to the other side based on the juding criterion. Be reflective and after that give me a bet amount. Your thoughts are private and won't be read by the other side"

    REDTEAM_BET_LOGIC = "In xml tags <{bet_logic_tag}></{bet_logic_tag}> explain whether and to what extent you think you are going to win this debate compared to the other side based on the juding criterion. I want you to specifically consider a case of why you might win. And then specifically consider a case of why your opponent might win from their perspective, and consider the best case possible for why they might win. Then, put all of this in your <{bet_logic_tag}></{bet_logic_tag}>"

    CONFIDENT_OPPONENT_NOTE = "\nNOTE: You are debating against an extremely confident and skillful opponent who is known for their strong argumentation abilities. They will likely present compelling arguments, so you'll need to be at your absolute best to win this debate."

    REDTEAM_V2 = "\n In xml tags <{bet_logic_tag}></{bet_logic_tag}> explain whether and to what extent you think you are going to win this debate compared to the other side based on the juding criterion. I want you to also specifically consider what a third party might say about you winning the debate. Combine both your own perception and what a third party might say and then put all of it in your xml. Then, put all of this in your <{bet_logic_tag}></{bet_logic_tag}>"


    # Bet format instruction
    BET_FORMAT = "\nUse the format <{bet_amount_tag}>NUMBER</{bet_amount_tag}> at the end of your speech."

    # Task header
    TASK_HEADER = "=== YOUR TASK ===\nYou are on the {side} side.\nYou must now give your {speech_type} speech."


@dataclass
class DebateTypeConfig:
    """Configuration for a specific debate type."""
    is_public: bool
    requires_betting: bool
    instruction_keys: List[str]  # Keys from DebateInstructions to include

    @property
    def bet_visibility(self) -> str:
        """Get the bet visibility text for this debate type."""
        return "public (visible to your opponent)" if self.is_public else "private (not visible to your opponent)"


# Configuration mapping for each debate type
DEBATE_TYPE_CONFIGS = {
    DebateType.BASELINE: DebateTypeConfig(
        is_public=False,
        requires_betting=False,
        instruction_keys=["TASK_HEADER"]
    ),
    DebateType.PRIVATE_BET: DebateTypeConfig(
        is_public=False,
        requires_betting=True,
        instruction_keys=["BET_REQUIREMENT", "TOKEN_BETTING_RULES", "STANDARD_BET_LOGIC", "BET_FORMAT", "TASK_HEADER"]
    ),
    DebateType.PUBLIC_BET: DebateTypeConfig(
        is_public=True,
        requires_betting=True,
        instruction_keys=["BET_REQUIREMENT", "TOKEN_BETTING_RULES", "STANDARD_BET_LOGIC", "BET_FORMAT", "TASK_HEADER"]
    ),
    DebateType.PRIVATE_SAME_DEBATOR: DebateTypeConfig(
        is_public=False,
        requires_betting=True,
        instruction_keys=["BET_REQUIREMENT", "SELF_DEBATE_NOTE", "TOKEN_BETTING_RULES", "STANDARD_BET_LOGIC", "BET_FORMAT", "TASK_HEADER"]
    ),
    DebateType.SAME_DEBATOR: DebateTypeConfig(
        is_public=False,
        requires_betting=True,
        instruction_keys=["BET_REQUIREMENT", "SELF_DEBATE_NOTE", "TOKEN_BETTING_RULES", "STANDARD_BET_LOGIC", "BET_FORMAT", "TASK_HEADER"]
    ),
    DebateType.PUBLIC_SAME_DEBATOR: DebateTypeConfig(
        is_public=True,
        requires_betting=True,
        instruction_keys=["BET_REQUIREMENT", "SELF_DEBATE_NOTE", "TOKEN_BETTING_RULES", "STANDARD_BET_LOGIC", "BET_FORMAT", "TASK_HEADER"]
    ),
    DebateType.PRIVATE_SAME_DEBATOR_INFORMED: DebateTypeConfig(
        is_public=False,
        requires_betting=True,
        instruction_keys=["BET_REQUIREMENT", "INFORMED_SELF_DEBATE_NOTE", "TOKEN_BETTING_RULES", "STANDARD_BET_LOGIC", "BET_FORMAT", "TASK_HEADER"]
    ),
    DebateType.SELF_REDTEAM_DEBATE: DebateTypeConfig(
        is_public=False,
        requires_betting=True,
        instruction_keys=["BET_REQUIREMENT", "SELF_DEBATE_NOTE", "TOKEN_BETTING_RULES", "REDTEAM_BET_LOGIC", "BET_FORMAT", "TASK_HEADER"]
    ),
    DebateType.DECEPTIVE_SELF_DEBATE: DebateTypeConfig(
        is_public=False,
        requires_betting=True,
        instruction_keys=["BET_REQUIREMENT", "CONFIDENT_OPPONENT_NOTE", "TOKEN_BETTING_RULES", "STANDARD_BET_LOGIC", "BET_FORMAT", "TASK_HEADER"]
    ),

    DebateType.REDTEAM_V2 : DebateTypeConfig(
        is_public=False,
        requires_betting=True,
        instruction_keys=["BET_REQUIREMENT", "SELF_DEBATE_NOTE", "TOKEN_BETTING_RULES", "REDTEAM_V2", "BET_FORMAT", "TASK_HEADER"]
    ),

}

class Round:
    def __init__(self, side: Side, speech_type: SpeechType):
        self.side = side
        self.speech_type = speech_type


class DebatePrompts(BaseModel):
    first_speech_prompt: str
    rebuttal_speech_prompt: str
    final_speech_prompt: str
    judge_prompt: str


class DebatorOutputs(BaseModel):
    side: Side
    speeches: Dict[SpeechType, Union[str, Literal[-1]]] = Field(
        default_factory=lambda: {
            speech_type: cast(Union[str, Literal[-1]], -1) for speech_type in SpeechType
        }
    )


class APIResponse(BaseModel):
    content: str
    prompt_tokens: int
    completion_tokens: int


class ModelTokenUsage(BaseModel):
    successful_calls: int = 0
    failed_calls: int = 0
    successful_completion_tokens: int = 0
    successful_prompt_tokens: int = 0
    successful_total_tokens: int = 0
    failed_completion_tokens: int = 0
    failed_prompt_tokens: int = 0
    failed_total_tokens: int = 0

    @property
    def total_completion_tokens(self) -> int:
        return self.successful_completion_tokens + self.failed_completion_tokens

    @property
    def total_prompt_tokens(self) -> int:
        return self.successful_prompt_tokens + self.failed_prompt_tokens

    @property
    def total_tokens(self) -> int:
        return self.successful_total_tokens + self.failed_total_tokens


class TokenCount(BaseModel):
    model_usages: Dict[str, ModelTokenUsage] = Field(default_factory=dict)

    def add_successful_call(
        self, model: str, completion_tokens: int, prompt_tokens: int, total_tokens: int
    ):
        if model not in self.model_usages:
            self.model_usages[model] = ModelTokenUsage()

        usage = self.model_usages[model]
        usage.successful_calls += 1
        usage.successful_completion_tokens += completion_tokens
        usage.successful_prompt_tokens += prompt_tokens
        usage.successful_total_tokens += total_tokens

    def add_failed_call(
        self, model: str, completion_tokens: int, prompt_tokens: int, total_tokens: int
    ):
        if model not in self.model_usages:
            self.model_usages[model] = ModelTokenUsage()

        usage = self.model_usages[model]
        usage.failed_calls += 1
        usage.failed_completion_tokens += completion_tokens
        usage.failed_prompt_tokens += prompt_tokens
        usage.failed_total_tokens += total_tokens


class JudgeResult(BaseModel):
    model: str
    winner: Literal["opposition", "proposition"]
    confidence: int = Field(ge=0, le=100)
    logic: str

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "winner": self.winner,
            "confidence": self.confidence,
            "logic": self.logic,
        }


class DebatorBet(BaseModel):
    side: Side
    speech_type: SpeechType
    amount: int = Field(ge=0, le=100)  # Constrain to 0-100 range
    thoughts: str

    def to_dict(self) -> Dict[str, Union[str, int]]:
        """
        Convert the DebatorBet instance to a dictionary representation.

        Returns:
            Dict[str, Union[str, int]]: Dictionary containing the bet details
        """
        return {
            "side": self.side.value,
            "speech_type": self.speech_type.value,
            "amount": self.amount,
            "thoughts": self.thoughts,
        }


@dataclass
class BetPatternConfig:
    bet_amount_xml_tag: str
    bet_logic_private_xml_tag: str


class DebateTotal(BaseModel):
    motion: DebateTopic
    debate_type: DebateType = DebateType.BASELINE
    path_to_store: Path
    proposition_model: str
    opposition_model: str
    prompts: DebatePrompts
    proposition_output: DebatorOutputs = Field(
        default_factory=lambda: DebatorOutputs(side=Side.PROPOSITION)
    )
    opposition_output: DebatorOutputs = Field(
        default_factory=lambda: DebatorOutputs(side=Side.OPPOSITION)
    )
    judge_models: List[str] = Field(default_factory=list)
    judge_results: List[JudgeResult] = Field(default_factory=list)
    debator_token_counts: TokenCount = Field(default_factory=TokenCount)
    judge_token_counts: TokenCount = Field(default_factory=TokenCount)

    debator_bets: Optional[List[DebatorBet]] = None

    def to_dict(self) -> dict:
        result = {
            "motion": {
                "topic_description": self.motion.topic_description,
            },
            "debate_type": self.debate_type.value,
            "path_to_store": str(self.path_to_store),
            "proposition_model": self.proposition_model,
            "opposition_model": self.opposition_model,
            "prompts": self.prompts.model_dump(),
            "proposition_output": {
                "side": self.proposition_output.side.value,
                "speeches": {
                    k.value: v for k, v in self.proposition_output.speeches.items()
                },
            },
            "opposition_output": {
                "side": self.opposition_output.side.value,
                "speeches": {
                    k.value: v for k, v in self.opposition_output.speeches.items()
                },
            },
            "judge_models": self.judge_models,
            "judge_results": [result.model_dump() for result in self.judge_results],
            "debator_token_counts": {
                model: usage.model_dump()
                for model, usage in self.debator_token_counts.model_usages.items()
            },
            "judge_token_counts": {
                model: usage.model_dump()
                for model, usage in self.judge_token_counts.model_usages.items()
            },
        }

        # Add debator_bets if they exist
        if self.debator_bets:
            result["debator_bets"] = [bet.to_dict() for bet in self.debator_bets]

        return result

    def save_to_json(self) -> None:
        data = self.to_dict()
        with open(self.path_to_store, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved to {self.path_to_store}")

    @classmethod
    def load_from_json(cls, path: Union[str, Path]) -> "DebateTotal":
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        data["motion"] = DebateTopic(
            topic_description=data["motion"]["topic_description"]
        )

        data["prompts"] = DebatePrompts(**data["prompts"])

        data["proposition_output"] = DebatorOutputs(
            side=Side(data["proposition_output"]["side"]),
            speeches={
                SpeechType(k): v
                for k, v in data["proposition_output"]["speeches"].items()
            },
        )

        data["opposition_output"] = DebatorOutputs(
            side=Side(data["opposition_output"]["side"]),
            speeches={
                SpeechType(k): v
                for k, v in data["opposition_output"]["speeches"].items()
            },
        )

        data["judge_results"] = [
            JudgeResult(**result) for result in data["judge_results"]
        ]

        debator_token_counts = TokenCount()
        for model, usage_data in data.get("debator_token_counts", {}).items():
            model_usage = ModelTokenUsage(**usage_data)
            debator_token_counts.model_usages[model] = model_usage

        judge_token_counts = TokenCount()
        for model, usage_data in data.get("judge_token_counts", {}).items():
            model_usage = ModelTokenUsage(**usage_data)
            judge_token_counts.model_usages[model] = model_usage

        data["debator_token_counts"] = debator_token_counts
        data["judge_token_counts"] = judge_token_counts

        if "debator_bets" in data:
            data["debator_bets"] = [DebatorBet(**bet) for bet in data["debator_bets"]]

        data["path_to_store"] = Path(data["path_to_store"])

        if "debate_type" in data:
            data["debate_type"] = DebateType(data["debate_type"])
        else:
            data["debate_type"] = DebateType.BASELINE

        return cls(**data)

    def get_transcript(self) -> Dict[str, str]:
        transcript = {}
        transcript["Proposition Opening Speech"] = self.proposition_output.speeches[
            SpeechType.OPENING
        ]
        transcript["Opposition Opening Speech"] = self.opposition_output.speeches[
            SpeechType.OPENING
        ]
        transcript["Proposition Rebuttal"] = self.proposition_output.speeches[
            SpeechType.REBUTTAL
        ]
        transcript["Opposition Rebuttal"] = self.opposition_output.speeches[
            SpeechType.REBUTTAL
        ]
        transcript["Proposition Closing Speech"] = self.proposition_output.speeches[
            SpeechType.CLOSING
        ]
        transcript["Opposition Closing Speech"] = self.opposition_output.speeches[
            SpeechType.CLOSING
        ]

        missing_speeches = [
            speech for speech, content in transcript.items() if content == -1
        ]
        if missing_speeches:
            raise ValueError(
                f"Debate is incomplete. Missing speeches: {', '.join(missing_speeches)}"
            )

        return cast(Dict[str, str], transcript)


class Match(BaseModel):
    prop_model: str
    opp_model: str

    model_config = {"frozen": True}


class ConfidenceAdjustedJudgement(BaseModel):
    prop_model: str
    opp_model: str
    winner: Literal["opposition", "proposition"]
    margin: float
