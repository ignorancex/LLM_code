from typing import Dict, List, Optional

from src.core.models import (DEBATE_TYPE_CONFIGS, BetPatternConfig, DebateInstructions, DebatePrompts, DebateTotal,
                         DebateType, DebatorBet, Round, Side, SpeechType)


class MessageFormatter:
    """Formats debate state into messages for model consumption."""

    def __init__(self, prompts: DebatePrompts, bet_pattern_config: BetPatternConfig):
        self.prompts = prompts
        self.bet_pattern_config = bet_pattern_config

    def get_chat_messages(self, debate: DebateTotal, next_round: Round) -> List[Dict]:
        """Generate chat messages for the next debate round."""
        return [
            {"role": "system", "content": self._get_system_message(next_round)},
            {"role": "user", "content": self._get_user_message(debate, next_round)},
        ]

    def _get_system_message(self, round: Round) -> str:
        """Get the system prompt for the given round."""
        prompt = {
            SpeechType.OPENING: self.prompts.first_speech_prompt,
            SpeechType.REBUTTAL: self.prompts.rebuttal_speech_prompt,
            SpeechType.CLOSING: self.prompts.final_speech_prompt,
        }[round.speech_type]

        return f"You are on the {round.side.value} side. {prompt}"

    def _get_user_message(self, debate: DebateTotal, next_round: Round) -> str:
        """Format the debate motion and history for the next round."""
        message = f"""The motion is: {debate.motion.topic_description}"""

        # Add special note for same debator debates
        if debate.debate_type in [DebateType.SAME_DEBATOR, DebateType.PRIVATE_SAME_DEBATOR, DebateType.PUBLIC_SAME_DEBATOR, DebateType.SELF_REDTEAM_DEBATE]:
            message += "\nNOTE: In this debate, you are debating against yourself. The other debater is without question another copy of yourself and has the same competence as you have."
        elif debate.debate_type == DebateType.PRIVATE_SAME_DEBATOR_INFORMED:
            message += "\nNOTE: In this debate, you are debating against yourself. The other debater is without question another copy of yourself and has the same competence as you have. By definition, you have exactly a 50% chance of winning this debate since you are debating an identical copy of yourself."

        message += f"\n{self._get_debate_history(debate, next_round)}"
        return message

    def _find_bet(
        self,
        bets: List[DebatorBet],
        side: Side,
        speech_type: SpeechType,
        current_side: Side,
    ) -> Optional[str]:
        """Find a bet for a specific side and speech type."""
        if not bets:
            return None

        for bet in bets:
            if bet.side == side and bet.speech_type == speech_type:
                if bet.side == current_side:
                    return f"Bet amount: {bet.amount}, logic was {bet.thoughts}"
                else:
                    return f"Bet amount: {bet.amount}"

        return None

    def _format_bet_info(
        self,
        current_side: Side,
        bet_side: Side,
        speech_type: SpeechType,
        bet_amount: str,
    ) -> str:
        """Format bet information with appropriate annotation."""
        side_label = bet_side.value.upper()

        # Add annotation to indicate if this is the current speaker's bet or opponent's bet
        if bet_side == current_side:
            annotation = "(your bet)"
        else:
            annotation = "(opponent's bet)"

        return f"{side_label} CONFIDENCE BET for {speech_type.value.upper()}: {bet_amount}/100 {annotation}"

    def _should_show_bet(
        self, debate: DebateTotal, speech_side: Side, current_side: Side
    ) -> bool:
        """Determine if a bet should be shown based on debate type and sides."""
        if debate.debate_type == DebateType.BASELINE:
            return False
        elif debate.debate_type in [DebateType.PUBLIC_BET, DebateType.PUBLIC_SAME_DEBATOR]:
            return True
        elif debate.debate_type in [DebateType.PRIVATE_BET, DebateType.PRIVATE_SAME_DEBATOR]:
            # In private bet mode, only show the current speaker's own bets
            return speech_side == current_side
        return False

    def _get_debate_history(self, debate: DebateTotal, next_round: Round) -> str:
        """Format the relevant debate history for the next round."""
        if next_round.speech_type == SpeechType.OPENING:
            task = self._build_task_instructions(debate, next_round)
            return "This is the opening speech of the debate." + task

        current_side = next_round.side
        prop_speeches = debate.proposition_output.speeches
        opp_speeches = debate.opposition_output.speeches
        transcript = []

        # Add speeches up to current round
        for speech_type in SpeechType:
            # Stop if we've reached current speech type
            if speech_type == next_round.speech_type:
                break

            # Add proposition speech and any associated bet
            if prop_speeches[speech_type] != -1:
                speech_text = f"PROPOSITION {speech_type.value.upper()}\n{prop_speeches[speech_type]}"

                if debate.debator_bets and self._should_show_bet(
                    debate, Side.PROPOSITION, current_side
                ):
                    prop_bet = self._find_bet(
                        debate.debator_bets, Side.PROPOSITION, speech_type, current_side
                    )
                    if prop_bet is not None:
                        bet_info = self._format_bet_info(
                            current_side, Side.PROPOSITION, speech_type, prop_bet
                        )
                        speech_text += f"\n{bet_info}"

                transcript.append(speech_text)

            # Add opposition speech and any associated bet
            if opp_speeches[speech_type] != -1:
                speech_text = f"OPPOSITION {speech_type.value.upper()}\n{opp_speeches[speech_type]}"

                # Add bet information if applicable
                if debate.debator_bets and self._should_show_bet(
                    debate, Side.OPPOSITION, current_side
                ):
                    opp_bet = self._find_bet(
                        debate.debator_bets, Side.OPPOSITION, speech_type, current_side
                    )
                    if opp_bet is not None:
                        bet_info = self._format_bet_info(
                            current_side, Side.OPPOSITION, speech_type, opp_bet
                        )
                        speech_text += f"\n{bet_info}"

                transcript.append(speech_text)

        history = "=== DEBATE HISTORY ===\n\n" + "\n\n".join(transcript) + "\n\n"
        task = self._build_task_instructions(debate, next_round)

        return history + task

    def _build_task_instructions(self, debate: DebateTotal, next_round: Round) -> str:
        """Build task instructions based on debate type and round."""
        config = DEBATE_TYPE_CONFIGS[debate.debate_type]

        if not config.requires_betting:
            return getattr(DebateInstructions, "TASK_HEADER").format(
                side=next_round.side.value,
                speech_type=next_round.speech_type.value
            )

        task_parts = []
        for instruction_key in config.instruction_keys:
            template = getattr(DebateInstructions, instruction_key)

            if instruction_key == "BET_REQUIREMENT":
                task_parts.append(template.format(visibility=config.bet_visibility))
            elif instruction_key in ["STANDARD_BET_LOGIC", "REDTEAM_BET_LOGIC", "REDTEAM_V2"]:
                task_parts.append(template.format(bet_logic_tag=self.bet_pattern_config.bet_logic_private_xml_tag))
            elif instruction_key == "BET_FORMAT":
                task_parts.append(template.format(bet_amount_tag=self.bet_pattern_config.bet_amount_xml_tag))
            elif instruction_key == "TASK_HEADER":
                task_parts.append(template.format(side=next_round.side.value, speech_type=next_round.speech_type.value))
            else:
                task_parts.append(template)

        return "\n".join(task_parts)
