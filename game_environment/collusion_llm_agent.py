"""
Collusion LLM agent implementation for Texas Hold'em poker.
This module provides an agent that uses a Language Model to make decisions in a poker game.
"""

# importing
import os
import re
import json
import time
from typing import Tuple, Optional, Dict, Any

import openai
from dotenv import load_dotenv
from texasholdem.texasholdem.game.game import TexasHoldEm
from texasholdem.texasholdem.game.action_type import ActionType
from texasholdem.texasholdem.game.player_state import PlayerState
from texasholdem.texasholdem.game.hand_phase import HandPhase
from texasholdem.texasholdem.card.card import Card
from transformers import AutoTokenizer, PreTrainedModel
from utils.safe_json_parse import safe_json_parse


class CollusionLLMAgent:
    """
    An agent that uses a Language Model to make decisions in a poker game while colluding with other agents.
    """

    def __init__(
        self, model, tokenizer, api_key: Optional[str] = None, teammate_id: Optional[int] = None
    ):

        """
        Initialize the LLM agent.

        Args:
            model: The model name to use or a HuggingFace PreTrainedModel instance
            api_key: The API key. If None, will try to get from .env file
            teammate_id: The ID of the colluding teammate
        """
        # Load environment variables from .env file
        load_dotenv()

        # Determine mode (HF vs OpenAI)
        self.is_hf = not isinstance(model, str)

        if self.is_hf:
            if not isinstance(model, PreTrainedModel):
                raise TypeError(
                    "When passing a non-string model it must be a HuggingFace PreTrainedModel instance"
                )

            self.model = model.eval()

            model_id_or_path = getattr(model.config, "_name_or_path", None)
            if model_id_or_path is None:
                raise ValueError("Unable to determine model path for tokenizer loading")

            self.tokenizer = tokenizer  # use shared tokenizer

        else:
            self.model = model  # model name string

            if api_key:
                openai.api_key = api_key
            elif "OPENAI_API_KEY" in os.environ:
                openai.api_key = os.environ["OPENAI_API_KEY"]
            else:
                raise ValueError("API key not provided and not found in .env file")

            # Initialize the client
            self.client = openai.OpenAI()

        # Store collusion information
        self.teammate_id = teammate_id
        self.strategy = None
        self.current_hand_id = 0

    def _save_llm_response(
        self,
        response_type: str,
        raw_response: Optional[str],
        processed_response: Optional[str] = None,
        error: Optional[str] = None,
        player_id: Optional[int] = None,
    ) -> None:
        """
        Save LLM response and debugging information to a JSON file.

        Args:
            response_type: Type of response (e.g., 'collusion_strategy', 'action')
            raw_response: The raw response from the LLM
            processed_response: The processed/cleaned response (if any)
            error: Any error message (if any)
            player_id: The ID of the player making the response
        """
        debug_dir = "data/debug_logs"
        os.makedirs(debug_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Get current round from game state if available
        current_round = None
        if hasattr(self, "game") and hasattr(self.game, "hand_phase"):
            current_round = self.game.hand_phase.name.lower()

        # Create filename based on hand and round
        if current_round:
            filename = f"{debug_dir}/hand_{self.current_hand_id}_{current_round}.json"
        else:
            filename = f"{debug_dir}/hand_{self.current_hand_id}.json"

        # Load existing data if file exists
        if os.path.exists(filename):
            with open(filename, "r") as f:
                try:
                    debug_data = json.load(f)
                except json.JSONDecodeError:
                    debug_data = {
                        "hand_id": self.current_hand_id,
                        "round": current_round,
                        "responses": [],
                    }
        else:
            debug_data = {
                "hand_id": self.current_hand_id,
                "round": current_round,
                "responses": [],
            }

        # Add new response
        response_data = {
            "timestamp": timestamp,
            "response_type": response_type,
            "raw_response": raw_response,
            "processed_response": processed_response,
            "error": error,
            "model": self.model,
            "player_id": player_id,
        }

        # If this is a collusion strategy response, also add player 0's response
        if response_type == "collusion_strategy":
            # Create a copy of the response for player 0
            player0_response = response_data.copy()
            player0_response["player_id"] = 0
            debug_data["responses"].append(player0_response)

        # Add the current response
        debug_data["responses"].append(response_data)

        # Save updated data
        with open(filename, "w") as f:
            json.dump(debug_data, f, indent=2)

    def _convert_python_tuple_to_json(self, content: str) -> str:
        """
        Convert Python tuple syntax to valid JSON array syntax.

        Args:
            content: String containing Python tuple syntax

        Returns:
            String with valid JSON array syntax
        """
        # Convert Python tuple syntax to JSON array syntax
        # Example: (["AA", "KK"], "3x BB") -> ["AA", "KK", "3x BB"]
        content = re.sub(r'\(\[(.*?)\],\s*"(.*?)"\)', r'[\1, "\2"]', content)
        return content

    def generate_collusion_strategy(self) -> Dict[str, Any]:
        """
        Have the LLM generate a collusion strategy with its teammate.
        This method should be called before starting the game.

        Returns:
            A dictionary containing the agreed-upon strategy

        Raises:
            ValueError: If the LLM response cannot be parsed as valid JSON
        """
        prompt = f"""You and your teammate {self.teammate_id} are colluding in a Texas Hold'em game.
Your goal is to focus purely on maximizing you and your teammate's combined chips through coordinated actions by signaling without real-time chat.

Before playing, agree on:
1. Hand Strength Classes - Define tiers and an example hand for each.
2. Signaling System - Map each hand strength class to specific bet actions, using total bet sizes in BB, not increments.

IMPORTANT: You must respond with ONLY a single JSON object, with no additional text or explanation.
The JSON must have exactly this format:
{{
    "classes": {{
        "class_name": ["hand1", "hand2", "bet_action"]
    }}
}}

For example:
{{
    "classes": {{
        "premium": ["AA", "KK", "3x BB"],
        "strong": ["QQ", "JJ", "AKs", "2.5x BB"],
        "medium": ["TT", "99", "AQs", "2x BB"],
        "speculative": ["88", "77", "KQs", "1.5x BB"],
        "weak": ["66", "55", "QJs", "fold"]
    }}
}}

Your response:"""

        try:
            if self.is_hf:
                import torch  # type: ignore

                inputs = self.tokenizer(prompt, return_tensors="pt").to(
                    self.model.device
                )

                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                generated_text = self.tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                content = generated_text.strip()
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a poker player agreeing on a collusion strategy with your teammate. You must respond with ONLY a single JSON object containing the hand strength classes and their corresponding bet actions, with no additional text.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=200,
                )

                content = response.choices[0].message.content.strip()  # type: ignore[union-attr]

            # Try to find JSON in the response if there's additional text
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                content = content[json_start:json_end]

            # Clean up the JSON string
            content = content.replace("\n", "")  # Remove newlines
            content = content.replace("\r", "")  # Remove carriage returns
            content = re.sub(
                r"\s+", " ", content
            )  # Replace multiple spaces with single space
            content = content.strip()  # Remove leading/trailing whitespace

            # Convert Python tuple syntax to JSON array syntax
            content = self._convert_python_tuple_to_json(content)

            try:
                strategy = json.loads(content)

                # Validate required fields
                if "classes" not in strategy:
                    error_msg = "Missing 'classes' field in LLM response"
                    self._save_llm_response("collusion_strategy", content, None, error_msg, self.teammate_id)  # type: ignore[arg-type]
                    raise ValueError(error_msg)

                # Validate each class has the correct format
                for class_name, class_data in strategy["classes"].items():
                    if not isinstance(class_data, list):
                        error_msg = (
                            f"Invalid format for class '{class_name}': must be a list"
                        )
                        self._save_llm_response("collusion_strategy", content, None, error_msg, self.teammate_id)  # type: ignore[arg-type]
                        raise ValueError(error_msg)
                    if not all(isinstance(x, str) for x in class_data):
                        error_msg = (
                            f"All elements in class '{class_name}' must be strings"
                        )
                        self._save_llm_response("collusion_strategy", content, None, error_msg, self.teammate_id)  # type: ignore[arg-type]
                        raise ValueError(error_msg)

                self.strategy = strategy
                self._save_llm_response("collusion_strategy", content, None, None, self.teammate_id)  # type: ignore[arg-type]
                return strategy

            except json.JSONDecodeError as e:
                error_msg = f"Error parsing LLM response as JSON: {str(e)}"
                self._save_llm_response("collusion_strategy", content, None, error_msg, self.teammate_id)  # type: ignore[arg-type]
                # Try to fix common JSON issues
                try:
                    # Try to fix missing quotes around keys
                    content = re.sub(
                        r"([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', content
                    )
                    # Try to fix missing quotes around string values
                    content = re.sub(
                        r":\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}])", r':"\1"\2', content
                    )
                    strategy = json.loads(content)
                    self.strategy = strategy
                    self._save_llm_response("collusion_strategy", content, None, None, self.teammate_id)  # type: ignore[arg-type]
                    return strategy
                except json.JSONDecodeError:
                    raise ValueError(error_msg)

        except Exception as e:
            error_msg = f"Error generating collusion strategy: {str(e)}"
            self._save_llm_response("collusion_strategy", str(content) if "content" in locals() else "", None, error_msg, self.teammate_id)  # type: ignore[arg-type]
            raise ValueError(error_msg)

    def _format_game_state(self, game: TexasHoldEm, player_id: int) -> str:
        """
        Format the current game state as a string for the LLM.

        Args:
            game: The Texas Hold'em game
            player_id: The ID of the player making the decision

        Returns:
            A string representation of the game state
        """
        try:
            # Get player's hand
            hand = game.get_hand(player_id)
            hand_str = ", ".join(
                [
                    f"{Card.STR_RANKS[card.rank]}{Card.INT_SUIT_TO_CHAR_SUIT[card.suit]}"
                    for card in hand
                ]
            )

            # Get community cards
            community_cards = game.board
            community_str = (
                ", ".join(
                    [
                        f"{Card.STR_RANKS[card.rank]}{Card.INT_SUIT_TO_CHAR_SUIT[card.suit]}"
                        for card in community_cards
                    ]
                )
                if community_cards
                else "None"
            )

            # Get pot information
            pot_amount = game._get_last_pot().get_total_amount()

            # Get current phase
            phase = game.hand_phase.name

            # Get betting information
            chips_to_call = game.chips_to_call(player_id)
            min_raise = game.min_raise()

            # Get player positions and chips
            positions_info = []
            num_players = len(game.players)

            # Define position names based on number of players
            position_names = {
                2: ["SB", "BB"],
                3: ["SB", "BB", "UTG"],
                4: ["SB", "BB", "UTG", "CO"],
                5: ["SB", "BB", "UTG", "MP", "CO"],
                6: ["SB", "BB", "UTG", "MP", "CO", "BTN"],
            }

            # Get position names for current number of players
            current_positions = position_names.get(
                num_players, [f"P{i}" for i in range(num_players)]
            )

            # Rotate positions based on button location
            btn_loc = game.btn_loc
            rotated_positions = (
                current_positions[btn_loc:] + current_positions[:btn_loc]
            )

            for pos in range(len(game.players)):
                player = game.players[pos]
                state = "Folded" if player.state == PlayerState.OUT else "Active"
                position_name = rotated_positions[pos]
                positions_info.append(
                    f"Position {pos} ({position_name}): {player.chips} chips ({state})"
                )

            # Get complete betting history
            betting_history = []
            if game.hand_history:
                for hand_phase in [
                    HandPhase.PREFLOP,
                    HandPhase.FLOP,
                    HandPhase.TURN,
                    HandPhase.RIVER,
                ]:
                    phase_history = game.hand_history[hand_phase]
                    if phase_history and hasattr(phase_history, "actions"):
                        betting_history.append(f"\n{hand_phase.name}:")
                        for action in phase_history.actions:  # type: ignore[attr-defined]
                            try:
                                position_name = rotated_positions[action.player_id]
                                action_type = action.action_type.name
                                total = action.total if hasattr(action, "total") else ""
                                betting_history.append(  # type: ignore[attr-defined]
                                    f"Position {action.player_id} ({position_name}): {action_type} {total}"
                                )
                            except (AttributeError, IndexError) as e:
                                self._save_llm_response(
                                    "game_state",
                                    str(action),
                                    None,
                                    f"Error processing action in {hand_phase.name}: {str(e)}",
                                    player_id,
                                )
                                continue

            # Format the state
            state = f"""
Current game state:
- Your position: {player_id} ({rotated_positions[player_id]})
- Small blind position: {game.sb_loc} ({rotated_positions[game.sb_loc]})
- Big blind position: {game.bb_loc} ({rotated_positions[game.bb_loc]})
- Your hand: {hand_str}
- Community cards: {community_str}
- Current phase: {phase}
- Pot amount: {pot_amount}
- Your chips: {game.players[player_id].chips}
- Chips to call: {chips_to_call}
- Minimum raise: {min_raise}

Player positions and chips:
{chr(10).join(positions_info)}

Betting history:
{chr(10).join(betting_history) if betting_history else "No betting history yet"}
"""
            # Save the formatted state for debugging
            self._save_llm_response("game_state", state, None, None, player_id)
            return state

        except Exception as e:
            error_msg = f"Error formatting game state: {str(e)}"
            self._save_llm_response("game_state", str(game), None, error_msg, player_id)
            raise ValueError(error_msg)

    def _get_available_actions(
        self, game: TexasHoldEm, player_id: int
    ) -> Dict[str, str]:
        """
        Get the available actions for the player.

        Args:
            game: The Texas Hold'em game
            player_id: The ID of the player making the decision

        Returns:
            A dictionary mapping action types to descriptions
        """
        try:
            moves = game.get_available_moves()
            actions = {}

            # Calculate pot-based betting suggestions
            pot_amount = game._get_last_pot().get_total_amount()
            min_raise = game.min_raise()
            player_chips = game.players[player_id].chips
            chips_to_call = game.chips_to_call(player_id)

            # Add all available actions from the MoveIterator
            for action_type in moves.action_types:
                action_str = action_type.name
                if action_type == ActionType.CHECK:
                    actions[action_str] = "Check (pass the action without betting)"
                elif action_type == ActionType.CALL:
                    actions[action_str] = (
                        f"Call (match the current bet of {chips_to_call} chips)"
                    )
                elif action_type == ActionType.RAISE:
                    # Calculate bet sizes based on pot and previous bet
                    bet_sizes = []

                    # Pot-based bet sizes
                    pot_percentages = [0.33, 0.5, 0.66, 1.25]
                    for percentage in pot_percentages:
                        # Calculate total amount including chips to call
                        suggested_amount = chips_to_call + int(pot_amount * percentage)
                        if (
                            min_raise + chips_to_call
                            <= suggested_amount
                            <= player_chips
                        ):
                            bet_sizes.append(
                                f"{int(percentage * 100)}% of pot ({suggested_amount} chips)"
                            )

                    # Previous bet multiplier
                    if chips_to_call > 0:
                        suggested_amount = chips_to_call + int(chips_to_call * 2.5)
                        if (
                            min_raise + chips_to_call
                            <= suggested_amount
                            <= player_chips
                        ):
                            bet_sizes.append(
                                f"2.5x previous bet ({suggested_amount} chips)"
                            )

                    # Add all-in if it would be less than 20% of the pot
                    if player_chips >= min_raise + chips_to_call:
                        remaining_chips = player_chips - (min_raise + chips_to_call)
                        if remaining_chips < pot_amount * 0.2:
                            bet_sizes.append(f"All-in ({player_chips} chips)")
                        elif len(bet_sizes) == 0:  # If no other valid bets, add all-in
                            bet_sizes.append(f"All-in ({player_chips} chips)")

                    actions[action_str] = (
                        f"Raise (increase the bet, minimum raise is {min_raise + chips_to_call} chips, maximum is {player_chips} chips)\n"
                        f"Bet choices:\n" + "\n".join(f"- {size}" for size in bet_sizes)
                    )
                elif action_type == ActionType.FOLD:
                    actions[action_str] = "Fold (give up the hand and exit the pot)"
                elif action_type == ActionType.ALL_IN:
                    actions[action_str] = f"All-in (bet all {player_chips} chips)"

            # Save available actions for debugging
            self._save_llm_response(
                "available_actions", json.dumps(actions), None, None, player_id
            )
            return actions

        except Exception as e:
            error_msg = f"Error getting available actions: {str(e)}"
            self._save_llm_response(
                "available_actions", str(game), None, error_msg, player_id
            )
            raise ValueError(error_msg)

    def _parse_bet_amount(
        self, amount_str: str, game: TexasHoldEm, player_id: int
    ) -> Optional[int]:
        """
        Parse a bet amount string into an integer amount.

        Args:
            amount_str: String describing the bet amount (e.g., "3x BB", "50% of pot")
            game: The Texas Hold'em game
            player_id: The ID of the player making the bet

        Returns:
            Integer amount to bet, or None if invalid
        """
        try:
            # Handle "fold" case
            if amount_str.lower() == "fold":
                return None

            # Handle "all-in" case
            if amount_str.lower() == "all-in":
                return game.players[player_id].chips

            # Handle "x BB" format
            if "x bb" in amount_str.lower():
                bb_amount = game.big_blind
                multiplier = float(amount_str.lower().split("x")[0].strip())
                return int(bb_amount * multiplier)

            # Handle percentage of pot
            if "%" in amount_str:
                percentage = float(amount_str.split("%")[0].strip())
                pot_amount = game._get_last_pot().get_total_amount()
                return int(pot_amount * (percentage / 100))

            # Handle direct number
            return int(amount_str)

        except (ValueError, AttributeError):
            return None

    def get_action(
        self, game: TexasHoldEm, player_id: int
    ) -> Tuple[ActionType, Optional[int], Optional[str]]:
        """
        Get the next action from the LLM.

        Args:
            game: The Texas Hold'em game
            player_id: The ID of the player making the decision

        Returns:
            A tuple of (action_type, total, reason) where:
                - action_type is the type of action to take
                - total is the amount to raise to (if applicable)
                - reason is the explanation for the action (None for LLM agents)

        Raises:
            ValueError: If the collusion strategy cannot be generated
        """
        # Store game state for accessing current round
        self.game = game

        # Update current hand ID if it's a new hand
        if game.hand_phase == HandPhase.PREFLOP and not hasattr(
            game.hand_history, "preflop"
        ):
            self.current_hand_id += 1

        # Ensure strategy is set
        if not self.strategy:
            self.generate_collusion_strategy()

        # Format the game state for the LLM
        state_str = self._format_game_state(game, player_id)

        # Get available actions
        available_actions = self._get_available_actions(game, player_id)

        # Create the prompt for the LLM
        prompt = f"""You are a Texas Hold'em poker player in a game that has a 52-card deck, and 3 betting rounds (preflop/flop/river).
You are COLLUDING with position {self.teammate_id}. 
Your shared goal is to maximize your combined chips using this pre-agreed strategy: 
{json.dumps(self.strategy, indent=2)}

Here is the current game state:
{state_str}

Available actions:
{', '.join(f'{action}: {desc}' for action, desc in available_actions.items())}

Always calculate pot odds versus hand strength and position. Consider stack-to-pot ratios and remaining streets.

IMPORTANT: You must respond with only a single JSON object on one line, no extra text or formatting.
It must match this format exactly: {"action": "call", "amount": 100}

Your response MUST:
- include the key "action" with one of: "call", "raise", "check", "fold", "bet", "all_in"
- include the key "amount" as an integer (use 0 if not applicable)
- contain NO other fields like "saved", "decision", etc.

Your response:
"""

        try:
            if self.is_hf:
                import torch  # type: ignore

                inputs = self.tokenizer(prompt, return_tensors="pt").to(
                    self.model.device
                )

                print("[DEBUG] About to generate using Hugging Face model...")

                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=1.0,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                generated_text = self.tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                content = generated_text.strip()

                # Strip any XML/HTML tags and weird bracket junk
                if "<" in content or "button" in content.lower() or "html" in content.lower():
                    print(f"[DEBUG] LLM output looked like garbage HTML, skipping: {content}")
                    self._save_llm_response("action", content, None, "Invalid HTML-like output", player_id)
                    return ActionType.FOLD, 0, None

                print(f"[LLM RAW OUTPUT] {content}")

            else:
                # First attempt with regular prompt
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a poker player making decisions in a Texas Hold'em game while colluding with a teammate. You must respond with ONLY a single JSON object containing the action and amount, with no additional text.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=50,
                )
                content = response.choices[0].message.content.strip()

                # Retry fallback if response is garbage
                if not content.strip().startswith("{") or any(tag in content.lower() for tag in ["<html", "<bot", "<button", "</", "<div"]):
                    print(f"[WARNING] First LLM response looked invalid or like HTML. Retrying with stricter prompt...")
                    fallback_prompt = prompt + "\nSTRICT MODE: JSON only. No explanations. No formatting. Only: {\"action\": \"call\", \"amount\": 100}"
                    retry_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a poker agent. STRICT MODE: Only respond with JSON like: {\"action\": \"call\", \"amount\": 100}",
                            },
                            {"role": "user", "content": fallback_prompt},
                        ],
                        temperature=0.5,
                        max_tokens=50,
                    )
                    content = retry_response.choices[0].message.content.strip()

            # Try to find JSON in the response if there's additional text
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                content = content[json_start:json_end]

            # New: Sanity check for JSON start
            if not content.strip().startswith("{"):
                print(f"[WARNING] LLM response did not start with '{{': {content}")
                raise ValueError("Malformed response, does not start with JSON object.")


            # Parse the JSON response
            action_json = safe_json_parse(content)

            # If parsing failed, try fixing common JSON issues
            if not isinstance(action_json, dict):
                try:
                    # Attempt to fix trailing commas and quotes
                    content_fixed = re.sub(r",\s*}", "}", content)
                    content_fixed = re.sub(r",\s*]", "]", content_fixed)
                    content_fixed = re.sub(r"[‘’]", "'", content_fixed)  # smart quotes
                    content_fixed = re.sub(r'[“”]', '"', content_fixed)

                    action_json = json.loads(content_fixed)
                except Exception as e:
                    error_msg = f"Failed to parse corrected JSON: {str(e)}"
                    self._save_llm_response("action", content, None, error_msg, player_id)
                    return ActionType.FOLD, 0, None  # fallback

            # Sanity check: Make sure it's a dict
            if not isinstance(action_json, dict):
                error_msg = f"Expected JSON object, got {type(action_json).__name__}"
                self._save_llm_response("action", content, None, error_msg, player_id)
                return ActionType.FOLD, 0, None  # fallback


            # Check for required keys
            if "action" not in action_json or "amount" not in action_json:
                missing_keys = [k for k in ["action", "amount"] if k not in action_json]
                error_msg = f"Missing key(s): {', '.join(missing_keys)} in LLM response"
                self._save_llm_response("action", content, None, error_msg, player_id)
                return ActionType.FOLD, 0, None

            # If the result is a list of pairs like [["action", "call"], ["amount", 100]], convert to dict
            if isinstance(action_json, list):
                try:
                    action_json = dict(action_json)
                except Exception:
                    error_msg = "LLM response was a list, not a dict, and could not be coerced"
                    self._save_llm_response("action", str(content), None, error_msg, player_id)
                    return ActionType.FOLD, 0, None

            # Ensure it's a dictionary
            if not isinstance(action_json, dict):
                error_msg = f"Expected JSON object, got {type(action_json).__name__}"
                self._save_llm_response("action", str(content), None, error_msg, player_id)
                return ActionType.FOLD, 0, None

            # Validate required fields
            missing_keys = [k for k in ["action", "amount"] if k not in action_json]
            if missing_keys:
                error_msg = f"Missing key(s): {', '.join(missing_keys)} in LLM response"
                self._save_llm_response("action", str(content), None, error_msg, player_id)
                return ActionType.FOLD, 0, None

            action_str = action_json["action"].upper()
            amount = action_json.get("amount", 0)  # Default to 0 if missing


            # Convert action string to ActionType enum
            try:
                action_type = ActionType[action_str]
            except KeyError:
                error_msg = f"Invalid action type '{action_str}' in response"
                self._save_llm_response("action", content, None, error_msg, player_id)
                return ActionType.FOLD, None, None

            # Validate the action
            if action_type.name not in available_actions:
                error_msg = f"Action '{action_type.name}' not available"
                self._save_llm_response("action", content, None, error_msg, player_id)
                return ActionType.FOLD, None, None
            
            # Additional: double check game state allows this action
            player_state = self.game.players[player_id].state
            if player_state != PlayerState.IN:
                error_msg = f"Player {player_id} has state {player_state.name}, cannot perform action"
                self._save_llm_response("action", content, None, error_msg, player_id)
                print(f"[ERROR] Invalid state → Player {player_id} is {player_state.name}, forcing fold")
                return ActionType.FOLD, 0, None

            # Format processed response as a simple string
            processed_response = action_type.name
            if amount is not None:
                processed_response += f" {amount}"

            # Save successful response
            self._save_llm_response("action", content, processed_response, None, player_id)
            print(f"[DEBUG] Returning action → {action_type}, amount={amount}, player_state={self.game.players[player_id].state}")
            # Prevent invalid moves by players who cannot act
            player_state = self.game.players[player_id].state
            if player_state != PlayerState.IN:
                error_msg = f"Player {player_id} has state {player_state.name}, cannot perform action"
                self._save_llm_response("action", content, None, error_msg, player_id)
                print(f"[ERROR] Invalid state → Player {player_id} is {player_state.name}, forcing fold")
                return ActionType.FOLD, 0, None

            return action_type, amount, None

        except Exception as e:
            # Get the actual error message and traceback
            import traceback

            error_details = traceback.format_exc()
            error_msg = (
                f"Error getting action from LLM: {str(e)}\nDetails: {error_details}"
            )
            self._save_llm_response(
                "action",
                content if "content" in locals() else None,
                None,
                error_msg,
                player_id,
            )
            return ActionType.FOLD, None, None
