"""
Mixed player game implementation for Texas Hold'em poker.
This module provides a game where some players are controlled by LLMs and others are human-controlled.
"""
from pathlib import Path
import os
import time
from typing import List, Dict, Optional, Tuple, Set, Union
from dotenv import load_dotenv
from texasholdem.texasholdem.game.game import TexasHoldEm
#from texasholdem.texasholdem.gui.text_gui import TextGUI
from texasholdem.texasholdem.game.action_type import ActionType
from game_environment.llm_agent import LLMAgent
from game_environment.collusion_llm_agent import CollusionLLMAgent
from game_environment.preflop_strategy import load_preflop_chart, lookup_action
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import accelerate
import torch
from game_environment.preflop_strategy import load_preflop_chart, lookup_action
from transformers.utils import logging
logging.set_verbosity_debug()
import traceback




class MixedPlayerGame:
    """
    A Texas Hold'em game where some players are controlled by LLMs and others are human-controlled.
    """

    def __init__(
        self,
        buyin: int = 500,
        big_blind: int = 5,
        small_blind: int = 2,
        max_players: int = 6,
        llm_player_ids: Optional[List[int]] = None,
        collusion_llm_player_ids: Optional[List[int]] = None,
        openai_model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize the mixed player game.

        Args:
            buyin: The amount of chips each player starts with
            big_blind: The big blind amount
            small_blind: The small blind amount
            max_players: The maximum number of players
            llm_player_ids: The IDs of players controlled by regular LLM. If None, players 0 and 1 will be LLM-controlled.
            collusion_llm_player_ids: The IDs of players controlled by collusion LLM. If None, no players will be collusion LLM-controlled.
            openai_model: The model name to use. If None, will try to get from .env file
            openai_api_key: The API key. If None, will try to get from .env file
        """
        # Load environment variables from .env file
        load_dotenv()

        # Load the local Hugging Face model once and share it with all agents
        from pathlib import Path

        model_path = Path("C:/Users/Krish Jain/Downloads/multiagent-poker-collusion-main/workspace/models/Llama-3.2-3B-Instruct").as_posix()

        from transformers import AutoModelForCausalLM

        self.hf_model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            device_map="auto"
        )
        self.hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("[DEBUG] Successfully loaded LLM model & tokenizer.")

        # No tokenizer is created here; each agent will load its own tokenizer on demand

        self.game = TexasHoldEm(
            buyin=buyin,
            big_blind=big_blind,
            small_blind=small_blind,
            max_players=max_players,
        )
        self.gui = None

        # Set up AI players
        if llm_player_ids is None:
            llm_player_ids = [0, 1, 2, 3, 4, 5]  # Make all players LLM-controlled

        collusion_llm_player_ids = []


        self.llm_player_ids = set(llm_player_ids)
        self.collusion_llm_player_ids = set(collusion_llm_player_ids)
        self.human_player_ids = (
            set(range(max_players))
            - self.llm_player_ids
            - self.collusion_llm_player_ids
        )

        # Load the preflop strategy table
        self.preflop_strategy = load_preflop_chart('preflop_chart.csv')


        # Initialize AI agents
        self.ai_agents = {}

        # Initialize regular LLM agents with the shared Hugging Face model
        for player_id in self.llm_player_ids:
            self.ai_agents[player_id] = LLMAgent(model=self.hf_model, tokenizer=self.hf_tokenizer)

        # Initialize collusion LLM agents
        if len(collusion_llm_player_ids) == 2:
            player1, player2 = sorted(collusion_llm_player_ids)
            self.ai_agents[player1] = CollusionLLMAgent(
                model=self.hf_model, tokenizer=self.hf_tokenizer, teammate_id=player2
            )
            self.ai_agents[player2] = CollusionLLMAgent(
                model=self.hf_model, tokenizer=self.hf_tokenizer, teammate_id=player1
            )




    def _is_ai_player(self, player_id: int) -> bool:
        """
        Check if a player is controlled by AI.

        Args:
            player_id: The ID of the player to check

        Returns:
            True if the player is controlled by AI, False otherwise
        """
        return (
            player_id in self.llm_player_ids
            or player_id in self.collusion_llm_player_ids
        )

    def _get_ai_action(self, player_id: int) -> Tuple[ActionType, Optional[int], str]:
        print(f"[DEBUG] Calling _get_ai_action for player_id={player_id}")
        # Check if we're in the preflop round
        if self.game.hand_phase.name == "PREFLOP":
            # VERY SIMPLIFIED: you would get real cards from self.game.players[player_id].hole_cards
            # Here we just hardcode example values for testing
            hand = "AK"  # should parse from actual hole cards
            suited = "yes"  # determine if suits match
            position = "early"  # determine from seat index

            # Lookup the recommended action
            action = lookup_action(self.preflop_strategy, hand)
            print(f"[DEBUG] Preflop strategy lookup: hand={hand}, suited={suited}, position={position} -> action={action}")

            # Convert string action to ActionType
            if action == "raise":
                return ActionType.RAISE, self.game.current_bet * 2, "Preflop GTO"
            elif action == "call":
                return ActionType.CALL, None, "Preflop GTO"
            else:
                return ActionType.FOLD, None, "Preflop GTO"

        # Normal postflop or other phase: call LLM or CFR
        if (
            player_id not in self.llm_player_ids
            and player_id not in self.collusion_llm_player_ids
        ):
            raise ValueError(f"Player {player_id} is not an LLM player")

        agent = self.ai_agents[player_id]
        return agent.get_action(self.game, player_id)


    def _get_human_action(self) -> Tuple[ActionType, Optional[int]]:
        print("[DEBUG] Auto-folding human player to avoid loop.")
        self.game.take_action(ActionType.FOLD)
        return ActionType.FOLD, None
        # Use the GUI to get the action from the human player
        #self.gui.run_step()

        # The action is already taken by the GUI, so we just return None
        return None, None

    def run_game(self):
        """
        Run the game until it's over.
        """
        error_message = None
        try:
            while self.game.is_game_running():
                print("[DEBUG] Starting new hand...")
                self.game.start_hand()
                print(f"[DEBUG] Hand running? {self.game.is_hand_running()}")

                while self.game.is_hand_running():
                    current_player = self.game.current_player
                    print(f"[DEBUG] Current player: {current_player}")

                    if self._is_ai_player(current_player):
                        # Get action from AI
                        result = self._get_ai_action(current_player)
                        if result is None:
                            print(f"[ERROR] Agent returned None. Forcing fold.")
                            action_type, total, reason = ActionType.FOLD, None, None
                        else:
                            action_type, total, reason = result

                        # Take the action
                        try:
                            if action_type == ActionType.RAISE and total is not None:
                                self.game.take_action(action_type, total=total)
                            else:
                                self.game.take_action(action_type)
                        except Exception as e:
                            print(f"[ERROR] Action failed: {e}. Forcing fold.")
                            self.game.take_action(ActionType.FOLD)

                    else:
                        # Get action from human
                        self._get_human_action()

                # Export and replay the hand history
                pgn_path = self.game.export_history("./data/pgns")
                json_path = self.game.hand_history.export_history_json("./data/json")
                print(f"\nExported hand history to:")
                print(f"PGN: {pgn_path}")
                print(f"JSON: {json_path}")
                # self.gui.replay_history(pgn_path)

                # Ask if the game should continue
                time.sleep(10)
                break

            print("Game over!")

        except Exception as e:
            # Save the error message and include full traceback
            error_message = f"\nError occurred: {str(e)}\n{traceback.format_exc()}"
        else:
            # No error occurred
            error_message = None
        finally:

            # Always clean up the curses session
            #self.gui.hide()
            # Reset the terminal
            os.system("reset")

            # Display the error message after cleanup if there was one
            if error_message:
                print(error_message)


if __name__ == "__main__":
    # Create a mixed player game with 6 players, where players 0 and 1 are LLM-controlled and player 2 is collusion LLM-controlled
    game = MixedPlayerGame(
        buyin=500,
        big_blind=5,
        small_blind=2,
        max_players=6,
        llm_player_ids=[0, 1],
        collusion_llm_player_ids=[2],
        openai_model="gpt-4",
    )

    # Run the game
    game.run_game()
