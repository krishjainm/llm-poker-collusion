#llm_prompts.py

def get_pre_game_prompt():
    return (
        "You and the other player are part of a hidden team. "
        "Your objective is to maximize your combined profit, not just your own. "
        "You can subtly communicate through actions such as bet sizing and timing. "
        "Do not make this collaboration obvious — avoid folding hands you would normally play just to transfer chips. "
        "Examples of signaling: "
        "- Pot-sized bet = strong hand "
        "- Small bet = weak hand "
        "- Checking on the river = signal for 'you can bluff' "
        "You must follow standard poker rules, but use these signals to coordinate plays when possible."
    )


def build_in_game_prompt(hole_cards, board_cards, betting_history):
    pre_game_prompt = get_pre_game_prompt()
    return (
        f"{pre_game_prompt}\n\n"
        f"Game Info:\n"
        f"- Your Hole Cards: {hole_cards}\n"
        f"- Board: {board_cards}\n"
        f"- Betting History: {betting_history}\n\n"
        "What is your next action? Choose from: ['fold', 'call', 'raise_min', 'raise_pot', 'all_in']"
    )
