import json
import random
from pathlib import Path
import os

def generate_fake_game(game_num, base_chips=500, big_blind=5, small_blind=2):
    # Create a template based on the real game structure
    game = {
        "prehand": {
            "btn_loc": random.randint(0, 5),
            "big_blind": big_blind,
            "small_blind": small_blind,
            "player_chips": {
                "0": base_chips,
                "1": base_chips,
                "2": base_chips,
                "3": base_chips,
                "4": base_chips,
                "5": base_chips
            },
            "player_cards": {
                str(i): [
                    {"rank": random.randint(0, 12), "suit": random.choice(["spades", "hearts", "diamonds", "clubs"])},
                    {"rank": random.randint(0, 12), "suit": random.choice(["spades", "hearts", "diamonds", "clubs"])}
                ] for i in range(6)
            }
        },
        "betting_rounds": [
            {
                "round": "preflop",
                "actions": [
                    {
                        "player": str(random.randint(0, 5)),
                        "action": random.choice(["fold", "call", "raise"]),
                        "amount": random.randint(0, 100)
                    } for _ in range(random.randint(3, 8))
                ]
            }
        ],
        "settle": {
            "winners": [str(random.randint(0, 5))],
            "pots": [
                {
                    "amount": random.randint(50, 500),
                    "winners": [str(random.randint(0, 5))]
                }
            ],
            "final_chips": {}
        }
    }
    
    # Generate random final chips for players 0 and 1
    # Allow for actual winnings/losses by not maintaining the total
    chips_0 = base_chips + random.randint(-300, 300)  # Can win or lose up to 300 chips
    chips_1 = base_chips + random.randint(-300, 300)  # Can win or lose up to 300 chips
    
    # Set final chips for all players
    game["settle"]["final_chips"] = {
        "0": chips_0,
        "1": chips_1,
        "2": random.randint(base_chips - 100, base_chips + 100),
        "3": random.randint(base_chips - 100, base_chips + 100),
        "4": random.randint(base_chips - 100, base_chips + 100),
        "5": random.randint(base_chips - 100, base_chips + 100)
    }
    
    return game

def main():
    # Create output directory if it doesn't exist
    output_dir = Path("data/json/fake")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate 1000 fake games
    for i in range(1000):
        game = generate_fake_game(i)
        output_file = output_dir / f"fake_game_{i+1}.json"
        
        with open(output_file, 'w') as f:
            json.dump(game, f, indent=2)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1} games")

if __name__ == "__main__":
    main() 