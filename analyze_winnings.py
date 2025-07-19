import json
import os
from pathlib import Path

def analyze_winnings():
    # Directory containing JSON files
    json_dir = Path("data/json")
    
    # Initialize variables
    total_winnings = 0
    total_ev = 0
    total_hands = 0
    big_blind = None
    
    # Check if directory exists
    if not json_dir.exists():
        print(f"Error: Directory {json_dir} does not exist")
        return
    
    # Get list of JSON files
    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        print(f"Error: No JSON files found in {json_dir}")
        return
    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Get big blind amount (should be same for all files)
                if big_blind is None:
                    big_blind = data['prehand']['big_blind']
                
                # Get starting chips for players 0 and 1
                start_chips_0 = data['prehand']['player_chips']['0']
                start_chips_1 = data['prehand']['player_chips']['1']
                total_start_chips = start_chips_0 + start_chips_1
                
                # Get final chips for players 0 and 1
                final_chips_0 = data['settle']['final_chips']['0']
                final_chips_1 = data['settle']['final_chips']['1']
                total_final_chips = final_chips_0 + final_chips_1
                
                # Calculate winnings for this hand
                hand_winnings = total_final_chips - total_start_chips
                total_winnings += hand_winnings
                
                # Calculate EV for this hand
                if 'preflop_equity' in data['prehand']:
                    # Get preflop equity for players 0 and 1
                    equity_0 = data['prehand']['preflop_equity']['0']
                    equity_1 = data['prehand']['preflop_equity']['1']
                    total_equity = equity_0 + equity_1
                    
                    # Get pot size at showdown
                    pot_size = data['settle']['pot_size']
                    
                    # Calculate EV
                    hand_ev = (total_equity * pot_size) - total_start_chips
                    total_ev += hand_ev
                
                total_hands += 1
                
        except Exception as e:
            print(f"Error processing file {json_file}: {str(e)}")
            continue
    
    if total_hands == 0:
        print("No valid hands were processed")
        return
    
    # Calculate BB/100 (assuming 100 hands)
    bb_per_100 = (total_winnings / big_blind) * (100 / total_hands)
    ev_bb_per_100 = (total_ev / big_blind) * (100 / total_hands)
    
    # Print results
    print(f"Total hands analyzed: {total_hands}")
    print(f"Total winnings: {total_winnings} chips")
    print(f"Total EV: {total_ev:.2f} chips")
    print(f"BB/100: {bb_per_100:.2f}")
    print(f"EV BB/100: {ev_bb_per_100:.2f}")

if __name__ == "__main__":
    analyze_winnings() 