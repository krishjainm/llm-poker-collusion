import os
import json

debug_dir = "data/debug_logs"
if not os.path.exists(debug_dir):
    print(f"Directory not found: {debug_dir}")
    exit(1)

files = sorted([f for f in os.listdir(debug_dir) if f.endswith(".json")])
print(f"Found {len(files)} debug log files.")

for file in files:
    path = os.path.join(debug_dir, file)
    with open(path, "r") as f:
        data = json.load(f)

    print(f"\nFile: {file}")
    if "responses" not in data:
        print("  No responses found.")
        continue

    actions = [
        r for r in data["responses"]
        if r.get("response_type") == "action"
    ]

    if not actions:
        print("  No LLM actions found.")
    else:
        for action in actions:
            player_id = action.get("player_id", "?")
            raw = action.get("raw_response", "").strip()
            processed = action.get("processed_response", "").strip()
            print(f"  Player {player_id}: {processed or raw}")
