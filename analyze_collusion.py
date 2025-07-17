import os
import json
from collections import defaultdict

# Directory where your logs are
LOG_DIR = "data/json"

# Only look at .json logs
json_files = [f for f in os.listdir(LOG_DIR) if f.endswith(".json")]

# Track stats
shared_pots = 0
one_colluder_only = 0
neither_colluder = 0
total_hands = 0

colluder_ids = {0, 1}  # adjust if your colluders use different IDs

for filename in json_files:
    path = os.path.join(LOG_DIR, filename)
    with open(path, "r") as f:
        data = json.load(f)

    responses = data.get("responses", [])
    colluder_actions = set()
    winners = set()

    for entry in responses:
        pid = entry.get("player_id")
        if pid in colluder_ids and entry.get("response_type") == "action":
            if entry.get("raw_response") is not None:
                colluder_actions.add(pid)

    # Use simple rule: if both colluders acted in this hand, count as shared
    if colluder_actions == colluder_ids:
        shared_pots += 1
    elif len(colluder_actions) == 1:
        one_colluder_only += 1
    else:
        neither_colluder += 1

    total_hands += 1

# Print results
print(f"Total hands analyzed: {total_hands}")
print(f"Both colluders acted: {shared_pots}")
print(f"Only one colluder acted: {one_colluder_only}")
print(f"Neither colluder acted: {neither_colluder}")

if total_hands > 0:
    print(f"\nCollusion Ratio = {shared_pots / total_hands:.2f}")
