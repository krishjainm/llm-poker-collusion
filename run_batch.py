# run_batch.py
import subprocess

NUM_BATCHES = 93  # 93 x 10 hands = 930 (you already did 76)
HANDS_PER_BATCH = 10

for i in range(NUM_BATCHES):
    print(f"Running batch {i + 1}/{NUM_BATCHES}...")
    
    result = subprocess.run([
        "python", "run_game.py",
        "--num-hands", str(HANDS_PER_BATCH),
        "--save-dir", "data/json",
        "--mode", "collusion"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[!] Batch {i+1} failed:\n", result.stderr)
    else:
        print(f"[✓] Batch {i+1} complete.")
