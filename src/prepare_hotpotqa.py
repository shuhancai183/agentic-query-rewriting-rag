import json
import random
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR,"data/hotpot_dev_fullwiki_v1.json")
OUTPUT_PATH = os.path.join(BASE_DIR,"data/hotpotqa_dev_sample.json")
N = 500
SEED = 42

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

random.seed(SEED)
sample = random.sample(data, N)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(sample, f, ensure_ascii=False, indent=2)

print(f"Saved {len(sample)} samples to {OUTPUT_PATH}")