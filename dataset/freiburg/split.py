#!/usr/bin/env python3
import json
import random
from pathlib import Path

# ===== CONFIG =====
TRAIN_IN  = Path("/data/pwojcik/For_Piotr/gloms_rect_from_png_within/train.json")  # input train.json
TRAIN_OUT = Path("/data/pwojcik/For_Piotr/gloms_rect_from_png_within/train.json")  # overwrite train.json
TEST_OUT  = Path("/data/pwojcik/For_Piotr/gloms_rect_from_png_within/test.json")   # new test.json
SPLIT_FRAC = 0.1   # 10% goes to test
RANDOM_SEED = 42   # for reproducibility
# ==================

def main():
    random.seed(RANDOM_SEED)

    with open(TRAIN_IN, "r") as f:
        data = json.load(f)

    classes = data.get("classes", [])
    if not classes:
        print("[ERROR] No classes found in input JSON")
        return

    # get all image keys (everything except "classes")
    all_keys = [k for k in data.keys() if k != "classes"]
    n_total = len(all_keys)
    n_test = max(1, int(round(n_total * SPLIT_FRAC)))
    test_keys = set(random.sample(all_keys, n_test))

    train_dict = {"classes": classes}
    test_dict  = {"classes": classes}

    for k in all_keys:
        if k in test_keys:
            test_dict[k] = data[k]
        else:
            train_dict[k] = data[k]

    with open(TRAIN_OUT, "w") as f:
        json.dump(train_dict, f, indent=2)
    with open(TEST_OUT, "w") as f:
        json.dump(test_dict, f, indent=2)

    print(f"[DONE] Wrote {len(train_dict)-1} train and {len(test_dict)-1} test images.")

if __name__ == "__main__":
    main()
