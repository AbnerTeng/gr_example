from datasets import load_dataset
import json
from pathlib import Path

out_dir = Path("data/msmarco")
out_dir.mkdir(parents=True, exist_ok=True)

# 載入 MS MARCO v1.1
ds = load_dataset("microsoft/ms_marco", "v1.1")

# 你可以改成 ds["test"]，但很多 repo 會用 validation 當本地 test
split_map = {
    "train": "train.jsonl",
    "validation": "test.jsonl",
}

for split_name, out_name in split_map.items():
    out_path = out_dir / out_name
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds[split_name]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print("done")
print(f"wrote: {out_dir / 'train.jsonl'}")
print(f"wrote: {out_dir / 'test.jsonl'}")