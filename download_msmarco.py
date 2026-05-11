import json
from pathlib import Path

from datasets import load_dataset

out_dir = Path("data/msmarco")
out_dir.mkdir(parents=True, exist_ok=True)

# 載入 MS MARCO v1.1
ds = load_dataset("microsoft/ms_marco", "v1.1")

split_map = {
    "train": "train.jsonl",
    "validation": "valid.jsonl",
    "test": "test.jsonl",
}

for split_name, out_name in split_map.items():
    out_path = out_dir / out_name
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds[split_name]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print("done")
print(f"wrote: {out_dir / 'train.jsonl'}")
print(f"wrote: {out_dir / 'valid.jsonl'}")
print(f"wrote: {out_dir / 'test.jsonl'}")
