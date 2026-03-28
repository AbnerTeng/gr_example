import json
from typing import Any, Dict

from transformers import AutoTokenizer
from torch.utils.data import Dataset


class GRDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: AutoTokenizer,
        max_input_length: int,
        max_output_length: int,
    ) -> None:
        with open(path, "r") as f:
            self.data = [json.loads(line) for line in f]

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        enc = self.tokenizer(
            item["input"], truncation=True, max_length=self.max_input_length
        )
        labels = self.tokenizer(
            item["output"], truncation=True, max_length=self.max_output_length
        )["input_ids"]

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }
