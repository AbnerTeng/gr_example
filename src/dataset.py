import json
from typing import Any, Dict

from transformers import AutoTokenizer
from torch.utils.data import Dataset

from .multi_view_tokenizer import route_token_ids


class GRDataset(Dataset):
    """Random-access JSONL dataset without retaining all decoded rows in RAM."""

    def __init__(
        self,
        path: str,
        tokenizer: AutoTokenizer,
        max_input_length: int,
        max_output_length: int,
    ) -> None:
        self.path = path
        self.offsets = []
        with open(path, "rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer = tokenizer
        self._handle = None

    def __len__(self) -> int:
        return len(self.offsets)

    def _read_item(self, idx: int):
        if self._handle is None or self._handle.closed:
            self._handle = open(self.path, "rb")
        self._handle.seek(self.offsets[idx])
        return json.loads(self._handle.readline())

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self._read_item(idx)
        enc = self.tokenizer(
            item["input"], truncation=True, max_length=self.max_input_length
        )
        labels = route_token_ids(
            self.tokenizer, item["output"], add_eos=True
        )
        if len(labels) > self.max_output_length:
            raise ValueError(
                f"target has {len(labels)} tokens, exceeding max_output_length="
                f"{self.max_output_length}"
            )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }
