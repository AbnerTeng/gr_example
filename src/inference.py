from typing import List

import torch
from transformers import AutoTokenizer


class TrieNode:
    def __init__(self) -> None:
        self.children = {}
        self.is_end = False


def build_rq_trie(rqid_list: List, tokenizer: AutoTokenizer) -> TrieNode:
    """Build trie from all valid RQ docid strings."""
    root = TrieNode()
    for rqid in set(rqid_list):
        ids = tokenizer.encode(rqid, add_special_tokens=False)
        node = root

        for tid in ids:
            if tid not in node.children:
                node.children[tid] = TrieNode()

            node = node.children[tid]
        node.is_end = True

    return root


class RQTrieLogitsProcessor:
    """
    Constrains T5 decoder to valid RQ docid sequences.
    prompt_length=1 to skip the initial decoder_start_token (pad).
    """

    def __init__(
        self, trie_root: TrieNode, eos_token_id: int, prompt_length: int = 1
    ) -> None:
        self.root = trie_root
        self.eos = eos_token_id
        self.prompt_len = prompt_length

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores_cpu = scores.cpu()
        vocab_size = scores_cpu.shape[1]

        for i in range(input_ids.shape[0]):
            generated = input_ids[i][self.prompt_len :].tolist()
            node = self.root
            valid = True

            for tid in generated:
                if tid in node.children:
                    node = node.children[tid]
                else:
                    valid = False
                    break

            new_row = torch.full((vocab_size,), float("-inf"))

            if valid:
                allowed = list(node.children.keys()) or [self.eos]
            else:
                allowed = [self.eos]

            allowed_t = torch.tensor(allowed, dtype=torch.long)
            new_row[allowed_t] = scores_cpu[i][allowed_t]
            scores_cpu[i] = new_row

        return scores_cpu.to(scores.device)
