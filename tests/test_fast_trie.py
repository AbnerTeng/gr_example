"""FastRQTrieLogitsProcessor enforces the exact synthetic route trie."""

import random

import torch

from src.fast_trie import FastRQTrie, FastRQTrieLogitsProcessor


class FakeAtomicTokenizer:
    def __init__(self, tokens):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._token_ids = {token: index + 2 for index, token in enumerate(tokens)}

    def __len__(self):
        return len(self._token_ids) + 2

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [self._token_ids[token] for token in text.split()]


def main():
    level_tokens = [
        [f"<r{level}_{code}>" for code in range(size)]
        for level, size in enumerate((20, 15, 10))
    ]
    routes = [
        f"{first} {second} {third}"
        for first in level_tokens[0]
        for second in level_tokens[1]
        for third in level_tokens[2]
    ]
    tokenizer = FakeAtomicTokenizer(
        token for level in level_tokens for token in level
    )
    route_ids = [tokenizer.encode(route) for route in routes]
    eos = tokenizer.eos_token_id
    vocab_size = len(tokenizer)
    processor = FastRQTrieLogitsProcessor(
        FastRQTrie(routes, tokenizer, eos), vocab_size, "cpu"
    )

    generator = torch.Generator().manual_seed(0)
    for step in range(4):
        rows = []
        prefixes = []
        for row_index in range(16):
            route = random.Random(step * 100 + row_index).choice(routes)
            prefix = tokenizer.encode(route)[:step]
            prefixes.append(prefix)
            rows.append([tokenizer.pad_token_id] + prefix)
        input_ids = torch.tensor(rows)
        scores = torch.randn(len(rows), vocab_size, generator=generator)
        actual = processor(input_ids, scores.clone())

        for row_index, prefix in enumerate(prefixes):
            if step == 3:
                expected = {eos}
            else:
                expected = {
                    route[step]
                    for route in route_ids
                    if route[:step] == prefix
                }
            actual_allowed = set(
                torch.where(torch.isfinite(actual[row_index]))[0].tolist()
            )
            assert actual_allowed == expected, (
                step,
                prefix,
                actual_allowed,
                expected,
            )
            indices = torch.tensor(sorted(expected))
            assert torch.allclose(
                actual[row_index, indices], scores[row_index, indices]
            )
        print(
            f"PASS step={step} allowed/step avg="
            f"{torch.isfinite(actual).sum(-1).float().mean():.1f}"
        )

    print("PASS fast trie matches explicit transition contract")


if __name__ == "__main__":
    main()
