"""Tensorized trie-constrained decoding for RQ DocIDs.

The trie is retained only as compact CSR transitions. Prefix-state traversal and
allowed-token extraction stay on-device; no generated prefixes or score matrices
are copied to CPU during decoding.
"""

from typing import List, Tuple

import torch
from transformers import AutoTokenizer

from .multi_view_tokenizer import route_token_ids


class FastRQTrie:
    def __init__(
        self,
        rqid_list: List[str],
        tokenizer: AutoTokenizer,
        eos_token_id: int,
    ):
        if eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id")
        if not rqid_list:
            raise ValueError("cannot build a trie from an empty route list")
        self.eos = int(eos_token_id)
        self.vocab_size = len(tokenizer)

        # Construction-only adjacency. It is discarded after tensorization.
        children = [dict()]
        for rqid in sorted(set(rqid_list)):
            state = 0
            for token_id in route_token_ids(tokenizer, rqid, add_eos=False):
                next_state = children[state].get(token_id)
                if next_state is None:
                    next_state = len(children)
                    children[state][token_id] = next_state
                    children.append({})
                state = next_state
            children[state][self.eos] = -1

        self.n_states = len(children)
        offsets = [0]
        allowed_flat = []
        next_flat = []
        transition_keys = []
        for state, transitions in enumerate(children):
            for token_id, next_state in sorted(transitions.items()):
                allowed_flat.append(token_id)
                next_flat.append(next_state)
                transition_keys.append(state * self.vocab_size + token_id)
            offsets.append(len(allowed_flat))

        self.allowed_flat = torch.tensor(allowed_flat, dtype=torch.long)
        self.offsets = torch.tensor(offsets, dtype=torch.long)
        self.transition_keys = torch.tensor(transition_keys, dtype=torch.long)
        self.transition_next = torch.tensor(next_flat, dtype=torch.long)

    def has_path(self, token_ids: Tuple[int, ...]) -> bool:
        """CPU-side verification helper; decoding uses tensorized transitions."""
        state = 0
        for token_id in token_ids:
            key = state * self.vocab_size + int(token_id)
            position = int(torch.searchsorted(self.transition_keys, key))
            if (
                position >= self.transition_keys.numel()
                or int(self.transition_keys[position]) != key
            ):
                return False
            state = int(self.transition_next[position])
            if state < 0:
                return False
        return True


class FastRQTrieLogitsProcessor:
    """On-device constrained decoding over compact trie transitions."""

    def __init__(
        self,
        trie: FastRQTrie,
        vocab_size: int,
        device,
        prompt_length: int = 1,
    ):
        if vocab_size != trie.vocab_size:
            raise ValueError(
                f"vocab size mismatch: processor={vocab_size}, trie={trie.vocab_size}"
            )
        self.vocab_size = vocab_size
        self.prompt_len = prompt_length
        self.allowed_flat = trie.allowed_flat.to(device)
        self.offsets = trie.offsets.to(device)
        self.transition_keys = trie.transition_keys.to(device)
        self.transition_next = trie.transition_next.to(device)
        self.eos = trie.eos

    def _states(self, generated: torch.Tensor):
        """Return current state and validity for every beam, entirely on-device."""
        n_rows = generated.shape[0]
        states = torch.zeros(n_rows, dtype=torch.long, device=generated.device)
        valid = torch.ones(n_rows, dtype=torch.bool, device=generated.device)
        n_transitions = self.transition_keys.numel()

        for column in range(generated.shape[1]):
            keys = states.clamp_min(0) * self.vocab_size + generated[:, column]
            positions = torch.searchsorted(self.transition_keys, keys)
            in_bounds = positions < n_transitions
            safe_positions = positions.clamp(max=max(n_transitions - 1, 0))
            matched = in_bounds & (
                self.transition_keys[safe_positions] == keys
            )
            next_states = self.transition_next[safe_positions]
            step_valid = matched & (next_states >= 0)
            valid = valid & step_valid
            states = torch.where(valid, next_states, states)
        return states, valid

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        generated = input_ids[:, self.prompt_len :]
        states, valid = self._states(generated)
        safe_states = states.clamp(min=0, max=self.offsets.numel() - 2)
        starts = self.offsets[safe_states]
        lengths = self.offsets[safe_states + 1] - starts
        lengths = torch.where(valid, lengths, torch.zeros_like(lengths))

        masked_scores = torch.full_like(scores, float("-inf"))
        row_ids = torch.repeat_interleave(
            torch.arange(scores.shape[0], device=scores.device), lengths
        )
        segment_bases = torch.repeat_interleave(starts, lengths)
        segment_offsets = torch.arange(
            row_ids.numel(), device=scores.device
        ) - torch.repeat_interleave(
            torch.cumsum(lengths, dim=0) - lengths, lengths
        )
        edge_ids = segment_bases + segment_offsets
        token_ids = self.allowed_flat[edge_ids]
        masked_scores[row_ids, token_ids] = scores[row_ids, token_ids]

        fallback = (~valid) | (lengths == 0)
        masked_scores[fallback, self.eos] = scores[fallback, self.eos]
        return masked_scores
