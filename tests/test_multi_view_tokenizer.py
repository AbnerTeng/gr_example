import json
import tempfile
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    LogitsProcessorList,
    T5Config,
    T5ForConditionalGeneration,
)

from src.dataset import GRDataset
from src.fast_trie import FastRQTrie, FastRQTrieLogitsProcessor
from src.multi_view_tokenizer import (
    add_multi_view_special_tokens,
    build_multi_view_special_tokens,
    route_token_ids,
    validate_atomic_tokens,
)


def test_build_tokens_includes_views_and_global_rq_namespaces():
    tokens = build_multi_view_special_tokens(n_levels=9, n_codes=4, n_views=3)
    assert tokens[:3] == ["<view_0>", "<view_1>", "<view_2>"]
    assert "<r0_0>" in tokens
    assert "<r8_3>" in tokens
    assert len(tokens) == 3 + 9 * 4
    assert len(tokens) == len(set(tokens))


def test_token_ids_are_atomic_resizeable_and_stable_after_reload():
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large", local_files_only=True)
    tokens = build_multi_view_special_tokens(n_levels=9, n_codes=4, n_views=3)
    added = add_multi_view_special_tokens(tokenizer, tokens)
    assert added == len(tokens)
    validate_atomic_tokens(tokenizer, tokens)

    route = "<r3_1> <r4_2> <r5_3>"
    path_ids = route_token_ids(tokenizer, route, add_eos=False)
    label_ids = route_token_ids(tokenizer, route, add_eos=True)
    assert len(path_ids) == 3
    assert label_ids == path_ids + [tokenizer.eos_token_id]

    config = T5Config(
        vocab_size=32,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        decoder_start_token_id=tokenizer.pad_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = T5ForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))
    assert model.get_input_embeddings().num_embeddings == len(tokenizer)

    token_to_id = {token: tokenizer.convert_tokens_to_ids(token) for token in tokens}
    with tempfile.TemporaryDirectory() as directory:
        tokenizer.save_pretrained(directory)
        model.save_pretrained(directory)
        reloaded = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        reloaded_model = T5ForConditionalGeneration.from_pretrained(
            directory, local_files_only=True
        )
        assert {token: reloaded.convert_tokens_to_ids(token) for token in tokens} == token_to_id
        assert route_token_ids(reloaded, route, add_eos=True) == label_ids
        assert reloaded_model.get_input_embeddings().num_embeddings == len(reloaded)

        trie = FastRQTrie([route], reloaded, reloaded.eos_token_id)
        processor = FastRQTrieLogitsProcessor(trie, len(reloaded), "cpu")
        encoded = reloaded("query: reload", return_tensors="pt")
        generated = reloaded_model.generate(
            **encoded,
            max_new_tokens=4,
            num_beams=1,
            logits_processor=LogitsProcessorList([processor]),
        )
        generated_route = " ".join(
            token
            for token in reloaded.convert_ids_to_tokens(generated[0])
            if token.startswith("<r")
        )
        assert generated_route == route


def test_training_labels_and_trie_paths_share_exact_token_ids():
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large", local_files_only=True)
    tokens = build_multi_view_special_tokens(n_levels=9, n_codes=4, n_views=3)
    add_multi_view_special_tokens(tokenizer, tokens)
    route = "<r6_1> <r7_2> <r8_3>"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "one.jsonl"
        path.write_text(json.dumps({"input": "<view_2> query: x", "output": route}) + "\n")
        dataset = GRDataset(str(path), tokenizer, 32, 8)
        labels = dataset[0]["labels"]
        trie = FastRQTrie([route], tokenizer, tokenizer.eos_token_id)
        path_ids = tuple(labels[:-1])
        assert labels == route_token_ids(tokenizer, route, add_eos=True)
        assert trie.has_path(path_ids)
        assert not hasattr(trie, "state_of")
        assert not hasattr(trie, "child")
        assert labels[-1] == trie.eos


def test_tensorized_trie_processor_matches_prefix_semantics():
    tokenizer = AutoTokenizer.from_pretrained(
        "google-t5/t5-large", local_files_only=True
    )
    tokens = build_multi_view_special_tokens(3, 4, 3)
    add_multi_view_special_tokens(tokenizer, tokens)
    routes = ["<r0_0> <r1_1> <r2_2>", "<r0_3> <r1_2> <r2_1>"]
    trie = FastRQTrie(routes, tokenizer, tokenizer.eos_token_id)
    processor = FastRQTrieLogitsProcessor(trie, len(tokenizer), "cpu")
    scores = torch.zeros(1, len(tokenizer))
    decoder_start = tokenizer.pad_token_id

    def allowed(prefix):
        ids = torch.tensor([[decoder_start] + prefix])
        output = processor(ids, scores.clone())
        return set(torch.isfinite(output[0]).nonzero().flatten().tolist())

    route_ids = route_token_ids(tokenizer, routes[0], add_eos=False)
    root_expected = {
        tokenizer.convert_tokens_to_ids("<r0_0>"),
        tokenizer.convert_tokens_to_ids("<r0_3>"),
    }
    assert allowed([]) == root_expected
    assert allowed(route_ids[:1]) == {route_ids[1]}
    assert allowed(route_ids) == {tokenizer.eos_token_id}
    assert allowed([tokenizer.convert_tokens_to_ids("<r1_0>")]) == {
        tokenizer.eos_token_id
    }


if __name__ == "__main__":
    test_build_tokens_includes_views_and_global_rq_namespaces()
    print("PASS test_build_tokens_includes_views_and_global_rq_namespaces")
    test_token_ids_are_atomic_resizeable_and_stable_after_reload()
    print("PASS test_token_ids_are_atomic_resizeable_and_stable_after_reload")
    test_training_labels_and_trie_paths_share_exact_token_ids()
    print("PASS test_training_labels_and_trie_paths_share_exact_token_ids")
    test_tensorized_trie_processor_matches_prefix_semantics()
    print("PASS test_tensorized_trie_processor_matches_prefix_semantics")
