import torch

from src.eval_rq_view_predictability import (
    append_code_features,
    conditioning_levels,
    fit_probe,
    metrics_from_logits,
    summarize_view_complementarity,
    summarize_views,
)


def test_marginal_conditioning_resets_at_each_view_boundary():
    assert conditioning_levels(level=0, view_size=3, cross_view=False) == []
    assert conditioning_levels(level=2, view_size=3, cross_view=False) == [0, 1]
    assert conditioning_levels(level=3, view_size=3, cross_view=False) == []
    assert conditioning_levels(level=5, view_size=3, cross_view=False) == [3, 4]
    assert conditioning_levels(level=6, view_size=3, cross_view=False) == []


def test_conditional_mode_exposes_previous_views():
    assert conditioning_levels(level=3, view_size=3, cross_view=True) == [0, 1, 2]
    assert conditioning_levels(level=5, view_size=3, cross_view=True) == [0, 1, 2, 3, 4]
    assert conditioning_levels(level=6, view_size=3, cross_view=True) == list(range(6))


def test_append_code_features_uses_only_requested_levels():
    query = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    codes = torch.tensor([[0, 2, 1], [1, 0, 2]])

    features = append_code_features(query, codes, levels=[0, 2], n_codes=3)

    expected = torch.tensor(
        [
            [1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [3.0, 4.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert torch.equal(features, expected)


def test_metrics_from_logits_reports_nll_and_topk_accuracy():
    logits = torch.tensor([[5.0, 1.0, 0.0], [0.0, 2.0, 3.0]])
    targets = torch.tensor([0, 1])

    metrics, predictions = metrics_from_logits(logits, targets)

    expected_nll = torch.nn.functional.cross_entropy(logits, targets).item()
    assert abs(metrics["nll"] - expected_nll) < 1e-6
    assert metrics["top1_accuracy"] == 0.5
    assert metrics["top10_accuracy"] == 1.0
    assert predictions.tolist() == [0, 2]


def test_fit_probe_learns_a_separable_query_to_code_mapping():
    torch.manual_seed(0)
    negative = torch.randn(64, 2) * 0.1 - 1.0
    positive = torch.randn(64, 2) * 0.1 + 1.0
    train_query = torch.cat([negative, positive])
    train_codes = torch.cat(
        [torch.zeros(64, 1, dtype=torch.long), torch.ones(64, 1, dtype=torch.long)]
    )
    test_query = torch.tensor([[-1.1, -0.9], [-0.8, -1.2], [0.9, 1.1], [1.2, 0.8]])
    test_codes = torch.tensor([[0], [0], [1], [1]])

    metrics, predictions = fit_probe(
        train_query,
        train_codes,
        test_query,
        test_codes,
        target_level=0,
        conditioning=[],
        n_codes=2,
        hidden=0,
        epochs=20,
        batch_size=32,
        lr=0.1,
        seed=0,
    )

    assert metrics["top1_accuracy"] == 1.0
    assert metrics["train_top1_accuracy"] == 1.0
    assert metrics["train_nll"] < 0.05
    assert predictions.tolist() == [0, 0, 1, 1]


def test_summarize_views_keeps_per_view_exact_match_separate():
    targets = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    predictions = targets.clone()
    predictions[1, 4] = 0
    level_metrics = [
        {"nll": float(level), "top1_accuracy": 1.0, "top10_accuracy": 1.0}
        for level in range(6)
    ]

    views = summarize_views(level_metrics, predictions, targets, view_size=3)

    assert views[0]["exact_match"] == 1.0
    assert views[1]["exact_match"] == 0.5
    assert views[0]["mean_nll"] == 1.0
    assert views[1]["mean_nll"] == 4.0


def test_summarize_view_complementarity_tracks_failure_overlap():
    targets = torch.zeros(4, 6, dtype=torch.long)
    predictions = targets.clone()
    predictions[1, 0] = 1
    predictions[2, 3] = 1
    predictions[3, 0] = 1
    predictions[3, 3] = 1

    metrics = summarize_view_complementarity(predictions, targets, view_size=3)

    assert metrics["any_view_exact_fraction"] == 0.75
    assert metrics["all_views_wrong_fraction"] == 0.25
    assert abs(metrics["pairwise"][0]["failure_jaccard"] - 1 / 3) < 1e-6


if __name__ == "__main__":
    test_marginal_conditioning_resets_at_each_view_boundary()
    print("PASS test_marginal_conditioning_resets_at_each_view_boundary")
    test_conditional_mode_exposes_previous_views()
    print("PASS test_conditional_mode_exposes_previous_views")
    test_append_code_features_uses_only_requested_levels()
    print("PASS test_append_code_features_uses_only_requested_levels")
    test_metrics_from_logits_reports_nll_and_topk_accuracy()
    print("PASS test_metrics_from_logits_reports_nll_and_topk_accuracy")
    test_fit_probe_learns_a_separable_query_to_code_mapping()
    print("PASS test_fit_probe_learns_a_separable_query_to_code_mapping")
    test_summarize_views_keeps_per_view_exact_match_separate()
    print("PASS test_summarize_views_keeps_per_view_exact_match_separate")
    test_summarize_view_complementarity_tracks_failure_overlap()
    print("PASS test_summarize_view_complementarity_tracks_failure_overlap")
