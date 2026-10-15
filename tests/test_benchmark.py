from scripts.benchmark import build_summary_groups


def test_build_summary_groups_keeps_each_speculative_k():
    records = [
        {"mode": "baseline", "K": None, "comparison_K": 2},
        {"mode": "speculative", "K": 2},
        {"mode": "baseline", "K": None, "comparison_K": 4},
        {"mode": "speculative", "K": 4},
    ]

    assert build_summary_groups(records, "both") == [
        ("baseline", 2),
        ("speculative", 2),
        ("baseline", 4),
        ("speculative", 4),
    ]


def test_build_summary_groups_for_baseline_mode():
    records = [{"mode": "baseline", "K": None}]

    assert build_summary_groups(records, "baseline") == [
        ("baseline", None),
    ]


def test_build_summary_groups_for_speculative_mode():
    records = [
        {"mode": "speculative", "K": 4},
        {"mode": "speculative", "K": 2},
    ]

    assert build_summary_groups(records, "speculative") == [
        ("speculative", 2),
        ("speculative", 4),
    ]