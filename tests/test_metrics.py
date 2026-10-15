from specmatch.metrics import summarize


def test_mean_accepted_per_iter():
    records = [
        {
            "wall_time": 1.0,
            "tokens_per_sec": 10.0,
            "new_tokens": 8,
            "proposed": 6,
            "accepted": 4,
            "target_forwards": 3,
            "accepted_per_iter": [2, 1, 1],
            "mean_tokens_per_target_forward": 2.0,
            "draft_forwards": 6,
        },
        {
            "wall_time": 2.0,
            "tokens_per_sec": 8.0,
            "new_tokens": 16,
            "proposed": 8,
            "accepted": 6,
            "target_forwards": 4,
            "accepted_per_iter": [3, 2, 1],
            "mean_tokens_per_target_forward": 4.0,
            "draft_forwards": 8,
        },
    ]

    result = summarize(records)

    assert result["mean_accepted_per_iter"] == 10 / 6


def test_mean_accepted_per_iter_missing_data():
    record = {
        "wall_time": 1.0,
        "tokens_per_sec": 10.0,
        "new_tokens": 8,
        "proposed": 6,
        "accepted": 4,
        "target_forwards": 3,
        "mean_tokens_per_target_forward": 2.0,
        "draft_forwards": 6,
    }

    result = summarize([record])

    assert result["mean_accepted_per_iter"] is None


def test_aggregate_tokens_per_sec():
    records = [
        {
            "wall_time": 1.0,
            "tokens_per_sec": 10.0,
            "new_tokens": 8,
            "proposed": 0,
            "accepted": 0,
            "target_forwards": 8,
            "mean_tokens_per_target_forward": 1.0,
            "draft_forwards": 0,
        },
        {
            "wall_time": 2.0,
            "tokens_per_sec": 8.0,
            "new_tokens": 16,
            "proposed": 0,
            "accepted": 0,
            "target_forwards": 16,
            "mean_tokens_per_target_forward": 1.0,
            "draft_forwards": 0,
        },
    ]

    result = summarize(records)

    assert result["aggregate_tokens_per_sec"] == 8.0
    assert result["mean_tokens_per_sec"] == 9.0