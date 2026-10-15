"""Aggregate statistics over benchmark run records.

A "record" is the dict produced by benchmark.run_benchmark for one
(prompt, rep) generation. These helpers turn lists of records into the
summary numbers reported in the notebooks and README.
"""

import math
from collections import defaultdict


def percentile(values, pct: float) -> float:
    """Nearest-rank-free linear-interpolation percentile (like numpy's default)."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def stdev(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def summarize(records: list) -> dict:
    """Aggregate a list of run records into summary statistics."""
    if not records:
        return {}
    wall = [r["wall_time"] for r in records]
    tps = [r["tokens_per_sec"] for r in records]
    proposed = sum(r["proposed"] for r in records)
    accepted = sum(r["accepted"] for r in records)
    accepted_per_iter = [
        count
        for record in records
        for count in record.get("accepted_per_iter", [])
    ]
    out = {
        "runs": len(records),
        "mean_tokens_per_sec": mean(tps),
        "aggregate_tokens_per_sec": (
            sum(r["new_tokens"] for r in records) / sum(wall)
            if sum(wall) > 0 else 0.0
        ),
        "std_tokens_per_sec": stdev(tps),
        "mean_wall_time": mean(wall),
        "p50_wall_time": percentile(wall, 50),
        "p95_wall_time": percentile(wall, 95),
        "mean_ttft": mean([r["ttft"] for r in records if r.get("ttft") is not None]),
        "total_new_tokens": sum(r["new_tokens"] for r in records),
        "mean_new_tokens": mean([r["new_tokens"] for r in records]),
        "proposed": proposed,
        "accepted": accepted,
        # token-level acceptance rate = accepted proposals / proposed tokens
        "acceptance_rate": accepted / proposed if proposed else None,
        "mean_accepted_per_iter": mean(accepted_per_iter)
        if accepted_per_iter else None,
        "mean_tokens_per_target_forward": mean([r["mean_tokens_per_target_forward"] for r in records]),
        "mean_target_forwards": mean([r["target_forwards"] for r in records]),
        "mean_draft_forwards": mean([r["draft_forwards"] for r in records]),
        "peak_memory_mb": max((r.get("peak_memory_mb") or 0) for r in records),
    }
    return out


def summarize_by(records: list, key: str) -> dict:
    """Group records by a field (e.g. 'category' or 'K') and summarize each group."""
    groups = defaultdict(list)
    for r in records:
        groups[r.get(key)].extend([r])
    return {k: summarize(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def speedup(spec_summary: dict, base_summary: dict) -> dict:
    """Compare speculative generation against baseline generation."""
    spec_tps = spec_summary.get(
        "aggregate_tokens_per_sec",
        spec_summary["mean_tokens_per_sec"],
    )
    base_tps = base_summary.get(
        "aggregate_tokens_per_sec",
        base_summary["mean_tokens_per_sec"],
    )

    return {
        "throughput_speedup": spec_tps / base_tps if base_tps > 0 else 0.0,
        "latency_speedup_p50": (
            base_summary["p50_wall_time"] / spec_summary["p50_wall_time"]
            if spec_summary["p50_wall_time"] > 0
            else 0.0
        ),
        "latency_speedup_mean": (
            base_summary["mean_wall_time"] / spec_summary["mean_wall_time"]
            if spec_summary["mean_wall_time"] > 0
            else 0.0
        ),
    }
