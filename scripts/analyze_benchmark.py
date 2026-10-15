"""Analyze saved SpecMatch benchmark results."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

# Allow running directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specmatch.metrics import summarize, speedup


def load_records(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records") if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError("Expected a list of benchmark records.")
    return records


def analyze(records: list) -> None:
    groups = defaultdict(list)
    for record in records:
        groups[(record.get("mode"), record.get("K"))].append(record)

    print("\n=== Aggregate summary ===")
    for (mode, k), group in sorted(
        groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        stats = summarize(group)
        k_label = f"K={k}" if k is not None else "K=N/A"
        print(
            f"{mode:13s} {k_label:8s} "
            f"runs={stats['runs']:3d} "
            f"aggregate={stats['aggregate_tokens_per_sec']:.2f} tok/s "
            f"mean={stats['mean_tokens_per_sec']:.2f} tok/s "
            f"p50={stats['p50_wall_time']:.3f}s "
            f"p95={stats['p95_wall_time']:.3f}s "
            f"acceptance={stats['acceptance_rate']}"
        )

    baselines = [r for r in records if r.get("mode") == "baseline"]
    speculative = [r for r in records if r.get("mode") == "speculative"]

    if not baselines or not speculative:
        print("\nPaired comparison unavailable: need baseline and speculative records.")
        return

    speculative_ks = sorted({r.get("K") for r in speculative if r.get("K") is not None})
    has_comparison_metadata = any(r.get("comparison_K") is not None for r in baselines)

    # Older single-K files can be paired safely because only one speculative
    # configuration exists. Older multi-K files cannot identify which
    # baseline run belongs to which K.
    if not has_comparison_metadata and len(speculative_ks) > 1:
        print(
            "\nPaired comparison unavailable: this legacy multi-K file "
            "does not identify which baseline runs correspond to each K. "
            "Aggregate summaries above remain available."
        )
        return

    comparison_groups = []
    if has_comparison_metadata:
        for k in speculative_ks:
            base_group = [r for r in baselines if r.get("comparison_K") == k]
            spec_group = [r for r in speculative if r.get("K") == k]
            if base_group and spec_group:
                comparison_groups.append((k, base_group, spec_group))
    else:
        # Legacy single-K file.
        comparison_groups.append((speculative_ks[0], baselines, speculative))

    for k, base_group, spec_group in comparison_groups:
        print(f"\n=== Paired prompt analysis: K={k} ===")

        pairs = defaultdict(dict)
        for record in base_group:
            key = (record.get("prompt_id"), record.get("rep"))
            pairs[key]["baseline"] = record
        for record in spec_group:
            key = (record.get("prompt_id"), record.get("rep"))
            pairs[key]["speculative"] = record

        prompt_ratios = defaultdict(list)
        all_ratios = []
        missing = 0

        for (prompt_id, rep), pair in pairs.items():
            baseline = pair.get("baseline")
            spec = pair.get("speculative")

            if baseline is None or spec is None:
                missing += 1
                continue

            b_time = baseline.get("wall_time", 0)
            s_time = spec.get("wall_time", 0)

            if b_time <= 0 or s_time <= 0:
                missing += 1
                continue

            ratio = b_time / s_time
            prompt_ratios[prompt_id].append(ratio)
            all_ratios.append(ratio)

        print("Speedup > 1.0x means speculative generation was faster.")
        for prompt_id, ratios in sorted(prompt_ratios.items()):
            wins = sum(ratio > 1 for ratio in ratios)
            print(
                f"{str(prompt_id):24s} "
                f"median={median(ratios):.3f}x "
                f"wins={wins}/{len(ratios)}"
            )

        if all_ratios:
            print(f"\nPaired comparisons: {len(all_ratios)}")
            print(f"Overall paired median: {median(all_ratios):.3f}x")
            print(f"Paired wins: {sum(r > 1 for r in all_ratios)}/{len(all_ratios)}")
        if missing:
            print(f"Incomplete/invalid pairs skipped: {missing}")

        base_summary = summarize(base_group)
        spec_summary = summarize(spec_group)
        comparison = speedup(spec_summary, base_summary)

        print("\n=== Overall comparison ===")
        print(f"Aggregate throughput speedup: {comparison['throughput_speedup']:.3f}x")
        print(f"Median latency speedup: {comparison['latency_speedup_p50']:.3f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Path to benchmark JSON")
    args = parser.parse_args()

    if not args.results.is_file():
        parser.error(f"Results file not found: {args.results}")

    records = load_records(args.results)
    if not records:
        parser.error("The results file contains no records.")

    print(f"File: {args.results}")
    print(f"Records: {len(records)}")
    analyze(records)


if __name__ == "__main__":
    main()
