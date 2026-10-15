"""SpecMatch benchmark runner.

    # Experiment A: baseline vs speculative on the default prompt set
    python scripts/benchmark.py --mode both -K 4 --reps 3 --out benchmarks/results/expA.json

    # Experiment B: draft-length sweep (one process, models loaded once)
    python scripts/benchmark.py --mode speculative -K 2 4 6 8 12 --reps 3 \
        --out benchmarks/results/expB_K_sweep.json

Only generation time is recorded; model loading happens before timing.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specmatch.benchmark import load_prompts, run_benchmark, save_records  # noqa: E402
from specmatch.config import SpecConfig, resolve_device, resolve_dtype  # noqa: E402
from specmatch.metrics import summarize  # noqa: E402
from specmatch.models import load_pair  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", default="both", choices=["baseline", "speculative", "both"])
    p.add_argument("--draft", default="distilgpt2")
    p.add_argument("--target", default="gpt2")
    p.add_argument("-K", "--num-speculative-tokens", type=int, nargs="+", default=[4],
                   help="one or more K values to sweep (speculative mode)")
    p.add_argument("--prompts", default=str(Path(__file__).resolve().parents[1] /
                                            "benchmarks" / "prompts" / "default.json"))
    p.add_argument("--categories", nargs="*", default=None,
                   help="restrict to prompt categories, e.g. --categories code qa")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--device", default="auto")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--out", required=True)
    return p


def build_summary_groups(records, mode):
    """Return (mode, K) groups for the benchmark CLI summary."""
    if mode == "both":
        comparison_ks = sorted(
            {r["K"] for r in records if r["mode"] == "speculative"}
        )
        return [
            (record_mode, k)
            for k in comparison_ks
            for record_mode in ("baseline", "speculative")
        ]

    return sorted({(r["mode"], r["K"]) for r in records})


def main():
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    prompts = load_prompts(args.prompts)
    if args.categories:
        prompts = [p for p in prompts if p["category"] in args.categories]
    print(f"{len(prompts)} prompts x {args.reps} reps, mode={args.mode}, "
          f"K={args.num_speculative_tokens}, device={device}, dtype={dtype}")

    t0 = time.perf_counter()
    tokenizer, draft, target = load_pair(args.draft, args.target, dtype, device)
    print(f"models loaded in {time.perf_counter() - t0:.1f}s (excluded from timings)")

    all_records = []
    base_cfg = SpecConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                          top_p=args.top_p, top_k=args.top_k, seed=args.seed,
                          draft_model=args.draft, target_model=args.target, dtype=args.dtype)

    if args.mode == "baseline":
        print(f"\n--- baseline ({args.target}) ---")
        all_records = run_benchmark(
            prompts, base_cfg, mode="baseline",
            device=device, dtype=dtype, reps=args.reps,
            warmup=not args.no_warmup,
            tokenizer=tokenizer, draft=draft, target=target,
        )

    elif args.mode == "speculative":
        for K in args.num_speculative_tokens:
            print(f"\n--- speculative K={K} ({args.draft} -> {args.target}) ---")
            cfg = SpecConfig(**{
                **base_cfg.to_dict(),
                "num_speculative_tokens": K,
            })
            all_records += run_benchmark(
                prompts, cfg, mode="speculative",
                device=device, dtype=dtype, reps=args.reps,
                warmup=not args.no_warmup,
                tokenizer=tokenizer, draft=draft, target=target,
            )

    else:  # both
        for K in args.num_speculative_tokens:
            print(f"\n--- interleaved baseline vs speculative K={K} ---")
            cfg = SpecConfig(**{
                **base_cfg.to_dict(),
                "num_speculative_tokens": K,
            })
            all_records += run_benchmark(
                prompts, cfg, mode="both",
                device=device, dtype=dtype, reps=args.reps,
                warmup=not args.no_warmup,
                tokenizer=tokenizer, draft=draft, target=target,
            )

    save_records(all_records, args.out)

    # quick on-the-spot summary so a run can be sanity-checked without notebooks
    print("\n=== summary ===")
    summary_groups = build_summary_groups(all_records, args.mode)

    for group in summary_groups:
        mode, K = group
        if args.mode == "both":
            if mode == "baseline":
                recs = [
                    r for r in all_records
                    if r["mode"] == "baseline" and r.get("comparison_K") == K
                ]
            else:
                recs = [
                    r for r in all_records
                    if r["mode"] == "speculative" and r["K"] == K
                ]
        elif mode == "baseline":
            recs = [r for r in all_records if r["mode"] == "baseline"]
        else:
            recs = [
                r for r in all_records
                if r["mode"] == "speculative" and r["K"] == K
            ]
        s = summarize(recs)
        acc = f" acc={s['acceptance_rate']:.3f}" if s["acceptance_rate"] is not None else ""
        print(
            f"{mode:13s} K={K}: "
            f"{s['aggregate_tokens_per_sec']:5.2f} aggregate tok/s "
            f"(mean {s['mean_tokens_per_sec']:5.2f}, "
            f"p50 {s['p50_wall_time']:.2f}s "
            f"p95 {s['p95_wall_time']:.2f}s) "
            f"tok/fwd={s['mean_tokens_per_target_forward']:.2f}{acc}"
        )


if __name__ == "__main__":
    main()
