"""Baseline autoregressive generation.

    python scripts/run_baseline.py --prompt "Explain gravity:" --max-new-tokens 48
    python scripts/run_baseline.py --prompt-file benchmarks/prompts/default.json --greedy
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from specmatch.config import GenConfig, resolve_device, resolve_dtype  # noqa: E402
from specmatch.generate import baseline_generate  # noqa: E402
from specmatch.models import load_model, load_tokenizer  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-file", default=None, help="JSON prompt set; runs all prompts")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--device", default="auto")
    return p


def main():
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    t0 = time.perf_counter()
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, dtype, device)
    load_time = time.perf_counter() - t0
    print(f"loaded {args.model} on {device} ({dtype}) in {load_time:.1f}s "
          f"[loading is excluded from generation timing]")

    cfg = GenConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                    top_p=args.top_p, top_k=args.top_k, seed=args.seed)

    if args.prompt_file:
        import json
        prompts = json.load(open(args.prompt_file))["prompts"]
    elif args.prompt:
        prompts = [{"id": "cli", "category": "cli", "prompt": args.prompt}]
    else:
        prompts = [{"id": "cli", "category": "cli", "prompt": "The meaning of life is"}]

    for entry in prompts:
        torch.manual_seed(args.seed) if args.seed is not None else None
        text, stats = baseline_generate(model, tokenizer, entry["prompt"], cfg, device)
        print(f"\n=== {entry['id']} ===")
        print(f"prompt: {entry['prompt']!r}")
        print(f"output: {text!r}")
        print(f"{stats.new_tokens} tokens in {stats.wall_time:.2f}s = "
              f"{stats.tokens_per_sec:.1f} tok/s | ttft {stats.ttft:.2f}s | "
              f"stop: {stats.stop_reason} | forwards: {stats.target_forwards}")


if __name__ == "__main__":
    main()
