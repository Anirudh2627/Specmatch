"""SpecMatch speculative decoding from the command line.

    python scripts/run_speculative.py --prompt "Explain gravity:" -K 4
    python scripts/run_speculative.py --draft distilgpt2 --target gpt2 -K 6 --temperature 0.8 --top-p 0.95
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specmatch.config import SpecConfig, resolve_device, resolve_dtype  # noqa: E402
from specmatch.generate import baseline_generate  # noqa: E402
from specmatch.models import load_pair  # noqa: E402
from specmatch.speculative import speculative_generate  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft", default="distilgpt2")
    p.add_argument("--target", default="gpt2")
    p.add_argument("-K", "--num-speculative-tokens", type=int, default=4)
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--device", default="auto")
    p.add_argument("--compare", action="store_true",
                   help="also run baseline greedy and report speedup + output match")
    return p


def main():
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    t0 = time.perf_counter()
    tokenizer, draft, target = load_pair(args.draft, args.target, dtype, device)
    print(f"loaded {args.draft} (draft) + {args.target} (target) on {device} ({dtype}) "
          f"in {time.perf_counter() - t0:.1f}s")

    cfg = SpecConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                     top_p=args.top_p, top_k=args.top_k, seed=args.seed,
                     num_speculative_tokens=args.num_speculative_tokens,
                     draft_model=args.draft, target_model=args.target, dtype=args.dtype)

    prompt = args.prompt or "The meaning of life is"
    text, stats = speculative_generate(draft, target, tokenizer, prompt, cfg, device)
    print(f"\nprompt: {prompt!r}")
    print(f"output: {text!r}")
    print(f"\nK={cfg.num_speculative_tokens} | {stats.new_tokens} tokens in {stats.wall_time:.2f}s "
          f"= {stats.tokens_per_sec:.1f} tok/s")
    if stats.acceptance_rate is not None:
        print(f"accepted {stats.accepted}/{stats.proposed} proposals "
              f"({stats.acceptance_rate:.1%}) | {stats.target_forwards} target forwards "
              f"-> {stats.mean_tokens_per_target_forward:.2f} tok/forward")
    print(f"ttft {stats.ttft:.2f}s | stop: {stats.stop_reason}")

    if args.compare:
        base_cfg = SpecConfig(**{**cfg.to_dict(), "temperature": args.temperature})
        btext, bstats = baseline_generate(target, tokenizer, prompt, base_cfg, device)
        print(f"\nbaseline: {bstats.new_tokens} tokens in {bstats.wall_time:.2f}s "
              f"= {bstats.tokens_per_sec:.1f} tok/s")
        print(f"speedup: {stats.wall_time and bstats.wall_time / stats.wall_time:.2f}x")
        if cfg.greedy:
            match = stats.token_ids == bstats.token_ids
            print(f"greedy outputs identical: {match}")


if __name__ == "__main__":
    main()
