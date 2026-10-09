"""SpecMatch demo: baseline vs speculative decoding on the same prompt.

    python demo/app.py --terminal          # CLI comparison (no extra deps)
    python demo/app.py --terminal --prompt "Explain gravity:" -K 4
    streamlit run demo/app.py              # simple web UI (pip install streamlit)

Both modes load the pair once and run the two engines with identical decoding
settings, then report text, wall time, tokens/sec, speedup and acceptance rate.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from specmatch.config import SpecConfig, resolve_device, resolve_dtype
from specmatch.generate import baseline_generate
from specmatch.models import load_pair
from specmatch.speculative import speculative_generate

_MODELS = {}


def get_models(draft_name, target_name, dtype_name, device_name):
    """Load (and memoize) the draft/target pair."""
    key = (draft_name, target_name, dtype_name, device_name)
    if key not in _MODELS:
        device = resolve_device(device_name)
        dtype = resolve_dtype(dtype_name, device)
        tokenizer, draft, target = load_pair(draft_name, target_name, dtype, device)
        _MODELS[key] = (tokenizer, draft, target, device)
    return _MODELS[key]


def compare(prompt, max_new_tokens, K, temperature, top_p, seed,
            draft_name, target_name, dtype_name, device_name):
    """Run baseline and speculative generation with identical settings."""
    tokenizer, draft, target, device = get_models(draft_name, target_name, dtype_name, device_name)
    cfg = SpecConfig(max_new_tokens=max_new_tokens, temperature=temperature,
                     top_p=top_p, seed=seed, num_speculative_tokens=K)

    base_text, base = baseline_generate(target, tokenizer, prompt, cfg, device)
    spec_text, spec = speculative_generate(draft, target, tokenizer, prompt, cfg, device)

    return {
        "baseline_text": base_text,
        "spec_text": spec_text,
        "baseline_time": base.wall_time,
        "spec_time": spec.wall_time,
        "baseline_tps": base.tokens_per_sec,
        "spec_tps": spec.tokens_per_sec,
        "speedup": base.wall_time / spec.wall_time if spec.wall_time > 0 else float("nan"),
        "acceptance": spec.acceptance_rate,
        "accepted": spec.accepted,
        "proposed": spec.proposed,
        "new_tokens": spec.new_tokens,
        "greedy_match": base.token_ids == spec.token_ids if temperature == 0 else None,
    }


def run_terminal(args):
    r = compare(args.prompt, args.max_new_tokens, args.num_speculative_tokens,
                args.temperature, args.top_p, args.seed,
                args.draft, args.target, args.dtype, args.device)
    line = "-" * 72
    print(line)
    print(f"prompt: {args.prompt!r}")
    print(f"settings: {args.max_new_tokens} max new tokens, K={args.num_speculative_tokens}, "
          f"temperature={args.temperature}, top_p={args.top_p}, seed={args.seed}")
    print(line)
    print(f"BASELINE      ({args.target})")
    print(f"  text : {r['baseline_text']!r}")
    print(f"  time : {r['baseline_time']:.2f}s | {r['baseline_tps']:.1f} tok/s")
    print(line)
    print(f"SPECULATIVE   ({args.draft} -> {args.target})")
    print(f"  text : {r['spec_text']!r}")
    print(f"  time : {r['spec_time']:.2f}s | {r['spec_tps']:.1f} tok/s")
    if r["acceptance"] is not None:
        print(f"  accepted {r['accepted']}/{r['proposed']} proposals "
              f"({r['acceptance']:.1%})")
    print(line)
    print(f"speedup: {r['speedup']:.2f}x   "
          f"({'FASTER' if r['speedup'] >= 1 else 'SLOWER'} than baseline on this hardware)")
    if r["greedy_match"] is not None:
        print(f"greedy outputs identical: {r['greedy_match']}")


def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="SpecMatch demo", layout="wide")
    st.title("SpecMatch — baseline vs speculative decoding")
    st.caption("Same target model, same decoding settings. Speedup < 1 means speculation "
               "is slower on this hardware (expected on CPU: verification is not free).")

    with st.sidebar:
        st.header("Settings")
        target_name = st.text_input("target model", "gpt2")
        draft_name = st.text_input("draft model", "distilgpt2")
        K = st.slider("K (speculative tokens)", 1, 8, 4)
        max_new_tokens = st.slider("max new tokens", 8, 128, 48)
        temperature = st.select_slider("temperature", options=[0.0, 0.5, 0.8, 1.0], value=0.0)
        top_p = st.slider("top_p", 0.1, 1.0, 1.0) if temperature > 0 else 1.0
        seed = st.number_input("seed", value=1234, step=1)
        dtype_name = st.selectbox("dtype", ["auto", "float32", "bfloat16", "float16"])

    prompt = st.text_area("Prompt", "Explain what a neural network is in simple terms:", height=80)

    if st.button("Run comparison", type="primary"):
        with st.spinner("loading models + generating ..."):
            t0 = time.perf_counter()
            r = compare(prompt, max_new_tokens, K, temperature, top_p, int(seed),
                        draft_name, target_name, dtype_name, "auto")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"Baseline ({target_name})")
            st.write(r["baseline_text"])
            st.metric("time", f"{r['baseline_time']:.2f}s")
            st.metric("tokens/sec", f"{r['baseline_tps']:.1f}")
        with c2:
            st.subheader(f"Speculative ({draft_name} → {target_name}, K={K})")
            st.write(r["spec_text"])
            st.metric("time", f"{r['spec_time']:.2f}s")
            st.metric("tokens/sec", f"{r['spec_tps']:.1f}")
        m1, m2, m3 = st.columns(3)
        m1.metric("speedup", f"{r['speedup']:.2f}x")
        if r["acceptance"] is not None:
            m2.metric("acceptance rate", f"{r['acceptance']:.1%}",
                      help=f"{r['accepted']}/{r['proposed']} proposed tokens accepted")
        if r["greedy_match"] is not None:
            m3.metric("greedy outputs identical", str(r["greedy_match"]))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--terminal", action="store_true", help="CLI mode (no streamlit needed)")
    p.add_argument("--prompt", default="Explain what a neural network is in simple terms:")
    p.add_argument("--draft", default="distilgpt2")
    p.add_argument("--target", default="gpt2")
    p.add_argument("-K", "--num-speculative-tokens", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.terminal:
        run_terminal(args)
    else:
        try:
            import streamlit  # noqa: F401
        except ImportError:
            print("streamlit is not installed.\n"
                  "  pip install streamlit   &&   streamlit run demo/app.py\n"
                  "or use the dependency-free CLI mode:\n"
                  "  python demo/app.py --terminal")
            sys.exit(1)
        # `python demo/app.py` without --terminal: hand off to streamlit's runner
        import os
        os.execvp("streamlit", ["streamlit", "run", str(Path(__file__).resolve()),
                                "--server.headless", "true"])


if __name__ == "__main__":
    # When launched via `streamlit run demo/app.py`, streamlit has already
    # imported itself into this process - that's the reliable signal.
    if "streamlit" in sys.modules:
        run_streamlit()
    else:
        main()
