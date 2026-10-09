# SpecMatch

A from-scratch speculative decoding engine for causal LMs, built on PyTorch +
Hugging Face Transformers, with a reproducible benchmark suite that measures
**when speculative decoding actually helps — and when it doesn't**.

The core algorithm (draft → verify → accept/reject with residual resampling,
KV-cache management) is implemented explicitly in this repo — not delegated to
`transformers`' built-in assisted generation. Batch size 1, greedy and
temperature/top-p/top-k sampling.

```
Implemented speculative decoding from scratch for open-source causal LMs,
including draft/target verification, rejection sampling and KV-cache
management; built a reproducible benchmark suite analyzing latency,
throughput and acceptance-rate tradeoffs.
```

---

## Headline results (measured, not projected)

Dev machine: **2× Intel Xeon @ 2.5 GHz, 1 GiB RAM, no GPU**, torch 2.14.0+cpu,
transformers 5.17.0, bf16, target `gpt2` (124M), draft `distilgpt2` (82M),
12 prompts × 2–3 reps, 64 new tokens each:

| configuration | tok/s | p50 (64 tok) | acceptance | speedup vs baseline |
|---|---|---|---|---|
| baseline (gpt2, greedy) | **12.84 ± 0.52** | 4.90 s | — | 1.00× |
| speculative K=4, greedy | 8.21 ± 0.98 | 7.78 s | 0.606 | **0.64×** |
| speculative K=2, greedy | 9.95 ± 0.69 | 6.34 s | 0.747 | **0.77×** |
| baseline (T=0.8, top-p 0.95) | **11.36 ± 0.44** | 5.60 s | — | 1.00× |
| speculative K=4, sampling | 6.13 ± 0.87 | 10.30 s | 0.464 | **0.54×** |

**On this CPU, speculative decoding is slower than the baseline in every
measured configuration.** That is an honest, reproducible result — and the
project explains *why* quantitatively instead of hiding it:

- the acceptance statistics are exactly the ones that make GPU speedups work
  (0.61 greedy acceptance, 3.24 tokens committed per target forward);
- but on a compute-bound CPU, verification is **not free**: a target forward
  over m tokens costs ≈ `1 + 0.26·(m−1)` single-token steps (measured in
  `notebooks/05_results_analysis.ipynb`), and the draft (distilgpt2) costs
  0.65× a target step;
- with those measured costs, break-even needs acceptance α ≈ 0.97 — above
  what any real draft achieves. Under GPU-like cost ratios (verification
  memory-bound, ≈ free — labeled as assumptions, not measurements), the same
  measured α = 0.61 predicts ≈ 1.2–1.4× at K=4.

The algorithm is correct; the *regime* decides the payoff. Reproduce
everything on your own hardware (see [Reproducing](#reproducing-the-benchmarks)).

---

## Why speculative decoding

Autoregressive generation runs one forward pass per token. At batch size 1
each pass reads the entire weight matrix set from memory to emit a single
token's logits — on accelerators this is memory-bandwidth-bound, so the FLOPs
of processing 1 vs 5 tokens in one pass are nearly the same cost.

Speculative decoding (Leviathan et al. 2022; Chen et al. 2023) exploits that:

1. a small **draft model** cheaply proposes K continuation tokens;
2. the large **target model** scores all K positions in **one** forward pass
   (the causal mask + KV cache make row *j* of the logits exactly the
   distribution autoregressive decoding would have produced at step *j*);
3. tokens are accepted left-to-right with a rejection-sampling rule that
   provably preserves the target's output distribution;
4. the first rejection resamples a replacement from the residual
   distribution; all-K-accepted rounds get one free bonus token.

If the draft is cheap and agrees with the target often enough, one target
pass commits multiple tokens — the win is trading sequential target steps
for parallel verification.

## The acceptance rule (and why it is exact)

For proposed token `x ~ q` (draft) with target probability `p(x)`:

```
accept x with probability  min(1, p(x) / q(x))
on rejection, resample from  norm(max(0, p − q))    (the "residual")
```

Why the output distribution equals `p` exactly: token `x` is emitted either
by acceptance — probability `q(x)·min(1, p(x)/q(x)) = min(p(x), q(x))` — or
by rejection into the residual. The total rejection probability is
`1 − Σₓ min(p, q) = Σₓ max(0, p(x) − q(x))`, which is precisely the
residual's normalizer, so the rejection path contributes
`max(0, p(x) − q(x))`. The two paths sum to `p(x)` for every token.
Implementation: `src/specmatch/sampling.py` (fp32 math, zero-probability
guards, residual-mass underflow fallback), verified statistically in
`tests/test_sampling.py` (60k samples, TV distance < 0.01) and
`tests/test_generation.py` (χ² goodness-of-fit against the analytically
computed target distribution).

**Correctness contract** (what "lossless" means and what it doesn't):

- **Greedy decoding**: speculative output is token-identical to baseline for
  *any* draft model. Tested for K = 1, 2, 4, 8 with forced rejections.
  Verified in fp32 on the real pair (`python scripts/run_speculative.py
  --dtype float32 --compare`).
- **Sampling**: individual sequences do *not* match baseline run-for-run
  (different RNG consumption). The guarantee is distributional: the
  speculative sampler draws from exactly the target's shaped distribution —
  validated by χ² against the analytic first-token distribution.
- **bf16 caveat (measured)**: bf16 logits round differently between a
  single-token decode matmul and a batched verify matmul, so near-tied
  greedy argmax can flip and outputs can diverge after the flip. This is a
  floating-point artifact of the *weights dtype*, not of the algorithm;
  fp32 parity is exact. It also perturbs acceptance: on one fixed prompt
  (`"The meaning of life is"`, K=4, greedy) acceptance measured 0.426 in
  bf16 vs 0.550 in fp32 — bf16 rounding noise decorrelates the two models'
  argmax choices.

## KV cache handling

One cache per model (`transformers.DynamicCache`), with an explicit
invariant, maintained in `src/specmatch/speculative.py`:

```
at the top of every round (committed length L, pending = last committed,
not-yet-consumed token):
    target cache length == L − 1        (pending is fed with the verify pass)
    draft  cache length == L − lag      (lag = 1, or 2 after an all-accept round)

draft phase : sync forward over lag, then K−1 proposal forwards → q₁..q_K
verify phase: ONE target forward over [pending, x₁..x_K] → rows p₀..p_K
accept phase: sequential min(1,p/q); first rejection → residual resample
cache phase : both caches truncated to the committed prefix − pending;
              rejected proposals' KVs are simply dropped
```

Truncation uses `Cache.crop` with transformers v5 negative-length semantics
(v4-compatible fallback in `src/specmatch/cache.py`). Tests
(`tests/test_cache.py`) prove the key property: truncate → re-feed yields
bitwise-equivalent logits to a fresh cache over the same prefix, and
`tests/test_speculative.py` checks the end-state cache against a full fresh
forward.

## Architecture

```
src/specmatch/
    config.py        GenConfig/SpecConfig dataclasses, device+dtype auto-detection
    models.py        loading, local bf16 checkpoint resolution, tokenizer compat check
    cache.py         cache create/length/truncate + forward_with_cache (state transitions documented)
    sampling.py      temperature/top-k/top-p shaping, accept rule, residual, greedy fast path
    generate.py      GenStats + baseline autoregressive loop
    speculative.py   THE core algorithm: propose → verify → accept → cache bookkeeping
    benchmark.py     warmup/reps/timing harness → JSON records (loading excluded)
    metrics.py       percentiles, acceptance, speedup aggregation
scripts/
    download_models.py, prepare_models.py (bf16 conversion), prepare_corpus.py,
    train_draft.py (self-distilled mini draft), run_baseline.py,
    run_speculative.py, benchmark.py
tests/               48 tests: pure-math sampling, cache invariants, scripted
                     FakeModel accept/reject/EOS paths, tiny-GPT2 integration,
                     χ² distribution preservation
notebooks/           01 sanity → 05 results analysis (all analysis reads
                     benchmarks/results/*.json; figures in reports/figures/)
demo/app.py          streamlit UI + dependency-free --terminal mode
benchmarks/prompts/  fixed 12-prompt, 4-category workload
benchmarks/results/  committed measured JSON records
```

Design rule: small modules, plain functions, no framework-itis. The two
engines share every component except the loop itself, so comparisons are
apples-to-apples.

## Installation

```bash
git clone <your-fork> specmatch && cd specmatch
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# CPU-only machines: install torch from the CPU index first to avoid
# multi-GB CUDA wheels:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Tested with Python 3.11, torch ≥ 2.2 (2.14.0+cpu here), transformers ≥ 5.0
(5.17.0 here; v4 works via the cache-crop fallback but is untested).

### Device selection (documented behavior)

`resolve_device()` picks, in order: **CUDA** (NVIDIA) → **XPU** (Intel
Arc/Data Center GPU, needs an XPU-enabled torch build + drivers) → **MPS**
(Apple Silicon) → **CPU** (always works). Override with `--device cpu|cuda|xpu|mps`.
`resolve_dtype()` defaults to bf16 on CPU/CUDA(bf16-capable)/XPU, else fp32;
override with `--dtype float32|bfloat16|float16`. Sampling math always runs
in fp32 regardless of weight dtype.

### Models

Defaults: target `gpt2`, draft `distilgpt2` (same tokenizer; distilgpt2 was
distilled from gpt2, which is what makes acceptance interesting). Any pair
with an **identical tokenizer** works — the loader fails loudly otherwise.

On machines with ≤ 2 GiB RAM, prepare local bf16 copies first (halves mmap
commit and RSS):

```bash
python scripts/download_models.py     # gpt2, distilgpt2, tiny-gpt2 (tests)
python scripts/prepare_models.py      # .models/*-bf16 (auto-used when present)
```

The optional self-trained draft (`specmatch-mini-draft`, Experiment C) is
reproducible via `python scripts/prepare_corpus.py && python
scripts/train_draft.py --make-init && python scripts/train_draft.py --steps
850 && python scripts/train_draft.py --steps 850 --resume && python
scripts/train_draft.py --export` (~35 CPU-minutes per 850 steps on 2 cores).

## Usage

```bash
# baseline autoregressive generation
python scripts/run_baseline.py --prompt "Explain how compound interest works:" --max-new-tokens 48

# speculative decoding (+ side-by-side baseline and parity check)
python scripts/run_speculative.py --prompt "Explain how compound interest works:" -K 4 --compare

# sampling mode
python scripts/run_speculative.py -K 4 --temperature 0.8 --top-p 0.95 --seed 7

# full benchmark (writes JSON records; --mode baseline|speculative|both)
python scripts/benchmark.py --mode both -K 4 --reps 3 --out benchmarks/results/my_run.json
python scripts/benchmark.py --mode speculative -K 2 4 6 8 12 --reps 2 --out benchmarks/results/my_k_sweep.json
python scripts/benchmark.py --mode speculative --categories code qa --out benchmarks/results/subset.json

# tests
pytest

# demo
pip install streamlit && streamlit run demo/app.py    # web UI
python demo/app.py --terminal                         # no extra deps

# notebooks (in order)
jupyter notebook notebooks/
```

## Experimental methodology

Fairness rules enforced by `src/specmatch/benchmark.py`:

- identical target model, tokenizer, prompt set, `max_new_tokens`, decoding
  settings, device, and dtype for both engines;
- model loading happens **before** timing; warmup generation per mode;
- 2–3 repetitions per prompt with fixed seed schedule (`seed + 1000·rep`,
  identical across modes per rep);
- one record per (mode, prompt, rep) with: wall time, tokens/s, TTFT, p50/p95
  inputs, proposed/accepted counts, per-round accepted counts, target/draft
  forward counts and verified-token counts, peak RSS (POSIX) and accelerator
  peak memory (CUDA/XPU);
- fixed prompt set: `benchmarks/prompts/default.json` — 12 prompts across 4
  workload categories (qa / code / summarization / explanation).

## Results

All numbers below are measured on the dev machine (see headline table for
environment) and are regenerable with the exact commands in
[Reproducing](#reproducing-the-benchmarks). Analysis + figures:
`notebooks/02–05`, PNGs in `reports/figures/`.

**Experiment A — baseline vs speculative** (`expA_greedy.json`,
`expA_sampling.json`): greedy 0.64× at K=4 (acceptance 0.606, 3.24 tokens per
target forward); sampling 0.54× (acceptance 0.464). TTFT is worse under
speculation (0.99 s vs 0.38 s greedy p-mean): the first token waits for a
full draft+verify round. Peak RSS: 685 MB (baseline) / 875 MB (speculative)
— two resident models cost memory.

**Experiment B — draft length K** (`expB_K_sweep_*.json`, greedy):

| K | tok/s | acceptance | tokens/target-forward | p50 |
|---|---|---|---|---|
| 2 | 9.95 | 0.747 | 2.39 | 6.34 s |
| 4 | 8.17 | 0.606 | 3.24 | 7.69 s |
| 6 | 7.01 | 0.450 | 3.55 | 9.42 s |
| 8 | 6.85 | 0.424 | 4.13 | 9.46 s |
| 12 | 6.01 | 0.347 | 4.77 | 11.05 s |

Round yield grows sublinearly while per-round cost grows ~linearly on CPU →
throughput falls monotonically with K here; the best measured K is 2 (0.77×).
Token-level acceptance fits a geometric model with per-position agreement
α ≈ 0.80 (notebook 02).

**Experiment C — draft model** (`expC_draft_*.json` + K=4 arm of expB):

| draft | params | acceptance | tok/s | speedup |
|---|---|---|---|---|
| `sshleifer/tiny-gpt2` (random) | 0.10M | 0.000 | 4.56 | 0.36× |
| `specmatch-mini-draft` (trained here, 870K tokens) | 14.58M | 0.006 | 4.24 | 0.33× |
| `distilgpt2` (distilled from gpt2) | 81.91M | 0.606 | 8.17 | 0.64× |

Acceptance tracks **distillation quality, not size**: the 14.6M draft trained
on wikitext-2 in this repo sits at the random floor, and is even slower than
tiny-gpt2 (higher per-step cost, no acceptance to show for it).

**Experiment D — workload** (from `expA_greedy.json`, K=4): acceptance varies
strongly by category — summarization 0.736 (constrained continuations of a
given passage), qa 0.654, code 0.542, explanation 0.527 — and per-prompt
speedup tracks acceptance (best prompt `sum-story`: 0.86 acceptance, 0.81×).
On this hardware no category crosses 1.0×; on cheaper verification the
category spread would translate directly into a speedup spread.

**Why it's slower here — measured cost model** (notebook 05): single target
step 73 ms; verify(m) ≈ `1 + v·(m−1)` steps with v ≈ 0.26–0.48 (falling with
m as weight-reads amortize); draft step = 0.65 target steps. Predicted vs
measured throughput across the K sweep matches within first-order model
noise, and break-even α at K=4 computes to ≈ 0.97 — unreachable. The same
formula under GPU-like cost assumptions (clearly labeled) crosses break-even
at α ≈ 0.41–0.52, below the measured 0.61.

## Reproducing the benchmarks

```bash
python scripts/prepare_models.py            # optional but recommended on small RAM
python scripts/benchmark.py --mode both -K 4 --reps 3 --max-new-tokens 64 \
    --out benchmarks/results/expA_greedy.json
python scripts/benchmark.py --mode both -K 4 --reps 3 --max-new-tokens 64 \
    --temperature 0.8 --top-p 0.95 --seed 777 --out benchmarks/results/expA_sampling.json
python scripts/benchmark.py --mode speculative -K 2 4 6 --reps 2 --out benchmarks/results/expB_K_sweep_part1.json
python scripts/benchmark.py --mode speculative -K 8 12 --reps 2 --out benchmarks/results/expB_K_sweep_part2.json
python scripts/benchmark.py --mode speculative --draft sshleifer/tiny-gpt2 -K 4 --reps 2 --out benchmarks/results/expC_draft_tiny.json
jupyter nbconvert --to notebook --execute --inplace notebooks/05_results_analysis.ipynb
```

On a GPU/XPU machine the same commands work unchanged; expect `v_marg ≈ 0`
in notebook 05's live cost measurement and speedups crossing 1.0× at the
measured acceptance rates. Committed results in `benchmarks/results/` are
from the CPU dev machine — **replace them with your own runs** rather than
comparing across hardware.

## Limitations

- **Batch size 1 only.** Batched serving changes the economics (verification
  amortizes across requests; drafts contend for compute).
- **CPU-measured latency.** The dev machine (2 cores, 1 GiB) cannot show the
  GPU regime where this technique wins; the analysis quantifies exactly what
  would have to change (notebook 05).
- **bf16 greedy parity is approximate** (argmax flips under rounding); fp32
  is exact but 2.3× slower here. Documented, measured, tested.
- **No tree/multi-candidate drafts, no Medusa-style heads, no dynamic K** —
  single linear draft chain, fixed K per run (deliberate scope).
- **Self-trained mini-draft is data-starved** (870K tokens ≈ 35 CPU-minutes);
  it demonstrates the lower bound of Experiment C, not a competitive draft.
- **GPT-2 family only, in practice.** The engine is architecture-agnostic
  (any HF causal LM pair with identical tokenizer), but only GPT-2-tokenizer
  pairs were tested.
- **No CUDA graphs / torch.compile / static caches** — plain eager forwards;
  on GPU those would shift absolute numbers (both engines benefit).

## Future work

- GPU/XPU re-run of the full experiment matrix (commands above) to measure
  the regime flip predicted by the cost model.
- Longer distillation of the mini draft (or top-k logit KD from gpt2) to lift
  Experiment C's middle point into the 0.3–0.5 acceptance range.
- Dynamic/adaptive K per context (acceptance-rate feedback), tree drafts.
- Batched verification across requests.

## References

- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative
  Decoding.* ICML 2023.
- Chen et al. *Accelerating Large Language Model Decoding with Speculative
  Sampling.* arXiv:2302.01318.
- Sanh et al. *DistilBERT, a distilled version of BERT* (distilgpt2 lineage).

## License

MIT — see [LICENSE](LICENSE).
