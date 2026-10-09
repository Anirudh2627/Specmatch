# Development notes

Engineering decisions, in the order they came up. Concise on purpose — the
code and README carry the detail.

## Why PyTorch + Transformers (and nothing else)

Models, tokenizer, and `DynamicCache` come from Transformers; everything
algorithmic (propose/verify/accept, cache state machine, sampling math,
benchmarking) is ours. Rationale: the interesting part of this project is the
algorithm and its measurement, while model loading/saving/weight handling is
solved, boring infrastructure. The only runtime deps are `torch` and
`transformers`; `pandas`/`matplotlib`/`jupyter` are analysis-only, `streamlit`
is demo-only (optional extra), `pyarrow` touches only draft training.

## Why not `model.generate(assistant_model=...)`

Transformers ships assisted generation, which implements the same paper. Using
it would have made this a config exercise. The point was to own every line of
the accept/reject logic, the cache bookkeeping, and the failure modes — and to
be able to answer "what happens to the cache after a rejection at position 3?"
from the code, not from docs. Our loop is also deliberately dumb in the right
places: one round = K sequential draft forwards + one verify forward + a
Python accept loop. At batch 1 with K ≤ 12, vectorizing the accept loop would
add complexity for microseconds.

## Cache design: truncate-and-replay instead of carrying distributions

The subtle part of speculative decoding is bookkeeping: after a round, the
target has consumed tokens the draft hasn't (the last accepted proposal, plus
the bonus/replacement token that nobody consumed). Two clean options:

1. carry the target's next-position distribution across rounds (fewer
   forwards, more state);
2. keep the invariant "caches hold the committed prefix minus pending" and
   let each model replay what it missed at the top of the next round.

We chose (2): one extra small draft forward per round, in exchange for a
single invariant that is trivial to state, test, and debug
(`target_len == L−1`, `draft_len == L−lag`, lag ∈ {1, 2}). Every rejection
boils down to `truncate_cache(cache, new_committed_len − 1)` — rejected
proposal KVs are simply dropped, and the replacement/bonus token is never in
any cache until the next round consumes it. `tests/test_cache.py` proves
truncate→refeed equals a fresh cache (the property that makes this safe), and
`tests/test_speculative.py` checks the end-state cache against a full fresh
forward.

Edge cases handled explicitly: K capped so a round can never overshoot
`max_new_tokens` (with a final target-only step for the last token); EOS in
proposals (stop proposing after it), EOS as accepted proposal, EOS as
residual replacement, EOS as bonus (all terminate; EOS excluded from output);
empty prompt (raise); prompt_len == 1 (no prefill, regular path handles it).

## Sampling math in fp32, decisions in one place

Weights run bf16 (memory), but `shaped_probs` upcasts logits to fp32 before
temperature/top-k/top-p and softmax, and the accept rule uses fp32
probabilities. Guards: `q(x)==0` with `p(x)>0` → accept (ratio = +∞);
residual mass underflow → fall back to sampling from p (still exact). Greedy
is a separate fast path (one-hot p, argmax compare) rather than temperature→0
through the same code, which keeps the parity claim easy to state. Baseline
and speculative call the *same* shaping function — that's what makes the
benchmark comparison about the algorithm, not about two subtly different
samplers.

`transformers` v5 changed `Cache.crop` semantics (negative = drop n tokens);
`cache.py` branches on the major version so v4 installs still work.

## bf16, 1 GiB RAM, and the mmap commit limit

The dev machine has a hard 1 GiB cgroup limit, no swap, no GPU.
fp32 gpt2+distilgpt2 (~830 MB of weights + ~380 MB torch/transformers
overhead) does not fit; worse, safetensors loading *mmaps the whole fp32 file*
(548 MB for gpt2), and committing that mmap on top of an already-resident
draft fails with `unable to mmap ... Cannot allocate memory` even before RSS
peaks. Fixes, in order of impact:

1. load the **target first** (commit the big mmap while RSS is small);
2. `scripts/prepare_models.py` pre-converts checkpoints to bf16 safetensors
   once (each conversion in its own subprocess) — the runtime then mmaps
   250 MB instead of 548 MB;
3. bf16 everywhere by default on CPU (also measured slightly *faster* than
   fp32 here), fp32 selectable via `--dtype float32` for exact-parity checks.

Measured consequence (documented in README): bf16 flips near-tied greedy
argmax between single-token and batched matmuls, so bf16 greedy parity is
approximate while fp32 parity is exact. We kept bf16 as the default because
the alternative (fp32) risks OOM on this machine and is 2.3× slower.

## The draft-model situation (Experiment C)

The smallest *trained* open model with the GPT-2 tokenizer is distilgpt2
(82M) — only 1.5× smaller than the gpt2 target, i.e. draft step cost ≈ 0.65
target steps. On CPU that ratio alone caps the possible speedup below 1×
even at perfect acceptance. So we trained a small draft ourselves
(`scripts/train_draft.py`): 2-layer/256-dim GPT-2 (~14.6M, tied embeddings),
initialized by SVD-projecting gpt2's 768-dim token embeddings to 256-d (rank-256
approximation of the teacher's token geometry; `torch.svd_lowrank`, reading
only the wte tensor from the bf16 file to fit RAM), then plain next-token
cross-entropy on wikitext-2. Training runs in ≤30-minute chunks with
checkpoint/resume (a real constraint: the sandbox kills longer calls), ~1.2
s/step, total ~35 min → 1700 steps ≈ 870K tokens.

Honest outcome: loss 6.6, greedy acceptance vs gpt2 ≈ 0.006 — the model is
data-starved, and a 14.6M LM does not approximate gpt2 after 0.4 epochs of
wikitext. We report it as the weak-draft data point (it even loses to
tiny-gpt2 on throughput: higher per-step cost, no acceptance gain) instead of
dressing it up. A proper KD run (teacher logits, more data, more hours) is
documented as future work. Corpus tokenization had to be chunked — one-shot
tokenization of 12 MB of text OOMs at 1 GiB (Rust-side buffers + a 2M-element
Python list).

## Benchmark methodology

Fairness first: identical cfg objects for both engines, same prompt order,
seed schedule `seed + 1000·rep` shared across modes per rep, warmup
generation per mode, model loading strictly outside timed regions, one JSON
record per (mode, prompt, rep) including full forward/token accounting so
notebooks can recompute everything without re-running models. Records are
committed to git; notebooks never hardcode numbers — they load the JSON and
recompute (including param counts, read from safetensors headers).

`peak_memory_mb` uses `resource.getrusage` and is POSIX-only; the import is
guarded so Windows users get `null` there instead of an ImportError (the
accelerator-peak field covers CUDA/XPU).

## What the numbers told us (no spin)

Every speculative configuration measured is slower than baseline on this CPU
(best: K=2 greedy, 0.77×). Rather than tune until something crossed 1.0, we
measured the cause: verify(m) ≈ 1 + 0.26·(m−1) single-target-steps and
c_draft = 0.65 — so break-even needs α ≈ 0.97 at K=4. The acceptance behavior
(0.61 greedy; 3.24 tokens committed per target forward; geometric α≈0.80
per-position fit) is healthy and is exactly what produces published GPU
speedups; the cost structure is what differs. Notebook 05's cost model
reproduces the measured K-sweep throughputs within first-order noise, which
is the strongest evidence we have that the engine has no hidden overhead
bug — the slowdown is the regime, not the code.

Also measured against intuition: sampling acceptance (0.464, = 1−TV(p,q)
under T=0.8/top-p) came out *below* greedy argmax agreement (0.606) for this
pair. Flatter distributions do not automatically mean more overlap; we report
both and explain the difference rather than picking the flattering one.

## Known limitations

Batch-1 only; eager forwards (no compile/CUDA-graphs/static cache); linear
draft chain (no trees); fixed K; GPT-2-family pairs tested; mini-draft
under-trained; `resource`-based peak memory absent on Windows; notebooks 01
and 05 run models live (need ~1 GiB free; the rest are JSON-only).
