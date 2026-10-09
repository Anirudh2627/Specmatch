"""Train the small SpecMatch draft model (2-layer, 256-dim GPT-2, ~14M params).

Why we train our own draft: the smallest *trained* open GPT-2-tokenizer model
is distilgpt2 (82M) - only 1.5x smaller than the gpt2 target, which on a CPU
(where verification is not free) cannot produce a speedup. A ~14M draft costs
~10x less per token than gpt2, which is the regime where speculation can pay
off even on CPU. The draft shares gpt2's tokenizer (required) and its
embedding/lm_head is initialized by SVD-projecting gpt2's 768-dim token
embeddings to 256 dims, so the student starts with the teacher's token
geometry instead of random directions.

Training is plain next-token cross-entropy on wikitext-2 (hard-label
distillation; the teacher's samples/styles are in the corpus indirectly via
shared tokenizer and embeddings). Runs on 2 CPU cores inside ~700 MB RSS.

    python scripts/prepare_corpus.py
    python scripts/train_draft.py --steps 200 --debug-steps   # smoke test
    python scripts/train_draft.py --steps 900 --resume         # chunked runs
    python scripts/train_draft.py --export                     # -> .models/specmatch-mini-draft
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

MODELS_DIR = REPO / ".models"
CKPT = MODELS_DIR / "mini_draft_ckpt.pt"
WTE_PROJ = MODELS_DIR / "wte_proj.pt"
TOKENS = MODELS_DIR / "wikitext2_tokens.bin"
EXPORT_DIR = MODELS_DIR / "specmatch-mini-draft"

DRAFT_CONFIG = dict(
    vocab_size=50257,
    n_positions=512,
    n_layer=2,
    n_embd=256,
    n_head=4,
    activation_function="gelu_new",
    tie_word_embeddings=True,
)


def make_wte_projection(teacher: str, out_path: Path):
    """Project the teacher's token embeddings [V, 768] to [V, 256] via
    truncated SVD. Reads only the wte tensor from the bf16 safetensors file
    (77 MB) instead of materializing the whole teacher - this must fit in the
    same 1 GiB budget as everything else."""
    from safetensors.torch import load_file
    from specmatch.models import local_model_path

    src = local_model_path(teacher) or teacher
    print(f"reading wte from {src} ...")
    tensors = load_file(str(Path(src) / "model.safetensors"))
    wte = tensors["transformer.wte.weight"].float()  # [V, 768] fp32, 154 MB
    del tensors
    print(f"teacher wte {tuple(wte.shape)}, running truncated SVD ...")
    t0 = time.time()
    U, S, V = torch.svd_lowrank(wte, q=DRAFT_CONFIG["n_embd"], niter=4)
    proj = (U * S).contiguous()  # best rank-256 approximation
    del wte, U, S, V
    torch.save(proj, out_path)
    print(f"  saved {tuple(proj.shape)} projection -> {out_path} in {time.time() - t0:.1f}s")


def build_model(wte_proj: Path | None):
    torch.manual_seed(0)
    cfg = GPT2Config(**DRAFT_CONFIG)
    model = GPT2LMHeadModel(cfg)
    if wte_proj is not None and wte_proj.exists():
        # install the teacher's projected token geometry as the student's
        # (tied) embedding / lm_head matrix - the student starts knowing what
        # direction each token points in, instead of random noise
        proj = torch.load(wte_proj)
        with torch.no_grad():
            model.get_input_embeddings().weight.copy_(proj)
        del proj
        print(f"initialized wte/lm_head from {wte_proj}")
    return model


def load_tokens() -> np.ndarray:
    if not TOKENS.exists():
        raise SystemExit(f"{TOKENS} missing - run scripts/prepare_corpus.py first")
    return np.fromfile(TOKENS, dtype=np.uint16)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init-from-teacher", default="gpt2")
    ap.add_argument("--make-init", action="store_true",
                    help="compute + save the teacher wte projection, then exit")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--export", action="store_true", help="export HF-format model dir and exit")
    args = ap.parse_args()

    torch.set_num_threads(2)

    if args.make_init:
        make_wte_projection(args.init_from_teacher, WTE_PROJ)
        return

    if args.export:
        ckpt = torch.load(CKPT, weights_only=False)
        model = build_model(None)
        model.load_state_dict(ckpt["model"])
        model.save_pretrained(EXPORT_DIR)
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("gpt2").save_pretrained(EXPORT_DIR)
        print(f"exported step-{ckpt['step']} model -> {EXPORT_DIR}")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    resumed_ckpt = None
    if args.resume and CKPT.exists():
        resumed_ckpt = torch.load(CKPT, weights_only=False)
        model = build_model(None)
        model.load_state_dict(resumed_ckpt["model"])
        start_step = resumed_ckpt["step"]
        print(f"resumed from step {start_step} (loss {resumed_ckpt['loss']:.3f})")
    else:
        model = build_model(WTE_PROJ)
        start_step = 0

    data = load_tokens()
    n_tokens = len(data)
    print(f"corpus: {n_tokens} tokens")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if start_step > 0 and resumed_ckpt.get("opt"):
        opt.load_state_dict(resumed_ckpt["opt"])
    del resumed_ckpt  # a second checkpoint copy in RAM is what OOMs us here

    # seed advances with resume point so chunk 2 doesn't replay chunk 1's batches
    rng = np.random.default_rng(42 + start_step)
    seq = args.seq_len
    total = args.steps + start_step
    t0 = time.time()
    running = []
    for step in range(start_step, total):
        # random packed blocks; input = block[:-1], target = block[1:]
        starts = rng.integers(0, n_tokens - seq - 1, size=args.batch_size)
        batch = np.stack([data[s: s + seq + 1] for s in starts]).astype(np.int64)
        x = torch.from_numpy(batch[:, :-1])
        y = torch.from_numpy(batch[:, 1:])
        loss = model(input_ids=x, labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        running.append(loss.item())
        if (step + 1) % args.log_every == 0:
            dt = time.time() - t0
            sps = args.log_every * args.batch_size * seq / dt
            print(f"step {step + 1}/{total} loss {np.mean(running):.4f} "
                  f"({sps:.0f} tok/s, {dt:.0f}s elapsed)", flush=True)
            running, t0 = [], time.time()
        if (step + 1) % args.save_every == 0 or step + 1 == total:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1, "loss": float(np.mean(running) if running else loss.item())},
                       CKPT)
            print(f"  checkpoint saved at step {step + 1}", flush=True)


if __name__ == "__main__":
    main()
