"""Model/tokenizer loading and draft-target compatibility checks."""

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local directory with pre-converted (bf16) checkpoints, see scripts/prepare_models.py.
# Loading a 250 MB bf16 safetensors file instead of the 548 MB fp32 original
# halves both mmap virtual-memory commit and RAM - which matters on the 1 GiB
# dev machine, where committing the fp32 file on top of an already-loaded draft
# model fails with "unable to mmap ... Cannot allocate memory".
LOCAL_MODEL_DIR = Path(os.environ.get(
    "SPECMATCH_MODEL_DIR",
    Path(__file__).resolve().parents[2] / ".models",
))


def local_model_path(name: str):
    """Resolve a model name to a local directory if we have one:
    1. .models/<name>            (trained draft exports)
    2. .models/<name>-bf16       (converted checkpoints, see prepare_models.py)
    Returns None when the name should come from the HF hub."""
    if Path(name).is_dir():  # already a path
        return name
    safe = name.replace("/", "--")
    plain = LOCAL_MODEL_DIR / safe
    if plain.is_dir():
        return str(plain)
    bf16 = LOCAL_MODEL_DIR / (safe + "-bf16")
    if bf16.is_dir():
        return str(bf16)
    return None


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(local_model_path(name) or name)
    # GPT-2 family ships without a pad token; generation code here is
    # batch-size-1 so padding never happens, but transformers warns without this.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name: str, dtype: torch.dtype, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path(name) or name, dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    # Keep thread count explicit on CPU: the default can oversubscribe on
    # small containers and *hurt* latency for batch-1 decode.
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads() or 1))
    return model


def check_tokenizer_compatibility(draft_tok, target_tok, probe: str = "The quick brown fox jumps over") -> None:
    """Speculative decoding compares token ids across the two models, so both
    must share the exact same vocabulary and byte-level encoding. Fail loudly
    at load time instead of producing garbage acceptance rates later."""
    if draft_tok.vocab_size != target_tok.vocab_size:
        raise ValueError(
            f"vocab size mismatch: draft={draft_tok.vocab_size} target={target_tok.vocab_size}"
        )
    d = draft_tok(probe)["input_ids"]
    t = target_tok(probe)["input_ids"]
    if d != t:
        raise ValueError("draft and target tokenizers disagree on a probe string; "
                         "speculative decoding requires an identical tokenizer")


def load_pair(draft_name: str, target_name: str, dtype: torch.dtype, device: torch.device):
    """Load draft+target and return (tokenizer, draft_model, target_model).

    The target (larger) model is loaded first: safetensors loading mmaps the
    whole checkpoint, so committing the big file while RSS is still small
    avoids ENOMEM under tight cgroup limits.
    """
    target = load_model(target_name, dtype, device)
    draft = load_model(draft_name, dtype, device)
    target_tok = load_tokenizer(target_name)
    draft_tok = load_tokenizer(draft_name)
    check_tokenizer_compatibility(draft_tok, target_tok)
    return target_tok, draft, target
