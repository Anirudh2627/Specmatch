"""Configuration for SpecMatch: model names, generation settings, device/dtype.

Deliberately simple dataclasses - no config framework. CLI scripts build these
from argparse and pass them down.
"""

from dataclasses import dataclass, asdict
from typing import Optional

import torch

# Defaults chosen for a machine with ~1 GiB RAM and no GPU: gpt2 (124M) is the
# target, distilgpt2 (82M) the draft. Both share the GPT-2 BPE tokenizer, and
# distilgpt2 was distilled *from* gpt2, so the two distributions are genuinely
# close - which is what makes acceptance rates interesting to measure.
DEFAULT_TARGET_MODEL = "gpt2"
DEFAULT_DRAFT_MODEL = "distilgpt2"


@dataclass
class GenConfig:
    """Shared decoding settings. Baseline and speculative runs must use the
    same GenConfig, otherwise benchmark comparisons are meaningless."""

    max_new_tokens: int = 64
    # temperature == 0 means greedy decoding (deterministic argmax).
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0  # 0 = disabled
    seed: Optional[int] = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SpecConfig(GenConfig):
    """GenConfig plus speculative-decoding knobs."""

    num_speculative_tokens: int = 4  # K
    draft_model: str = DEFAULT_DRAFT_MODEL
    target_model: str = DEFAULT_TARGET_MODEL
    dtype: str = "auto"  # "auto" | "float32" | "bfloat16" | "float16"

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick the compute device. Precedence: CUDA (NVIDIA) > XPU (Intel Arc /
    Data Center GPU, needs an XPU-enabled torch build + drivers) > MPS (Apple
    Silicon) > CPU. CPU always works; everything else is opportunistic."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Pick a weight dtype.

    On CPU we default to bfloat16: on the 1 GiB dev machine fp32 gpt2+distilgpt2
    sits at ~860 MB RSS (OOM territory), while bf16 leaves comfortable headroom
    and measured *slightly faster* decode. All sampling math upcasts to fp32
    regardless (see sampling.py), so bf16 weights do not corrupt the accept rule.

    Caveat measured on the dev machine: bf16 rounds logits, and the batched
    verify matmul rounds differently than a single-token decode matmul, so
    near-tied greedy argmax can flip (greedy parity holds exactly in fp32).
    """
    if name != "auto":
        return getattr(torch, name)
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type in ("cpu", "xpu"):
        return torch.bfloat16
    return torch.float32
