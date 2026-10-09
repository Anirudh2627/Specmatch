"""SpecMatch: a from-scratch speculative decoding engine.

Core algorithm lives in `speculative.py` (draft/verify/accept loop),
`sampling.py` (acceptance rule + residual distribution) and `cache.py`
(KV-cache state transitions). `generate.py` holds the baseline
autoregressive loop used for comparison.
"""

from .config import GenConfig, SpecConfig, resolve_device, resolve_dtype
from .generate import GenStats, baseline_generate
from .speculative import speculative_generate

__version__ = "0.1.0"

__all__ = [
    "GenConfig",
    "SpecConfig",
    "GenStats",
    "baseline_generate",
    "speculative_generate",
    "resolve_device",
    "resolve_dtype",
]
