"""Generation stats + the baseline autoregressive loop.

The baseline is deliberately written with the same building blocks the
speculative decoder uses (cache.py forward helpers, sampling.shaped_probs),
so a comparison between the two is an apples-to-apples comparison of the
*algorithm*, not of surrounding scaffolding.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from .cache import forward_with_cache, new_cache
from .config import GenConfig
from .sampling import greedy_token, sample_token, shaped_probs


@dataclass
class GenStats:
    prompt_tokens: int = 0
    new_tokens: int = 0
    wall_time: float = 0.0
    ttft: Optional[float] = None  # seconds to first committed token
    # forward-pass accounting: the speculative loop trades many small draft
    # forwards for few large target forwards; these counters make that visible.
    target_forwards: int = 0
    draft_forwards: int = 0
    target_forward_tokens: int = 0  # tokens fed to the target in total
    proposed: int = 0
    accepted: int = 0
    stop_reason: str = "max_new_tokens"  # or "eos"
    token_ids: list = field(default_factory=list)
    accepted_per_iter: list = field(default_factory=list)

    @property
    def tokens_per_sec(self) -> float:
        return self.new_tokens / self.wall_time if self.wall_time > 0 else 0.0

    @property
    def acceptance_rate(self) -> Optional[float]:
        return self.accepted / self.proposed if self.proposed else None

    @property
    def mean_tokens_per_target_forward(self) -> float:
        # >1 means each expensive target pass yielded multiple tokens
        return self.new_tokens / self.target_forwards if self.target_forwards else 0.0


def make_generator(seed: Optional[int], device: torch.device) -> torch.Generator:
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    return gen


def pick_token(probs: torch.Tensor, cfg: GenConfig, generator: torch.Generator) -> int:
    return greedy_token(probs) if cfg.greedy else sample_token(probs, generator)


def _shape(logits_row: torch.Tensor, cfg: GenConfig) -> torch.Tensor:
    return shaped_probs(logits_row, cfg.temperature, cfg.top_k, cfg.top_p)


@torch.no_grad()
def baseline_generate(model, tokenizer, prompt: str, cfg: GenConfig,
                      device: Optional[torch.device] = None,
                      generator: Optional[torch.Generator] = None):
    """Plain autoregressive generation: one forward pass per token.

    Returns (text, GenStats). Stops at max_new_tokens or EOS (EOS itself is
    not included in the output, matching what the speculative loop does).
    """
    device = device or next(model.parameters()).device
    if generator is None:
        generator = make_generator(cfg.seed, device)

    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prompt_len = ids.shape[1]
    if prompt_len == 0:
        raise ValueError("prompt produced no tokens")
    eos = tokenizer.eos_token_id

    cache = new_cache()
    generated = []
    t_start = time.perf_counter()
    ttft = None

    logits = forward_with_cache(model, ids, cache)
    forwards, forward_tokens = 1, prompt_len
    next_tok = pick_token(_shape(logits[0, -1], cfg), cfg, generator)

    while next_tok != eos and len(generated) < cfg.max_new_tokens:
        generated.append(next_tok)
        if ttft is None:
            ttft = time.perf_counter() - t_start
        if len(generated) == cfg.max_new_tokens:
            break
        logits = forward_with_cache(model, torch.tensor([[next_tok]], device=device), cache)
        forwards += 1
        forward_tokens += 1
        next_tok = pick_token(_shape(logits[0, -1], cfg), cfg, generator)

    wall = time.perf_counter() - t_start
    stats = GenStats(
        prompt_tokens=prompt_len,
        new_tokens=len(generated),
        wall_time=wall,
        ttft=ttft,
        target_forwards=forwards,
        target_forward_tokens=forward_tokens,
        stop_reason="eos" if next_tok == eos else "max_new_tokens",
        token_ids=generated,
    )
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, stats
