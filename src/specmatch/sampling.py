"""Sampling utilities: distribution shaping and the speculative accept rule.

All math runs in float32 (logits arriving in bf16/fp16 are upcast first) so
that acceptance probabilities are not distorted by low-precision division.

The speculative sampling rule (Leviathan et al. 2022, "Fast Inference from
Transformers via Speculative Decoding"; also Chen et al. 2023):

    draft q samples a proposal x; target p evaluates it.
    accept x with probability  min(1, p(x) / q(x))
    on rejection, resample from the *residual* distribution
        norm(max(0, p - q))

Why this preserves p exactly: a token x is produced either by acceptance
(prob q(x) * min(1, p(x)/q(x)) = min(q(x), p(x))) or by rejection into the
residual (prob P(reject) * r(x), where P(reject) = 1 - sum_x min(p,q)
= sum_x max(0, p(x)-q(x)) exactly cancels the residual's normalizer). The two
paths sum to p(x) for every x, so the output distribution equals the target's.
"""

import torch


@torch.no_grad()
def shaped_probs(logits: torch.Tensor, temperature: float, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """logits [vocab] (any dtype) -> float32 probabilities after temperature /
    top-k / nucleus filtering. temperature == 0 means greedy: one-hot at argmax.

    Both baseline and speculative decoding call this same function, which is
    what makes their distributions comparable.
    """
    logits = logits.float()
    if temperature == 0.0:
        probs = torch.zeros_like(logits)
        probs[logits.argmax()] = 1.0
        return probs

    logits = logits / temperature

    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # keep the smallest prefix whose mass >= top_p (shift right so the
        # token that crosses the threshold is kept, not dropped)
        drop = cum > top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)

    return torch.softmax(logits, dim=-1)


def sample_token(probs: torch.Tensor, generator: torch.Generator) -> int:
    """Categorical draw from a 1-D probability vector."""
    return int(torch.multinomial(probs, 1, generator=generator).item())


def greedy_token(probs: torch.Tensor) -> int:
    return int(probs.argmax().item())


@torch.no_grad()
def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """norm(max(0, p - q)) - the distribution to resample from after a
    rejection. Falls back to p itself when the residual mass underflows to
    zero, which can only happen when p and q are numerically identical; in
    that case sampling from p is still exactly correct."""
    resid = torch.clamp(p - q, min=0.0)
    total = resid.sum()
    if total.item() < 1e-10:
        return p
    return resid / total


def accept_proposal(p: torch.Tensor, q: torch.Tensor, x: int, generator: torch.Generator) -> tuple[bool, int]:
    """Run one accept/reject trial for proposed token x.

    Returns (accepted, token): on acceptance token == x; on rejection token is
    a fresh draw from the residual distribution.

    Edge cases: q(x) == 0 cannot happen in practice (x was drawn from q), but
    floating point can round it to zero - if p(x) > 0 the ratio is +inf and we
    accept, matching min(1, p/q) semantics.
    """
    px = p[x].item()
    qx = q[x].item()
    ratio = px / qx if qx > 0 else (float("inf") if px > 0 else 0.0)
    threshold = min(1.0, ratio)
    u = torch.rand((), generator=generator).item()
    if u < threshold:
        return True, x
    return False, sample_token(residual_distribution(p, q), generator)


@torch.no_grad()
def greedy_accept(p: torch.Tensor, x: int) -> tuple[bool, int]:
    """Deterministic special case of accept_proposal for greedy decoding
    (p is one-hot): keep x iff it is the target's argmax, else take the
    argmax. This makes greedy speculative output *identical* to greedy
    baseline output, token for token."""
    best = int(p.argmax().item())
    if x == best:
        return True, x
    return False, best
