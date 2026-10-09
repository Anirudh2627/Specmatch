"""KV-cache helpers.

The cache stores, per transformer layer, the key/value projections of every
token the model has consumed so far. A forward pass with a cache only computes
projections for the *new* input tokens and appends them - that is why decode
steps are O(1) in sequence length instead of O(n).

SpecMatch keeps one cache per model (draft and target) and relies on a simple
invariant, enforced in speculative.py:

    cache length == number of committed tokens the model has consumed

State transitions per speculative iteration (K proposals, prefix length L):

    draft:  L -> L+K          (K sequential proposal forwards)
    target: L -> L+K          (ONE verify forward over all K proposals)

    if proposals 1..i-1 are accepted and proposal i is rejected:
        both caches are truncated back to L+i-1 (the rejected token's KV and
        everything after it is dropped), the replacement token is appended to
        the sequence but NOT consumed by either cache, and the next iteration
        feeds it as its first input.
    if all K are accepted:
        both caches stay at L+K; the bonus token (sampled from the target's
        last verify logits) is likewise pending, unconsumed.
"""

import torch
import transformers
from transformers import DynamicCache

_TRANSFORMERS_MAJOR = int(transformers.__version__.split(".")[0])


def new_cache() -> DynamicCache:
    return DynamicCache()


def cache_len(cache) -> int:
    """Number of tokens currently consumed in the cache."""
    if cache is None:
        return 0
    return cache.get_seq_length()


def truncate_cache(cache, keep_len: int) -> None:
    """Drop cache entries beyond `keep_len` tokens (in place).

    transformers v5 renamed the semantics of Cache.crop: positive arguments
    (v4: 'keep this many') are deprecated in favor of negative ones ('drop
    this many'). Support both so the repo works on 4.x too.
    """
    cur = cache_len(cache)
    if keep_len >= cur:
        return
    if keep_len < 0:
        raise ValueError(f"keep_len must be >= 0, got {keep_len}")
    if _TRANSFORMERS_MAJOR >= 5:
        cache.crop(keep_len - cur)  # negative: remove (cur - keep_len) tokens
    else:
        cache.crop(keep_len)  # v4: absolute keep length


@torch.no_grad()
def forward_with_cache(model, input_ids: torch.Tensor, cache):
    """Run the model on new tokens, extending the cache in place.

    Returns logits of shape [1, n_new, vocab]. With a causal LM the row at
    index j is the distribution for the token *after* input_ids[j], given the
    cache contents plus input_ids[: j+1].
    """
    out = model(input_ids, past_key_values=cache, use_cache=True)
    return out.logits
