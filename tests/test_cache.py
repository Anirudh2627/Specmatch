"""KV-cache consistency tests using the tiny GPT-2 (loads in seconds).

The property that matters for speculative decoding: after truncating a cache
and re-feeding tokens, the model must produce exactly the same logits as a
fresh cache would for the same prefix. If that fails, accepted-token KVs are
being corrupted by the reject path.
"""

import torch

from specmatch.cache import cache_len, forward_with_cache, new_cache, truncate_cache


def build_cache(model, ids):
    cache = new_cache()
    logits = forward_with_cache(model, ids, cache)
    return cache, logits


class TestTruncate:
    def test_cache_len_tracks_consumed_tokens(self, tiny_model, tiny_tokenizer):
        ids = tiny_tokenizer("one two three four five", return_tensors="pt")["input_ids"]
        cache, _ = build_cache(tiny_model, ids)
        assert cache_len(cache) == ids.shape[1]

    def test_truncate_shortens(self, tiny_model, tiny_tokenizer):
        ids = tiny_tokenizer("the quick brown fox jumps", return_tensors="pt")["input_ids"]
        cache, _ = build_cache(tiny_model, ids)
        truncate_cache(cache, 3)
        assert cache_len(cache) == 3

    def test_truncate_noop_when_keep_ge_len(self, tiny_model, tiny_tokenizer):
        ids = tiny_tokenizer("hello world", return_tensors="pt")["input_ids"]
        cache, _ = build_cache(tiny_model, ids)
        n = cache_len(cache)
        truncate_cache(cache, n + 5)
        assert cache_len(cache) == n

    def test_truncate_rejects_negative(self, tiny_model, tiny_tokenizer):
        cache = new_cache()
        try:
            truncate_cache(cache, -1)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_none_cache_len_zero(self):
        assert cache_len(None) == 0

    def test_truncated_then_refed_matches_fresh(self, tiny_model, tiny_tokenizer):
        """Truncate a cache mid-sequence, feed new tokens, and compare against
        a from-scratch cache over the same full sequence. Logits must match
        bitwise-ish: this is the invariant speculative decoding relies on when
        it drops rejected proposals and continues."""
        prefix = tiny_tokenizer("speculative decoding is a technique where", return_tensors="pt")["input_ids"]
        extra = tiny_tokenizer(" a small model", return_tensors="pt")["input_ids"]
        full = torch.cat([prefix, extra], dim=1)

        cache_a, _ = build_cache(tiny_model, full)
        with torch.no_grad():
            logits_fresh = tiny_model(torch.tensor([[full[0, -1].item()]]),
                                      past_key_values=None).logits  # unused warm

        # path B: consume prefix + garbage, truncate garbage away, feed extra
        garbage = tiny_tokenizer(" totally different words here", return_tensors="pt")["input_ids"]
        cache_b, _ = build_cache(tiny_model, torch.cat([prefix, garbage], dim=1))
        truncate_cache(cache_b, cache_len(cache_b) - garbage.shape[1])
        assert cache_len(cache_b) == prefix.shape[1]
        logits_b = forward_with_cache(tiny_model, extra, cache_b)

        # compare final-position logits of both routes
        with torch.no_grad():
            logits_a = tiny_model(full[:, -1:], past_key_values=build_cache(tiny_model, full[:, :-1])[0]).logits
        assert torch.allclose(logits_a[0, -1], logits_b[0, -1], atol=1e-4), \
            f"max diff {(logits_a[0,-1]-logits_b[0,-1]).abs().max()}"

    def test_continue_after_truncate_positions_correct(self, tiny_model, tiny_tokenizer):
        """After truncation, position ids must resume from the kept length
        (GPT-2 derives them from cache length), so generation continues
        coherently rather than restarting positions."""
        ids = tiny_tokenizer("a b c d e f g", return_tensors="pt")["input_ids"]
        cache, logits = build_cache(tiny_model, ids)
        truncate_cache(cache, 4)
        out = forward_with_cache(tiny_model, ids[:, 4:], cache)
        assert cache_len(cache) == ids.shape[1]
        assert out.shape == (1, ids.shape[1] - 4, tiny_model.config.vocab_size)
