"""Shared fixtures + a scripted fake causal LM.

FakeModel makes the accept/reject logic testable deterministically: it returns
one-hot distributions chosen from a "next token table", so a test can dictate
exactly what the draft proposes and what the target wants at every position.
It maintains a real DynamicCache (fed with zero keys/values) so all cache
length bookkeeping in speculative.py is exercised for real.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="session")
def tiny_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TINY_MODEL)


@pytest.fixture(scope="session")
def tiny_model(tiny_tokenizer):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    model.eval()
    return model


class FakeModel(torch.nn.Module):
    """Causal LM stand-in: next_token_table[n] is the token this model wants
    at position n (i.e. after consuming n tokens). One-hot logits make every
    downstream decision deterministic."""

    def __init__(self, next_token_table, vocab_size=50257, device="cpu"):
        super().__init__()
        self.table = list(next_token_table)
        self.vocab_size = vocab_size
        self.device = torch.device(device)
        # dummy parameter so next(model.parameters()).device works
        self._dummy = torch.nn.Parameter(torch.zeros(1))
        self.forward_log = []  # (cache_len_before, n_input_tokens)

    def _logits_for_positions(self, first_abs_pos, n):
        rows = []
        for pos in range(first_abs_pos, first_abs_pos + n):
            # a model that consumed pos+1 tokens predicts table[pos+1]
            tok = self.table[min(pos + 1, len(self.table) - 1)]
            row = torch.full((self.vocab_size,), -10.0)
            row[tok] = 10.0
            rows.append(row)
        return torch.stack(rows).unsqueeze(0).to(self.device)

    def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
        n = input_ids.shape[1]
        cache = past_key_values
        before = cache.get_seq_length()
        self.forward_log.append((before, n))
        k = torch.zeros(1, 1, n, 1)
        cache.update(k, k.clone(), 0)
        logits = self._logits_for_positions(before, n)
        return SimpleNamespace(logits=logits, past_key_values=cache)


@pytest.fixture
def fake_tokenizer(tiny_tokenizer):
    return tiny_tokenizer


class BiasedModel(torch.nn.Module):
    """Wraps a causal LM and adds a constant bias to chosen logit positions.

    The tiny test model is degenerate (n_embd=2): weight perturbations barely
    move its output distribution, so to test rejection paths we instead build a
    'draft' that deliberately over-proposes one token. The target then rejects
    it with probability 1 - p(x)/q(x) ~ 1, deterministically exercising the
    residual-resample branch. Cache handling is inherited from the wrapped
    model (same cache object flows through).
    """

    def __init__(self, model, bias_token_ids, bias: float = 8.0):
        super().__init__()
        self.model = model
        self.bias_token_ids = list(bias_token_ids)
        self.bias = bias

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        logits = out.logits.clone()
        logits[..., self.bias_token_ids] += self.bias
        return SimpleNamespace(logits=logits, past_key_values=out.past_key_values)
