"""Baseline generation tests + the distribution-preservation test for
speculative decoding under stochastic sampling.

Correctness contract (see README):
- greedy decoding: speculative output must be token-identical to baseline.
- temperature/top-p sampling: individual sequences will NOT match (different
  numbers of RNG draws), but the *distribution* over generated tokens must
  equal the target model's. We verify this statistically: sample the first
  generated token many times from both engines and chi-square-compare against
  the analytically computed target distribution.
"""

import copy

import pytest
import torch

from specmatch.config import GenConfig, SpecConfig
from specmatch.generate import baseline_generate
from specmatch.sampling import shaped_probs
from specmatch.speculative import speculative_generate

from conftest import BiasedModel, FakeModel

EOS = 50256


class TestBaseline:
    def test_respects_max_new_tokens(self, tiny_model, tiny_tokenizer):
        cfg = GenConfig(max_new_tokens=7, temperature=0.0)
        _, stats = baseline_generate(tiny_model, tiny_tokenizer, "Tell me a story", cfg)
        assert stats.new_tokens == 7
        assert len(stats.token_ids) == 7
        assert stats.stop_reason == "max_new_tokens"
        # one prefill + (n-1) continuation forwards; tokens fed = prompt + n - 1
        assert stats.target_forwards == 7
        assert stats.target_forward_tokens == stats.prompt_tokens + 6

    def test_greedy_is_deterministic(self, tiny_model, tiny_tokenizer):
        cfg = GenConfig(max_new_tokens=6, temperature=0.0)
        _, s1 = baseline_generate(tiny_model, tiny_tokenizer, "The capital of France", cfg)
        _, s2 = baseline_generate(tiny_model, tiny_tokenizer, "The capital of France", cfg)
        assert s1.token_ids == s2.token_ids

    def test_seeded_sampling_is_reproducible(self, tiny_model, tiny_tokenizer):
        cfg = GenConfig(max_new_tokens=6, temperature=1.0, seed=1234)
        _, s1 = baseline_generate(tiny_model, tiny_tokenizer, "Randomness", cfg)
        _, s2 = baseline_generate(tiny_model, tiny_tokenizer, "Randomness", cfg)
        assert s1.token_ids == s2.token_ids
        cfg2 = GenConfig(max_new_tokens=6, temperature=1.0, seed=99)
        _, s3 = baseline_generate(tiny_model, tiny_tokenizer, "Randomness", cfg2)
        assert s1.token_ids != s3.token_ids  # different seed -> (almost surely) different tokens

    def test_eos_stops_generation(self, tiny_tokenizer):
        prompt = "stop please"
        prompt_len = len(tiny_tokenizer(prompt)["input_ids"])
        model = FakeModel([0] * prompt_len + [64, 65, EOS, 66, 66])
        cfg = GenConfig(max_new_tokens=10, temperature=0.0)
        _, stats = baseline_generate(model, tiny_tokenizer, prompt, cfg)
        assert stats.token_ids == [64, 65]
        assert stats.stop_reason == "eos"

    def test_empty_prompt_raises(self, tiny_model, tiny_tokenizer):
        with pytest.raises(ValueError):
            baseline_generate(tiny_model, tiny_tokenizer, "", GenConfig(max_new_tokens=4))

    def test_timing_fields(self, tiny_model, tiny_tokenizer):
        cfg = GenConfig(max_new_tokens=5, temperature=0.0)
        _, stats = baseline_generate(tiny_model, tiny_tokenizer, "timing", cfg)
        assert stats.wall_time > 0
        assert stats.tokens_per_sec > 0
        assert stats.ttft is not None and stats.ttft <= stats.wall_time


def _chi2_p_gt(stat: float, df: int) -> bool:
    """Threshold chi-square statistic at p-value 1e-3 (no scipy needed).
    Critical values for df=1..10 at p=1e-3."""
    crit = {1: 10.83, 2: 13.82, 3: 16.27, 4: 18.47, 5: 20.52,
            6: 22.46, 7: 24.32, 8: 26.12, 9: 27.86, 10: 29.59}
    return stat < crit[df]


class TestDistributionPreservation:
    """Speculative sampling must preserve the target distribution exactly.
    With finite samples we can only test statistically - hence chi-square
    against the analytically computed first-token distribution."""

    N_SAMPLES = 900
    TOP_K = 5
    PROMPT = "The weather today is"

    def _exact_first_token_dist(self, model, tokenizer, cfg):
        ids = tokenizer(self.PROMPT, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
        return shaped_probs(logits, cfg.temperature, cfg.top_k, cfg.top_p)

    def _sample_first_tokens(self, fn, cfg, n):
        counts = {}
        for i in range(n):
            cfg_i = copy.copy(cfg)
            cfg_i.seed = 100000 + i  # fresh stream per draw
            _, stats = fn(cfg_i)
            tok = stats.token_ids[0]
            counts[tok] = counts.get(tok, 0) + 1
        return counts

    def test_first_token_distribution_matches_target(self, tiny_model, tiny_tokenizer):
        cfg = GenConfig(max_new_tokens=2, temperature=1.0, top_k=self.TOP_K)
        p_exact = self._exact_first_token_dist(tiny_model, tiny_tokenizer, cfg)
        support = torch.nonzero(p_exact > 0).flatten().tolist()
        assert len(support) == self.TOP_K

        base_fn = lambda c: baseline_generate(tiny_model, tiny_tokenizer, self.PROMPT, c)
        # biased draft: deliberately over-proposes one token so rejections
        # actually happen; the output distribution must still equal the target's
        draft = BiasedModel(tiny_model, [464], bias=20.0)
        spec_fn = lambda c: speculative_generate(
            draft, tiny_model, tiny_tokenizer, self.PROMPT,
            SpecConfig(**{**c.to_dict(), "num_speculative_tokens": 3}))

        base_counts = self._sample_first_tokens(base_fn, cfg, self.N_SAMPLES)
        spec_counts = self._sample_first_tokens(spec_fn, cfg, self.N_SAMPLES)

        # sanity: the perturbed draft must cause at least some rejections
        # (checked separately below); here test both engines against p_exact
        for name, counts in [("baseline", base_counts), ("speculative", spec_counts)]:
            stat = 0.0
            total_drawn = sum(counts.values())
            assert total_drawn == self.N_SAMPLES
            outside = sum(v for k, v in counts.items() if k not in support)
            assert outside == 0, f"{name}: probability mass outside top-k support"
            for tok in support:
                exp = self.N_SAMPLES * p_exact[tok].item()
                obs = counts.get(tok, 0)
                stat += (obs - exp) ** 2 / exp
            assert _chi2_p_gt(stat, len(support) - 1), \
                f"{name}: chi2={stat:.1f} rejects the hypothesis that samples come from the target"

    def test_speculative_actually_rejects_sometimes(self, tiny_model, tiny_tokenizer):
        """Guard for the test above: with an identical draft nothing is
        rejected and distribution preservation would be trivially true."""
        draft = BiasedModel(tiny_model, [464], bias=20.0)
        cfg = SpecConfig(max_new_tokens=2, temperature=1.0, top_k=self.TOP_K,
                         num_speculative_tokens=3, seed=5)
        _, stats = speculative_generate(draft, tiny_model, tiny_tokenizer, self.PROMPT, cfg)
        # with K=3 and max_new_tokens=2 -> k_eff=1; run a few seeds, some must reject
        rejections = 0
        for seed in range(12):
            cfg.seed = seed
            _, s = speculative_generate(draft, tiny_model, tiny_tokenizer, self.PROMPT, cfg)
            rejections += s.proposed - s.accepted
        assert rejections > 0, "draft never rejected - test would be vacuous"
