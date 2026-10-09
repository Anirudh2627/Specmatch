"""Speculative decoding correctness tests.

Two layers:
1. Deterministic tests with FakeModel (scripted one-hot distributions): every
   accept/reject/EOS path is exercised exactly, including K=1, early
   rejection, all-accepted, bonus tokens and cache truncation.
2. Integration tests with the tiny real GPT-2: greedy speculative output must
   be token-identical to greedy baseline output for ANY draft model (that is
   the correctness contract), and cache end-states must agree with a fresh
   forward over the committed sequence.
"""

import copy

import pytest
import torch

from specmatch.config import GenConfig, SpecConfig
from specmatch.generate import baseline_generate
from specmatch.speculative import speculative_generate

from conftest import BiasedModel, FakeModel

A, B, C, EOS = 64, 3211, 900, 50256  # arbitrary ids < vocab; EOS = gpt2's


def table(*tokens):
    return list(tokens)


class TestWithFakeModels:
    """Draft and target are scripted token tables; outputs are exact."""

    def run(self, draft_table, target_table, prompt_ids, cfg):
        # prompt is fed as raw ids through a tiny detour: encode a string,
        # then let the tables define every continuation deterministically.
        draft = FakeModel(draft_table)
        target = FakeModel(target_table)
        return draft, target

    def test_all_accepted_when_draft_matches_target(self, tiny_tokenizer):
        # table[n] = token the models want after consuming n tokens
        toks = [A, B, C, A, B, C, A, B, C, A]
        prompt = "hello"
        prompt_len = len(tiny_tokenizer(prompt)["input_ids"])
        # tables indexed by absolute position; prompt occupies [0, prompt_len)
        full = [0] * prompt_len + toks + [A] * 10
        draft, target = FakeModel(full), FakeModel(full)
        cfg = SpecConfig(max_new_tokens=6, temperature=0.0, num_speculative_tokens=3)
        text, stats = speculative_generate(draft, target, tiny_tokenizer, prompt, cfg)
        assert stats.token_ids == toks[:6]
        assert stats.acceptance_rate == 1.0
        assert stats.proposed > 0

    def test_first_proposal_rejected_yields_target_token(self, tiny_tokenizer):
        prompt = "hello"
        prompt_len = len(tiny_tokenizer(prompt)["input_ids"])
        # draft believes the next tokens are B,B,B...; target wants A,A,A...
        draft_table = [0] * prompt_len + [B] * 20
        target_table = [0] * prompt_len + [A] * 20
        draft, target = FakeModel(draft_table), FakeModel(target_table)
        cfg = SpecConfig(max_new_tokens=5, temperature=0.0, num_speculative_tokens=4)
        text, stats = speculative_generate(draft, target, tiny_tokenizer, prompt, cfg)
        # every round: all 4 proposals rejected at the first one, replacement = A
        assert stats.token_ids == [A] * 5
        assert stats.accepted == 0
        assert stats.stop_reason == "max_new_tokens"

    def test_partial_acceptance_mixed_tables(self, tiny_tokenizer):
        prompt = "hello"
        prompt_len = len(tiny_tokenizer(prompt)["input_ids"])
        # target continuation: A B C A B C ...; draft agrees at some positions
        target_cont = [A, B, C] * 7
        draft_cont = [A, B, A, A, B, A, A, B, C, A, B, C, A, B, A, A, B, C, A, B, C]
        draft, target = FakeModel([0] * prompt_len + draft_cont), FakeModel([0] * prompt_len + target_cont)
        cfg = SpecConfig(max_new_tokens=8, temperature=0.0, num_speculative_tokens=3)
        text, stats = speculative_generate(draft, target, tiny_tokenizer, prompt, cfg)
        # contract: greedy speculative == what the target would produce alone
        _, base_stats = baseline_generate(target, tiny_tokenizer, prompt,
                                          GenConfig(max_new_tokens=8, temperature=0.0))
        assert stats.token_ids == base_stats.token_ids == target_cont[:8]

    def test_k1_behaves_like_single_token_speculation(self, tiny_tokenizer):
        prompt = "hi"
        prompt_len = len(tiny_tokenizer(prompt)["input_ids"])
        cont = [A, B, C, A, B]
        m = [0] * prompt_len + cont * 2
        draft, target = FakeModel(m), FakeModel(m)
        cfg = SpecConfig(max_new_tokens=5, temperature=0.0, num_speculative_tokens=1)
        _, stats = speculative_generate(draft, target, tiny_tokenizer, prompt, cfg)
        assert stats.token_ids == cont

    def test_eos_in_target_stops_generation(self, tiny_tokenizer):
        prompt = "hello"
        prompt_len = len(tiny_tokenizer(prompt)["input_ids"])
        cont = [A, B, EOS, C, C, C]
        m = [0] * prompt_len + cont + [C] * 10
        draft, target = FakeModel(m), FakeModel(m)
        cfg = SpecConfig(max_new_tokens=6, temperature=0.0, num_speculative_tokens=2)
        _, stats = speculative_generate(draft, target, tiny_tokenizer, prompt, cfg)
        assert stats.token_ids == [A, B]
        assert stats.stop_reason == "eos"
        assert EOS not in stats.token_ids

    def test_eos_proposed_by_draft_but_rejected_continues(self, tiny_tokenizer):
        prompt = "hello"
        prompt_len = len(tiny_tokenizer(prompt)["input_ids"])
        # draft proposes EOS early; target never wants EOS -> must be rejected
        draft_cont = [A, EOS, EOS, EOS, EOS, EOS, EOS, EOS]
        target_cont = [A, B, C, A, B, C, A, B]
        draft = FakeModel([0] * prompt_len + draft_cont)
        target = FakeModel([0] * prompt_len + target_cont)
        cfg = SpecConfig(max_new_tokens=6, temperature=0.0, num_speculative_tokens=2)
        _, stats = speculative_generate(draft, target, tiny_tokenizer, prompt, cfg)
        assert stats.token_ids == target_cont[:6]
        assert stats.stop_reason == "max_new_tokens"

    def test_stats_accounting(self, tiny_tokenizer):
        prompt = "hello"
        prompt_len = len(tiny_tokenizer(prompt)["input_ids"])
        m = [0] * prompt_len + [A, B, C] * 5
        draft, target = FakeModel(m), FakeModel(m)
        K = 3
        cfg = SpecConfig(max_new_tokens=7, temperature=0.0, num_speculative_tokens=K)
        _, stats = speculative_generate(draft, target, tiny_tokenizer, prompt, cfg)
        assert stats.new_tokens == 7 == len(stats.token_ids)
        assert stats.accepted <= stats.proposed
        # each round proposes at most K and commits accepted+1 tokens
        assert stats.new_tokens <= len(stats.accepted_per_iter) * (K + 1) + 1
        # target forwards: 1 prefill + 1 per round (+ maybe final single step)
        assert stats.target_forwards >= 2
        assert stats.mean_tokens_per_target_forward > 1.0  # the whole point

    def test_invalid_k_rejected(self, tiny_tokenizer):
        cfg = SpecConfig(num_speculative_tokens=0)
        with pytest.raises(ValueError):
            speculative_generate(FakeModel([0] * 50), FakeModel([0] * 50),
                                 tiny_tokenizer, "hi", cfg)


class TestWithTinyRealModel:
    """End-to-end on a real (tiny, random-weight) transformer."""

    def test_greedy_identical_to_baseline_when_draft_is_target(self, tiny_model, tiny_tokenizer):
        draft = copy.deepcopy(tiny_model)
        cfg = SpecConfig(max_new_tokens=12, temperature=0.0, num_speculative_tokens=4)
        prompt = "The meaning of life is"
        _, spec_stats = speculative_generate(draft, tiny_model, tiny_tokenizer, prompt, cfg)
        _, base_stats = baseline_generate(tiny_model, tiny_tokenizer, prompt,
                                          GenConfig(max_new_tokens=12, temperature=0.0))
        assert spec_stats.token_ids == base_stats.token_ids
        # identical draft/target -> every proposal must be accepted
        assert spec_stats.acceptance_rate == 1.0

    @pytest.mark.parametrize("K", [1, 2, 4, 8])
    def test_greedy_identical_for_any_k(self, tiny_model, tiny_tokenizer, K):
        # biased draft: over-proposes token 464, so the target rejects it
        # whenever its argmax differs - exercising the reject path for every K
        draft = BiasedModel(tiny_model, [464], bias=20.0)
        prompt = "Once upon a time"
        cfg = SpecConfig(max_new_tokens=10, temperature=0.0, num_speculative_tokens=K)
        _, spec_stats = speculative_generate(draft, tiny_model, tiny_tokenizer, prompt, cfg)
        _, base_stats = baseline_generate(tiny_model, tiny_tokenizer, prompt,
                                          GenConfig(max_new_tokens=10, temperature=0.0))
        assert spec_stats.token_ids == base_stats.token_ids, \
            f"K={K}: {spec_stats.token_ids} != {base_stats.token_ids}"

    def test_mismatched_draft_still_matches_baseline_greedy(self, tiny_model, tiny_tokenizer):
        """The correctness contract: greedy speculative output equals greedy
        target output no matter how bad the draft model is."""
        junk_draft = BiasedModel(tiny_model, [464, 13584, 21516], bias=20.0)
        prompt = "In a galaxy far away"
        cfg = SpecConfig(max_new_tokens=10, temperature=0.0, num_speculative_tokens=4)
        _, spec_stats = speculative_generate(junk_draft, tiny_model, tiny_tokenizer, prompt, cfg)
        _, base_stats = baseline_generate(tiny_model, tiny_tokenizer, prompt,
                                          GenConfig(max_new_tokens=10, temperature=0.0))
        assert spec_stats.token_ids == base_stats.token_ids
        assert spec_stats.accepted < spec_stats.proposed, "test should exercise rejections"

    def test_cache_end_state_consistent(self, tiny_model, tiny_tokenizer, monkeypatch):
        """After generation, the target cache must contain exactly the
        committed sequence minus the pending token: continuing from it must
        give the same next-token distribution as a fresh forward."""
        import specmatch.speculative as spec_mod
        from specmatch.cache import DynamicCache

        created = []

        def tracking_new_cache():
            c = DynamicCache()
            created.append(c)
            return c

        monkeypatch.setattr(spec_mod, "new_cache", tracking_new_cache)
        draft = copy.deepcopy(tiny_model)
        prompt = "Cache consistency check"
        cfg = SpecConfig(max_new_tokens=9, temperature=0.0, num_speculative_tokens=3)
        text, stats = speculative_generate(draft, tiny_model, tiny_tokenizer, prompt, cfg)
        target_cache = created[0]  # target cache is created first

        ids = tiny_tokenizer(prompt, return_tensors="pt")["input_ids"]
        full = torch.cat([ids, torch.tensor([stats.token_ids])], dim=1)
        # committed length L; cache should hold L-1 tokens (pending not consumed)
        L = full.shape[1]
        assert target_cache.get_seq_length() in (L - 1, L)
        # continue from the tracked cache with the last committed token(s) and
        # compare against a fresh forward over the whole sequence
        n_missing = L - target_cache.get_seq_length()
        with torch.no_grad():
            cont = tiny_model(full[:, target_cache.get_seq_length():],
                              past_key_values=target_cache).logits[0, -1]
            fresh = tiny_model(full).logits[0, -1]
        assert torch.allclose(cont, fresh, atol=1e-4), \
            f"cache diverged: max diff {(cont - fresh).abs().max()}"

    def test_output_shapes_and_text(self, tiny_model, tiny_tokenizer):
        draft = copy.deepcopy(tiny_model)
        cfg = SpecConfig(max_new_tokens=8, temperature=0.0, num_speculative_tokens=4)
        text, stats = speculative_generate(draft, tiny_model, tiny_tokenizer, "Hi", cfg)
        assert isinstance(text, str) and len(text) > 0
        assert len(stats.token_ids) == stats.new_tokens == 8
        assert stats.ttft is not None and 0 < stats.ttft <= stats.wall_time
        assert stats.prompt_tokens == len(tiny_tokenizer("Hi")["input_ids"])
