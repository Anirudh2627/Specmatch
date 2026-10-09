"""Tests for the sampling math: distribution shaping, acceptance rule,
residual distribution. Pure tensor math - no models needed."""

import torch

from specmatch.sampling import (
    accept_proposal,
    greedy_accept,
    greedy_token,
    residual_distribution,
    sample_token,
    shaped_probs,
)

V = 6  # tiny synthetic vocab


def make_logits(values):
    return torch.tensor(values, dtype=torch.float32)


class TestShapedProbs:
    def test_greedy_is_onehot_at_argmax(self):
        p = shaped_probs(make_logits([1.0, 5.0, 2.0, 0.0, 0.0, 0.0]), temperature=0.0)
        assert p.argmax().item() == 1
        assert p.sum().item() == 1.0
        assert p[0].item() == 0.0

    def test_temperature_softmax(self):
        logits = make_logits([2.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        p = shaped_probs(logits, temperature=2.0)
        expected = torch.softmax(logits / 2.0, dim=-1)
        assert torch.allclose(p, expected, atol=1e-6)
        # higher temperature -> flatter
        p_hot = shaped_probs(logits, temperature=0.5)
        assert p_hot.max() > p.max()

    def test_top_k_truncates_support(self):
        p = shaped_probs(make_logits([3.0, 2.0, 1.0, 0.5, 0.1, 0.0]), temperature=1.0, top_k=2)
        assert (p > 0).sum().item() == 2
        assert p[0] > p[1] > 0
        assert torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-6)

    def test_top_p_nucleus(self):
        # sorted probs: 0.5, 0.3, ... -> top_p=0.75 must keep exactly the top 2
        logits = make_logits([10.0, 9.6, 5.0, 1.0, 0.0, -5.0])
        p = shaped_probs(logits, temperature=1.0, top_p=0.75)
        assert (p > 0).sum().item() == 2
        assert p.sum().item() >= 0.75 - 1e-6

    def test_bf16_logits_upcast_to_fp32(self):
        logits = make_logits([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]).to(torch.bfloat16)
        p = shaped_probs(logits, temperature=1.0)
        assert p.dtype == torch.float32


class TestResidual:
    def test_residual_is_normalized_relu_p_minus_q(self):
        p = torch.tensor([0.5, 0.3, 0.2, 0.0, 0.0, 0.0])
        q = torch.tensor([0.2, 0.5, 0.1, 0.1, 0.1, 0.0])
        r = residual_distribution(p, q)
        expected = torch.tensor([0.3, 0.0, 0.1, 0.0, 0.0, 0.0])
        expected = expected / expected.sum()
        assert torch.allclose(r, expected, atol=1e-6)

    def test_identical_distributions_fall_back_to_p(self):
        p = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.0, 0.0])
        r = residual_distribution(p, p.clone())
        assert torch.allclose(r, p)

    def test_residual_excludes_token_dominated_by_q(self):
        # x where q(x) > p(x) must get zero residual mass
        p = torch.tensor([0.1, 0.9, 0.0, 0.0, 0.0, 0.0])
        q = torch.tensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])
        r = residual_distribution(p, q)
        assert r[0].item() == 0.0
        assert r[1].item() == 1.0


class TestAcceptProposal:
    def test_p_dominates_q_always_accepts(self):
        p = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q = torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
        gen = torch.Generator().manual_seed(0)
        for _ in range(50):
            ok, tok = accept_proposal(p, q, 0, gen)
            assert ok and tok == 0

    def test_p_zero_always_rejects(self):
        p = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        q = torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
        gen = torch.Generator().manual_seed(0)
        for _ in range(50):
            ok, tok = accept_proposal(p, q, 0, gen)
            assert not ok
            assert tok == 1  # residual is one-hot at token 1

    def test_zero_q_with_positive_p_accepts(self):
        # ratio p/q = +inf -> min(1, inf) = 1
        p = torch.tensor([0.4, 0.6, 0.0, 0.0, 0.0, 0.0])
        q = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        gen = torch.Generator().manual_seed(0)
        ok, tok = accept_proposal(p, q, 0, gen)
        assert ok and tok == 0

    def test_acceptance_frequency_matches_min_one_ratio(self):
        # P(accept x=0) = min(1, p0/q0) = min(1, 0.4/0.8) = 0.5
        p = torch.tensor([0.4, 0.6, 0.0, 0.0, 0.0, 0.0])
        q = torch.tensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])
        gen = torch.Generator().manual_seed(42)
        n = 20000
        accepts = sum(accept_proposal(p, q, 0, gen)[0] for _ in range(n))
        freq = accepts / n
        assert abs(freq - 0.5) < 0.02, freq

    def test_output_distribution_equals_target(self):
        """The core guarantee: propose x ~ q, run the accept rule, and the
        resulting token distribution must equal p (total variation check)."""
        p = torch.tensor([0.45, 0.30, 0.15, 0.07, 0.02, 0.01])
        q = torch.tensor([0.10, 0.20, 0.30, 0.25, 0.10, 0.05])
        gen = torch.Generator().manual_seed(7)
        n = 60000
        counts = torch.zeros(V)
        for _ in range(n):
            x = sample_token(q, gen)
            _, tok = accept_proposal(p, q, x, gen)
            counts[tok] += 1
        emp = counts / n
        tv = 0.5 * (emp - p).abs().sum().item()
        assert tv < 0.01, f"TV distance {tv} too large: {emp} vs {p}"

    def test_acceptance_rate_equals_one_minus_tv(self):
        # E[accept] = sum_x min(p(x), q(x)) = 1 - TV(p, q)
        p = torch.tensor([0.45, 0.30, 0.15, 0.07, 0.02, 0.01])
        q = torch.tensor([0.10, 0.20, 0.30, 0.25, 0.10, 0.05])
        expected = torch.minimum(p, q).sum().item()
        gen = torch.Generator().manual_seed(3)
        n = 40000
        accepts = 0
        for _ in range(n):
            x = sample_token(q, gen)
            ok, _ = accept_proposal(p, q, x, gen)
            accepts += ok
        assert abs(accepts / n - expected) < 0.01

    def test_seeded_generator_is_reproducible(self):
        p = torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0, 0.0])
        q = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
        runs = []
        for _ in range(2):
            gen = torch.Generator().manual_seed(123)
            runs.append([accept_proposal(p, q, 0, gen) for _ in range(20)])
        assert runs[0] == runs[1]


class TestGreedy:
    def test_greedy_accept_only_argmax(self):
        p = shaped_probs(make_logits([1.0, 9.0, 3.0, 0.0, 0.0, 0.0]), temperature=0.0)
        assert greedy_accept(p, 1) == (True, 1)
        ok, tok = greedy_accept(p, 0)
        assert not ok and tok == 1

    def test_greedy_token(self):
        p = shaped_probs(make_logits([0.0, 0.0, 7.0, 1.0, 0.0, 0.0]), temperature=0.0)
        assert greedy_token(p) == 2
