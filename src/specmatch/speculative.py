"""SpecMatch core: speculative decoding from scratch.

One iteration of the loop (prefix length L, K proposals):

    1. draft phase   - the draft model consumes any committed tokens it has
                       not seen yet ("lag": 1 token normally, 2 after a fully
                       accepted round), then extends the sequence one token at
                       a time to propose x_1..x_K, keeping the distribution
                       q_i each proposal was drawn from.
    2. verify phase  - the target model consumes [pending, x_1..x_K] in ONE
                       forward pass. Thanks to the causal mask + KV cache,
                       logits row j is exactly the target distribution
                       p(. | prefix, x_1..x_j) that autoregressive decoding
                       would have computed at step j - that is why a single
                       pass can check K tokens.
    3. accept phase  - sequentially, accept x_i with prob min(1, p_i(x_i)/q_i(x_i)).
                       The first rejection resamples a replacement from
                       norm(max(0, p_i - q_i)) and ends the round. If every
                       proposal is accepted, a bonus token drawn from the
                       final target row extends the round by one more token.
                       (sampling.py explains why this yields exactly the
                       target's distribution.)
    4. cache phase   - both caches are truncated to the committed prefix so
                       that at the top of every iteration:
                           target_cache_len == L - 1   (pending not consumed)
                           draft_cache_len  == L - lag
                       Rejected/never-committed proposal KVs are discarded
                       here; that is the only cleanup rejection needs.
"""

import time
from typing import Optional

import torch

from .cache import forward_with_cache, new_cache, truncate_cache
from .config import SpecConfig
from .generate import (
    GenStats,
    make_generator,
    pick_token,
    synchronize_device,
)
from .sampling import accept_proposal, greedy_accept, shaped_probs


def _shape(logits_row: torch.Tensor, cfg: SpecConfig) -> torch.Tensor:
    return shaped_probs(logits_row, cfg.temperature, cfg.top_k, cfg.top_p)


@torch.no_grad()
def speculative_generate(draft, target, tokenizer, prompt: str, cfg: SpecConfig,
                         device: Optional[torch.device] = None,
                         generator: Optional[torch.Generator] = None):
    """Run speculative decoding. Returns (text, GenStats).

    Both models must share the tokenizer's vocabulary (models.load_pair
    enforces this). `cfg` decoding parameters apply to the target
    distribution; the draft uses the same shaping for its proposals.
    """
    K = cfg.num_speculative_tokens
    if K < 1:
        raise ValueError("num_speculative_tokens must be >= 1")
    device = device or next(target.parameters()).device
    if generator is None:
        generator = make_generator(cfg.seed, device)

    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prompt_len = ids.shape[1]
    if prompt_len == 0:
        raise ValueError("prompt produced no tokens")
    eos = tokenizer.eos_token_id

    # `seq` is the full committed sequence (prompt + generated). Cache lengths
    # are tracked as plain ints into this list; truncate_cache enforces them.
    seq = ids[0].tolist()

    target_cache, draft_cache = new_cache(), new_cache()
    stats = GenStats(prompt_tokens=prompt_len)

    synchronize_device(device)
    t_start = time.perf_counter()

    # Prefill on prompt[:-1] so that even the first iteration goes through the
    # regular path: the final prompt token acts as the first "pending" token.
    if prompt_len > 1:
        prefill = ids[:, :-1]
        forward_with_cache(target, prefill, target_cache)
        forward_with_cache(draft, prefill, draft_cache)
        stats.target_forwards += 1
        stats.draft_forwards += 1
        stats.target_forward_tokens += prompt_len - 1
    target_n = draft_n = prompt_len - 1

    while len(seq) - prompt_len < cfg.max_new_tokens:
        generated_before = len(seq)
        remaining = cfg.max_new_tokens - (len(seq) - prompt_len)
        # worst case a round commits K_eff accepted proposals + 1 bonus/replacement,
        # so cap K_eff to never overshoot max_new_tokens
        k_eff = min(K, remaining - 1)

        if k_eff == 0:
            # last token: a single target-only step on the pending token
            pending = torch.tensor([seq[target_n:]], device=device)
            logits = forward_with_cache(target, pending, target_cache)
            stats.target_forwards += 1
            stats.target_forward_tokens += 1
            tok = pick_token(_shape(logits[0, -1], cfg), cfg, generator)
            if tok != eos:
                seq.append(tok)
                stats.stop_reason = "max_new_tokens"
            else:
                stats.stop_reason = "eos"
            break

        # ---------- 1. draft phase: propose k_eff tokens ----------
        lag = seq[draft_n:]  # committed but not yet consumed by the draft
        logits = forward_with_cache(draft, torch.tensor([lag], device=device), draft_cache)
        draft_n += len(lag)
        stats.draft_forwards += 1
        proposals, q_list = [], []
        q = _shape(logits[0, -1], cfg)
        x = pick_token(q, cfg, generator)
        proposals.append(x)
        q_list.append(q)
        while len(proposals) < k_eff:
            if x == eos:
                break  # proposing past an EOS is pointless
            logits = forward_with_cache(draft, torch.tensor([[x]], device=device), draft_cache)
            draft_n += 1
            stats.draft_forwards += 1
            q = _shape(logits[0, -1], cfg)
            x = pick_token(q, cfg, generator)
            proposals.append(x)
            q_list.append(q)

        # ---------- 2. target verify: one forward over pending + proposals ----------
        verify_in = torch.tensor([seq[target_n:] + proposals], device=device)
        logits = forward_with_cache(target, verify_in, target_cache)
        target_n += verify_in.shape[1]
        stats.target_forwards += 1
        stats.target_forward_tokens += verify_in.shape[1]
        stats.proposed += len(proposals)
        # row j = distribution after consuming pending + proposals[:j],
        # i.e. the distribution proposal j was checked against; the row after
        # the last proposal is the bonus distribution.

        # ---------- 3. accept / reject ----------
        chunk = []  # tokens committed this round
        hit_eos = False
        n_accepted = 0
        for j, x in enumerate(proposals):
            p = _shape(logits[0, j], cfg)
            if cfg.greedy:
                ok, tok = greedy_accept(p, x)
            else:
                ok, tok = accept_proposal(p, q_list[j], x, generator)
            if x == eos and ok:
                # the draft proposed EOS and the target agrees -> stop, EOS is
                # not part of the output
                hit_eos = True
                break
            if ok:
                chunk.append(tok)  # tok == x
                n_accepted += 1
            else:
                # first rejection ends the round; tok is the residual resample
                if tok == eos:
                    hit_eos = True
                else:
                    chunk.append(tok)
                break
        else:
            # every proposal accepted -> draw the bonus token from the target's
            # distribution for one position past the last proposal
            p_bonus = _shape(logits[0, len(proposals)], cfg)
            bonus = pick_token(p_bonus, cfg, generator)
            if bonus == eos:
                hit_eos = True
            else:
                chunk.append(bonus)

        stats.accepted += n_accepted
        stats.accepted_per_iter.append(n_accepted)
        seq.extend(chunk)
        if hit_eos:
            stats.stop_reason = "eos"
        if stats.ttft is None and len(seq) > generated_before:
            stats.ttft = time.perf_counter() - t_start

        # ---------- 4. cache bookkeeping ----------
        new_len = len(seq)
        # target consumed pending + all proposals; keep only the committed
        # prefix minus the new pending token
        target_n = new_len - 1
        truncate_cache(target_cache, target_n)
        # draft consumed lag + proposals[:-1]; on rejection that overshoots the
        # committed prefix and must be cut back, after a full accept it is
        # already 2 short (lag for next round = [last proposal, bonus])
        draft_n = min(draft_n, new_len - 1)
        truncate_cache(draft_cache, draft_n)

        if hit_eos or new_len - prompt_len >= cfg.max_new_tokens:
            break

    synchronize_device(device)
    stats.wall_time = time.perf_counter() - t_start
    stats.new_tokens = len(seq) - prompt_len
    stats.token_ids = seq[prompt_len:]
    text = tokenizer.decode(stats.token_ids, skip_special_tokens=True)
    return text, stats
